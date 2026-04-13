"""
Native Dynamic Router - 使用 sglang 原生 FusedMoE kernel 的动态路由.

职责单一：执行专家计算
- 根据调度器的执行计划，执行 GPU/CPU 专家计算
- 异步触发预测并通知调度器
- 不包含调度决策逻辑（由通用 ExpertScheduler 负责）

调度流程：
- Layer N 推理时：
  1. 获取当前层实际激活的专家
  2. 调用调度器做最终决策 (finalize_decision)
  3. 根据决策执行 GPU/CPU 专家计算
  4. 异步预测下一层并通知调度器 (receive_prediction)

性能对比：
- 朴素实现 (F.linear 循环): ~2-4ms per layer
- 原生 FusedMoE kernel: ~0.5ms per layer
- 加速比: 4-8x
"""

from concurrent.futures import ThreadPoolExecutor
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
import threading
import torch
import torch.nn.functional as F
import numpy as np

from ..logger import log_once, append_log
from ..core.expert_scheduler import ExpertScheduler
from .native_gpu_cache import get_native_cache
from .native_scheduler import get_native_scheduler

if TYPE_CHECKING:
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput, CombineInput

# ============================================================
# Native Dynamic Router
# ============================================================

# 全局线程池用于异步predict
_predict_executor: Optional[ThreadPoolExecutor] = None
_predict_executor_lock = threading.Lock()

# 全局线程池用于异步prefetch (IO密集型)
_prefetch_executor: Optional[ThreadPoolExecutor] = None
_prefetch_executor_lock = threading.Lock()

# ========== 优化: 专用 CUDA stream 用于 H2D 传输 ==========
# 使用独立 stream 可以让 H2D 传输与 GPU 计算并行
_prefetch_stream: Optional[torch.cuda.Stream] = None
_prefetch_stream_lock = threading.Lock()


def _get_prefetch_stream() -> torch.cuda.Stream:
    """获取或创建用于 prefetch 的 CUDA stream (低优先级)."""
    global _prefetch_stream
    with _prefetch_stream_lock:
        if _prefetch_stream is None:
            # 使用低优先级 stream，减少对计算的影响
            # priority: 数值越小优先级越高，默认为 0
            # 通常 GPU 支持 priority 范围: (low, high) = (-1, 0) 或更宽
            _prefetch_stream = torch.cuda.Stream(priority=-1)  # 低优先级
        return _prefetch_stream


def _get_predict_executor() -> ThreadPoolExecutor:
    """获取或创建predict线程池."""
    global _predict_executor
    with _predict_executor_lock:
        if _predict_executor is None:
            _predict_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")
        return _predict_executor


def _get_prefetch_executor() -> ThreadPoolExecutor:
    """获取或创建prefetch线程池（IO密集型操作）."""
    global _prefetch_executor
    with _prefetch_executor_lock:
        if _prefetch_executor is None:
            _prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")
        return _prefetch_executor


class NativeDynamicRouter:
    """
    使用原生 FusedMoE kernel 的动态路由器.
    
    职责：执行专家计算，不做调度决策。
    
    工作流程：
    1. 获取当前层实际激活的专家
    2. 调用通用调度器做最终决策
    3. 根据决策执行 GPU/CPU 专家计算
    4. 异步预测下一层并通知调度器
    """
    
    def __init__(
        self,
        scheduler: Optional[ExpertScheduler] = None,
        log_path: Optional[str] = None,
        enable_prefetch: bool = False,  # 默认禁用 prefetch
        prefetch_mode: str = "batch_gap",  # prefetch 模式: "layer", "batch_gap", "disabled"
    ):
        self.log_path = log_path
        self._lock = threading.Lock()
        
        # 通用调度器（外部注入或使用全局实例）
        self._scheduler = scheduler
        
        # 预测函数（外部设置）
        self._predict_fn: Optional[Callable] = None
        
        # deferral 警告标记
        self._warned_deferral = False
        self._deferral_enabled = False
        
        # ========== Prefetch 配置 ==========
        # enable_prefetch: 是否启用专家预加载
        # prefetch_mode:
        #   - "disabled": 完全禁用
        #   - "layer": 每层都尝试 prefetch (影响吞吐)
        #   - "batch_gap": 仅在 batch 间隙 prefetch (推荐)
        self._enable_prefetch = enable_prefetch
        self._prefetch_mode = prefetch_mode
        
        self.mlp_end_ms = {}
        
        # 专家热度计数: layer_idx -> {expert_idx: count}
        self._expert_heat: Dict[int, Dict[int, int]] = {}
        self._heat_lock = threading.Lock()
        # 计算时间统计: per-layer 总耗时与计数（ms）
        # _compute_time_totals: layer_idx -> {'total_ms': float, 'count': int}
        # _layer_total_time_totals: layer_idx -> {'total_ms': float, 'count': int}
        # _inter_layer_time_totals: layer_idx -> {'total_ms': float, 'count': int}
        #   记录两次 native_dynamic_apply 调用之间的间隔（即 Attention+LayerNorm 等非MLP部分耗时）
        self._compute_time_totals: Dict[int, Dict[str, float]] = {}
        self._layer_total_time_totals: Dict[int, Dict[str, float]] = {}
        self._inter_layer_time_totals: Dict[int, Dict[str, float]] = {}
        # 上一层 MLP 结束时刻（ms），用于计算层间间隔
        self._last_mlp_end_ms: Optional[float] = None
        # 提前调度结果缓存: layer_idx -> (topk_ids, topk_weights, reroute_stats)
        self._precomputed_decisions: Dict[int, tuple] = {}
        self._decisions_lock = threading.Lock()
        self._time_stats_lock = threading.Lock()
        # Request-level aggregated stats
        self._request_stats = {
            'total_layers': 0,
            'total_compute_ms': 0.0,
            'total_gpu_compute_ms': 0.0,
            'total_cpu_compute_ms': 0.0,
            'total_layer_time_ms': 0.0,
            'total_decision_ms': 0.0,        # 调度决策总耗时
            'total_gpu_positions': 0,        # GPU token-expert 位置总数
            'total_cpu_positions': 0,        # CPU token-expert 位置总数
            'gpu_load_change_ratio': [],  # 存储每层的GPU负载变化比例（正数=增加，负数=减少）
            'cpu_load_change_ratio': [],  # 存储每层的CPU负载变化比例（正数=增加，负数=减少）
            'expert_migration_count': 0,     # 专家迁移次数
            'reroute_count': 0,              # 重路由层数
            'total_position_reroutes': 0,    # 总位置重路由数
            'total_positions': 0,            # 总位置数 (T*k)
            'total_token_reroutes': 0,       # 被重路由的token数
            'total_inter_layer_ms': 0.0,     # 层间间隔总耗时（两次MLP调用之间）
            # ===== 高分专家负载占比（按层） =====
            # 每层记录若干次（prefill 1次，decode 每步1次），最终取每层均值
            # layer_hs_pct[layer_idx]   = [pct_step0, pct_step1, ...]  (各 forward 步的值)
            # layer_conc_ratio[layer_idx] = [conc_step0, conc_step1, ...]
            'layer_hs_pct': {},     # {layer_idx: [pct_values...]}，每层高分专家负载占比
            'layer_conc_ratio': {}, # {layer_idx: [conc_values...]}，每层集中度比值
            'random_expected_pct': 0.0,  # 随机期望占比（固定值 HIGH_K/N*100，只需存一次）
        }
        self._request_stats_lock = threading.Lock()
        # ========== 专家评分统计 ==========
        self._expert_scores: Dict[int, Dict[int, float]] = {}
        self._expert_scores_lock = threading.Lock()
        # gamma: 动态分数权重 (0-1), 1-gamma 为历史分数权重
        self._score_gamma: float = 0.3  # 默认30%动态分数 + 70%历史分数
        self._score_sigma: float = 0.5  # 默认50%平均分 + 50%P75分数
        self._score_beta: float = 0.5   # 默认50%平均分 + 50%历史分数

        # 记录已打印的 layer_hs_pct 长度，避免同一请求内重复打印相同列表
        self._last_layer_hs_log_len: Dict[int, int] = {}

        log_once('native_router_init', f'NativeDynamicRouter initialized (prefetch={enable_prefetch}, mode={prefetch_mode})')
    
    def set_scheduler(self, scheduler: ExpertScheduler):
        """设置调度器."""
        self._scheduler = scheduler
    
    def get_scheduler(self) -> Optional[ExpertScheduler]:
        """获取调度器（优先使用注入的，否则使用 native 全局的）."""
        return self._scheduler or get_native_scheduler()
    
    def set_predict_fn(self, fn: Callable):
        """设置预测函数."""
        self._predict_fn = fn
    
    def _update_expert_heat(self, layer_idx: int, expert_ids: List[int]):
        """异步更新专家热度统计."""
        try:
            with self._heat_lock:
                layer_heat = self._expert_heat.setdefault(layer_idx, {})
                for eid in expert_ids:
                    layer_heat[eid] = layer_heat.get(eid, 0) + 1
        except Exception:
            pass
    
    def _update_stats_async(
        self,
        layer_idx: int,
        compute_time_ms: float,
        layer_total_ms: float,
        gpu_compute_ms: float,
        cpu_compute_ms: float,
    ):
        """异步更新统计信息和日志."""
        try:
            # 记录耗时
            self._record_compute_time(layer_idx, compute_time_ms)
            self._record_layer_total_time(layer_idx, layer_total_ms)
            
            # 日志
            append_log(
                f"L{layer_idx} compute: {compute_time_ms:.2f}ms "
                f"(GPU {gpu_compute_ms:.2f}ms, CPU {cpu_compute_ms:.2f}ms), "
                f"total: {layer_total_ms:.2f}ms",
                self.log_path,
                level=2
            )
        except Exception:
            pass
    
    def _post_layer_stats_async(
        self,
        layer_idx: int,
        compute_time_ms: float,
        layer_total_ms: float,
        gpu_compute_ms: float,
        cpu_compute_ms: float,
        decision_time_ms: float,
        final_gpu_load: int,
        final_cpu_load: int,
        orig_gpu_load: int,
        orig_cpu_load: int,
        reroute_stats: dict,
        has_rerouting: bool,
        migration_count: int,
        inter_layer_ms: float = 0.0,
    ):
        """异步执行所有层级统计和日志（不在推理主路径上）."""
        try:
            # 聚合请求级统计
            self._aggregate_layer_stats(
                compute_ms = compute_time_ms,
                gpu_compute_ms=gpu_compute_ms,
                cpu_compute_ms=cpu_compute_ms,
                decision_time_ms=decision_time_ms,
                gpu_load=final_gpu_load,
                cpu_load=final_cpu_load,
                orig_gpu_load=orig_gpu_load,
                orig_cpu_load=orig_cpu_load,
                reroute_stats=reroute_stats,
                layer_total_ms=layer_total_ms,
                has_rerouting=has_rerouting,
                migration_count=migration_count,
                inter_layer_ms=inter_layer_ms,
            )
            # 记录每层耗时
            self._record_compute_time(layer_idx, compute_time_ms)
            self._record_layer_total_time(layer_idx, layer_total_ms)
            if inter_layer_ms > 0:
                self._record_inter_layer_time(layer_idx, inter_layer_ms)
            # 日志
            append_log(
                f"L{layer_idx} compute: {compute_time_ms:.2f}ms "
                f"(GPU {gpu_compute_ms:.2f}ms, CPU {cpu_compute_ms:.2f}ms), "
                f"total: {layer_total_ms:.2f}ms, inter_layer: {inter_layer_ms:.2f}ms",
                self.log_path,
                level=2
            )
        except Exception:
            pass

    def _update_expert_scores_async(self, layer_idx: int, router_probs: torch.Tensor):
        """异步更新专家评分."""
        try:
            expert_scores = self._compute_expert_scores(layer_idx, router_probs)
            # 只在 level=3 时输出详细评分信息
            if self.log_path:
                # 简化日志：只显示更新的专家数量，不显示完整评分字典
                append_log(
                    f'NativeRouter: updated {len(expert_scores)} expert scores for layer {layer_idx}',
                    self.log_path,
                    level=3
                )
        except Exception as e:
            if self.log_path:
                append_log(f'NativeRouter: failed to update expert scores for layer {layer_idx}: {e}', self.log_path)

    # 高分专家 Top-K
    HIGH_SCORE_TOP_K: int = 4

    def _record_score_load_ratio(
        self,
        layer_idx: int,
        hs_pct: float,
        conc_ratio: float,
        random_pct: float,
        high_k: int,
        num_experts: int,
        total_load: int,
    ):
        """记录高分专家负载占比统计（仅做轻量更新与日志）。"""
        try:
            with self._request_stats_lock:
                self._request_stats['layer_hs_pct'].setdefault(layer_idx, []).append(hs_pct)
                self._request_stats['layer_conc_ratio'].setdefault(layer_idx, []).append(conc_ratio)
                self._request_stats['random_expected_pct'] = random_pct  # 常量，覆盖写即可

            if total_load > 0:
                append_log(
                    f"L{layer_idx} high-score load: "
                    f"top{high_k}/{num_experts} experts hold {hs_pct:.1f}% load "
                    f"(random={random_pct:.1f}%, conc={conc_ratio:.3f}x)",
                    self.log_path,
                    level=2
                )
        except Exception as e:
            if self.log_path:
                append_log(f'NativeRouter: score-load ratio error at L{layer_idx}: {e}', self.log_path)

    def _record_score_load_ratio_async(
        self,
        layer_idx: int,
        router_probs: torch.Tensor,
        orig_topk_ids: torch.Tensor,
    ):
        try:
            _, num_experts = router_probs.shape
            high_k = self.HIGH_SCORE_TOP_K

            expert_score = router_probs.sum(dim=0)
            high_score_indices = expert_score.topk(min(high_k, num_experts)).indices
            expert_load = torch.bincount(orig_topk_ids.view(-1), minlength=num_experts)

            high_score_load = expert_load[high_score_indices].sum()
            all_load = expert_load.sum()

            append_log(f"Layer {layer_idx}:\n"
                       f"expert scores: {expert_score}\n"
                       f"high_score_indices {high_score_indices}\n"
                       f"expert_load: {expert_load}\n"
                       f"high score load: {high_score_load}, all load: {all_load}",
                       self.log_path, level=3)

            agg = torch.stack([high_score_load.float(), all_load.float()]).cpu()
            hs_load = int(agg[0].item())
            tot_load = int(agg[1].item())

            if tot_load > 0:
                hs_pct = hs_load / tot_load * 100
                random_pct = min(high_k, num_experts) / num_experts * 100
                conc_ratio = hs_pct / random_pct if random_pct > 0 else 0.0
            else:
                hs_pct = random_pct = conc_ratio = 0.0

            self._record_score_load_ratio(
                layer_idx,
                hs_pct,
                conc_ratio,
                random_pct,
                high_k,
                num_experts,
                tot_load,
            )
        except Exception as e:
            if self.log_path:
                append_log(f'NativeRouter: score-load ratio error at L{layer_idx}: {e}', self.log_path)

    def _aggregate_layer_stats(
        self,
        compute_ms: float,
        gpu_compute_ms: float,
        cpu_compute_ms: float,
        decision_time_ms: float,
        gpu_load: int,
        cpu_load: int,
        orig_gpu_load: int = 0,
        orig_cpu_load: int = 0,
        reroute_stats: dict = {},
        layer_total_ms: float = 0.0,
        has_rerouting: bool = False,
        migration_count: int = 0,
        inter_layer_ms: float = 0.0,
    ):
        """
        聚合层级统计信息
        
        Args:
            gpu_compute_ms: GPU 计算耗时（毫秒）
            cpu_compute_ms: CPU 计算耗时（毫秒）
            decision_time_ms: 调度决策耗时（毫秒）
            gpu_load: GPU token assignments 数量（调度后）
            cpu_load: CPU token assignments 数量（调度后）
            orig_gpu_load: 原始GPU负载（调度前）
            orig_cpu_load: 原始CPU负载（调度前）
            layer_total_ms: 层总耗时
            has_rerouting: 是否发生重路由
            migration_count: 专家迁移数量
            inter_layer_ms: 层间间隔（两次 MLP 调用之间，含 Attention 等）
        """
        try:
            # 更新请求级别统计
            with self._request_stats_lock:
                self._request_stats['total_layers'] += 1
                self._request_stats['total_compute_ms'] += compute_ms
                self._request_stats['total_gpu_compute_ms'] += gpu_compute_ms
                self._request_stats['total_cpu_compute_ms'] += cpu_compute_ms
                self._request_stats['total_layer_time_ms'] += layer_total_ms
                self._request_stats['total_decision_ms'] += decision_time_ms
                self._request_stats['total_gpu_positions'] += gpu_load
                self._request_stats['total_cpu_positions'] += cpu_load
                self._request_stats['total_inter_layer_ms'] += inter_layer_ms
                
                if has_rerouting:
                    self._request_stats['reroute_count'] += 1
                    # 从 reroute_stats 中提取详细统计
                    if reroute_stats:
                        self._request_stats['total_position_reroutes'] += reroute_stats.get('total_reroutes', 0)
                        self._request_stats['total_token_reroutes'] += reroute_stats.get('rerouted_tokens', 0)
                        # 估计total_positions (T*k)，如果reroute_stats有position_reroute_rate可以推算
                        pos_rate = reroute_stats.get('position_reroute_rate') or reroute_stats.get('reroute_rate')
                        if pos_rate and pos_rate > 0:
                            total_pos = reroute_stats.get('total_reroutes', 0) / pos_rate
                            self._request_stats['total_positions'] += int(total_pos)
                
                if migration_count > 0:
                    self._request_stats['expert_migration_count'] += migration_count
                
                # 计算负载变化比例
                if orig_gpu_load + orig_cpu_load > 0:  # 避免除零
                    # GPU负载变化：正数表示增加，负数表示减少
                    gpu_load_change = (gpu_load - orig_gpu_load) / (orig_gpu_load + orig_cpu_load)
                    self._request_stats['gpu_load_change_ratio'].append(gpu_load_change)
                    
                    # CPU负载变化：正数表示增加，负数表示减少
                    cpu_load_change = (cpu_load - orig_cpu_load) / (orig_gpu_load + orig_cpu_load)
                    self._request_stats['cpu_load_change_ratio'].append(cpu_load_change)

                append_log(f"self._request_stats: {self._request_stats}", self.log_path, level=3)
        except Exception:
            pass
    
    def get_expert_scores(self, layer_idx: Optional[int] = None) -> Dict[int, Dict[int, float]]:
        """
        获取专家评分.
        
        Args:
            layer_idx: 层索引，如果为 None 则返回所有层的评分
        
        Returns:
            如果指定 layer_idx，返回该层的评分 {expert_idx: score}
            否则返回所有层的评分 {layer_idx: {expert_idx: score}}
        """
        with self._expert_scores_lock:
            if layer_idx is not None:
                return dict(self._expert_scores.get(layer_idx, {}))
            else:
                return {lid: dict(scores) for lid, scores in self._expert_scores.items()}
    
    def set_score_gamma(self, gamma: float):
        """
        设置动态分数权重 gamma.
        
        Args:
            gamma: 动态分数权重 (0-1)，1-gamma 为历史分数权重
        """
        if not 0 <= gamma <= 1:
            raise ValueError(f'gamma must be in [0, 1], got {gamma}')
        self._score_gamma = gamma
        if self.log_path:
            append_log(f'NativeRouter: score_gamma set to {gamma}', self.log_path)
    
    def get_request_stats(self) -> Dict[str, Any]:
        """获取当前请求的统计信息."""
        with self._request_stats_lock:
            stats = dict(self._request_stats)
            
            # 计算平均值和比例
            if stats['total_layers'] > 0:
                stats['avg_total_compute_ms'] = stats['total_compute_ms'] / stats['total_layers']
                stats['avg_gpu_compute_ms'] = stats['total_gpu_compute_ms'] / stats['total_layers']
                stats['avg_cpu_compute_ms'] = stats['total_cpu_compute_ms'] / stats['total_layers']
                stats['avg_layer_time_ms'] = stats['total_layer_time_ms'] / stats['total_layers']
                stats['avg_decision_ms'] = stats['total_decision_ms'] / stats['total_layers']
                stats['avg_gpu_positions'] = stats['total_gpu_positions'] / stats['total_layers']
                stats['avg_cpu_positions'] = stats['total_cpu_positions'] / stats['total_layers']
                stats['avg_inter_layer_ms'] = stats['total_inter_layer_ms'] / stats['total_layers']
                # 使用position reroute rate
                if stats['total_positions'] > 0:
                    stats['reroute_rate'] = stats['total_position_reroutes'] / stats['total_positions']
                else:
                    # fallback: 使用层级别计数
                    stats['reroute_rate'] = stats['reroute_count'] / stats['total_layers']
            else:
                stats['avg_gpu_compute_ms'] = 0.0
                stats['avg_cpu_compute_ms'] = 0.0
                stats['avg_layer_time_ms'] = 0.0
                stats['avg_decision_ms'] = 0.0
                stats['avg_inter_layer_ms'] = 0.0
                stats['reroute_rate'] = 0.0
            
            # 计算负载变化统计
            if stats['gpu_load_change_ratio']:
                ratios = stats['gpu_load_change_ratio']
                stats['avg_gpu_load_change'] = np.mean(ratios)
                stats['max_gpu_load_change'] = np.max(ratios)
                stats['min_gpu_load_change'] = np.min(ratios)
            else:
                stats['avg_gpu_load_change'] = 0.0
                stats['max_gpu_load_change'] = 0.0
                stats['min_gpu_load_change'] = 0.0
            
            if stats['cpu_load_change_ratio']:
                ratios = stats['cpu_load_change_ratio']
                stats['avg_cpu_load_change'] = np.mean(ratios)
                stats['max_cpu_load_change'] = np.max(ratios)
                stats['min_cpu_load_change'] = np.min(ratios)
            else:
                stats['avg_cpu_load_change'] = 0.0
                stats['max_cpu_load_change'] = 0.0
                stats['min_cpu_load_change'] = 0.0
            
            # 计算高分专家负载占比（按层分布）
            layer_hs_dict   = stats.get('layer_hs_pct', {})    # {layer_idx: [pct...]}
            layer_conc_dict = stats.get('layer_conc_ratio', {}) # {layer_idx: [conc...]}
            random_pct  = stats.get('random_expected_pct', 0.0)
            # 仅在列表长度变化时打印，避免同一请求内重复日志
            for lid, vals in layer_hs_dict.items():
                cur_len = len(vals)
                prev_len = self._last_layer_hs_log_len.get(lid, -1)
                if cur_len != prev_len:
                    # print(f'layer_hs_dict layer {lid}: ', vals)
                    self._last_layer_hs_log_len[lid] = cur_len
            # 每层跨所有 forward 步求均值，得到 {layer_idx: avg_pct}
            layer_pct_avg  = {lid: float(np.mean(v)) for lid, v in sorted(layer_hs_dict.items())}
            layer_conc_avg = {lid: float(np.mean(v)) for lid, v in sorted(layer_conc_dict.items())}
            stats['layer_hs_pct_avg']    = {lid: round(v, 3) for lid, v in layer_pct_avg.items()}
            stats['layer_conc_ratio_avg'] = {lid: round(v, 4) for lid, v in layer_conc_avg.items()}

            layer_pct  = list(layer_pct_avg.values())
            layer_conc = list(layer_conc_avg.values())

            if layer_pct:
                arr = np.array(layer_pct, dtype=np.float32)
                stats['hs_load_pct_mean'] = round(float(np.mean(arr)), 3)
                stats['hs_load_pct_max']  = round(float(np.max(arr)), 3)
                stats['hs_load_pct_std']  = round(float(np.std(arr)), 3)
                stats['hs_load_pct_p75']  = round(float(np.percentile(arr, 75)), 3)
                stats['hs_load_pct_p90']  = round(float(np.percentile(arr, 90)), 3)
                conc_arr = np.array(layer_conc, dtype=np.float32)
                stats['hs_conc_ratio_mean'] = round(float(np.mean(conc_arr)), 4)
                stats['hs_conc_ratio_max']  = round(float(np.max(conc_arr)), 4)
                stats['hs_conc_ratio_std']  = round(float(np.std(conc_arr)), 4)
                stats['random_expected_pct'] = round(random_pct, 3)
            else:
                for k in ('hs_load_pct_mean','hs_load_pct_max','hs_load_pct_std',
                          'hs_load_pct_p75','hs_load_pct_p90',
                          'hs_conc_ratio_mean','hs_conc_ratio_max','hs_conc_ratio_std'):
                    stats[k] = 0.0
                stats['random_expected_pct'] = 0.0
                stats['layer_hs_pct_avg'] = {}
                stats['layer_conc_ratio_avg'] = {}
            
            return stats
    
    def reset_request_stats(self):
        """重置请求级别统计（新请求开始时调用）."""
        with self._request_stats_lock:
            self._request_stats = {
                'total_layers': 0,
                'total_compute_ms': 0.0,
                'total_gpu_compute_ms': 0.0,
                'total_cpu_compute_ms': 0.0,
                'total_layer_time_ms': 0.0,
                'total_decision_ms': 0.0,
                'total_gpu_positions': 0,
                'total_cpu_positions': 0,
                'gpu_load_change_ratio': [],
                'cpu_load_change_ratio': [],
                'expert_migration_count': 0,
                'reroute_count': 0,
                'total_position_reroutes': 0,
                'total_positions': 0,
                'total_token_reroutes': 0,
                'total_inter_layer_ms': 0.0,
                'layer_hs_pct': {},
                'layer_conc_ratio': {},
                'random_expected_pct': 0.0,
            }
            # 重置打印去重状态
            self._last_layer_hs_log_len = {}
    
    def native_dynamic_apply(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """
        使用原生 FusedMoE kernel 的动态路由.
        
        核心流程：
        1. 快速路径检查
        2. 预处理：提取路由信息（GPU操作）
        3. 调度器决策：重路由 + 专家分配
        4. GPU/CPU 并行计算
        5. 后台任务：统计、日志（异步）
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        
        layer_idx = getattr(wrapper.kt_config, 'layer_idx', -1)
        
        # ========== Deferral 警告 ==========
        max_deferred = getattr(wrapper.kt_config, 'max_deferred_experts_per_token', 0) or 0
        if max_deferred > 0 and not self._warned_deferral:
            self._deferral_enabled = True
            self._warned_deferral = True
            log_once(
                'deferral_warning',
                f'WARNING: deferral enabled (max_deferred={max_deferred}). '
                f'Consider --kt-max-deferred-experts-per-token 0 for simpler debugging.'
            )
        
        native_cache = get_native_cache()
        scheduler = self.get_scheduler()
        
        # ========== 快速路径：无调度器且默认映射 ==========
        if scheduler is None:
            if native_cache is None or native_cache.all_layers_default_mapping():
                append_log(f"NativeRouter: layer {layer_idx} fast path (no scheduler, default mapping)", self.log_path, level=3)
                return self._original_apply(wrapper, layer, dispatch_output)
        
        layer_start_ms = time.time() * 1000

        # ========== 计算层间间隔（两次 native_dynamic_apply 调用之间的时间）==========
        # 代表一个 Transformer 层中 MLP 以外的部分（Attention、LayerNorm、残差等）的耗时
        inter_layer_ms = (layer_start_ms - self._last_mlp_end_ms) if self._last_mlp_end_ms is not None else 0.0

        # ========== 获取当前层实际激活的专家 ==========
        topk_output = dispatch_output.topk_output
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        append_log(f"NativeRouter: Layer {layer_idx} original topk_ids: {topk_ids}", self.log_path, level=3)

        # 保存路由前原始 topk_ids/weights（必须 clone，避免 finalize_decision 的就地修改影响统计）
        _orig_topk_ids = topk_ids.clone()
        _orig_topk_weights = topk_weights.clone()

        router_logits = topk_output.router_logits
        T, N = router_logits.shape
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # ========== 获取当前GPU/CPU专家分布 ==========
        cur_gpu_experts = native_cache.get_gpu_experts(layer_idx=layer_idx)
        
        # ========== 优化：全 GPU 路径计算负载 ==========
        # 使用 bincount + tensor 操作替代 unique→cpu→tolist→Python set/sum
        # 所有计算在 GPU 上完成，仅在调度器边界处转为 Python set
        if topk_ids.numel() > 0:
            device = topk_ids.device
            flat_ids = topk_ids.view(-1)
            
            # GPU 上计算每个专家的 token 计数 (bincount，零 CPU 同步)
            expert_load_counts = torch.bincount(flat_ids, minlength=N)  # [N]
            
            # 获取实际激活的专家 tensor（GPU 上）
            actual_experts_tensor = torch.nonzero(expert_load_counts > 0, as_tuple=False).view(-1)  # [num_active]
            
            # GPU 上用 tensor 计算原始 GPU/CPU 负载
            # 构建 GPU 成员 mask 向量：gpu_membership[e] = True if e in cur_gpu_experts
            gpu_membership = torch.zeros(N, dtype=torch.bool, device=device)
            if cur_gpu_experts:
                gpu_expert_tensor = torch.tensor(sorted(cur_gpu_experts), dtype=torch.long, device=device)
                gpu_membership[gpu_expert_tensor] = True
            
            # 向量化负载计算：按专家 membership 分组求和
            gpu_loads = expert_load_counts * gpu_membership  # GPU 专家的 load
            cpu_loads = expert_load_counts * (~gpu_membership)  # CPU 专家的 load
            
            # 单次 GPU→CPU 传输获取两个标量
            load_pair = torch.stack([gpu_loads.sum(), cpu_loads.sum()]).cpu()
            orig_gpu_load = load_pair[0].item()
            orig_cpu_load = load_pair[1].item()
            
            # 仅在调度器边界处转为 Python set (finalize_decision 要求 Set[int])
            actual_experts_set = set(actual_experts_tensor.cpu().tolist())
        else:
            actual_experts_set = set()
            actual_experts_tensor = torch.tensor([], dtype=torch.long, device=topk_ids.device)
            orig_gpu_load = 0
            orig_cpu_load = 0
        
        # ========== 异步更新热度统计 ==========
        if actual_experts_set:
            _get_predict_executor().submit(
                self._update_expert_heat,
                layer_idx,
                list(actual_experts_set)  # 复用已传输的数据
            )

        append_log(f"NativeRouter: cur_gpu_experts: {cur_gpu_experts}", self.log_path, level=3)

        # ========== 使用预计算的决策结果（提前调度）==========
        # 运行时decision overhead移除，使用预测阶段提前计算的结果
        decision_start_ms = time.time() * 1000
        
        # 尝试获取预计算的决策结果
        with self._decisions_lock:
            precomputed = self._precomputed_decisions.pop(layer_idx, None)
        
        if precomputed is not None:
            # 使用预计算的结果
            topk_ids, topk_weights, reroute_stats = precomputed
            append_log(f"NativeRouter: Layer {layer_idx} using precomputed decision", self.log_path, level=3)
            
            # 仍需获取plan（包含GPU/CPU expert分配信息）
            # 这里只是轻量级的查询，不做重路由
            plan = scheduler.get_plan(layer_idx)
            if plan is None:
                # Fallback: 如果没有plan，重新生成（通常不会发生）
                append_log(f"NativeRouter: Layer {layer_idx} precomputed decision missing plan, regenerating", self.log_path, level=1 )
                plan, _, _, _ = scheduler.finalize_decision(
                    layer_idx=layer_idx,
                    actual_experts=actual_experts_set,
                    router_probs=router_probs,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
        else:
            # Fallback: 没有预计算结果，运行时决策（第一层或预测失败）
            append_log(f"NativeRouter: Layer {layer_idx} no precomputed decision, fallback to runtime", self.log_path, level=2)
            plan, topk_ids, topk_weights, reroute_stats = scheduler.finalize_decision(
                layer_idx=layer_idx,
                actual_experts=actual_experts_set,
                router_probs=router_probs,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        
        # plan, topk_ids, topk_weights, reroute_stats = scheduler.finalize_decision(
        #     layer_idx=layer_idx,
        #     actual_experts=actual_experts_set,
        #     router_probs=router_probs,
        #     topk_ids=topk_ids,
        #     topk_weights=topk_weights,
        # )
        decision_time_ms = time.time() * 1000 - decision_start_ms
        append_log(f"NativeRouter: Layer {layer_idx} decision time: {decision_time_ms:.2f}ms", self.log_path, level=3)

        if topk_ids is not None:
            # 更新 dispatch_output 中的 topk_output，使用重路由后的 ids 和权重
            # 这确保 GPU 和 CPU 计算使用一致的专家分配
            append_log(f"NativeRouter: Layer {layer_idx} finalized topk_ids: {topk_ids}", self.log_path, level=3)
            dispatch_output = dispatch_output._replace(
                topk_output=topk_output._replace(
                    topk_ids=topk_ids,
                    topk_weights=topk_weights
                )
            )
    
        gpu_experts = plan.final_gpu_experts
        cpu_experts = plan.final_cpu_experts
        actual_experts = plan.actual_experts
        new_pending = plan.pending_gpu_experts  # 需要立即迁移到GPU的专家
        
        append_log(f"NativeRouter: Layer {layer_idx} plan: " 
                   f"GPU experts {gpu_experts}, CPU experts {cpu_experts}, "
                   f"Actual experts {actual_experts}, "
                   f"pending {new_pending}", self.log_path, level=3)
        
        # ========== 构建执行掩码（向量化）==========
        # 使用 torch.isin 替代 Python for 循环，单次 kernel 调用
        device = topk_ids.device
        
        if gpu_experts:
            gpu_expert_tensor = torch.tensor(sorted(gpu_experts), dtype=topk_ids.dtype, device=device)
            gpu_mask = torch.isin(topk_ids, gpu_expert_tensor)
        else:
            gpu_mask = torch.zeros_like(topk_ids, dtype=torch.bool)
        
        if cpu_experts:
            cpu_expert_tensor = torch.tensor(sorted(cpu_experts), dtype=topk_ids.dtype, device=device)
            cpu_mask = torch.isin(topk_ids, cpu_expert_tensor)
        else:
            cpu_mask = torch.zeros_like(topk_ids, dtype=torch.bool)

        # ========== GPU 侧快速判断 has_gpu/has_cpu ==========
        # 使用 .any() 避免 sum + cpu transfer，只需 bool 标量
        has_gpu = gpu_mask.any().item()
        has_cpu = cpu_mask.any().item()
        
        # 延迟计算 final_gpu_load / final_cpu_load，仅用于异步统计
        # 在计算路径上只需 has_gpu/has_cpu 即可
        
        # 检查是否发生重路由（使用 reroute_stats 来准确判断）
        # reroute_stats 只有在实际执行了重路由时才不为 None
        has_rerouting = (reroute_stats is not None and reroute_stats.get('total_reroutes', 0) > 0)
        migration_count = len(new_pending) if new_pending else 0
        
        compute_start_ms = time.time() * 1000
        cpu_prepare_time_ms = 0
        gpu_prepare_time_ms = 0
        
        # ========== Step 1: 提交 CPU 计算 (非阻塞) ==========
        if has_cpu and wrapper.tp_rank == 0:
            cpu_prepare_time_ms = self._submit_cpu_experts(wrapper, dispatch_output, gpu_mask, cpu_mask)
        
        # ========== Step 2: GPU 专家计算（流水线并行） ==========
        gpu_start_ms = time.time() * 1000
        gpu_prepare_start_ms = gpu_start_ms
        gpu_kernel_start_ms = gpu_start_ms
        
        if has_gpu and native_cache is not None:
            remapped_ids, _ = native_cache.remap_topk_ids(layer_idx, topk_ids)
            append_log(f"NativeRouter: Layer {layer_idx} remapped topk_ids: {remapped_ids}", self.log_path, level=3)
            masked_dispatch = dispatch_output._replace(
                topk_output=topk_output._replace(topk_ids=remapped_ids)
            )
            gpu_kernel_start_ms = time.time() * 1000
            gpu_prepare_time_ms = gpu_kernel_start_ms - gpu_prepare_start_ms
            output = wrapper.gpu_method.apply(layer, masked_dispatch).hidden_states
            gpu_kernel_end_ms = time.time() * 1000
        else:
            output = torch.zeros_like(dispatch_output.hidden_states)
            gpu_kernel_end_ms = time.time() * 1000
        
        gpu_compute_ms = gpu_kernel_end_ms - gpu_kernel_start_ms
        
        # ========== Step 3: 同步 CPU 结果 ==========
        cpu_compute_ms = 0.0
        if has_cpu and wrapper.tp_rank == 0:
            cpu_sync_start_ms = time.time() * 1000
            cpu_output = self._sync_cpu_experts(wrapper, dispatch_output.hidden_states)
            torch.cuda.synchronize()  # 确保 H2D copy 完成
            cpu_sync_end_ms = time.time() * 1000
            output = output + cpu_output
            
            # 使用实际的 wall-clock 时间（包含 D2H + CPU 计算 + H2D）
            # 而不是 C++ 侧的 _last_cpu_ms（只包含纯 CPU kernel 时间）
            cpu_compute_ms = cpu_sync_end_ms - cpu_sync_start_ms
            
            # 可选：记录 C++ 侧的纯 CPU kernel 时间用于对比
            cpp_cpu_kernel_ms = getattr(wrapper.wrapper, '_last_cpu_ms', 0.0)
            if self.log_path and cpp_cpu_kernel_ms > 0:
                append_log(
                    f"Layer {layer_idx} CPU timing: wall-clock={cpu_compute_ms:.2f}ms, "
                    f"cpp_kernel={cpp_cpu_kernel_ms:.2f}ms, "
                    f"overhead={cpu_compute_ms - cpp_cpu_kernel_ms:.2f}ms",
                    self.log_path, level=3
                )
        
        compute_time_ms = time.time() * 1000 - compute_start_ms
        layer_total_ms = time.time() * 1000 - layer_start_ms

        # ========== 异步统计和日志（全部移出推理主路径）==========
        # 延迟计算 final_gpu_load/final_cpu_load（仅异步统计需要）
        # 单次 GPU→CPU 传输两个标量
        _mask_stats = torch.stack([gpu_mask.sum(), cpu_mask.sum()]).cpu()
        _gpu_mask_cpu = _mask_stats[0].item()
        _cpu_mask_cpu = _mask_stats[1].item()
        _get_predict_executor().submit(
            self._post_layer_stats_async,
            layer_idx,
            compute_time_ms, layer_total_ms,
            gpu_compute_ms, cpu_compute_ms,
            decision_time_ms,
            _gpu_mask_cpu, _cpu_mask_cpu,
            orig_gpu_load, orig_cpu_load,
            reroute_stats,
            has_rerouting, migration_count,
            inter_layer_ms,
        )
        
        # # ========== 异步统计 per-token 负载集中度 ==========
        # if _orig_topk_ids.numel() > 0:
        #     router_probs_cpu = router_probs.detach().cpu()
        #     orig_topk_ids_cpu = _orig_topk_ids.detach().cpu()
        #     _get_predict_executor().submit(
        #         self._record_score_load_ratio_async,
        #         layer_idx,
        #         router_probs_cpu,
        #         orig_topk_ids_cpu,
        #     )
        
        # ========== 标记层完成，清理 ==========
        scheduler.mark_completed(layer_idx)
        scheduler.cleanup_old_plans()
        
        self.mlp_end_ms[layer_idx] = time.time() * 1000
        self._last_mlp_end_ms = self.mlp_end_ms[layer_idx]

        # ========== 异步计算并更新专家评分 ==========
        # 使用 router_probs 计算 P75 分位数评分（异步执行，避免阻塞推理）
        # _get_predict_executor().submit(
        #     self._update_expert_scores_async,
        #     layer_idx,
        #     router_probs.clone()  # 克隆以避免异步访问时的数据竞争
        # )
        
        return StandardCombineInput(hidden_states=output)
    
    def _predict_and_schedule(
        self,
        num_hidden_layers: int,
        hidden_states: "StandardDispatchOutput",
        current_layer_idx: int,
    ):
        """
        执行预测并更新调度器，然后触发专家预加载.
        
        这在后台线程中异步执行：
        1. 调用预测函数获取下一层需要的专家
        2. 将预测结果发送给调度器，生成 plan
        3. 根据 plan，预加载需要迁移到 GPU 的专家
        """
        try:
            scheduler = self.get_scheduler()
            if scheduler is None:
                return
            
            # 计算下一层索引
            next_layer_idx = (current_layer_idx + 1) % num_hidden_layers
            
            # ========== Step 1: 调用预测函数 ========== 
            predict_start_ms = time.time() * 1000
            predict_result = self._predict_fn(next_layer_idx, hidden_states)
            predict_end_ms = time.time() * 1000
            
            if predict_result is None:
                return
            

            topk_ids, topk_weights, router_probs = predict_result
            num_experts = router_probs.size(-1)
            # wzq todo: 统计专家分数分布，以及高负载低分专家占比
            expert_scores_tensor = router_probs.mean(dim=0)
            
            expert_scores = {
                eid: expert_scores_tensor[eid].item()
                for eid in range(num_experts)
            }
            pri_expert_loads = torch.bincount(topk_ids.view(-1), minlength=num_experts)
            # pri_active_experts = torch.nonzero(pri_expert_loads > 0).view(-1)
            pri_active_experts = set(torch.unique(topk_ids).cpu().tolist())
            append_log(f"NativeRouter: predict layer {next_layer_idx}: {pri_active_experts}, completed in {predict_end_ms - predict_start_ms:.2f} ms. ", self.log_path, level=3)
            
            # ========== Step 2: 将预测结果发送给调度器 ==========
            # wzq todo: plan.predicted_experts是tensor
            # plan = scheduler.receive_prediction(
            #     layer_idx=next_layer_idx,
            #     predicted_experts=pri_active_experts,
            #     predicted_loads = pri_expert_loads,
            #     predicted_scores=expert_scores,
            # )
            
            # ========== Step 2.5: 提前调度 - 基于预测结果做决策 ==========
            # 在预测阶段就完成重路由决策，将decision overhead移出关键路径
            decision_start = time.time() * 1000
            plan, precomputed_topk_ids, precomputed_topk_weights, precomputed_reroute_stats = scheduler.finalize_decision(
                layer_idx=next_layer_idx,
                actual_experts=pri_active_experts,
                router_probs=router_probs,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
            decision_end = time.time() * 1000
            
            # 保存预计算的决策结果
            with self._decisions_lock:
                self._precomputed_decisions[next_layer_idx] = (
                    precomputed_topk_ids,
                    precomputed_topk_weights,
                    precomputed_reroute_stats
                )
            
            append_log(
                f"NativeRouter: precomputed decision for layer {next_layer_idx} in {decision_end - decision_start:.2f}ms",
                self.log_path, level=3
            )
                 
            # ========== Step 3: 根据 prefetch 模式处理 ==========
            if plan is not None and plan.pending_gpu_experts:
                if self._enable_prefetch and self._prefetch_mode == "layer":
                    # 每层立即 prefetch (会影响吞吐)
                    from .native_migration import get_native_migration_manager
                    migration_manager = get_native_migration_manager()
                    
                    if migration_manager is not None:
                        # 构建专家列表
                        expert_list = [(next_layer_idx, exp_idx) for exp_idx in plan.pending_gpu_experts]
                        expert_scores = getattr(plan, 'expert_scores', {})
                        
                        planned_gpu = getattr(plan, 'planned_gpu_experts', set())
                        
                        # 计算预取优先级：基于预测的距离
                        # next_layer_idx - current_layer_idx，距离越近优先级越高
                        # 优先级范围：1-2 (中等优先级，低于当前层的0，高于背景任务的>2)
                        priority = 1  # 预取默认使用中等优先级
                        
                        # 异步执行预取
                        prefetch_executor = _get_prefetch_executor()
                        prefetch_executor.submit(
                            migration_manager.prefetch_experts,
                            expert_list,
                            min(len(expert_list), 2),  # max_concurrent
                            expert_scores,
                            planned_gpu,  # wzq: 这里保护的planned_gpu只是基于预测规划的gpu专家，不一定是实际要用的，可能保护不够准确
                            priority,  # 使用中等优先级，避免干扰当前层IO
                        )
                        
                        if self.log_path:
                            append_log(
                                f'NativeRouter: triggered prefetch for layer {next_layer_idx}, '
                                f'{len(plan.pending_gpu_experts)} experts',
                                self.log_path,
                                level=3
                            )        
                
        except Exception as e:
            if self.log_path:
                append_log(f'NativeRouter: predict failed: {e}', self.log_path)
            import traceback
            traceback.print_exc()
    
    def set_prefetch_mode(self, enable: bool, mode: str = "batch_gap"):
        """
        动态设置 prefetch 模式.
        
        Args:
            enable: 是否启用
            mode: "disabled", "layer", "batch_gap"
        """
        self._enable_prefetch = enable
        self._prefetch_mode = mode
        if self.log_path:
            append_log(f'NativeRouter: prefetch mode set to {mode} (enable={enable})', self.log_path)

    def _submit_cpu_experts(
        self,
        wrapper: "KTEPWrapperMethod",
        dispatch_output: "StandardDispatchOutput",
        gpu_mask: torch.Tensor,
        cpu_mask: torch.Tensor,
    ) -> float:
        """提交 CPU 专家计算."""
        if wrapper.wrapper is None:
            return 0.0
        
        cpu_prepare_start_ms = time.time() * 1000
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        x = dispatch_output.hidden_states
        
        # Only keep CPU-assigned positions, set others to -1
        if not cpu_mask.any().item():
            return 0.0

        cpu_topk_ids = topk_ids.clone()
        cpu_topk_ids[~cpu_mask] = -1
        
        # 提交CPU专家计算（异步）
        # submit_with_cuda_stream 通过 cudaLaunchHostFunc 注册回调：
        # 当 CUDA stream 排空后（包括之前队列中的决策阶段 GPU 操作），
        # 回调触发 CPU task 执行。
        # 
        # 注意：_slot_submit_ts 在 submit_forward 内部的 Python 侧记录，
        # 而非 CUDA 回调内部。因此 _last_cpu_ms 不是真正的 CPU 计算时间。
        # 真正的计时在 native_dynamic_apply 的 Step 3 中通过
        # cuda.synchronize() + cpu_infer.sync() 实现。
        
        # 提交CPU专家计算（异步）
        submit_stream = torch.cuda.current_stream(x.device).cuda_stream
        try:
            wrapper.wrapper.submit_forward(
                x,
                cpu_topk_ids,
                topk_weights,
                submit_stream
            )
        except RuntimeError as err:
            if "doesn't match the broadcast shape" not in str(err):
                raise
            if self.log_path:
                append_log(
                    f"NativeRouter: CPU buffer mismatch detected (ids {cpu_topk_ids.shape}, weights {topk_weights.shape}). Clearing cache and retrying.",
                    self.log_path,
                    level=1,
                )
            # try:
            #     type(wrapper.wrapper).clear_buffer_cache()
            # except Exception as clear_err:
            #     if self.log_path:
            #         append_log(
            #             f"NativeRouter: failed to clear CPU buffer cache: {clear_err}",
            #             self.log_path,
            #             level=1,
            #         )
            #     raise err
            # wrapper.wrapper.submit_forward(
            #     x,
            #     cpu_topk_ids,
            #     topk_weights,
            #     submit_stream
            # )
        
        cpu_prepare_end_ms = time.time() * 1000
        cpu_prepare_time_ms = cpu_prepare_end_ms - cpu_prepare_start_ms
        append_log(f"NativeRouter: CPU prepare time: {cpu_prepare_time_ms:.2f}ms", self.log_path, level=3)
        return cpu_prepare_time_ms
    
    def _sync_cpu_experts(
        self,
        wrapper: "KTEPWrapperMethod",
        x: torch.Tensor
    ) -> torch.Tensor:
        """同步 CPU 专家结果."""
        if wrapper.wrapper is None:
            return torch.zeros_like(x)
        
        return wrapper.wrapper.sync_forward(
            x,
            torch.cuda.current_stream(x.device).cuda_stream
        )
    
    def _original_apply(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from ..hooks import _orig_apply
        return _orig_apply(wrapper, layer, dispatch_output)

    def _record_compute_time(self, layer_idx: int, duration_ms: float) -> None:
        """Record compute_time_ms for a specific layer."""
        with self._time_stats_lock:
            ent = self._compute_time_totals.setdefault(layer_idx, {'total_ms': 0.0, 'count': 0})
            ent['total_ms'] += float(duration_ms)
            ent['count'] += 1

    def _record_layer_total_time(self, layer_idx: int, duration_ms: float) -> None:
        """Record layer total time for a specific layer."""
        with self._time_stats_lock:
            ent = self._layer_total_time_totals.setdefault(layer_idx, {'total_ms': 0.0, 'count': 0})
            ent['total_ms'] += float(duration_ms)
            ent['count'] += 1

    def _record_inter_layer_time(self, layer_idx: int, duration_ms: float) -> None:
        """Record inter-layer interval for a specific layer (time between two consecutive MLP calls)."""
        with self._time_stats_lock:
            ent = self._inter_layer_time_totals.setdefault(layer_idx, {'total_ms': 0.0, 'count': 0})
            ent['total_ms'] += float(duration_ms)
            ent['count'] += 1
    
    # def get_stats(self) -> Dict[str, Any]:
    #     """获取统计信息."""
    #     with self._lock:
    #         # 初始化 stats（如果 _stats 不存在，创建空字典）
    #         if not hasattr(self, '_stats'):
    #             self._stats = {}
    #         stats = dict(self._stats)
    #         scheduler = self.get_scheduler()
    #         if scheduler is not None:
    #             stats['scheduler'] = scheduler.get_stats()

    #     # 计算并附加每层平均耗时（compute 和 layer total）
    #     with self._time_stats_lock:
    #         compute_per_layer = {}
    #         compute_total_ms = 0.0
    #         compute_total_count = 0
    #         for lid, v in self._compute_time_totals.items():
    #             c = int(v.get('count', 0))
    #             t = float(v.get('total_ms', 0.0))
    #             if c > 0:
    #                 compute_per_layer[int(lid)] = t / c
    #                 compute_total_ms += t
    #                 compute_total_count += c

    #         layer_total_per_layer = {}
    #         layer_total_ms = 0.0
    #         layer_total_count = 0
    #         for lid, v in self._layer_total_time_totals.items():
    #             c = int(v.get('count', 0))
    #             t = float(v.get('total_ms', 0.0))
    #             if c > 0:
    #                 layer_total_per_layer[int(lid)] = t / c
    #                 layer_total_ms += t
    #                 layer_total_count += c

    #     stats['compute_avg_ms_per_layer'] = compute_per_layer
    #     stats['compute_overall_avg_ms'] = (compute_total_ms / compute_total_count) if compute_total_count > 0 else None
    #     stats['layer_total_avg_ms_per_layer'] = layer_total_per_layer
    #     stats['layer_total_overall_avg_ms'] = (layer_total_ms / layer_total_count) if layer_total_count > 0 else None

    #     inter_layer_per_layer = {}
    #     inter_layer_total_ms = 0.0
    #     inter_layer_total_count = 0
    #     for lid, v in self._inter_layer_time_totals.items():
    #         c = int(v.get('count', 0))
    #         t = float(v.get('total_ms', 0.0))
    #         if c > 0:
    #             inter_layer_per_layer[int(lid)] = t / c
    #             inter_layer_total_ms += t
    #             inter_layer_total_count += c
    #     stats['inter_layer_avg_ms_per_layer'] = inter_layer_per_layer
    #     stats['inter_layer_overall_avg_ms'] = (inter_layer_total_ms / inter_layer_total_count) if inter_layer_total_count > 0 else None

    #     # Attach expert scores (P75 percentile + historical)
    #     stats['expert_scores'] = self.get_expert_scores()
    #     stats['score_gamma'] = self._score_gamma
        
    #     # 添加当前请求的统计信息
    #     stats['current_request'] = self.get_request_stats()

    #     return stats
    
    def log_request_summary(self, request_id: str = "unknown") -> str:
        """
        记录请求级别统计摘要.
        
        Args:
            request_id: 请求ID
            
        Returns:
            格式化的统计信息字符串
        """
        try:
            request_stats = self.get_request_stats()
            
            summary = (
                f"Request {request_id} Summary:\n"
                f"  Layers processed: {request_stats['total_layers']}\n"
                f"  Avg GPU compute: {request_stats['avg_gpu_compute_ms']:.2f}ms\n"
                f"  Avg CPU compute: {request_stats['avg_cpu_compute_ms']:.2f}ms\n"
                f"  Avg decision time: {request_stats['avg_decision_ms']:.2f}ms\n"
                f"  Avg inter-layer gap: {request_stats.get('avg_inter_layer_ms', 0.0):.2f}ms\n"
                f"  Reroute rate: {request_stats['reroute_rate']:.1%}\n"
                f"  GPU load change: {request_stats['avg_gpu_load_change']:.1%}\n"
                f"  CPU load change: {request_stats['avg_cpu_load_change']:.1%}\n"
                f"  High-score load (top{self.HIGH_SCORE_TOP_K}, random={request_stats.get('random_expected_pct',0):.2f}%): "
                f"mean={request_stats.get('hs_load_pct_mean',0):.2f}% "
                f"max={request_stats.get('hs_load_pct_max',0):.2f}% "
                f"std={request_stats.get('hs_load_pct_std',0):.2f}% "
                f"p90={request_stats.get('hs_load_pct_p90',0):.2f}% | "
                f"conc mean={request_stats.get('hs_conc_ratio_mean',0):.4f}x "
                f"max={request_stats.get('hs_conc_ratio_max',0):.4f}x"
            )

            # 每层平均高分专家负载占比明细
            layer_pct_avg  = request_stats.get('layer_hs_pct_avg', {})
            layer_conc_avg = request_stats.get('layer_conc_ratio_avg', {})
            if layer_pct_avg:
                lines = [f"  Per-layer hs_pct (top{self.HIGH_SCORE_TOP_K}, random={request_stats.get('random_expected_pct',0):.2f}%):"]
                for lid in sorted(layer_pct_avg.keys()):
                    pct  = layer_pct_avg[lid]
                    conc = layer_conc_avg.get(lid, 0.0)
                    n_steps = len(self._request_stats.get('layer_hs_pct', {}).get(lid, []))
                    lines.append(f"    L{lid:2d}: {pct:6.2f}%  conc={conc:.3f}x  (n={n_steps})")
                summary += "\n" + "\n".join(lines)
            
            if self.log_path:
                append_log(summary, self.log_path)
            
            return summary
            
        except Exception as e:
            error_msg = f"Failed to generate request summary for {request_id}: {e}"
            if self.log_path:
                append_log(error_msg, self.log_path)
            return error_msg



# ============================================================
# 全局实例管理
# ============================================================
_native_router: Optional[NativeDynamicRouter] = None
_init_lock = threading.Lock()

def get_native_router() -> Optional[NativeDynamicRouter]:
    """获取全局 Native 路由器."""
    return _native_router

def init_native_router(
    log_path: Optional[str] = None,
    scheduler: Optional[ExpertScheduler] = None,
    enable_prefetch: bool = True,
    prefetch_mode: str = "layer",
) -> NativeDynamicRouter:
    """
    初始化全局 Native 路由器.
    
    Args:
        log_path: 日志路径
        scheduler: 调度器（重路由策略通过 scheduler 的 RerouteConfig 配置）
        enable_prefetch: 是否启用专家预加载
        prefetch_mode: prefetch 模式 ("disabled", "layer", "batch_gap")
    """
    global _native_router
    
    with _init_lock:
        if _native_router is None:
            _native_router = NativeDynamicRouter(
                scheduler=scheduler,
                log_path=log_path,
                enable_prefetch=enable_prefetch,
                prefetch_mode=prefetch_mode,
            )
        _native_router.set_prefetch_mode(enable_prefetch, prefetch_mode)
        
        return _native_router
