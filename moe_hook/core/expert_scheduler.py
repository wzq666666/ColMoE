"""
Expert Scheduler - 通用的专家调度器.

职责：
1. 基于预测结果，规划未来层的专家分布
2. 基于实际激活，调整当前层的专家分布

这是一个纯决策模块，不依赖任何 backend (native/legacy)。
具体的专家迁移执行由各 backend 的 router/migration 模块负责。

调度流程：
  Layer N 推理时：
  1. 预测器预测 Layer N+1 的激活专家
  2. 调度器基于预测，规划 Layer N+1 的专家分布 (receive_prediction)
  3. Layer N+1 推理开始时，获取实际激活专家
  4. 调度器基于实际激活，调整最终执行计划 (finalize_decision)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union
from ..logger import append_log
import threading
import time
import torch


class SchedulePhase(Enum):
    """调度阶段."""
    PREDICTED = "predicted"    # 基于预测生成的计划
    FINALIZED = "finalized"    # 结合实际激活后的最终计划
    COMPLETED = "completed"    # 执行完成


@dataclass
class RerouteConfig:
    """
    重路由策略配置.
    
    统一管理所有策略的参数，避免 finalize_decision 入参膨胀。
    通过 strategy 字段选择策略，各策略的参数通过对应字段配置。
    """
    # === 策略选择 ===
    strategy: str = "token_reroute"
    # "none" / "static": 不做重路由
    # "io_free": 专家级 IO-free 替换
    # "token_reroute": Token 级重路由（默认）
    # "expert_low_score": 专家级低分重路由（粗粒度）
    # "token_low_score": Token级低分重路由（细粒度）
    
    # === 通用参数 ===
    alpha: float = 0.05               # 评分相似性阈值（相对差距）
    
    # === token_reroute 专用参数 ===
    allow_duplicate: bool = False      # 是否允许重复专家
    use_limited_reroute: bool = True   # 是否使用带限制的重路由
    max_duplicates_per_expert: int = 2 # 每个专家最多重复次数
    min_unique_experts: Optional[int] = None  # 每个 token 最小唯一专家数, None=auto(k//2)
    
    # === io_free 专用参数 ===
    score_threshold_ratio: float = 0.5 # 低分阈值比例: score < median * ratio 才替换

    # === load_aware_token 专用参数 ===
    max_gpu_duplicates: int = 2           # l: 每个GPU专家在单个token中的最大出现总次数
    dominance_threshold: float = 0.5      # 主导性阈值: 原专家得分/topk总得分 > 此值则跳过


@dataclass 
class LayerPlan:
    """
    单层的执行计划.
    
    生命周期：
    1. PREDICTED: 由 receive_prediction() 创建，基于预测
    2. FINALIZED: 由 finalize_decision() 更新，结合实际激活
    3. COMPLETED: 由 mark_completed() 标记，执行完成
    """
    layer_idx: int
    
    # === 预测阶段 (PREDICTED) ===
    predicted_experts: Set[int] = field(default_factory=set)     # 预测激活的专家
    planned_gpu_experts: Set[int] = field(default_factory=set)   # 规划放到 GPU 的专家
    planned_cpu_experts: Set[int] = field(default_factory=set)   # 规划在 CPU 执行的专家 (predicted - planned_gpu)
    pending_gpu_experts: Set[int] = field(default_factory=set)   # 需要迁移到 GPU 的专家 (planned_gpu - current_gpu)
    expert_scores: Dict[int, float] = field(default_factory=dict)  # 每个专家的预测概率/分数
    
    # === 最终决策阶段 (FINALIZED) ===
    actual_experts: Set[int] = field(default_factory=set)        # 实际激活的专家
    final_gpu_experts: Set[int] = field(default_factory=set)     # 最终在 GPU 执行的专家
    final_cpu_experts: Set[int] = field(default_factory=set)     # 最终在 CPU 执行的专家
    
    # === 元信息 ===
    phase: SchedulePhase = SchedulePhase.PREDICTED
    created_at: float = field(default_factory=time.time)
    finalized_at: Optional[float] = None
    
    # === 统计 ===
    prediction_hits: int = 0    # 预测命中数 (预测了且实际激活了)
    prediction_misses: int = 0  # 预测未命中数 (实际激活了但未预测)


class GPUStateProvider(Protocol):
    """
    GPU 状态提供者接口.
    
    调度器通过此接口查询当前 GPU 上的专家分布，
    而不直接依赖具体的 backend 实现。
    """
    
    def get_gpu_experts(self, layer_idx: int) -> Set[int]:
        """获取指定层当前在 GPU 上的专家."""
        ...
    
    def get_num_gpu_slots(self, layer_idx: int) -> int:
        """获取指定层的 GPU 槽位数."""
        ...


class ExpertScheduler:
    """
    通用专家调度器.
    
    核心功能：
    1. receive_prediction(): 接收预测，生成规划
    2. finalize_decision(): 结合实际激活，生成最终决策
    
    不包含：
    - 专家迁移执行（由 backend 的 migration manager 负责）
    - GPU 缓存管理（由 backend 的 cache manager 负责）
    """
    
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        num_gpu_slots: int,
        gpu_state_provider: Optional[GPUStateProvider] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        reroute_config: Optional[RerouteConfig] = None,
    ):
        """
        初始化调度器.
        
        Args:
            num_layers: MoE 层数
            num_experts: 每层专家数
            num_gpu_slots: 每层 GPU 槽位数
            gpu_state_provider: GPU 状态查询接口（可选，用于优化决策）
            log_fn: 日志函数（可选）
            reroute_config: 重路由策略配置（可选，默认 token_reroute）
        """
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_gpu_slots = num_gpu_slots
        self._gpu_state = gpu_state_provider
        self._log_fn = log_fn
        self._reroute_config = reroute_config or RerouteConfig()
        
        self._lock = threading.Lock()
        self._plans: Dict[int, LayerPlan] = {}
        self._current_layer = -1
        self._config_path: Optional[str] = None  # 配置文件路径，用于热重载
        
        # 统计
        self._stats = {
            'total_predictions': 0,
            'prediction_hits': 0,
            'prediction_misses': 0,
        }
        
        self._log(
            f'ExpertScheduler initialized: '
            f'{num_layers} layers, {num_experts} experts, {num_gpu_slots} GPU slots, '
            f'reroute_strategy={self._reroute_config.strategy}'
        )
    
    def _log(self, msg: str):
        """输出日志."""
        if self._log_fn is not None:
            self._log_fn(msg)
    
    # ================================================================
    # 核心接口 1: 接收预测，生成规划
    # ================================================================
    
    def receive_prediction(
        self,
        layer_idx: int,
        predicted_experts: Set[int],
        predicted_loads: Optional[Set[int]] = set(),
        predicted_scores: Optional[Dict[int, float]] = {},
    ) -> LayerPlan:
        """
        接收预测结果，生成该层的专家规划.
        
        这通常在 Layer N 推理时被调用，为 Layer N+1 做准备。
        
        Args:
            layer_idx: 目标层索引
            predicted_experts: 预测会被激活的专家列表
            confidence_scores: 可选的专家置信度分数字典 {expert_idx: score}
            
        Returns:
            生成的执行计划
        """
                
        with self._lock:
            # 查询当前 GPU 上的专家
            current_gpu = set()
            if self._gpu_state is not None:
                current_gpu = self._gpu_state.get_gpu_experts(layer_idx)
            
            # 选择哪些专家规划放到 GPU
            # planned_gpu = self._plan_gpu_experts(
            #     layer_idx=layer_idx,
            #     predicted_experts=predicted_experts,
            #     predicted_loads=predicted_loads,
            #     predicted_scores=predicted_scores,
            # )
            
            # # 计算需要迁移的专家 (planned_gpu 中当前不在 GPU 上的)
            # pending_gpu = planned_gpu - current_gpu

            pending_gpu = set()  # 暂不考虑迁移，后续根据策略添加
            # 计算规划在 CPU 执行的专家 (预测中但不在 planned_gpu 中的)
            current_cpu = set(range(self.num_experts)) - current_gpu
            
            plan = LayerPlan(
                layer_idx=layer_idx,
                predicted_experts=predicted_experts,
                planned_gpu_experts=current_gpu,
                planned_cpu_experts=current_cpu,
                pending_gpu_experts=pending_gpu,
                expert_scores=predicted_scores,
                phase=SchedulePhase.PREDICTED,
            )
            
            self._plans[layer_idx] = plan
            self._stats['total_predictions'] += 1
            
            self._log(
                f'[Scheduler] Layer {layer_idx}: '
                f'planned_gpu={sorted(current_gpu)}, '
                f'pending_gpu={sorted(pending_gpu)}, '
                f'planned_cpu={sorted(current_cpu)}'
            )
            
            return plan
    
    def _plan_gpu_experts(
        self,
        layer_idx: int,
        predicted_experts: Set[int],
        predicted_loads: Optional[Set[int]] = set(),
        predicted_scores: Optional[Dict[int, float]] = None,
    ) -> Set[int]:
        """
        规划哪些专家放到 GPU.

        目标:
        MIN(MAX(T_cpu, T_gpu))，即最小化 CPU 和 GPU 的最大计算时间.
        约束：
        T_migration 能够在layer_idx计算开始前完成
        策略：
        1. 先根据当前layer_idx的专家位置，分成gpu专家和cpu专家，对于每个专家，基于其位置和负载大小，估算计算时间
        2. 得到当前的T_cpu, T_gpu
        3. 按照负载顺序倒序排列predicted_loads
        4. 从头遍历predicted_loads，若专家的预测得分低于某阈值（与未激活专家分数相近），则丢弃，若得分高，则
        寻找GPU中是否存在它的可替换专家（分数相近），选择最相近的那个进行替换，避免迁移
        5. 若既不可丢弃，也不可替换，则加入待迁移队列Q，同时加入gpu专家，从cpu专家中去除
        6. 重新计算当前的T_cpu, T_gpu，T_migration
        7. 当遍历到的负载大小小于某个阈值时，停止遍历

        """
        selected = set()
        
        # 查询当前 GPU 上的专家
        current_gpu = set()
        if self._gpu_state is not None:
            current_gpu = self._gpu_state.get_gpu_experts(layer_idx)
        
        # 策略 1: 优先保留已在 GPU 且被预测的专家
        for exp_idx in current_gpu:
            if exp_idx in predicted_experts and len(selected) < self.num_gpu_slots:
                selected.add(exp_idx)
        
        # 对于不在GPU的预测专家，选取分数最高的那个专家加载
        remaining_slots = self.num_gpu_slots - len(selected)
        if remaining_slots > 0 and predicted_scores is not None:
            # 排序预测专家，按分数从高到低
            sorted_experts = sorted(
                (exp for exp in predicted_experts if exp not in selected),
                key=lambda e: predicted_scores.get(e, 0.0),
                reverse=True
            )
            for exp_idx in sorted_experts[:remaining_slots]:
                selected.add(exp_idx)
        
        return selected
    
    # ================================================================
    # 核心接口 2: 结合实际激活，生成最终决策
    # ================================================================
    
    def get_plan(self, layer_idx: int) -> Optional[LayerPlan]:
        """
        获取已生成的计划（不重新计算）.
        
        用于提前调度场景，运行时直接获取预测阶段生成的plan。
        
        Args:
            layer_idx: 层索引
            
        Returns:
            如果存在则返回LayerPlan，否则返回None
        """
        with self._lock:
            return self._plans.get(layer_idx)
    
    def finalize_decision(
        self,
        layer_idx: int,
        actual_experts: Set[int],
        router_probs: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[LayerPlan, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        结合实际激活情况，生成最终执行决策.
        
        这在每层推理开始时调用，此时已知实际激活的专家。
        重路由策略及其参数由 self._reroute_config 统一控制。
        
        Args:
            layer_idx: 当前层索引
            actual_experts: 实际被激活的专家集合
            router_probs: 路由概率 [T, E]，用于重路由
            topk_ids: Top-k 专家ID [T, k]，用于重路由
            topk_weights: Top-k 权重 [T, k]，用于重路由
            
        Returns:
            (plan, rerouted_topk_ids, rerouted_topk_weights, reroute_stats)
            - plan: LayerPlan with final GPU/CPU expert assignments
            - rerouted_topk_ids: Rerouted expert IDs (or original if no rerouting)
            - rerouted_topk_weights: Rerouted weights (or original if no rerouting)
            - reroute_stats: Dict with reroute statistics, or None if no rerouting
        """
        cfg = self._reroute_config
    
        with self._lock:
            self._current_layer = layer_idx
            
            # 获取预测阶段的计划（可能不存在）
            plan = self._plans.get(layer_idx)
            
            # 查询当前 GPU 状态（实时）
            current_gpu = set()
            if self._gpu_state is not None:
                current_gpu = self._gpu_state.get_gpu_experts(layer_idx)
            
            k = topk_ids.size(1) if topk_ids is not None else 0
            # ========== 执行重路由优化 ==========
            rerouted_topk_ids = topk_ids
            rerouted_topk_weights = topk_weights
            reroute_stats = None  # Will be set if rerouting is performed
            
            current_cpu = set(range(self.num_experts)) - current_gpu
            self._log(
                f"[Scheduler] Layer {layer_idx} current GPU experts: {sorted(current_gpu)}, "
                f"current CPU experts: {sorted(current_cpu)}"
            )
            
            # ========== 策略分发 ==========
            strategy = cfg.strategy.lower().strip()
            
            if strategy in ("none", "static"):
                # ===== 静态策略：不做重路由 =====
                self._log(f"[Scheduler] Layer {layer_idx} strategy=static, no rerouting")
                gpu_experts = actual_experts & current_gpu
                cpu_experts = actual_experts - current_gpu
                
            elif strategy == "io_free":
                # ===== IO-free 专家级替换策略 =====
                if router_probs is not None and topk_ids is not None and topk_weights is not None:
                    reroute_start_ms = time.time() * 1000
                    try:
                        rerouted_topk_ids, rerouted_topk_weights, actual_experts, cpu_experts, gpu_experts, reroute_stats = self.expert_level_reroute(
                            router_probs, topk_ids, topk_weights,
                            current_gpu, current_cpu,
                            alpha=cfg.alpha,
                            score_threshold_ratio=cfg.score_threshold_ratio,
                        )
                        self._log(
                            f"L{layer_idx} io_free reroute time: {time.time() * 1000 - reroute_start_ms:.2f}ms"
                        )
                        if reroute_stats:
                            self._log(
                                f"[IOFree] Layer {layer_idx} stats: "
                                f"replacements={reroute_stats['expert_replacements']}, "
                                f"position_rate={reroute_stats.get('position_reroute_rate', reroute_stats.get('reroute_rate', 0)):.1%}, "
                                f"inactive_gpu={reroute_stats['inactive_gpu_experts']}, "
                                f"activated_cpu={reroute_stats['activated_cpu_experts']}"
                            )
                    except Exception as e:
                        self._log(f"[Scheduler] Layer {layer_idx} io_free reroute failed: {e}")
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                else:
                    gpu_experts = actual_experts & current_gpu
                    cpu_experts = actual_experts - current_gpu
            
            elif strategy == "token_reroute":
                # ===== Token 级重路由策略（默认） =====
                if router_probs is not None and topk_ids is not None and topk_weights is not None:
                    reroute_start_ms = time.time() * 1000
                    try:
                        rerouted_topk_ids, rerouted_topk_weights, actual_experts, cpu_experts, gpu_experts, reroute_stats = self.token_level_reroute(
                            router_probs, topk_ids, topk_weights,
                            current_gpu, current_cpu,
                            alpha=cfg.alpha,
                            allow_duplicate=cfg.allow_duplicate,
                            use_limited_reroute=cfg.use_limited_reroute,
                            max_duplicates_per_expert=cfg.max_duplicates_per_expert,
                            min_unique_experts=cfg.min_unique_experts,
                        )
                        self._log(
                            f"L{layer_idx} token reroute time: {time.time() * 1000 - reroute_start_ms:.2f}ms"
                        )
                        
                        if reroute_stats:
                            self._log(
                                f"[Reroute] Layer {layer_idx} stats: "
                                f"position_rate={reroute_stats.get('position_reroute_rate', reroute_stats.get('reroute_rate', 0)):.1%}, "
                                f"blocked_dup={reroute_stats['blocked_by_duplicate_limit']}, "
                                f"blocked_uniq={reroute_stats['blocked_by_unique_limit']}"
                            )
                    except Exception as e:
                        self._log(f"[Scheduler] Layer {layer_idx} token reroute failed: {e}")
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                else:
                    gpu_experts = actual_experts & current_gpu
                    cpu_experts = actual_experts - current_gpu
            
            elif strategy == "expert_low_score":
                # ===== 专家级低分重路由策略（粗粒度） =====
                if router_probs is not None and topk_ids is not None and topk_weights is not None:
                    reroute_start_ms = time.time() * 1000
                    try:
                        from ..core.routing_redirection import expert_level_low_score_reroute
                        rerouted_topk_ids, rerouted_topk_weights, reroute_mask, reroute_stats = expert_level_low_score_reroute(
                            router_probs, topk_ids, topk_weights,
                            list(current_gpu), list(current_cpu),
                            alpha=cfg.alpha,
                        )
                        self._log(
                            f"L{layer_idx} expert_low_score reroute time: {time.time() * 1000 - reroute_start_ms:.2f}ms"
                        )
                        # 重新计算实际激活专家
                        actual_experts = set(rerouted_topk_ids[rerouted_topk_ids >= 0].unique().cpu().tolist())
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                        if reroute_stats:
                            self._log(
                                f"[ExpertLowScore] Layer {layer_idx} stats: "
                                f"replacements={reroute_stats['expert_replacements']}, "
                                f"position_reroute_rate={reroute_stats['position_reroute_rate']:.1%}, "
                                f"token_reroute_rate={reroute_stats['token_reroute_rate']:.1%}, "
                                f"low_score_cpu={reroute_stats['low_score_cpu_experts']}, "
                                f"replaceable_gpu={reroute_stats['replaceable_gpu_experts']}"
                            )
                    except Exception as e:
                        self._log(f"[Scheduler] Layer {layer_idx} expert_low_score reroute failed: {e}")
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                else:
                    gpu_experts = actual_experts & current_gpu
                    cpu_experts = actual_experts - current_cpu
            
            elif strategy == "token_low_score":
                # ===== Token级低分重路由策略（细粒度） =====
                if router_probs is not None and topk_ids is not None and topk_weights is not None:
                    reroute_start_ms = time.time() * 1000
                    try:
                        from ..core.routing_redirection import token_level_low_score_reroute
                        rerouted_topk_ids, rerouted_topk_weights, reroute_mask, reroute_stats = token_level_low_score_reroute(
                            router_probs, topk_ids, topk_weights,
                            list(current_gpu), list(current_cpu),
                            alpha=cfg.alpha,
                        )
                        self._log(
                            f"L{layer_idx} token_low_score reroute time: {time.time() * 1000 - reroute_start_ms:.2f}ms"
                        )
                        # 重新计算实际激活专家
                        actual_experts = set(rerouted_topk_ids[rerouted_topk_ids >= 0].unique().cpu().tolist())
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                        if reroute_stats:
                            self._log(
                                f"[TokenLowScore] Layer {layer_idx} stats: "
                                f"total={reroute_stats['total_reroutes']}, "
                                f"replaceable_gpu_found={reroute_stats['replaceable_gpu_found']}, "
                                f"position_reroute_rate={reroute_stats['position_reroute_rate']:.1%}, "
                                f"token_reroute_rate={reroute_stats['token_reroute_rate']:.1%}, "
                                f"success_rate={reroute_stats['success_rate']:.1%}, "
                            )
                    except Exception as e:
                        self._log(f"[Scheduler] Layer {layer_idx} token_low_score reroute failed: {e}")
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                else:
                    gpu_experts = actual_experts & current_gpu
                    cpu_experts = actual_experts - current_gpu
            elif strategy == "load_aware_token":
                # ===== Load-aware token级重路由策略 =====
                if router_probs is not None and topk_ids is not None and topk_weights is not None:
                    reroute_start_ms = time.time() * 1000
                    try:
                        # 实时计算当前批次每个专家的激活次数作为负载指标（单次 GPU reduce）
                        flat = topk_ids.reshape(-1)
                        flat = flat[flat >= 0]
                        if flat.numel() > 0:
                            counts = torch.bincount(flat.long(), minlength=self.num_experts)
                            counts_cpu = counts.cpu().numpy()
                            realtime_load: Dict[int, float] = {
                                int(eid): float(counts_cpu[eid])
                                for eid in range(self.num_experts)
                                if counts_cpu[eid] > 0
                            }
                        else:
                            realtime_load = {}
                        self._log(f"[Scheduler] alpha={cfg.alpha}, max_gpu_duplicates={cfg.max_gpu_duplicates}, dominance_threshold={cfg.dominance_threshold}")
                        rerouted_topk_ids, rerouted_topk_weights, actual_experts, cpu_experts, gpu_experts, reroute_stats = self.load_aware_token_reroute(
                            router_probs, topk_ids, topk_weights,
                            current_gpu, current_cpu,
                            expert_load=realtime_load,
                            alpha=cfg.alpha,
                            max_gpu_duplicates=cfg.max_gpu_duplicates,
                            dominance_threshold=cfg.dominance_threshold,
                        )
                        self._log(
                            f"L{layer_idx} load_aware_token reroute time: {time.time() * 1000 - reroute_start_ms:.2f}ms"
                        )
                        if reroute_stats:
                            self._log(
                                f"[LoadAware] Layer {layer_idx} stats: "
                                f"total={reroute_stats['total_reroutes']}, "
                                f"blocked_dom={reroute_stats['blocked_by_dominance']}, "
                                f"blocked_dup={reroute_stats['blocked_by_duplicate_limit']}, "
                                f"blocked_alpha={reroute_stats['blocked_by_alpha']}, "
                                f"position_rate={reroute_stats['position_reroute_rate']:.1%}, "
                                f"token_rate={reroute_stats['token_reroute_rate']:.1%}"
                            )
                    except Exception as e:
                        self._log(f"[Scheduler] Layer {layer_idx} load_aware_token reroute failed: {e}")
                        gpu_experts = actual_experts & current_gpu
                        cpu_experts = actual_experts - current_gpu
                else:
                    gpu_experts = actual_experts & current_gpu
                    cpu_experts = actual_experts - current_gpu
            else:
                self._log(f"[Scheduler] Layer {layer_idx} unknown strategy '{cfg.strategy}', falling back to static")
                gpu_experts = actual_experts & current_gpu
                cpu_experts = actual_experts - current_gpu

            # ========== 获取最终的执行计划 ==========
            final_gpu = gpu_experts
            final_cpu = cpu_experts
            new_pending = set()
            
            # 更新或创建计划
            if plan is not None:
                plan.actual_experts = actual_experts
                plan.final_gpu_experts = final_gpu
                plan.final_cpu_experts = final_cpu
                plan.pending_gpu_experts = new_pending  # 更新为重新规划后需要迁移的专家
                plan.phase = SchedulePhase.FINALIZED
                plan.finalized_at = time.time()
            else:
                # 没有预测计划，创建新的
                self._log(f"[Scheduler] Layer {layer_idx} has no prior plan, creating default.")
                plan = LayerPlan(
                    layer_idx=layer_idx,
                    actual_experts=actual_experts,
                    final_gpu_experts=final_gpu,
                    final_cpu_experts=final_cpu,
                    pending_gpu_experts=new_pending,
                    phase=SchedulePhase.FINALIZED,
                    finalized_at=time.time(),
                )
                self._plans[layer_idx] = plan
            
            return plan, rerouted_topk_ids, rerouted_topk_weights, reroute_stats
    
    def token_level_reroute(self, router_probs, topk_ids, topk_weights,
                    current_gpu, current_cpu,
                    alpha=0.05, allow_duplicate=False,
                    use_limited_reroute=True, max_duplicates_per_expert=2, min_unique_experts=None):
        '''
        基于 token 级别的 GPU 优先重路由策略.
        Args:
            router_probs: [T, E] 路由概率
            topk_ids: [T, k] 原始 top-k 专家 ID
            topk_weights: [T, k] 对应权重
            current_gpu: 当前在 GPU 上的专家集合
            cpu_experts: 当前在 CPU 上的专家集合
            alpha: 容忍度参数
            allow_duplicate: 是否允许重复专家（默认False）
                - False: 每个GPU专家在每个token的topk中最多出现一次，保持原始权重
                - True: 允许重复，同步更新权重后合并
            use_limited_reroute: 是否使用带限制的重路由（默认True）
            max_duplicates_per_expert: 每个专家最多重复次数
            min_unique_experts: 每个token最小唯一专家数
        Returns:
            rerouted_topk_ids: 重路由后的专家 ID
            rerouted_topk_weights: 重路由后的权重
            actual_experts: 重新计算的实际激活专家集合
        '''
        from ..core.routing_redirection import (
            token_level_gpu_preferred_reroute, 
            token_level_gpu_preferred_reroute_with_duplicate,
            token_level_gpu_preferred_reroute_with_limits,
            merge_duplicate_experts_with_weights
        )
        
        T, N = router_probs.shape
        k = topk_ids.size(1)
        
        # 自动计算最小唯一专家数
        if min_unique_experts is None:
            min_unique_experts = max(k // 2, 3)  # 至少保持一半的多样性，最少3个
        
        reroute_stats = None
        
        if use_limited_reroute:
            # 使用带限制的重路由策略
            new_topk_ids, new_topk_weights, _, reroute_stats = token_level_gpu_preferred_reroute_with_limits(
                router_probs, topk_ids, topk_weights,
                list(current_gpu), list(current_cpu),
                alpha, eps=1e-6,
                max_duplicates_per_expert=max_duplicates_per_expert,
                min_unique_experts=min_unique_experts
            )
            self._log(f"[Reroute] new_topk_ids:\n{new_topk_ids}")
            self._log(f"[Reroute] Limited reroute stats: {reroute_stats}")
            
            # 带限制的重路由不需要额外合并，直接使用结果
            rerouted_topk_ids = new_topk_ids
            rerouted_topk_weights = new_topk_weights
            
        elif allow_duplicate:
            # 允许重复专家的重路由：同步更新权重
            new_topk_ids, new_topk_weights, _ = token_level_gpu_preferred_reroute_with_duplicate(
                router_probs, topk_ids, topk_weights,
                list(current_gpu), list(current_cpu),
                alpha, eps=1e-6
            )
            self._log(f"[Reroute] New topk_ids after reroute (allow_duplicate=True):\n{new_topk_ids}")
            
            # 合并重复专家（权重已同步更新，可以正确合并）
            rerouted_topk_ids, rerouted_topk_weights = merge_duplicate_experts_with_weights(
                new_topk_ids, new_topk_weights, N
            )
            self._log(f"[Reroute] After merge:\n{rerouted_topk_ids}")
        else:
            # 不允许重复的重路由：保持原始权重
            new_topk_ids, _ = token_level_gpu_preferred_reroute(
                router_probs, topk_ids, topk_weights,
                list(current_gpu), list(current_cpu),
                alpha, eps=1e-6
            )
            self._log(f"[Reroute] New topk_ids after reroute:\n{new_topk_ids}")
            
            # 直接使用重路由后的 topk_ids 和原始的 topk_weights
            rerouted_topk_ids = new_topk_ids
            rerouted_topk_weights = topk_weights

        # 计算实际激活的专家集合
        valid = rerouted_topk_ids.view(-1)
        if valid.numel() == 0:
            actual_experts = set()
        else:
            valid = valid[valid >= 0]
            if valid.numel() == 0:
                actual_experts = set()
            else:
                load = torch.bincount(valid, minlength=N)
                actual_experts = set(torch.nonzero(load > 0).view(-1).cpu().tolist())
        
        # 权重验证：仅在详细日志模式下执行（每次调用触发多次 GPU→CPU 同步）
        from ..logger import get_log_level
        if get_log_level() >= 3:
            if torch.isnan(rerouted_topk_weights).any() or torch.isinf(rerouted_topk_weights).any():
                self._log(f"[WARNING] Layer {self._current_layer} has invalid weights after reroute!")
            weight_sum = rerouted_topk_weights.sum(dim=1)
            if (weight_sum < 0.99).any() or (weight_sum > 1.01).any():
                self._log(f"[WARNING] Layer {self._current_layer} weight sum not normalized! min={weight_sum.min():.4f}, max={weight_sum.max():.4f}")
                self._log(f"[DEBUG] topk_ids sample: {rerouted_topk_ids[0].tolist()}")
                self._log(f"[DEBUG] topk_weights sample: {rerouted_topk_weights[0].tolist()}")

        # ========== 关键修复：正确计算GPU/CPU专家分配 ==========
        # 所有在 actual_experts 中的专家都需要被执行
        # - 如果专家在 current_gpu 中 → GPU执行
        # - 否则 → CPU执行（无论当前是在CPU内存还是在磁盘）
        gpu_experts_after = actual_experts & current_gpu
        cpu_experts_after = actual_experts - current_gpu  # 不在GPU上的都由CPU处理

        self._log(
            f"[Scheduler] Layer {self._current_layer} after reroute: "
            f"Actual={sorted(actual_experts)}, "
            f"GPU={sorted(gpu_experts_after)}, "
            f"CPU={sorted(cpu_experts_after)}"
        )

        return rerouted_topk_ids, rerouted_topk_weights, actual_experts, cpu_experts_after, gpu_experts_after, reroute_stats

    def expert_level_reroute(self, router_probs, topk_ids, topk_weights,
                    current_gpu, current_cpu, alpha=0.2, score_threshold_ratio=0.5):
        '''
        IO-free 专家级替换策略.
        
        与 token_level_reroute 的区别：
        - token_level: 逐 token 逐位置决策
        - expert_level: 逐专家决策，一个 CPU 专家的所有 token 统一替换到同一个 GPU 专家
        - 只替换低分 CPU 专家，高分专家保留在 CPU 执行
        
        Args:
            router_probs: [T, E] 路由概率
            topk_ids: [T, k] 原始 top-k 专家 ID
            topk_weights: [T, k] 对应权重
            current_gpu: 当前在 GPU 上的专家集合
            current_cpu: 当前在 CPU 上的专家集合
            alpha: 评分相似性阈值
            score_threshold_ratio: 低分阈值比例
        Returns:
            rerouted_topk_ids, rerouted_topk_weights, actual_experts,
            cpu_experts_after, gpu_experts_after, reroute_stats
        '''
        from ..core.routing_redirection import expert_level_io_free_reroute

        T, N = router_probs.shape

        new_topk_ids, new_topk_weights, _, reroute_stats = expert_level_io_free_reroute(
            router_probs, topk_ids, topk_weights,
            list(current_gpu), list(current_cpu),
            alpha=alpha, eps=1e-6,
            score_threshold_ratio=score_threshold_ratio,
        )

        self._log(f"[IOFree] new_topk_ids:\n{new_topk_ids}")
        if reroute_stats:
            self._log(f"[IOFree] reroute stats: {reroute_stats}")

        rerouted_topk_ids = new_topk_ids
        rerouted_topk_weights = new_topk_weights

        # 计算实际激活的专家集合
        valid = rerouted_topk_ids.view(-1)
        if valid.numel() == 0:
            actual_experts = set()
        else:
            valid = valid[valid >= 0]
            if valid.numel() == 0:
                actual_experts = set()
            else:
                load = torch.bincount(valid, minlength=N)
                actual_experts = set(torch.nonzero(load > 0).view(-1).cpu().tolist())

        # GPU/CPU 分配
        gpu_experts_after = actual_experts & current_gpu
        cpu_experts_after = actual_experts - current_gpu

        self._log(
            f"[IOFree] Layer {self._current_layer} after reroute: "
            f"Actual={sorted(actual_experts)}, "
            f"GPU={sorted(gpu_experts_after)}, "
            f"CPU={sorted(cpu_experts_after)}"
        )

        return rerouted_topk_ids, rerouted_topk_weights, actual_experts, cpu_experts_after, gpu_experts_after, reroute_stats

    def load_aware_token_reroute(
        self, router_probs, topk_ids, topk_weights,
        current_gpu, current_cpu,
        expert_load: Dict[int, float],
        alpha: float = 0.05,
        max_gpu_duplicates: int = 2,
        dominance_threshold: float = 0.5,
    ):
        '''
        负载感知的GPU优先重路由策略（token级别）。

        优先将 CPU 上负载最高的专家替换为得分相近的 GPU 专家，
        同时遵守重复次数限制、得分阈值和主导性约束。

        Args:
            router_probs:       [T, E] 路由概率矩阵
            topk_ids:           [T, k] 原始 top-k 专家 ID
            topk_weights:       [T, k] 对应权重
            current_gpu:        当前在 GPU 上的专家集合
            current_cpu:        当前在 CPU 上的专家集合
            expert_load:        专家历史负载字典 {expert_id: ema_load}
            alpha:              得分相对差距阈值
            max_gpu_duplicates: l，每个 GPU 专家在单个 token 中最大出现总次数
            dominance_threshold: 原专家得分占 topk 总得分比例超过此值则不替换
        Returns:
            rerouted_topk_ids, rerouted_topk_weights, actual_experts,
            cpu_experts_after, gpu_experts_after, reroute_stats
        '''
        from ..core.routing_redirection import token_level_load_aware_reroute

        T, N = router_probs.shape

        new_topk_ids, new_topk_weights, reroute_mask, reroute_stats = token_level_load_aware_reroute(
            router_probs, topk_ids, topk_weights,
            list(current_gpu), list(current_cpu),
            expert_load=expert_load,
            alpha=alpha,
            eps=1e-6,
            max_gpu_duplicates=max_gpu_duplicates,
            dominance_threshold=dominance_threshold,
        )

        self._log(f"[LoadAware] new_topk_ids:\n{new_topk_ids}")
        if reroute_stats:
            self._log(f"[LoadAware] reroute stats: {reroute_stats}")

        rerouted_topk_ids = new_topk_ids
        rerouted_topk_weights = new_topk_weights

        # 计算实际激活的专家集合
        valid = rerouted_topk_ids.view(-1)
        if valid.numel() == 0:
            actual_experts = set()
        else:
            valid = valid[valid >= 0]
            if valid.numel() == 0:
                actual_experts = set()
            else:
                load = torch.bincount(valid, minlength=N)
                actual_experts = set(torch.nonzero(load > 0).view(-1).cpu().tolist())

        # GPU/CPU 分配
        gpu_experts_after = actual_experts & current_gpu
        cpu_experts_after = actual_experts - current_gpu

        self._log(
            f"[LoadAware] Layer {self._current_layer} after reroute: "
            f"Actual={sorted(actual_experts)}, "
            f"GPU={sorted(gpu_experts_after)}, "
            f"CPU={sorted(cpu_experts_after)}"
        )

        return rerouted_topk_ids, rerouted_topk_weights, actual_experts, cpu_experts_after, gpu_experts_after, reroute_stats

    # ================================================================
    # 辅助接口
    # ================================================================
    
    def get_plan(self, layer_idx: int) -> Optional[LayerPlan]:
        """获取指定层的执行计划."""
        with self._lock:
            return self._plans.get(layer_idx)
    
    def get_planned_migrations(self, layer_idx: int) -> Set[int]:
        """
        获取规划的迁移专家列表.
        
        返回：规划放到 GPU 但当前不在 GPU 上的专家
        """
        with self._lock:
            plan = self._plans.get(layer_idx)
            if plan is None:
                return set()
            
            current_gpu = set()
            if self._gpu_state is not None:
                current_gpu = self._gpu_state.get_gpu_experts(layer_idx)
            
            return plan.planned_gpu_experts - current_gpu
    
    def mark_completed(self, layer_idx: int):
        """标记层执行完成."""
        with self._lock:
            plan = self._plans.get(layer_idx)
            if plan is not None:
                plan.phase = SchedulePhase.COMPLETED
    
    def cleanup_old_plans(self, keep_recent: int = 3):
        """清理旧的执行计划，避免内存泄漏."""
        with self._lock:
            if self._current_layer < 0:
                return
            
            to_remove = []
            to_keep = [(self._current_layer + i) % self.num_layers for i in range(keep_recent)]
            for layer_idx in self._plans:
                if layer_idx not in to_keep:
                    to_remove.append(layer_idx)
            
            for layer_idx in to_remove:
                del self._plans[layer_idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        with self._lock:
            stats = dict(self._stats)
            total = stats['prediction_hits'] + stats['prediction_misses']
            if total > 0:
                stats['prediction_accuracy'] = stats['prediction_hits'] / total
            return stats
    
    def set_gpu_state_provider(self, provider: GPUStateProvider):
        """设置 GPU 状态提供者."""
        self._gpu_state = provider

    def set_reroute_config(self, config: RerouteConfig):
        """运行时更新重路由配置."""
        self._reroute_config = config
        self._log(f'RerouteConfig updated: strategy={config.strategy}')
    
    def reload_config_from_file(self, config_path: str) -> bool:
        """
        从配置文件重新加载重路由配置.
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            是否成功重新加载
        """
        import os
        try:
            if not os.path.exists(config_path):
                self._log(f"Config file not found: {config_path}")
                return False
                
            from ..config import load_config
            cfg = load_config(config_path)
            
            # 保存配置路径用于后续检查
            self._config_path = config_path
            
            max_gpu_duplicates = cfg.get('reroute_max_gpu_duplicates')
            if max_gpu_duplicates is None:
                max_gpu_duplicates = cfg.get('max_gpu_duplicates', 2)
            max_dup_per_token = cfg.get('reroute_max_duplicates_per_expert')
            if max_dup_per_token is None:
                max_dup_per_token = max_gpu_duplicates
            dominance_threshold = cfg.get('reroute_dominance_threshold')
            if dominance_threshold is None:
                dominance_threshold = cfg.get('dominance_threshold', 0.5)

            new_config = RerouteConfig(
                strategy=cfg.get('reroute_strategy', 'token_reroute'),
                alpha=cfg.get('reroute_alpha', 0.05),
                allow_duplicate=cfg.get('reroute_allow_duplicate', False),
                use_limited_reroute=cfg.get('reroute_use_limited', True),
                max_duplicates_per_expert=max_dup_per_token,
                min_unique_experts=cfg.get('reroute_min_unique_experts', None),
                score_threshold_ratio=cfg.get('reroute_score_threshold_ratio', 0.5),
                max_gpu_duplicates=max_gpu_duplicates,
                dominance_threshold=dominance_threshold,
            )
            
            self.set_reroute_config(new_config)
            self._log(f"Successfully reloaded config from {config_path}: strategy={new_config.strategy}")
            return True
            
        except Exception as e:
            self._log(f"Failed to reload config from {config_path}: {e}")
            return False
    
    @property
    def reroute_config(self) -> RerouteConfig:
        """获取当前重路由配置."""
        return self._reroute_config


# ============================================================
# 全局实例管理
# ============================================================

_scheduler: Optional[ExpertScheduler] = None
_init_lock = threading.Lock()


def get_scheduler() -> Optional[ExpertScheduler]:
    """获取全局调度器实例."""
    return _scheduler


def init_scheduler(
    num_layers: int,
    num_experts: int,
    num_gpu_slots: int,
    gpu_state_provider: Optional[GPUStateProvider] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    reroute_config: Optional[RerouteConfig] = None,
) -> ExpertScheduler:
    """初始化全局调度器."""
    global _scheduler
    
    with _init_lock:
        if _scheduler is None:
            _scheduler = ExpertScheduler(
                num_layers=num_layers,
                num_experts=num_experts,
                num_gpu_slots=num_gpu_slots,
                gpu_state_provider=gpu_state_provider,
                log_fn=log_fn,
                reroute_config=reroute_config,
            )
        return _scheduler


def reset_scheduler():
    """重置全局调度器（主要用于测试）."""
    global _scheduler
    with _init_lock:
        _scheduler = None
