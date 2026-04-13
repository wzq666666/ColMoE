"""
Native Expert Migration Manager - 专门用于 native 缓存的专家迁移.

核心功能：
1. 从 HF checkpoint 加载专家权重
2. 转换权重格式 (HF → sglang Triton 格式)
3. 调用 NativeGPUCacheManager.swap_expert() 替换槽位中的专家

权重格式说明：
- HuggingFace 格式:
  - gate_proj.weight: [intermediate_size, hidden_size]
  - up_proj.weight: [intermediate_size, hidden_size]
  - down_proj.weight: [hidden_size, intermediate_size]

- sglang FusedMoE Triton 格式 (已转置，用于高效 GEMM):
  - w13_weight[slot]: [hidden_size, intermediate_size * 2]
    - 前半部分是 gate_proj.T，后半部分是 up_proj.T
  - w2_weight[slot]: [intermediate_size, hidden_size]
    - 即 down_proj.T
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
import time
import torch

from ..logger import log_once, append_log
from ..core.expert_resolver import ExpertResolver
from .native_gpu_cache import NativeGPUCacheManager, get_native_cache


class NativeMigrationStatus(Enum):
    """迁移任务状态."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NativeMigrationTask:
    """Native 缓存的迁移任务."""
    task_id: int
    layer_idx: int
    expert_idx: int
    target_slot: int  # 目标槽位
    status: NativeMigrationStatus = NativeMigrationStatus.PENDING
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None
    cancelled: bool = False
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


def convert_hf_to_sglang_format(
    w1: torch.Tensor,  # gate_proj: [intermediate, hidden]
    w2: torch.Tensor,  # down_proj: [hidden, intermediate]
    w3: torch.Tensor,  # up_proj: [intermediate, hidden]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 HuggingFace 权重格式转换为 sglang FusedMoE Triton 格式.
    
    Args:
        w1: gate_proj.weight [intermediate_size, hidden_size]
        w2: down_proj.weight [hidden_size, intermediate_size]
        w3: up_proj.weight [intermediate_size, hidden_size]
        
    Returns:
        (w13_weight, w2_weight):
        - w13_weight: [intermediate_size * 2, hidden_size] (gate + up 拼接)
        - w2_weight: [hidden_size, intermediate_size] (down 原样)
        
    Note:
        sglang FusedMoE 存储格式 (每个 slot):
        - w13_weight[slot]: [intermediate*2, hidden] = cat([gate_proj, up_proj], dim=0)
        - w2_weight[slot]: [hidden, intermediate] = down_proj
    """
    # sglang FusedMoE 的 w13 存储方式:
    # w13_weight[slot] = cat([gate_proj, up_proj], dim=0)
    # 即 [intermediate_size * 2, hidden_size]
    
    # gate_proj (w1): [intermediate, hidden]
    # up_proj (w3): [intermediate, hidden]
    # 拼接后: [intermediate*2, hidden]
    
    w13_weight = torch.cat([w1, w3], dim=0)  # [intermediate*2, hidden]
    
    # down_proj (w2): [hidden, intermediate] - 原样使用
    w2_weight = w2  # [hidden, intermediate]
    
    return w13_weight, w2_weight


def convert_hf_to_sglang_format_contiguous(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    转换并确保结果是 contiguous 的（某些 kernel 需要）.
    """
    w13_weight, w2_weight = convert_hf_to_sglang_format(w1, w2, w3)
    return w13_weight.contiguous(), w2_weight.contiguous()


class NativeExpertMigrationManager:
    """
    Native 缓存的专家迁移管理器.
    
    专门用于 native backend (使用 sglang FusedMoE kernel):
    1. 加载专家权重 (通过 ExpertResolver)
    2. 转换权重格式 (HF → sglang Triton)
    3. 替换槽位中的专家 (通过 NativeGPUCacheManager.swap_expert)
    """
    
    def __init__(
        self,
        expert_resolver: ExpertResolver,
        native_cache: Optional[NativeGPUCacheManager] = None,
        log_path: Optional[str] = None,
    ):
        """
        初始化 Native 迁移管理器.
        
        Args:
            expert_resolver: 用于加载 HF 权重的 resolver
            native_cache: NativeGPUCacheManager 实例 (可选，否则使用全局)
            log_path: 日志路径
        """
        self.expert_resolver = expert_resolver
        self._native_cache = native_cache
        self.log_path = log_path
        
        self._lock = threading.Lock()
        self._migration_count = 0
        self._task_counter = 0
        self._task_history: Dict[int, NativeMigrationTask] = {}
        
        # ========== Event-based 迁移追踪机制 ==========
        # 每个 (layer_idx, expert_idx) 对应一个 Event，当迁移完成时 set()
        self._migration_events: Dict[Tuple[int, int], threading.Event] = {}  # (layer, expert) -> Event
        self._migration_lock = threading.Lock()  # 保护 _migration_events
        
        # ========== 优化的预取取消机制 ==========
        # 仅取消未开始的 prefetch，保留已完成和正在进行的
        self._cancel_events: Dict[int, threading.Event] = {}  # layer_idx -> cancel_event
        self._active_prefetch_layers: Set[int] = set()  # 当前正在预取的层
        self._prefetch_completed: Dict[int, Set[int]] = {}  # layer_idx -> {completed expert_ids}
        self._prefetch_in_progress: Dict[int, Set[int]] = {}  # layer_idx -> {in-progress expert_ids}
        # self._prefetch_completed: Dict[int, torch.Tensor] = {}      # layer_idx -> [num_completed]
        # self._prefetch_in_progress: Dict[int, torch.Tensor] = {}    # layer_idx -> [num_in_progress]
        
        log_once(
            'native_migration_init',
            f'NativeExpertMigrationManager initialized'
        )
    
    @property
    def native_cache(self) -> Optional[NativeGPUCacheManager]:
        """获取 native cache 实例."""
        if self._native_cache is not None:
            return self._native_cache
        return get_native_cache()
    
    def cancel_prefetch(self, 
                        layer_idx: int, 
                        actual_experts: Optional[Set[int]] = None
                    ):
        """
        优化的预取取消：仅取消未命中的专家 IO，保留已完成和正在进行的.
        
        策略：
        1. 已完成的专家：保留（已在 GPU 上）
        2. 正在进行的专家：允许继续（避免浪费已投入的 IO）
        3. 未开始的专家：取消（避免无用 IO）
        
        Args:
            layer_idx: 层索引
            actual_experts: 实际需要的专家集合（用于判断哪些是命中的）
        """
        if actual_experts is not None:
            if isinstance(actual_experts, torch.Tensor):
                # 注意：放在非推理关键路径
                actual_experts = set(actual_experts.cpu().tolist())
            else:
                actual_experts = set(actual_experts)

        with self._lock:
            # 获取已完成和正在进行的专家
            completed = self._prefetch_completed.get(layer_idx, set())
            in_progress = self._prefetch_in_progress.get(layer_idx, set())
            
            # 如果提供了 actual_experts，计算命中和未命中
            if actual_experts is not None:
                hit = completed & actual_experts  # 预取命中
                miss = completed - actual_experts  # 预取未命中（可能需要替换）
                ongoing_hit = in_progress & actual_experts  # 正在进行且需要
                ongoing_miss = in_progress - actual_experts  # 正在进行但不需要
                
                if self.log_path:
                    append_log(
                        f'[Prefetch] Layer {layer_idx}: '
                        f'hit={len(hit)}, miss={len(miss)}, '
                        f'ongoing_hit={len(ongoing_hit)}, ongoing_miss={len(ongoing_miss)}',
                        self.log_path, level=2
                    )
            
            # 设置取消标志（未开始的任务会检查此标志）
            if layer_idx in self._cancel_events:
                self._cancel_events[layer_idx].set()
                if self.log_path:
                    append_log(
                        f'NativeMigration: cancel prefetch for layer {layer_idx} '
                        f'(keep {len(completed)} completed, {len(in_progress)} in-progress)',
                        self.log_path
                    )

    # def cancel_prefetch(
    #     self,
    #     layer_idx: int,
    #     actual_experts: Optional[torch.Tensor] = None,  # Tensor [num_actual]
    # ):
    #     """
    #     优化的预取取消：仅取消未命中的专家 IO，保留已完成和正在进行的。
    #     Tensor-native 版本，可在关键路径安全调用。

    #     策略：
    #         1. 已完成的专家：保留（已在 GPU 上）
    #         2. 正在进行的专家：允许继续（避免浪费已投入的 IO）
    #         3. 未开始的专家：取消（避免无用 IO）

    #     Args:
    #         layer_idx: 层索引
    #         actual_experts: 实际需要的专家 Tensor [num_actual]，可为 None
    #     """
    #     with self._lock:
    #         # --- 获取已完成和正在进行的专家 ---
    #         completed = self._prefetch_completed.get(layer_idx, torch.tensor([], dtype=torch.long))
    #         in_progress = self._prefetch_in_progress.get(layer_idx, torch.tensor([], dtype=torch.long))

    #         # --- 统计命中和未命中（Tensor版本） ---
    #         if actual_experts is not None and actual_experts.numel() > 0:
    #             # completed & actual_experts
    #             hit_mask = torch.isin(completed, actual_experts)
    #             hit = completed[hit_mask]

    #             # completed - actual_experts
    #             miss = completed[~hit_mask]

    #             # in_progress & actual_experts
    #             ongoing_hit_mask = torch.isin(in_progress, actual_experts)
    #             ongoing_hit = in_progress[ongoing_hit_mask]

    #             # in_progress - actual_experts
    #             ongoing_miss = in_progress[~ongoing_hit_mask]

    #             if self.log_path:
    #                 append_log(
    #                     f'[Prefetch] Layer {layer_idx}: '
    #                     f'hit={hit.numel()}, miss={miss.numel()}, '
    #                     f'ongoing_hit={ongoing_hit.numel()}, ongoing_miss={ongoing_miss.numel()}',
    #                     self.log_path, level=2
    #                 )
    #         else:
    #             hit = miss = ongoing_hit = ongoing_miss = torch.tensor([], dtype=torch.long)

    #         # --- 设置取消标志（未开始的任务会检查此标志） ---
    #         if layer_idx in self._cancel_events:
    #             self._cancel_events[layer_idx].set()
    #             if self.log_path:
    #                 append_log(
    #                     f'NativeMigration: cancel prefetch for layer {layer_idx} '
    #                     f'(keep {completed.numel()} completed, {in_progress.numel()} in-progress)',
    #                     self.log_path
    #                 )
    
    def load_expert_to_slot(
        self,
        layer_idx: int,
        expert_idx: int,
        slot_idx: int,
        device: str = "cuda",
    ) -> bool:
        """
        将专家加载到指定槽位.
        
        这是核心的专家加载函数：
        1. 从 HF checkpoint 加载权重 (可能走 CPU 缓存)
        2. 转换权重格式为 sglang Triton 格式
        3. 调用 native_cache.swap_expert() 替换槽位
        
        Args:
            layer_idx: 层索引
            expert_idx: 专家索引
            slot_idx: 目标槽位索引
            device: 中间设备 (用于格式转换)
            
        Returns:
            是否成功
        """
        cache = self.native_cache
        if cache is None:
            if self.log_path:
                append_log(
                    f'NativeMigration: cache not available for expert[{layer_idx}][{expert_idx}]',
                    self.log_path
                )
            return False
        
        try:
            # Step 1: 加载专家权重（已是 sglang 格式 + pinned memory）
            # ExpertResolver 现在直接返回 {'w13': pinned_tensor, 'w2': pinned_tensor}
            weights = self.expert_resolver.load_expert_weights_from_hf(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device="cpu",       # 加载到 CPU pinned memory
                use_cache=True,     # 使用 CPU 缓存
                cache_on_cpu=True,  # 缓存到 CPU 内存
            )
            
            if weights is None:
                if self.log_path:
                    append_log(
                        f'NativeMigration: failed to load weights for '
                        f'expert[{layer_idx}][{expert_idx}]',
                        self.log_path
                    )
                return False
            
            # ===== 关键优化：不再需要格式转换！=====
            # weights 已经是 sglang 格式且 pinned:
            # - weights['w13']: [intermediate*2, hidden], pinned
            # - weights['w2']: [hidden, intermediate], pinned
            # 
            # 删除旧代码:
            # w13_weight, w2_weight = convert_hf_to_sglang_format_contiguous(...)
            
            w13_weight = weights['w13']  # 已是 sglang 格式 + pinned
            w2_weight = weights['w2']    # 已是 sglang 格式 + pinned
            
            # Step 2: 直接传输到 GPU（享受 pinned memory DMA 加速）
            success = cache.swap_expert(
                layer_idx=layer_idx,
                slot_idx=slot_idx,
                new_expert_idx=expert_idx,
                w13_weight=w13_weight,  # ✅ pinned tensor, 直接异步 H2D
                w2_weight=w2_weight,    # ✅ pinned tensor, 直接异步 H2D
                non_blocking=True,      # 启用异步传输
                use_pinned_staging=False,  # ✅ 不需要 staging，源已 pinned
            )
            
            if success:
                with self._lock:
                    self._migration_count += 1
                
                # ✅ Event-based 同步：设置迁移完成 Event
                cache_key = (layer_idx, expert_idx)
                with self._migration_lock:
                    if cache_key not in self._migration_events:
                        self._migration_events[cache_key] = threading.Event()
                    self._migration_events[cache_key].set()  # 通知等待者
                
                if self.log_path:
                    append_log(
                        f'NativeMigration: loaded expert[{layer_idx}][{expert_idx}] '
                        f'to slot {slot_idx}',
                        self.log_path
                    )
            
            return success
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'NativeMigration: error loading expert[{layer_idx}][{expert_idx}]: {e}',
                    self.log_path
                )
            import traceback
            traceback.print_exc()
            return False
    
    def load_expert_to_slot_async(
        self,
        layer_idx: int,
        expert_idx: int,
        slot_idx: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> bool:
        """
        异步将专家加载到指定槽位 (优化版).
        
        优化点:
        1. 使用独立 CUDA stream 进行 H2D 传输
        2. 自动使用 native_cache 的 pinned staging buffer
        3. non_blocking copy 减少同步开销
        
        Args:
            layer_idx: 层索引
            expert_idx: 专家索引
            slot_idx: 目标槽位索引
            stream: 用于传输的 CUDA stream (可选)
            
        Returns:
            是否成功启动传输 (不等待完成)
        """
        cache = self.native_cache
        if cache is None:
            return False
        
        try:
            # Step 1: 加载专家权重（sglang 格式 + pinned）
            weights = self.expert_resolver.load_expert_weights_from_hf(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device="cpu",
                use_cache=True,
                cache_on_cpu=True,
            )
            
            if weights is None:
                return False
            
            # ===== 优化：权重已是 sglang 格式 + pinned，无需转换 =====
            w13_weight = weights['w13']
            w2_weight = weights['w2']
            
            # Step 2: 异步传输到 GPU（使用独立 stream）
            success = cache.swap_expert(
                layer_idx=layer_idx,
                slot_idx=slot_idx,
                new_expert_idx=expert_idx,
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                non_blocking=True,
                stream=stream,
                use_pinned_staging=False,  # 源已 pinned，不需要 staging
            )
            
            if success:
                with self._lock:
                    self._migration_count += 1
                
                # ✅ Event-based 同步：设置迁移完成 Event
                cache_key = (layer_idx, expert_idx)
                with self._migration_lock:
                    if cache_key not in self._migration_events:
                        self._migration_events[cache_key] = threading.Event()
                    self._migration_events[cache_key].set()
            
            return success
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'NativeMigration: async load expert[{layer_idx}][{expert_idx}] failed: {e}',
                    self.log_path
                )
            return False
    
    def swap_experts(
        self,
        layer_idx: int,
        old_expert_idx: int,
        new_expert_idx: int,
    ) -> bool:
        """
        替换一个槽位中的专家.
        
        找到 old_expert 所在的槽位，用 new_expert 替换它。
        
        Args:
            layer_idx: 层索引
            old_expert_idx: 要被替换的专家
            new_expert_idx: 新专家
            
        Returns:
            是否成功
        """
        cache = self.native_cache
        if cache is None:
            return False
        
        # 找到 old_expert 所在的槽位
        slot_idx = cache.get_slot_for_expert(layer_idx, old_expert_idx)
        if slot_idx is None:
            if self.log_path:
                append_log(
                    f'NativeMigration: expert {old_expert_idx} not on GPU '
                    f'in layer {layer_idx}',
                    self.log_path
                )
            return False
        
        return self.load_expert_to_slot(
            layer_idx=layer_idx,
            expert_idx=new_expert_idx,
            slot_idx=slot_idx,
        )
    
    def prefetch_experts(
        self,
        expert_list: List[Tuple[int, int]],  # [(layer_idx, expert_idx), ...]
        max_concurrent: int = 4,
        expert_scores: Optional[Dict[int, float]] = None,  # 专家分数（用于排序）
        exclude_experts: Optional[Set[int]] = None,  # 不应被替换的专家
        priority: int = 1,  # IO优先级：0=最高（当前层），1-2=中等（预取），>2=低（背景）
    ) -> int:
        """
        预取专家到 GPU（异步 pipeline 模式）.
        
        类似 HybriMoE 的 prefetch_expert 机制:
        - 在当前层计算时，异步加载下一层的专家
        - 使用独立 stream 避免阻塞计算
        - 支持按分数排序，优先加载高分专家
        
        Args:
            expert_list: 要预取的专家列表 [(layer_idx, expert_idx), ...]
            max_concurrent: 最大并发加载数
            expert_scores: 专家分数字典 {expert_idx: score}，用于优先级排序
            exclude_experts: 不应被替换的专家集合（用于槽位选择）
            priority: IO优先级（0=最高，数值越大优先级越低）
            
        Returns:
            成功预取的数量
        """
        cache = self.native_cache
        if cache is None:
            return 0
        
        # 按层分组
        layer_experts: Dict[int, List[int]] = {}
        for layer_idx, expert_idx in expert_list:
            if layer_idx not in layer_experts:
                layer_experts[layer_idx] = []
            if expert_idx not in layer_experts[layer_idx]:
                layer_experts[layer_idx].append(expert_idx)
        
        success_count = 0
        
        for layer_idx, experts in layer_experts.items():
            # 为该层创建取消事件
            with self._lock:
                if layer_idx not in self._cancel_events:
                    self._cancel_events[layer_idx] = threading.Event()
                self._active_prefetch_layers.add(layer_idx)
                cancel_event = self._cancel_events[layer_idx]
            
            # 等待目标层被注册（最多等待 500ms）
            max_wait_ms = 500
            wait_interval_ms = 10
            waited_ms = 0
            while not cache.is_layer_registered(layer_idx) and waited_ms < max_wait_ms:
                time.sleep(wait_interval_ms / 1000.0)
                waited_ms += wait_interval_ms
            
            if not cache.is_layer_registered(layer_idx):
                if self.log_path:
                    append_log(
                        f'NativeMigration: layer {layer_idx} not registered, skipping prefetch',
                        self.log_path
                    )
                continue
            
            # 检查哪些已经在 GPU 上
            current_gpu = cache.get_gpu_experts(layer_idx)
            to_load = [exp for exp in experts if exp not in current_gpu]
            
            if not to_load:
                success_count += len(experts)  # 已经在 GPU 上
                continue
            
            # 按分数排序（如果提供了 expert_scores）
            if expert_scores:
                to_load = sorted(
                    to_load,
                    key=lambda x: expert_scores.get(x, 0.0),
                    reverse=True
                )
            
            # 准备加载列表
            migrations = []
            # 使用 get_available_slots 获取可替换的槽位（基于 LRU）
            # exclude_experts 参数指定哪些专家不应被替换
            exclude_set = exclude_experts or set()
            # 注意：不需要 exclude to_load，因为它们不在 current_gpu 中
            
            available_slots = cache.get_available_slots(
                layer_idx=layer_idx,
                exclude_experts=exclude_set,
                num_slots=min(len(to_load), max_concurrent)
            )
            
            # 边界检查：确保槽位索引合法
            num_slots_total = cache.num_gpu_slots
            for expert_idx, slot_idx in zip(to_load[:len(available_slots)], available_slots):
                # ✅ 检查是否已取消（在标记为 in_progress 之前）
                if cancel_event.is_set():
                    if self.log_path:
                        append_log(
                            f'NativeMigration: prefetch cancelled for layer {layer_idx}, '
                            f'skipping remaining {len(to_load) - len(migrations)} experts',
                            self.log_path
                        )
                    break
                
                if slot_idx < 0 or slot_idx >= num_slots_total:
                    if self.log_path:
                        append_log(
                            f'NativeMigration: ERROR invalid slot_idx {slot_idx} '
                            f'for layer {layer_idx} (valid: 0-{num_slots_total-1})',
                            self.log_path
                        )
                    continue
                
                # ✅ 标记为 in_progress（在真正开始加载之前）
                with self._lock:
                    if layer_idx not in self._prefetch_in_progress:
                        self._prefetch_in_progress[layer_idx] = set()
                        # self._prefetch_in_progress[layer_idx] = torch.tensor([], dtype=torch.long)
                    self._prefetch_in_progress[layer_idx].add(expert_idx)
                    # new_expert = torch.tensor([expert_idx], dtype=torch.long, device=self._prefetch_in_progress[layer_idx].device)
                    # self._prefetch_in_progress[layer_idx] = torch.unique(
                    #     torch.cat([self._prefetch_in_progress[layer_idx], new_expert])
                    # )
                
                migrations.append((layer_idx, expert_idx, slot_idx))
            
            # 异步加载（使用指定的优先级）
            if migrations:
                # ✅ 提前准备权重（避免在 _batch_load_with_stream 中重复加载）
                swaps = []
                swap_to_expert = {}
                for layer_idx, expert_idx, slot_idx in migrations:
                    try:
                        # 加载权重（走 CPU 缓存）
                        weights = self.expert_resolver.load_expert_weights_from_hf(
                            layer_idx=layer_idx,
                            expert_idx=expert_idx,
                            device="cpu",
                            use_cache=True,
                            cache_on_cpu=True,
                        )
                        
                        if weights is None:
                            continue
                        
                        # 权重已是 sglang 格式 + pinned
                        w13_weight = weights['w13']
                        w2_weight = weights['w2']
                        
                        swap_idx = len(swaps)
                        swaps.append((layer_idx, slot_idx, expert_idx, w13_weight, w2_weight))
                        swap_to_expert[swap_idx] = (layer_idx, expert_idx)
                        
                    except Exception as e:
                        if self.log_path:
                            append_log(
                                f'NativeMigration: failed to prepare expert[{layer_idx}][{expert_idx}]: {e}',
                                self.log_path
                            )
                        continue
                
                # 调用 _batch_load_with_stream，传入已准备好的 5元组
                loaded_count, successful_experts = self._batch_load_with_stream(
                    swaps,  # 5元组，已包含权重
                    stream=self._get_priority_stream(priority), 
                    priority=priority,
                    threading_events=self._migration_events
                )
                success_count += loaded_count
                
                # ✅ 将成功加载的专家从 in_progress 移到 completed
                with self._lock:
                    if layer_idx not in self._prefetch_completed:
                        self._prefetch_completed[layer_idx] = set()
                        # self._prefetch_completed[layer_idx] = torch.tensor([], dtype=torch.long)
                    # if layer_idx not in self._prefetch_in_progress:
                    #     self._prefetch_in_progress[layer_idx] = torch.tensor([], dtype=torch.long)
                    
                    # 使用返回的具体成功专家列表
                    successful_expert_ids = {exp_idx for lay_idx, exp_idx in successful_experts if lay_idx == layer_idx}
                    # successful_expert_ids = torch.tensor(
                    #     [exp_idx for lay, exp_idx in successful_experts if lay == layer_idx],
                    #     dtype=torch.long,
                    #     device=self._prefetch_completed[layer_idx].device
                    # )
                    
                    for exp_idx in successful_expert_ids:
                        if layer_idx in self._prefetch_in_progress:
                            self._prefetch_in_progress[layer_idx].discard(exp_idx)
                        self._prefetch_completed[layer_idx].add(exp_idx)
                    # if successful_expert_ids.numel() > 0:
                    #     # 从 in_progress 移除
                    #     mask = ~torch.isin(self._prefetch_in_progress[layer_idx], successful_expert_ids)
                    #     self._prefetch_in_progress[layer_idx] = self._prefetch_in_progress[layer_idx][mask]

                    #     # 添加到 completed
                    #     self._prefetch_completed[layer_idx] = torch.unique(
                    #         torch.cat([self._prefetch_completed[layer_idx], successful_expert_ids])
                    #     )
                    
                    # 清理失败的任务（从 in_progress 移除）
                    all_attempted = {exp_idx for _, exp_idx, _ in migrations}
                    failed_experts = all_attempted - successful_expert_ids
                    for exp_idx in failed_experts:
                        if layer_idx in self._prefetch_in_progress:
                            self._prefetch_in_progress[layer_idx].discard(exp_idx)
                    # all_attempted = torch.tensor(
                    #     [exp_idx for _, exp_idx, _ in migrations],
                    #     dtype=torch.long,
                    #     device=self._prefetch_in_progress[layer_idx].device
                    # )

                    # failed_experts = torch.tensor(
                    #     [exp_idx for exp_idx in all_attempted.tolist() if exp_idx not in successful_expert_ids.tolist()],
                    #     dtype=torch.long,
                    #     device=self._prefetch_in_progress[layer_idx].device
                    # )

                    # if failed_experts.numel() > 0:
                    #     mask = ~torch.isin(self._prefetch_in_progress[layer_idx], failed_experts)
                    #     self._prefetch_in_progress[layer_idx] = self._prefetch_in_progress[layer_idx][mask]
            
            # 清理该层的取消事件
            with self._lock:
                if layer_idx in self._active_prefetch_layers:
                    self._active_prefetch_layers.remove(layer_idx)
                # 保留一段时间再删除 event，避免竞争
                # self._cancel_events 会在后续统一清理
        
        if self.log_path:
            total_requested = sum(len(exps) for exps in layer_experts.values())
            append_log(
                f'NativeMigration: prefetch completed, '
                f'{success_count}/{total_requested} experts prefetched',
                self.log_path
            )
        
        return success_count
    
    def trigger_immediate_migration(
        self,
        layer_idx: int,
        expert_ids: List[int],
        priority: int = 0,
    ):
        """
        立即触发专家迁移到GPU（高优先级，用于当前层计算）.
        
        与 prefetch_experts 的区别：
        - prefetch 是低优先级的预加载（为未来层准备）
        - trigger_immediate 是高优先级的即时迁移（当前层需要）
        
        优先级机制：
        - priority=0: 使用高优先级 CUDA Stream (当前层计算)
        - priority>0: 使用低优先级 CUDA Stream (预取)
        
        Args:
            layer_idx: 层索引
            expert_ids: 需要迁移的专家ID列表
            priority: IO任务优先级（0=最高优先级，用于当前层计算）
        """
        cache = self.native_cache
        if cache is None:
            return
        
        if not expert_ids:
            return
        
        # 等待层注册
        max_wait_ms = 500
        wait_interval_ms = 10
        waited_ms = 0
        while not cache.is_layer_registered(layer_idx) and waited_ms < max_wait_ms:
            time.sleep(wait_interval_ms / 1000.0)
            waited_ms += wait_interval_ms
        
        if not cache.is_layer_registered(layer_idx):
            if self.log_path:
                append_log(
                    f'NativeMigration: layer {layer_idx} not registered for immediate migration',
                    self.log_path
                )
            return
        
        # 筛选需要加载的专家（排除已在GPU的）
        current_gpu = cache.get_gpu_experts(layer_idx)
        to_load = [exp for exp in expert_ids if exp not in current_gpu]
        
        if not to_load:
            if self.log_path:
                append_log(
                    f'NativeMigration: all experts already on GPU for layer {layer_idx}',
                    self.log_path
                )
            return
        
        # 获取可用槽位
        available_slots = cache.get_available_slots(
            layer_idx=layer_idx,
            exclude_experts=current_gpu,  # 保护已在GPU的专家
            num_slots=len(to_load)
        )
        
        # 构建迁移任务
        migrations = []
        num_slots_total = cache.num_gpu_slots
        for expert_idx, slot_idx in zip(to_load[:len(available_slots)], available_slots):
            if 0 <= slot_idx < num_slots_total:
                migrations.append((layer_idx, expert_idx, slot_idx))
        
        if migrations:
            # ========== 关键：根据优先级选择 CUDA Stream ==========
            # 创建或获取对应优先级的 Stream
            stream = self._get_priority_stream(priority)
            
            # 使用指定优先级的 stream 进行异步加载
            loaded, _ = self._batch_load_with_stream(migrations, stream, priority, threading_events=self._migration_events)
            
            if self.log_path:
                append_log(
                    f'NativeMigration: immediate migration for layer {layer_idx}, '
                    f'loaded {loaded}/{len(to_load)} experts (priority={priority}, '
                    f'stream_priority={stream.priority if hasattr(stream, "priority") else "default"})',
                    self.log_path
                )
    
    def _get_priority_stream(self, priority: int) -> torch.cuda.Stream:
        """
        获取指定优先级的 CUDA Stream.
        
        优先级映射：
        - priority=0 (最高): CUDA stream priority = low (最高优先级，数值最小如-3)
        - priority=1-2 (中): CUDA stream priority = 中间值
        - priority>2 (低): CUDA stream priority = high (最低优先级，数值最大如0)
        
        Returns:
            对应优先级的 CUDA Stream
        """
        # 检查是否已缓存该优先级的 stream
        if not hasattr(self, '_priority_streams'):
            self._priority_streams = {}
        
        if priority in self._priority_streams:
            return self._priority_streams[priority]
        
        # 获取 GPU 支持的优先级范围
        low, high = torch.cuda.Stream.priority_range()
        # low 是最高优先级（数值最小，如-3），high 是最低优先级（数值最大，如0）
        
        # 根据任务优先级映射到 CUDA Stream 优先级
        if priority == 0:
            # 最高优先级任务 -> 使用最高优先级 Stream
            cuda_priority = low  # 数值最小，如-3
        elif priority <= 2:
            # 中等优先级 -> 使用中间优先级
            # 线性插值: priority 1-2 映射到 low 和 high 之间
            if low < high:
                # 计算中间值 (priority 1 -> 靠近 low, priority 2 -> 靠近 high)
                cuda_priority = low + (high - low) * priority // 3
            else:
                cuda_priority = high  # 如果 GPU 不支持多级优先级，使用默认
        else:
            # 低优先级 -> 使用最低优先级 Stream
            cuda_priority = high  # 数值最大，如0
        
        # 创建对应优先级的 Stream
        stream = torch.cuda.Stream(priority=cuda_priority)
        self._priority_streams[priority] = stream
        
        if self.log_path:
            append_log(
                f'NativeMigration: created CUDA stream for priority {priority} '
                f'(CUDA priority={cuda_priority}, range=[{low}, {high}])',
                self.log_path,
                level=3
            )
        
        return stream
    
    def _batch_load_with_stream(
        self,
        migrations: List[Tuple],
        stream: torch.cuda.Stream,
        priority: int,
        threading_events: Optional[Dict[Tuple[int, int], threading.Event]] = None,  # ✅ 新增参数
    ) -> Tuple[int, List[Tuple[int, int]]]:
        """
        使用指定 Stream 批量加载专家.
        
        Args:
            migrations: 可以是以下两种格式之一：
                - [(layer_idx, expert_idx, slot_idx), ...] (3元组，需要加载权重)
                - [(layer_idx, slot_idx, expert_idx, w13_weight, w2_weight), ...] (5元组，已有权重)
            stream: 用于传输的 CUDA Stream
            priority: 任务优先级（用于日志）
            threading_events: Event 字典，用于在传输完成后通知等待者
            
        Returns:
            (成功加载的数量, 成功的专家列表 [(layer_idx, expert_idx), ...])
        """
        cache = self.native_cache
        if cache is None:
            return 0
        
        if not migrations:
            return 0
        
        # 检查第一个元素的长度，判断是3元组还是5元组
        first_item = migrations[0]
        is_prepared_swaps = len(first_item) == 5
        
        swap_to_expert = {}  # swap_idx -> (layer_idx, expert_idx)
        
        if is_prepared_swaps:
            # 已经准备好权重的 swaps，直接使用
            swaps = migrations
            cpu_prepare_ms = 0.0
            # 从 swaps 中提取映射关系 (格式: layer_idx, slot_idx, expert_idx, w13, w2)
            for idx, swap in enumerate(swaps):
                swap_to_expert[idx] = (swap[0], swap[2])  # (layer_idx, expert_idx)
        else:
            # 未准备权重的 migrations，需要加载
            cpu_start_ms = time.time() * 1000.0
            swaps = []
            for migration in migrations:
                layer_idx, expert_idx, slot_idx = migration
                try:
                    # 加载权重
                    weights = self.expert_resolver.load_expert_weights_from_hf(
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        device="cpu",
                        use_cache=True,
                        cache_on_cpu=True,
                    )
                    
                    if weights is None:
                        continue
                    
                    # ===== 优化：权重已是 sglang 格式，无需转换 =====
                    w13_weight = weights['w13']
                    w2_weight = weights['w2']
                    
                    swap_idx = len(swaps)
                    swaps.append((layer_idx, slot_idx, expert_idx, w13_weight, w2_weight))
                    swap_to_expert[swap_idx] = (layer_idx, expert_idx)
                    
                except Exception as e:
                    if self.log_path:
                        append_log(
                            f'NativeMigration: failed to prepare expert[{layer_idx}][{expert_idx}]: {e}',
                            self.log_path
                        )
                    continue
            cpu_prepare_ms = time.time() * 1000.0 - cpu_start_ms
        
        # 使用 batch_swap_experts 进行批量传输
        h2d_start_ms = time.time() * 1000.0
        # 直接传递 stream 参数，而非使用 with context
        success_count = cache.batch_swap_experts(
            swaps=swaps,
            use_pipeline=False,  # pipeline 优化，会开多个stream
            priority_stream=stream,  # ✅ 传递优先级 stream
            sync_after=False,  # ✅ 不同步，由 Event 机制处理
            threading_events=threading_events,  # ✅ 传入 Event 字典
        )
        h2d_end_ms = time.time() * 1000.0
        h2d_time_ms = h2d_end_ms - h2d_start_ms
        total_time_ms = h2d_time_ms + cpu_prepare_ms
        
        # 计算传输带宽
        if success_count > 0 and h2d_time_ms > 0:
            # 每个专家约52.5MB
            total_mb = success_count * 52.5
            bandwidth_gbps = (total_mb / 1024.0) / (h2d_time_ms / 1000.0)
            
            if self.log_path:
                append_log(
                    f"NativeMigration: batch load {len(swaps)} experts with priority={priority}. "
                    f"CPU: {cpu_prepare_ms:.2f}ms, H2D: {h2d_time_ms:.2f}ms ({bandwidth_gbps:.2f} GB/s), "
                    f"total: {total_time_ms:.2f}ms",
                    self.log_path,
                    level=3
                )
        else:
            if self.log_path:
                append_log(
                    f"NativeMigration: batch load {len(swaps)} experts with priority={priority}. "
                    f"CPU: {cpu_prepare_ms:.2f}ms, H2D: {h2d_time_ms:.2f}ms, total: {total_time_ms:.2f}ms",
                    self.log_path,
                    level=3
                )
        with self._lock:
            self._migration_count += success_count
        
        # ✅ 返回成功的专家列表
        successful_experts = [swap_to_expert[i] for i in range(min(success_count, len(swap_to_expert)))]
        
        if self.log_path:
            append_log(
                f'NativeMigration: batch load with priority={priority} stream completed, '
                f'{success_count}/{len(migrations)} succeeded',
                self.log_path,
                level=3
            )
        
        return success_count, successful_experts
    
    def wait_for_experts(
        self,
        layer_idx: int,
        expert_ids: List[int],
        timeout: float = 10.0,
    ):
        """
        等待指定专家的迁移完成（Event-based，零轮询）.
        
        使用 Event 机制：
        1. 每个 (layer_idx, expert_idx) 有一个 Event
        2. 迁移完成时 Event.set()
        3. 此处 Event.wait(timeout) 高效等待
        
        Args:
            layer_idx: 层索引
            expert_ids: 专家ID列表
            timeout: 超时时间（秒，每个专家独立计时）
        """
        cache = self.native_cache
        if cache is None:
            return
        
        for expert_idx in expert_ids:
            start_time = time.time()
            
            # ✅ 快速路径：检查是否已在 GPU
            if cache.is_expert_on_gpu(layer_idx, expert_idx):
                # 已在 GPU，无需等待
                continue
            
            # ✅ 优化：使用 Event 机制等待迁移完成
            cache_key = (layer_idx, expert_idx)
            
            # 获取或创建该专家的 Event
            with self._migration_lock:
                if cache_key not in self._migration_events:
                    self._migration_events[cache_key] = threading.Event()
                event = self._migration_events[cache_key]
            
            # ✅ Event.wait() 高效阻塞，零 CPU 消耗
            if not event.wait(timeout=timeout):
                # 超时
                if self.log_path:
                    append_log(
                        f'NativeMigration: timeout waiting for expert[{layer_idx}][{expert_idx}] '
                        f'(waited {time.time() - start_time:.2f}s)',
                        self.log_path
                    )
            else:
                # 成功等到
                elapsed = time.time() - start_time
                if self.log_path and elapsed > 0.01:  # 仅记录显著等待
                    append_log(
                        f'NativeMigration: expert[{layer_idx}][{expert_idx}] ready '
                        f'(waited {elapsed*1000:.1f}ms)',
                        self.log_path, level=3
                    )
    
    def batch_swap_experts(
        self,
        swaps: List[Tuple[int, int, int]],  # [(layer_idx, old_expert, new_expert), ...]
    ) -> int:
        """
        批量替换专家.
        
        Args:
            swaps: 替换列表，每项为 (layer_idx, old_expert_idx, new_expert_idx)
            
        Returns:
            成功替换的数量
        """
        success_count = 0
        
        for layer_idx, old_expert, new_expert in swaps:
            if self.swap_experts(layer_idx, old_expert, new_expert):
                success_count += 1
        
        if self.log_path:
            append_log(
                f'NativeMigration: batch swap completed, '
                f'{success_count}/{len(swaps)} succeeded',
                self.log_path
            )
        
        return success_count
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """获取迁移统计信息."""
        with self._lock:
            return {
                'total_migrations': self._migration_count,
                'task_history_size': len(self._task_history),
            }


# 全局实例
_native_migration_manager: Optional[NativeExpertMigrationManager] = None
_native_migration_lock = threading.Lock()


def get_native_migration_manager() -> Optional[NativeExpertMigrationManager]:
    """获取全局 Native 迁移管理器."""
    return _native_migration_manager


def init_native_migration_manager(
    expert_resolver: ExpertResolver,
    native_cache: Optional[NativeGPUCacheManager] = None,
    log_path: Optional[str] = None,
) -> NativeExpertMigrationManager:
    """
    初始化全局 Native 迁移管理器.
    
    Args:
        expert_resolver: 用于加载 HF 权重的 resolver
        native_cache: NativeGPUCacheManager 实例 (可选)
        log_path: 日志路径
        
    Returns:
        迁移管理器实例
    """
    global _native_migration_manager
    
    with _native_migration_lock:
        if _native_migration_manager is None:
            _native_migration_manager = NativeExpertMigrationManager(
                expert_resolver=expert_resolver,
                native_cache=native_cache,
                log_path=log_path,
            )
            
            if log_path:
                append_log(
                    'NativeExpertMigrationManager initialized',
                    log_path
                )
        
        return _native_migration_manager
