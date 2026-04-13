"""
Native GPU Cache Manager - 直接操作 sglang FusedMoE 权重实现动态专家调度.

核心思路：
1. 复用 sglang 的 w13_weight/w2_weight tensor 作为 GPU 缓存
2. 通过直接 copy_ 操作实现专家热替换
3. 维护 expert_idx → slot_idx 映射

优势：
- 使用 FusedMoE Triton kernel 的全部优化
- 零额外 GPU 内存开销
- 与 sglang 原生实现完全兼容
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import threading
import torch
from collections import deque

from ..logger import log_once, append_log


@dataclass
class NativeGPUSlot:
    """表示 sglang FusedMoE 中的一个专家槽位."""
    slot_idx: int          # 槽位索引 (0 ~ num_gpu_experts-1)
    expert_idx: int = -1   # 当前加载的专家ID (-1 表示空)
    is_dirty: bool = False # 是否被修改过 (用于追踪)
    last_access_time: float = 0.0  # 最后访问时间 (用于 LRU)


@dataclass
class LayerNativeCache:
    """单层的原生 GPU 缓存."""
    layer_idx: int
    num_slots: int
    slots: List[NativeGPUSlot] = field(default_factory=list)
    
    # 映射：expert_idx → slot_idx
    expert_to_slot: Dict[int, int] = field(default_factory=dict)
    
    # 反向映射：slot_idx → expert_idx (原始映射，用于恢复)
    original_slot_mapping: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self.slots = [
            NativeGPUSlot(slot_idx=i, expert_idx=i)  # 初始: slot_i 存 expert_i
            for i in range(self.num_slots)
        ]
        # 初始化映射
        for i in range(self.num_slots):
            self.expert_to_slot[i] = i
            self.original_slot_mapping[i] = i


class NativeGPUCacheManager:
    """
    原生 GPU 缓存管理器 - 直接操作 sglang 的 FusedMoE 权重.
    
    工作原理：
    1. sglang 使用 --kt-num-gpu-experts N 时，会为前 N 个专家分配 GPU 内存
    2. 这些权重存储在 layer.w13_weight[0:N] 和 layer.w2_weight[0:N]
    3. 我们可以直接用 .copy_() 替换这些权重来实现动态专家调度
    4. 同时需要维护 topk_ids 的重映射，让 kernel 使用正确的槽位
    
    使用方式：
    1. 初始化时设置 num_gpu_slots (= --kt-num-gpu-experts)
    2. 调用 register_layer() 注册每层的 MoE 模块
    3. 调用 swap_expert() 替换槽位中的专家
    4. 在推理时调用 remap_topk_ids() 重映射专家ID
    
    IO 优化 (仿照 HybriMoE):
    - Pinned memory pool: 预分配锁页内存，加速 H2D 传输 (3-4x 带宽)
    - 异步 CUDA streams: 独立 stream 进行传输，避免阻塞计算
    - Event-based 同步: 精确控制依赖关系，减少等待
    - 批量传输优化: 合并小传输，减少 PCIe overhead
    """
    
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        num_gpu_slots: int,  # = --kt-num-gpu-experts
        log_path: Optional[str] = None,
        # IO 优化参数
        enable_pinned_memory: bool = True,
        num_transfer_streams: int = 2,  # 传输用的 stream 数量
        # CPU 侧 pinned pool 大小（用于 ExpertResolver 共享）
        cpu_pinned_pool_size: int = 0,  # 0 表示只分配 GPU slots 的 pinned buffer
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_gpu_slots = num_gpu_slots
        self.log_path = log_path
        self.enable_pinned_memory = enable_pinned_memory
        self.cpu_pinned_pool_size = cpu_pinned_pool_size
        
        self._lock = threading.Lock()
        
        # 每层的缓存状态
        self._layer_caches: Dict[int, LayerNativeCache] = {}
        
        # 每层的 MoE 模块引用
        self._layer_modules: Dict[int, torch.nn.Module] = {}
        
        # 统计
        self._swap_count = 0
        
        # 标记是否有任何层发生过专家替换
        self._any_swap_happened = False
        
        # ===== IO 优化: CUDA Streams =====
        # 使用独立 stream 进行专家传输，避免阻塞计算 stream
        self._transfer_streams: List[torch.cuda.Stream] = []
        self._stream_pool: deque = deque()  # 可用 stream 池
        for i in range(num_transfer_streams):
            stream = torch.cuda.Stream()
            self._transfer_streams.append(stream)
            self._stream_pool.append(stream)
        
        # ===== IO 优化: 统一 Pinned Memory Pool (HybriMoE-style) =====
        # 为 H2D 传输和 CPU 缓存提供统一的 pinned memory pool
        # 分为两部分：
        # 1. GPU transfer buffers: num_gpu_slots 个槽位用于 H2D 传输
        # 2. CPU cache pool: cpu_pinned_pool_size 个槽位用于 ExpertResolver 缓存
        self._pinned_w13_storage: Optional[torch.UntypedStorage] = None  # 连续存储
        self._pinned_w2_storage: Optional[torch.UntypedStorage] = None
        self._pinned_w13_views: List[torch.Tensor] = []  # GPU transfer views (0 ~ num_gpu_slots-1)
        self._pinned_w2_views: List[torch.Tensor] = []   # GPU transfer views
        self._pinned_buffer_initialized = False
        
        # CPU cache pool views (num_gpu_slots ~ num_gpu_slots+cpu_pinned_pool_size-1)
        self._cpu_pinned_w13_views: List[torch.Tensor] = []  # CPU cache views
        self._cpu_pinned_w2_views: List[torch.Tensor] = []   # CPU cache views
        self._cpu_pinned_slot_map: Dict[Tuple[int, int], int] = {}  # (layer, expert) -> cpu_slot_idx
        
        # ===== IO 优化: GPU 侧 Storage Views (避免重复索引) =====
        # 预创建 GPU tensor views，避免运行时 .data[slot_idx] 开销
        self._gpu_w13_views: Dict[int, List[torch.Tensor]] = {}  # layer_idx -> [view per slot]
        self._gpu_w2_views: Dict[int, List[torch.Tensor]] = {}
        
        # ===== IO 优化: Event-based 同步 =====
        # 每个槽位对应的传输完成 events
        self._transfer_events: Dict[int, List[torch.cuda.Event]] = {}  # layer_idx -> [event per slot]
        
        log_once(
            'native_cache_init',
            f'NativeGPUCacheManager: {num_layers} layers, '
            f'{num_experts} experts, {num_gpu_slots} GPU slots, '
            f'pinned_memory={enable_pinned_memory}, streams={num_transfer_streams}'
        )
    
    def is_default_mapping(self, layer_idx: int) -> bool:
        """
        检查指定层是否使用默认映射（expert_i -> slot_i）。
        
        如果是默认映射，可以直接使用原始 apply，无需重映射。
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return True  # 未注册的层使用默认映射
            
            # 检查是否所有 slot 都是默认映射
            for slot in cache.slots:
                if slot.slot_idx != slot.expert_idx:
                    return False
            return True
    
    def all_layers_default_mapping(self) -> bool:
        """检查是否所有层都使用默认映射。"""
        return not self._any_swap_happened
    
    def _init_pinned_buffers(
        self, 
        w13_shape: torch.Size, 
        w2_shape: torch.Size,
        target_dtype: torch.dtype
    ) -> None:
        """
        初始化统一的 pinned memory 缓冲区（HybriMoE-style）.
        
        优化策略：
        1. 使用 UntypedStorage 分配连续内存（避免碎片化）
        2. 为 GPU transfer 和 CPU cache 预创建 tensor views
        3. 使用 pin_memory() 实现零拷贝 DMA 传输
        
        内存布局：
        [GPU transfer buffers: 0 ~ num_gpu_slots-1]
        [CPU cache pool: num_gpu_slots ~ num_gpu_slots+cpu_pinned_pool_size-1]
        
        Args:
            w13_shape: 单个专家的 w13_weight shape [intermediate*2, hidden]
            w2_shape: 单个专家的 w2_weight shape [hidden, intermediate]
            target_dtype: 目标 GPU tensor 的 dtype
        """
        if self._pinned_buffer_initialized or not self.enable_pinned_memory:
            return
        
        try:
            # 计算单个专家所需的字节数
            element_size = torch.tensor([], dtype=target_dtype).element_size()
            w13_bytes_per_expert = w13_shape.numel() * element_size
            w2_bytes_per_expert = w2_shape.numel() * element_size
            
            # ===== 统一 pinned memory pool: GPU transfer + CPU cache =====
            # 总槽位数 = GPU transfer buffers + CPU cache pool
            total_slots = self.num_gpu_slots + self.cpu_pinned_pool_size
            total_w13_bytes = w13_bytes_per_expert * total_slots
            total_w2_bytes = w2_bytes_per_expert * total_slots
            
            # 创建统一的 pinned UntypedStorage
            self._pinned_w13_storage = torch.UntypedStorage(total_w13_bytes).pin_memory()
            self._pinned_w2_storage = torch.UntypedStorage(total_w2_bytes).pin_memory()
            
            # ===== Part 1: GPU transfer buffers (0 ~ num_gpu_slots-1) =====
            self._pinned_w13_views = []
            self._pinned_w2_views = []
            
            for i in range(self.num_gpu_slots):
                w13_storage_slice = self._pinned_w13_storage[
                    i * w13_bytes_per_expert : (i + 1) * w13_bytes_per_expert
                ]
                w13_view = torch.as_tensor(
                    w13_storage_slice, dtype=target_dtype, device='cpu'
                ).view(w13_shape)
                self._pinned_w13_views.append(w13_view)
                
                w2_storage_slice = self._pinned_w2_storage[
                    i * w2_bytes_per_expert : (i + 1) * w2_bytes_per_expert
                ]
                w2_view = torch.as_tensor(
                    w2_storage_slice, dtype=target_dtype, device='cpu'
                ).view(w2_shape)
                self._pinned_w2_views.append(w2_view)
            
            # ===== Part 2: CPU cache pool (num_gpu_slots ~ total_slots-1) =====
            self._cpu_pinned_w13_views = []
            self._cpu_pinned_w2_views = []
            
            for i in range(self.num_gpu_slots, total_slots):
                w13_storage_slice = self._pinned_w13_storage[
                    i * w13_bytes_per_expert : (i + 1) * w13_bytes_per_expert
                ]
                w13_view = torch.as_tensor(
                    w13_storage_slice, dtype=target_dtype, device='cpu'
                ).view(w13_shape)
                self._cpu_pinned_w13_views.append(w13_view)
                
                w2_storage_slice = self._pinned_w2_storage[
                    i * w2_bytes_per_expert : (i + 1) * w2_bytes_per_expert
                ]
                w2_view = torch.as_tensor(
                    w2_storage_slice, dtype=target_dtype, device='cpu'
                ).view(w2_shape)
                self._cpu_pinned_w2_views.append(w2_view)
            
            self._pinned_buffer_initialized = True
            
            if self.log_path:
                total_mb = (total_w13_bytes + total_w2_bytes) / (1024**2)
                gpu_mb = (w13_bytes_per_expert + w2_bytes_per_expert) * self.num_gpu_slots / (1024**2)
                cpu_mb = (w13_bytes_per_expert + w2_bytes_per_expert) * self.cpu_pinned_pool_size / (1024**2)
                append_log(
                    f'NativeCache: unified pinned pool initialized, '
                    f'dtype={target_dtype}, '
                    f'GPU_transfer={self.num_gpu_slots} slots ({gpu_mb:.1f}MB), '
                    f'CPU_cache={self.cpu_pinned_pool_size} slots ({cpu_mb:.1f}MB), '
                    f'total={total_mb:.1f}MB',
                    self.log_path
                )
        except Exception as e:
            if self.log_path:
                append_log(
                    f'NativeCache: failed to init pinned storage: {e}, '
                    f'falling back to non-pinned',
                    self.log_path
                )
            self.enable_pinned_memory = False
    
    def _acquire_stream(self) -> Optional[torch.cuda.Stream]:
        """从 pool 中获取一个可用的 stream."""
        with self._lock:
            if self._stream_pool:
                return self._stream_pool.popleft()
            return None
    
    def _release_stream(self, stream: torch.cuda.Stream) -> None:
        """归还 stream 到 pool."""
        with self._lock:
            self._stream_pool.append(stream)
    
    def register_layer(
        self, 
        layer_idx: int, 
        moe_layer: torch.nn.Module,
        num_slots: Optional[int] = None
    ) -> bool:
        """
        注册一个 MoE 层的模块引用.
        
        这应该在模型加载完成后调用，让我们能够直接访问 w13_weight/w2_weight.
        
        Args:
            layer_idx: 层索引
            moe_layer: MoE 层模块 (包含 w13_weight, w2_weight)
            num_slots: 该层的 GPU 槽位数 (默认使用全局 num_gpu_slots)
            
        Returns:
            是否成功
        """
        with self._lock:
            if layer_idx in self._layer_modules:
                return True
            
            # 验证模块有必要的属性
            if not hasattr(moe_layer, 'w13_weight') or not hasattr(moe_layer, 'w2_weight'):
                if self.log_path:
                    append_log(
                        f'NativeCache: layer {layer_idx} missing w13_weight/w2_weight',
                        self.log_path
                    )
                return False
            
            # 确定槽位数（不能超过全局限制）
            slots = num_slots if num_slots is not None else self.num_gpu_slots
            if slots > self.num_gpu_slots:
                if self.log_path:
                    append_log(
                        f'NativeCache: WARNING layer {layer_idx} requested {slots} slots, '
                        f'clamping to global limit {self.num_gpu_slots}',
                        self.log_path
                    )
                slots = self.num_gpu_slots
            
            self._layer_modules[layer_idx] = moe_layer
            self._layer_caches[layer_idx] = LayerNativeCache(
                layer_idx=layer_idx,
                num_slots=slots
            )
            
            # 初始化该层的 transfer events
            self._transfer_events[layer_idx] = [
                torch.cuda.Event() for _ in range(slots)
            ]
            
            # ===== 优化：预创建 GPU tensor views =====
            # 避免运行时 moe_layer.w13_weight[slot_idx] 的索引开销
            # 直接使用预分配的 view 进行 copy_，减少内存访问
            self._gpu_w13_views[layer_idx] = []
            self._gpu_w2_views[layer_idx] = []
            for i in range(slots):
                # 注意：这里创建的是 view，不是 copy
                # view 和原 tensor 共享底层存储
                self._gpu_w13_views[layer_idx].append(moe_layer.w13_weight[i])
                self._gpu_w2_views[layer_idx].append(moe_layer.w2_weight[i])
            
            # 初始化 pinned memory buffers (如果还没初始化)
            # ===== 关键：传入目标 dtype =====
            w13_shape = moe_layer.w13_weight[0].shape  # 单个专家的 shape
            w2_shape = moe_layer.w2_weight[0].shape
            target_dtype = moe_layer.w13_weight.dtype  # 获取目标 dtype
            self._init_pinned_buffers(w13_shape, w2_shape, target_dtype)
            
            if self.log_path:
                w13_shape_full = moe_layer.w13_weight.shape
                w2_shape_full = moe_layer.w2_weight.shape
                append_log(
                    f'NativeCache: registered layer {layer_idx}, '
                    f'slots={slots}, w13={w13_shape_full}, w2={w2_shape_full}',
                    self.log_path
                )
            
            return True
    
    def is_layer_registered(self, layer_idx: int) -> bool:
        """检查层是否已注册."""
        with self._lock:
            return layer_idx in self._layer_caches
    
    def get_gpu_experts(self, layer_idx: int) -> Set[int]:
        """获取当前在 GPU 上的专家集合."""
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return set(range(self.num_gpu_slots))  # 默认: 前 N 个
            return set(cache.expert_to_slot.keys())
    
    def get_slot_for_expert(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        """获取专家对应的槽位索引."""
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                # 未注册时，使用默认映射
                if expert_idx < self.num_gpu_slots:
                    return expert_idx
                return None
            return cache.expert_to_slot.get(expert_idx)
    
    def is_expert_on_gpu(self, layer_idx: int, expert_idx: int) -> bool:
        """检查专家是否在 GPU 上."""
        return self.get_slot_for_expert(layer_idx, expert_idx) is not None
    
    def update_access_time(self, layer_idx: int, expert_indices: Set[int]) -> None:
        """
        更新专家的访问时间（用于 LRU 策略）.
        
        Args:
            layer_idx: 层索引
            expert_indices: 被访问的专家集合
        """
        import time
        current_time = time.time()
        
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return
            
            for expert_idx in expert_indices:
                slot_idx = cache.expert_to_slot.get(expert_idx)
                if slot_idx is not None and slot_idx < len(cache.slots):
                    cache.slots[slot_idx].last_access_time = current_time
    
    def get_available_slots(
        self,
        layer_idx: int,
        exclude_experts: Set[int],
        num_slots: int
    ) -> List[int]:
        """
        获取可用的槽位列表.

        用于决定哪些槽位可以被替换。
        
        Args:
            layer_idx: 层索引
            exclude_experts: 不应被替换的专家集合（如计划中的专家）
            num_slots: 需要的槽位数
            
        Returns:
            可替换的槽位索引列表（保证在 [0, num_gpu_slots) 范围内）
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return []
            
            # 收集可替换的槽位（不在 exclude_experts 中的）
            # 注意：使用 enumerate 的索引作为 slot_idx，而不是 slot.slot_idx 字段
            replaceable_slots = []
            for idx, slot in enumerate(cache.slots):
                if slot.expert_idx not in exclude_experts:
                    replaceable_slots.append((idx, slot.last_access_time))
            
            # 按访问时间排序（最早访问的在前 = LRU）
            replaceable_slots.sort(key=lambda x: x[1])
            
            # 返回需要的槽位数（这些索引保证在 [0, len(cache.slots)) 范围内）
            result = [slot_idx for slot_idx, _ in replaceable_slots[:num_slots]]
            
            # 诊断日志
            if len(result) < num_slots and self.log_path:
                append_log(
                    f'[Cache] Layer {layer_idx}: requested {num_slots} slots, '
                    f'only {len(result)} available (exclude={len(exclude_experts)} experts, '
                    f'total_slots={len(cache.slots)})',
                    self.log_path, level=2
                )
            
            return result

    def swap_expert(
        self,
        layer_idx: int,
        slot_idx: int,
        new_expert_idx: int,
        w13_weight: torch.Tensor,  # [intermediate*2, hidden] or transposed
        w2_weight: torch.Tensor,   # [hidden, intermediate] or transposed
        non_blocking: bool = True,   # 默认使用异步传输
        stream: Optional[torch.cuda.Stream] = None,  # 指定 CUDA stream
        use_pinned_staging: bool = True,  # 使用 pinned staging buffer
    ) -> bool:
        """
        将新专家加载到指定槽位 (HybriMoE-style 优化版).
        
        IO 优化技术（借鉴 HybriMoE）：
        1. Continuous Pinned Memory: 使用预分配的连续 UntypedStorage
        2. Pre-allocated Views: 使用预创建的 tensor views，零运行时分配
        3. 异步传输: 使用独立 stream + non_blocking copy
        4. Event 同步: 记录 event 供后续等待
        5. Storage-level Copy: 直接操作底层 storage，减少元数据开销
        
        Args:
            layer_idx: 层索引
            slot_idx: 目标槽位 (0 ~ num_gpu_slots-1)
            new_expert_idx: 新专家的ID
            w13_weight: 新专家的 gate_up 权重
            w2_weight: 新专家的 down 权重
            non_blocking: 是否使用非阻塞H2D传输
            stream: 使用指定的 CUDA stream
            use_pinned_staging: 是否使用 pinned staging buffer
            
        Returns:
            是否成功
        """
        # 快速验证（不持锁）
        if slot_idx >= self.num_gpu_slots:
            if self.log_path:
                append_log(
                    f'NativeCache: invalid slot_idx {slot_idx} >= {self.num_gpu_slots}',
                    self.log_path
                )
            return False
        
        # 获取必要的引用（最小化锁持有时间）
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            
            if cache is None:
                if self.log_path:
                    append_log(
                        f'NativeCache: layer {layer_idx} not registered',
                        self.log_path
                    )
                return False
            
            # 获取旧专家和 event
            old_expert_idx = cache.slots[slot_idx].expert_idx
            transfer_event = self._transfer_events[layer_idx][slot_idx]
            
            # 获取预分配的 GPU views
            gpu_w13_view = self._gpu_w13_views[layer_idx][slot_idx]
            gpu_w2_view = self._gpu_w2_views[layer_idx][slot_idx]
        
        # 在锁外执行 tensor 操作（避免锁内同步）
        try:
            # 选择 stream：优先用指定 stream，否则从 pool 获取
            use_stream = stream
            acquired_stream = False
            if use_stream is None and non_blocking:
                use_stream = self._acquire_stream()
                acquired_stream = True
            
            # ===== 方案选择优先级（避免不必要的拷贝）=====
            # 1. 优先检查输入是否已 pinned → 直接异步传输（零拷贝）
            # 2. 如果不是 pinned，且 staging buffer 可用 → 使用 staging
            # 3. 否则 → 同步传输
            
            # ===== 诊断日志：检查输入 tensor 状态 =====
            w13_pinned = w13_weight.is_pinned()
            w2_pinned = w2_weight.is_pinned()
            w13_contig = w13_weight.is_contiguous()
            w2_contig = w2_weight.is_contiguous()
            
            if self.log_path:
                append_log(
                    f'NativeCache: swap_expert[{layer_idx}][{slot_idx}] expert {new_expert_idx}: '
                    f'w13(pinned={w13_pinned}, contig={w13_contig}), '
                    f'w2(pinned={w2_pinned}, contig={w2_contig})',
                    self.log_path, level=3  # 详细日志
                )
            
            # ===== 方案 A: 源 tensor 已经在 pinned memory（最优，零 CPU 拷贝）=====
            if w13_pinned and w2_pinned:
                # 输入已经是 pinned（如 ExpertResolver 提供的），直接异步传输
                if use_stream is not None:
                    with torch.cuda.stream(use_stream):
                        gpu_w13_view.copy_(w13_weight, non_blocking=True)
                        gpu_w2_view.copy_(w2_weight, non_blocking=True)
                        
                        transfer_event.record(use_stream)
                    if self.log_path:
                        log_once('swap_path_A_async', 'NativeCache: using path A (pinned input, async)')
                else:
                    # fallback: 同步传输
                    gpu_w13_view.copy_(w13_weight)
                    gpu_w2_view.copy_(w2_weight)
                    if self.log_path:
                        log_once('swap_path_A_sync', 'NativeCache: using path A (pinned input, sync)')
            
            # ===== 方案 B: 使用预分配的 pinned staging buffer =====
            elif use_pinned_staging and self.enable_pinned_memory and self._pinned_buffer_initialized:
                # 输入不是 pinned，通过 staging buffer 中转
                # Step 1: CPU copy 到预分配的 pinned view (快速 CPU memcpy)
                pinned_w13 = self._pinned_w13_views[slot_idx]
                pinned_w2 = self._pinned_w2_views[slot_idx]
                
                # ✅ 优化：直接 copy_，PyTorch 会自动处理 dtype 转换（如果需要）
                # 避免显式 .to() 创建中间 tensor
                pinned_w13.copy_(w13_weight)
                pinned_w2.copy_(w2_weight)
                
                # Step 2: 异步 H2D 传输到预分配的 GPU view
                if use_stream is not None:
                    with torch.cuda.stream(use_stream):
                        # ===== 关键优化：直接使用 pre-allocated GPU view =====
                        # 避免 moe_layer.w13_weight[slot_idx] 的索引操作
                        gpu_w13_view.copy_(pinned_w13, non_blocking=True)
                        gpu_w2_view.copy_(pinned_w2, non_blocking=True)
                        
                        # 记录 event
                        transfer_event.record(use_stream)
                    if self.log_path:
                        log_once('swap_path_B_async', 'NativeCache: using path B (staging buffer, async)')
                else:
                    # fallback: 同步传输
                    gpu_w13_view.copy_(pinned_w13)
                    gpu_w2_view.copy_(pinned_w2)
                    if self.log_path:
                        log_once('swap_path_B_sync', 'NativeCache: using path B (staging buffer, sync)')
            
            # ===== 方案 C: 普通 CPU tensor（最慢，同步传输）=====
            else:
                # 输入不是 pinned，且 staging buffer 不可用
                # 使用 .to() 会触发同步拷贝
                gpu_w13 = w13_weight.to(device=gpu_w13_view.device, dtype=gpu_w13_view.dtype)
                gpu_w13_view.copy_(gpu_w13)
                
                gpu_w2 = w2_weight.to(device=gpu_w2_view.device, dtype=gpu_w2_view.dtype)
                gpu_w2_view.copy_(gpu_w2)
                
                if self.log_path:
                    log_once('swap_path_C', 'WARNING: NativeCache using path C (non-pinned, sync) - SLOW!')
            
            # 释放 stream
            if acquired_stream and use_stream is not None:
                self._release_stream(use_stream)
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'NativeCache: copy failed layer {layer_idx} slot {slot_idx}: {e}',
                    self.log_path
                )
            # 释放 stream
            if acquired_stream and use_stream is not None:
                self._release_stream(use_stream)
            return False
        
        # 更新映射（最小化锁持有时间）
        with self._lock:
            # 从映射中移除旧专家
            if old_expert_idx >= 0 and old_expert_idx in cache.expert_to_slot:
                del cache.expert_to_slot[old_expert_idx]
            
            # 更新映射
            cache.slots[slot_idx].expert_idx = new_expert_idx
            cache.slots[slot_idx].is_dirty = True
            cache.expert_to_slot[new_expert_idx] = slot_idx
            
            self._swap_count += 1
            self._any_swap_happened = True
        
        if self.log_path:
            # ✅ 记录实际 tensor 的 pinned 状态，而不是 use_pinned_staging 参数
            actual_pinned = w13_weight.is_pinned() and w2_weight.is_pinned()
            append_log(
                f'NativeCache: swapped layer {layer_idx} slot {slot_idx}: '
                f'expert {old_expert_idx} → {new_expert_idx} '
                f'(async={non_blocking}, pinned={actual_pinned})',
                self.log_path
            )
        
        return True
    
    def wait_transfer_complete(self, layer_idx: int, slot_idx: Optional[int] = None, timeout: Optional[float] = None) -> bool:
        """
        等待指定槽位的传输完成.
        
        在使用异步传输的权重前，需要调用此函数确保传输完成。
        
        Args:
            layer_idx: 层索引
            slot_idx: 槽位索引，如果为 None 则等待该层所有槽位
            timeout: 超时时间（秒），仅用于诊断
            
        Returns:
            是否成功等待（总是 True，除非 events 不存在）
        """
        events = self._transfer_events.get(layer_idx)
        if events is None:
            return True
        
        if slot_idx is not None:
            # 等待单个槽位
            if slot_idx < len(events):
                events[slot_idx].synchronize()
        else:
            # 等待所有槽位
            for event in events:
                event.synchronize()
        
        return True
    
    def batch_swap_experts(
        self,
        swaps: List[Tuple[int, int, int, torch.Tensor, torch.Tensor]],
        # [(layer_idx, slot_idx, expert_idx, w13_weight, w2_weight), ...]
        use_pipeline: bool = False,  # ✅ 默认禁用 pipeline（避免带宽竞争）
        priority_stream: Optional[torch.cuda.Stream] = None,  # ✅ 支持外部指定 stream
        sync_after: bool = False,  # ✅ 是否在完成后同步（默认不同步，由 Event 机制处理）
        threading_events: Optional[Dict[Tuple[int, int], 'threading.Event']] = None,  # ✅ threading.Event 字典（用于通知等待者）
    ) -> int:
        """
        批量专家替换（修复版 - 顺序传输避免 PCIe 带宽竞争）.
        
        关键修复：
        1. 禁用多 stream 并行传输（默认 use_pipeline=False）
        2. 使用顺序传输，避免 PCIe 带宽竞争
        3. 保持稳定的 11+ GB/s 高带宽
        
        其他优化：
        1. 支持外部指定优先级 stream
        2. 可选的同步（默认异步，由 Event 机制处理）
        3. 可选的 threading.Event 通知（用于 wait_for_experts）
        
        Args:
            swaps: 替换列表
            use_pipeline: 是否使用 pipeline 优化
            priority_stream: 外部指定的 CUDA stream（优先使用，用于优先级控制）
            sync_after: 是否在传输完成后同步（默认 False，使用 Event 机制）
            threading_events: threading.Event 字典 (layer_idx, expert_idx) -> Event
                             用于在传输完成后通知等待者
            
        Returns:
            成功数量
        """
        if not swaps:
            return 0
        
        success_count = 0
        completed_experts = []  # 记录成功的 (layer_idx, expert_idx)
        
        # ✅ 优先使用外部指定的 stream（用于优先级控制）
        if priority_stream is not None:
            # 使用外部指定的优先级 stream 进行所有传输
            for layer_idx, slot_idx, expert_idx, w13, w2 in swaps:
                use_pinned_staging = not (w13.is_pinned() and w2.is_pinned())  # 源已 pinned 则不使用 staging
                if use_pinned_staging:
                    append_log(f"NativeCache: batch_swap_experts using priority_stream without pinned input for layer {layer_idx} expert {expert_idx}", self.log_path, level=3)
                if self.swap_expert(
                    layer_idx, slot_idx, expert_idx, w13, w2,
                    non_blocking=True, stream=priority_stream, use_pinned_staging=use_pinned_staging
                ):
                    success_count += 1
                    completed_experts.append((layer_idx, expert_idx))
            
            # ✅ 可选同步
            if sync_after:
                priority_stream.synchronize()
            
        elif use_pipeline and len(self._transfer_streams) > 1:
            # ===== 关键修复：禁用多 stream 并行传输，避免 PCIe 带宽竞争 =====
            # 问题：多个异步传输同时运行，PCIe 带宽被分散（0.1-1 GB/s）
            # 解决：使用单个 stream 顺序传输，保持稳定高带宽（11+ GB/s）
            # 
            # 原因：PCIe 总带宽固定，多个传输竞争会导致：
            # - 每个传输分到的带宽不均（有的快有的慢）
            # - 总吞吐量反而降低（调度开销）
            # - 时间测量不准确（异步完成时间不同）
            # 
            # 最优策略：顺序传输 + pinned memory DMA
            # - 每个传输独占 PCIe 带宽
            # - 稳定的 11+ GB/s
            # - 总时间 = N * 单次时间（可预测）
            
            # 使用第一个 stream 进行顺序传输
            stream = self._transfer_streams[0]
            for layer_idx, slot_idx, expert_idx, w13, w2 in swaps:
                if self.swap_expert(
                    layer_idx, slot_idx, expert_idx, w13, w2,
                    non_blocking=True, stream=stream, use_pinned_staging=False  # ✅ 不使用 staging，源已 pinned
                ):
                    success_count += 1
                    completed_experts.append((layer_idx, expert_idx))
            
            # ✅ 可选同步
            if sync_after:
                stream.synchronize()
        else:
            # 顺序模式
            for layer_idx, slot_idx, expert_idx, w13, w2 in swaps:
                if self.swap_expert(layer_idx, slot_idx, expert_idx, w13, w2):
                    success_count += 1
                    completed_experts.append((layer_idx, expert_idx))
        
        # ✅ 通知 threading.Event（在后台异步等待 CUDA Event）
        if threading_events is not None and completed_experts:
            # 在后台线程中等待 CUDA Event 完成后 set threading.Event
            def _async_notify_completion():
                # 等待所有 CUDA Event 完成
                for layer_idx, expert_idx in completed_experts:
                    events = self._transfer_events.get(layer_idx)
                    if events is not None:
                        cache = self._layer_caches.get(layer_idx)
                        if cache is not None:
                            slot_idx = cache.expert_to_slot.get(expert_idx)
                            if slot_idx is not None and slot_idx < len(events):
                                events[slot_idx].synchronize()  # 等待 CUDA Event
                
                # CUDA Event 完成后，set 所有 threading.Event
                for layer_idx, expert_idx in completed_experts:
                    cache_key = (layer_idx, expert_idx)
                    if cache_key in threading_events:
                        threading_events[cache_key].set()  # 通知等待者
            
            # 启动后台线程
            import threading as th
            th.Thread(target=_async_notify_completion, daemon=True).start()
        
        if self.log_path:
            append_log(
                f'NativeCache: batch swap completed, '
                f'{success_count}/{len(swaps)} succeeded',
                self.log_path
            )
        
        return success_count
    
    def remap_topk_ids(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重映射 topk_ids 以使用正确的槽位.
        
        FusedMoE kernel 使用 topk_ids 作为索引访问 w13_weight[topk_ids]。
        如果我们把 expert_5 放在了 slot_1，需要把 topk_ids 中的 5 改成 1。
        
        同时返回 GPU mask 和 CPU mask。
        
        Args:
            layer_idx: 层索引
            topk_ids: 原始 expert IDs [batch, top_k]
            
        Returns:
            (remapped_ids, gpu_mask)
            - remapped_ids: 重映射后的 IDs (GPU专家映射到槽位，CPU专家标记为-1)
            - gpu_mask: bool tensor，标记哪些是 GPU 专家
        """
        # 快速路径：获取映射表（最小化锁持有时间）
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                expert_to_slot = None
            else:
                # 复制映射表，避免在锁外使用时被修改
                expert_to_slot = dict(cache.expert_to_slot)
        
        # 在锁外执行所有 tensor 操作
        if expert_to_slot is None:
            # 未注册时，使用默认映射：expert_i -> slot_i (if i < num_gpu_slots)
            gpu_mask = topk_ids < self.num_gpu_slots
            remapped = topk_ids.clone()
            remapped[~gpu_mask] = -1
            return remapped, gpu_mask
        
        # 使用向量化操作代替 Python 循环
        # 创建映射 tensor：expert_idx -> slot_idx (不存在的映射到 -1)
        device = topk_ids.device
        mapping_size = self.num_experts
        
        # 创建映射表：expert_idx -> slot_idx，默认 -1 表示 CPU
        mapping_table = torch.full((mapping_size,), -1, dtype=topk_ids.dtype, device=device)
        for expert_idx, slot_idx in expert_to_slot.items():
            if expert_idx < mapping_size:
                mapping_table[expert_idx] = slot_idx
        
        # 向量化重映射（需要处理越界情况）
        flat_ids = topk_ids.flatten()
        
        # 安全索引：将越界的 expert_id clip 到有效范围
        # 越界的会被映射到 mapping_table[-1] = -1 (CPU)
        flat_ids_safe = torch.clamp(flat_ids, 0, mapping_size - 1)
        flat_remapped = mapping_table[flat_ids_safe]
        
        # 将原本就越界的 expert_id 强制标记为 -1 (CPU)
        out_of_bounds = (flat_ids < 0) | (flat_ids >= mapping_size)
        flat_remapped[out_of_bounds] = -1
        
        remapped = flat_remapped.view_as(topk_ids)
        
        # GPU mask: remapped >= 0 表示在 GPU 上
        gpu_mask = remapped >= 0
        
        return remapped, gpu_mask
    
    def get_cpu_pinned_tensors(
        self,
        layer_idx: int,
        expert_idx: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 CPU pinned memory 中的专家权重（供 ExpertResolver 使用）.
        
        使用哈希槽位分配策略，允许冲突（性能优先）。
        
        Args:
            layer_idx: 层索引
            expert_idx: 专家索引
            w13_weight: 源权重 w13
            w2_weight: 源权重 w2
            
        Returns:
            (pinned_w13, pinned_w2): pinned memory 中的权重副本
        """
        if not self.enable_pinned_memory or not self._pinned_buffer_initialized:
            # Fallback: 返回非 pinned 副本
            return w13_weight.clone(), w2_weight.clone()
        
        if self.cpu_pinned_pool_size == 0:
            # 没有 CPU cache pool，返回非 pinned
            return w13_weight.clone(), w2_weight.clone()
        
        # 哈希槽位分配
        cache_key = (layer_idx, expert_idx)
        slot_idx = hash(cache_key) % self.cpu_pinned_pool_size
        
        # 获取 pinned views
        pinned_w13 = self._cpu_pinned_w13_views[slot_idx]
        pinned_w2 = self._cpu_pinned_w2_views[slot_idx]
        
        # Copy 数据到 pinned memory
        pinned_w13.copy_(w13_weight)
        pinned_w2.copy_(w2_weight)
        
        # 更新映射记录（仅用于诊断）
        with self._lock:
            self._cpu_pinned_slot_map[cache_key] = slot_idx
        
        return pinned_w13, pinned_w2

    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息."""
        with self._lock:
            layer_stats = {}
            for layer_idx, cache in self._layer_caches.items():
                gpu_experts = list(cache.expert_to_slot.keys())
                slot_mapping = {
                    slot.slot_idx: slot.expert_idx 
                    for slot in cache.slots
                }
                layer_stats[layer_idx] = {
                    'gpu_experts': gpu_experts,
                    'slot_mapping': slot_mapping,
                }
            
            return {
                'num_layers': self.num_layers,
                'num_gpu_slots': self.num_gpu_slots,
                'cpu_pinned_slots': self.cpu_pinned_pool_size,
                'cpu_cached_experts': len(self._cpu_pinned_slot_map),
                'total_swaps': self._swap_count,
                'registered_layers': len(self._layer_modules),
                'layer_stats': layer_stats,
            }


# 全局实例
_native_cache: Optional[NativeGPUCacheManager] = None
_native_cache_lock = threading.Lock()


def get_native_cache() -> Optional[NativeGPUCacheManager]:
    """获取全局原生缓存管理器."""
    return _native_cache


def init_native_cache(
    num_layers: int,
    num_experts: int,
    hidden_size: int = 0,
    intermediate_size: int = 0,
    num_gpu_slots: int = 0,
    log_path: Optional[str] = None,
    enable_pinned_memory: bool = True,
    num_transfer_streams: int = 2,
    cpu_pinned_pool_size: int = 0,  # CPU 侧 pinned pool 大小
) -> NativeGPUCacheManager:
    """
    初始化全局原生缓存管理器.
    
    Args:
        num_layers: 层数
        num_experts: 每层专家数
        hidden_size: hidden size (可选，用于日志)
        intermediate_size: intermediate size (可选，用于日志)
        num_gpu_slots: 全局 GPU 槽位数 (可选，每层可单独设置)
        log_path: 日志路径
        enable_pinned_memory: 启用 pinned memory pool (IO 优化)
        num_transfer_streams: 传输用的 CUDA stream 数量 (IO 优化)
        cpu_pinned_pool_size: CPU 侧 pinned pool 大小
        
    Returns:
        缓存管理器实例
    """
    global _native_cache
    
    with _native_cache_lock:
        if _native_cache is None:
            _native_cache = NativeGPUCacheManager(
                num_layers=num_layers,
                num_experts=num_experts,
                num_gpu_slots=num_gpu_slots,
                log_path=log_path,
                enable_pinned_memory=enable_pinned_memory,
                num_transfer_streams=num_transfer_streams,
                cpu_pinned_pool_size=cpu_pinned_pool_size,
            )
            
            if log_path:
                append_log(
                    f'NativeGPUCacheManager initialized: '
                    f'{num_layers} layers, {num_experts} experts, '
                    f'{num_gpu_slots} GPU slots, '
                    f'hidden={hidden_size}, intermediate={intermediate_size}, '
                    f'pinned_memory={enable_pinned_memory}, streams={num_transfer_streams}, '
                    f'cpu_pinned_pool_size={cpu_pinned_pool_size}',
                    log_path
                )
        
        return _native_cache
