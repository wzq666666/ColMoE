"""
Expert resolver for extracting and managing individual expert weights.

Supports extracting expert weights from:
1. SGLang loaded models (runtime)
2. HuggingFace model files (from disk)
"""

from typing import Any, Dict, Optional, Tuple
import threading
import os
import torch

from ..logger import log_once, append_log


class ExpertResolver:
    """
    Resolves and extracts individual expert weights from MOE layers.
    
    Provides methods to locate and retrieve expert model weights by
    layer index and expert index. Supports both runtime model access
    and direct loading from HuggingFace checkpoint files.
    
    IO Optimization Features:
    - CPU memory cache: avoids repeated disk reads
    - safetensors file handle cache: avoids repeated file opens
    - Direct GPU loading option: skip CPU intermediate step
    - Batch loading: minimize file seeks
    """
    
    def __init__(self, cfg: Dict[str, Any], log_path: Optional[str] = None):
        """
        Initialize expert resolver.
        
        Args:
            cfg: Configuration dictionary
            log_path: Optional path for logging
        """
        self.cfg = cfg
        self.log_path = log_path
        self._expert_cache: Dict[Tuple[int, int], Any] = {}
        self._lock = threading.Lock()
        self._model_ref: Optional[Any] = None
        self._hf_model_path: Optional[str] = None
        self._shard_index: Optional[Dict[str, Any]] = None
        
        # Cache for open safetensors files (reduces file open overhead)
        self._safetensors_handles: Dict[str, Any] = {}
        self._handles_lock = threading.Lock()
        
        # ========== IO优化: CPU内存缓存 (HybriMoE-style pinned memory) ==========
        # 缓存已加载的专家权重在CPU内存中，避免重复磁盘IO
        # 使用 pinned memory 存储，后续传输到GPU时可享受DMA加速 (3-4x带宽)
        # 
        # 缓存格式: sglang 格式 {'w13', 'w2'}
        # - w13: [intermediate*2, hidden] = cat([gate_proj, up_proj], dim=0)
        # - w2: [hidden, intermediate] = down_proj
        # 
        # 直接在 pinned pool 中存储 sglang 格式,避免 torch.cat 破坏 pinned 状态
        # Key: (layer_idx, expert_idx), Value: {'w13': tensor, 'w2': tensor}
        self._cpu_weight_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._cpu_cache_lock = threading.Lock()
        self._cpu_cache_enabled = cfg.get('enable_cpu_weight_cache', True)
        self._cpu_cache_max_experts = cfg.get('cpu_cache_max_experts', 256)  # 最多缓存256个专家
        
        # ========== Pinned Memory 配置 ==========
        # use_pinned_memory: 是否使用 pinned memory（从 native_cache 获取）
        # allocate_own_pinned_pool: 是否分配独立的 pinned pool（已废弃，改用 native_cache 统一管理）
        self._use_pinned_memory = cfg.get('use_pinned_memory', True)  # 是否使用pinned memory
        self._allocate_own_pool = cfg.get('allocate_own_pinned_pool', False)  # 是否分配独立pool（废弃）
        
        # ========== Pinned Memory Pool (统一管理) ==========
        # Note: Pinned memory pool 现在由 native_gpu_cache 统一管理
        # ExpertResolver 通过 native_gpu_cache.get_cpu_pinned_tensors() 获取 pinned memory
        # 不再在 ExpertResolver 内部分配独立的 pinned pool
        
        # Try to get HF model path from config
        if 'model_path' in cfg:
            self._hf_model_path = cfg['model_path']
        
        log_once(
            'expert_resolver_init', 
            f'ExpertResolver initialized (cpu_cache={self._cpu_cache_enabled}, '
            f'use_pinned_memory={self._use_pinned_memory}, allocate_own_pool={self._allocate_own_pool})'
        )
    
    def _init_pinned_pool(
        self,
        w13_shape: torch.Size,  # sglang format: [intermediate*2, hidden]
        w2_shape: torch.Size,   # [hidden, intermediate]
        dtype: torch.dtype
    ) -> None:
        """
        初始化 pinned memory pool (HybriMoE-style, sglang 格式).
        
        优化策略:
        1. 预分配连续的 pinned memory（避免运行时 pin_memory() 开销）
        2. 使用 UntypedStorage 实现内存复用
        3. 直接存储 sglang 格式（w13+w2），避免 torch.cat 破坏 pinned 状态
        4. 后续 H2D 传输可享受 DMA 加速（3-4x 带宽提升）
        
        Args:
            w13_shape: w13 (gate_proj + up_proj 拼接后) 的 shape [intermediate*2, hidden]
            w2_shape: w2 (down_proj) 的 shape [hidden, intermediate]
            dtype: 权重的 dtype
        """
        if self._pinned_pool_initialized or not self._use_pinned_memory:
            return
        
        try:
            element_size = torch.tensor([], dtype=dtype).element_size()
            
            # 计算每个权重所需字节数（sglang 格式）
            w13_bytes = w13_shape.numel() * element_size
            w2_bytes = w2_shape.numel() * element_size
            
            # 分配连续 pinned memory pool
            # pool_size 个专家的连续存储空间
            total_w13_bytes = w13_bytes * self._pinned_pool_size
            total_w2_bytes = w2_bytes * self._pinned_pool_size
            
            self._pinned_w13_storage = torch.UntypedStorage(total_w13_bytes).pin_memory()
            self._pinned_w2_storage = torch.UntypedStorage(total_w2_bytes).pin_memory()
            
            self._pinned_pool_initialized = True
            
            total_mb = (total_w13_bytes + total_w2_bytes) / (1024**2)
            
            if self.log_path:
                append_log(
                    f'ExpertResolver: pinned pool initialized (sglang format), '
                    f'dtype={dtype}, pool_size={self._pinned_pool_size}, '
                    f'w13_shape={w13_shape}, w2_shape={w2_shape}, '
                    f'total={total_mb:.1f}MB',
                    self.log_path
                )
            
            log_once(
                'expert_resolver_pinned_pool',
                f'Pinned memory pool: {total_mb:.1f}MB for {self._pinned_pool_size} experts (sglang format)'
            )
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'ExpertResolver: failed to init pinned pool: {e}, falling back to non-pinned',
                    self.log_path
                )
            self._use_pinned_memory = False
    
    def _get_pinned_tensor(
        self,
        weight_name: str,  # 'w13' or 'w2' (sglang format)
        shape: torch.Size,
        dtype: torch.dtype,
        slot_idx: int
    ) -> torch.Tensor:
        """
        从 pinned pool 中获取一个 tensor view (sglang 格式).
        
        Args:
            weight_name: 权重名称 ('w13', 'w2')
            shape: tensor shape
            dtype: tensor dtype
            slot_idx: pool 中的槽位索引 (0 ~ pool_size-1)
            
        Returns:
            pinned memory tensor view
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_bytes = shape.numel() * element_size
        
        if weight_name == 'w13':
            storage = self._pinned_w13_storage
        elif weight_name == 'w2':
            storage = self._pinned_w2_storage
        else:
            raise ValueError(f"Unknown weight name: {weight_name} (expected 'w13' or 'w2')")
        
        # 切片出这个 slot 的存储空间
        storage_slice = storage[slot_idx * num_bytes : (slot_idx + 1) * num_bytes]
        
        # 创建 tensor view
        tensor = torch.as_tensor(storage_slice, dtype=dtype, device='cpu').view(shape)
        
        return tensor
    
    def set_model_reference(self, model: Any) -> None:
        """
        Set reference to the full model for expert extraction.
        
        Args:
            model: The MOE model instance
        """
        with self._lock:
            self._model_ref = model
            if self.log_path:
                append_log(f'ExpertResolver: model reference set', self.log_path)
    
    def get_expert_weights(
        self, 
        layer_idx: int, 
        expert_idx: int,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Extract weights for a specific expert.
        
        Args:
            layer_idx: Layer index (0-based)
            expert_idx: Expert index within the layer (0-based)
            use_cache: Whether to use cached weights if available
            
        Returns:
            Dictionary containing expert weights, or None if not found
        """
        cache_key = (layer_idx, expert_idx)
        
        # Check cache first
        if use_cache and cache_key in self._expert_cache:
            return self._expert_cache[cache_key]
        
        if self._model_ref is None:
            log_once('expert_resolver_no_model', 'ExpertResolver: no model reference set')
            return None
        
        try:
            weights = self._extract_expert_weights(layer_idx, expert_idx)
            
            # Cache the weights
            if use_cache and weights is not None:
                with self._lock:
                    self._expert_cache[cache_key] = weights
            
            if self.log_path:
                append_log(
                    f'ExpertResolver: extracted expert[{layer_idx}][{expert_idx}]',
                    self.log_path
                )
            
            return weights
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'ExpertResolver: failed to extract expert[{layer_idx}][{expert_idx}]: {e}',
                    self.log_path
                )
            log_once(
                f'expert_resolver_err_{layer_idx}_{expert_idx}',
                f'Failed to extract expert[{layer_idx}][{expert_idx}]: {e}'
            )
            return None
    
    def _extract_expert_weights(
        self, 
        layer_idx: int, 
        expert_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method to extract expert weights from model.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            Dictionary of expert weights or None
        """
        if self._model_ref is None:
            return None
        
        try:
            # Try different model structures
            # For transformers models with decoder layers
            if hasattr(self._model_ref, 'model'):
                model = self._model_ref.model
            else:
                model = self._model_ref
            
            # Access decoder/transformer layers
            if hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'decoder'):
                layers = model.decoder.layers
            else:
                log_once('expert_resolver_no_layers', 'ExpertResolver: cannot find model layers')
                return None
            
            if layer_idx >= len(layers):
                if self.log_path:
                    append_log(
                        f'ExpertResolver: layer_idx {layer_idx} out of range (max: {len(layers)-1})',
                        self.log_path
                    )
                return None
            
            layer = layers[layer_idx]
            
            # Find MOE/expert module in the layer
            moe_module = None
            for attr_name in ['block_sparse_moe', 'moe', 'mlp', 'feed_forward']:
                if hasattr(layer, attr_name):
                    moe_module = getattr(layer, attr_name)
                    break
            
            if moe_module is None:
                log_once('expert_resolver_no_moe', f'ExpertResolver: no MOE module found in layer {layer_idx}')
                return None
            
            # Extract expert weights
            if hasattr(moe_module, 'experts'):
                experts = moe_module.experts
                if expert_idx >= len(experts):
                    if self.log_path:
                        append_log(
                            f'ExpertResolver: expert_idx {expert_idx} out of range (max: {len(experts)-1})',
                            self.log_path
                        )
                    return None
                
                expert = experts[expert_idx]
                
                # Extract state dict for this expert
                expert_weights = {}
                if hasattr(expert, 'state_dict'):
                    expert_weights = expert.state_dict()
                else:
                    # Manually extract parameters
                    for name, param in expert.named_parameters():
                        expert_weights[name] = param
                
                return expert_weights
            
            else:
                log_once('expert_resolver_no_experts', f'ExpertResolver: no experts found in MOE module')
                return None
                
        except Exception as e:
            if self.log_path:
                append_log(f'ExpertResolver._extract_expert_weights failed: {e}', self.log_path)
            raise
    
    def get_expert_model(
        self, 
        layer_idx: int, 
        expert_idx: int
    ) -> Optional[Any]:
        """
        Get the expert module instance directly.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            Expert module instance or None
        """
        if self._model_ref is None:
            log_once('expert_resolver_no_model_ref', 'ExpertResolver: no model reference')
            return None
        
        try:
            # Access model layers
            if hasattr(self._model_ref, 'model'):
                model = self._model_ref.model
            else:
                model = self._model_ref
            
            if hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'decoder'):
                layers = model.decoder.layers
            else:
                return None
            
            if layer_idx >= len(layers):
                return None
            
            layer = layers[layer_idx]
            
            # Find MOE module
            moe_module = None
            for attr_name in ['block_sparse_moe', 'moe', 'mlp', 'feed_forward']:
                if hasattr(layer, attr_name):
                    moe_module = getattr(layer, attr_name)
                    break
            
            if moe_module is None or not hasattr(moe_module, 'experts'):
                return None
            
            experts = moe_module.experts
            if expert_idx >= len(experts):
                return None
            
            return experts[expert_idx]
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'ExpertResolver.get_expert_model failed for [{layer_idx}][{expert_idx}]: {e}',
                    self.log_path
                )
            return None
    
    def set_hf_model_path(self, model_path: str) -> None:
        """
        Set the HuggingFace model path for direct weight loading.
        
        Args:
            model_path: Path to HuggingFace model directory
        """
        self._hf_model_path = model_path
        self._shard_index = None  # Reset shard index
        if self.log_path:
            append_log(f'ExpertResolver: HF model path set to {model_path}', self.log_path)
    
    def _load_shard_index(self) -> Optional[Dict[str, Any]]:
        """Load the model shard index if available."""
        if self._shard_index is not None:
            return self._shard_index
        
        if self._hf_model_path is None:
            return None
        
        import json
        
        # Try to load shard index
        index_path = os.path.join(self._hf_model_path, 'model.safetensors.index.json')
        if not os.path.exists(index_path):
            index_path = os.path.join(self._hf_model_path, 'pytorch_model.bin.index.json')
        
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self._shard_index = json.load(f)
            if self.log_path:
                append_log(f'ExpertResolver: loaded shard index from {index_path}', self.log_path)
        
        return self._shard_index
    
    def load_expert_weights_from_hf(
        self,
        layer_idx: int,
        expert_idx: int,
        device: str = "cpu",
        use_cache: bool = True,
        cache_on_cpu: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load expert weights directly from HuggingFace checkpoint files.
        
        IO优化特性:
        1. CPU内存缓存: 首次从磁盘加载后缓存在CPU内存，后续直接从内存读取
        2. safetensors文件句柄缓存: 避免重复打开文件
        3. 支持直接加载到GPU: device="cuda" 跳过CPU中转
        
        加载路径:
        - 有缓存时:  CPU缓存 → 目标设备 (快速，~1ms)
        - 无缓存时:  磁盘 → CPU缓存 → 目标设备 (首次较慢)
        - 直接GPU:   磁盘 → GPU (device="cuda", 跳过CPU)
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            device: Device to load weights to ("cpu", "cuda", "cuda:0", etc.)
            use_cache: Whether to check/use CPU memory cache
            cache_on_cpu: Whether to cache loaded weights in CPU memory
            
        Returns:
            Dictionary with 'w1', 'w2', 'w3' weight tensors, or None if failed
        """
        if self._hf_model_path is None:
            log_once('expert_resolver_no_hf_path', 'ExpertResolver: no HF model path set')
            return None
        
        cache_key = (layer_idx, expert_idx)
        
        # ========== 1. 检查CPU内存缓存 ==========
        if use_cache and self._cpu_cache_enabled:
            with self._cpu_cache_lock:
                if cache_key in self._cpu_weight_cache:
                    append_log(f"ExpertResolver: CPU cache hit for expert[{layer_idx}][{expert_idx}]", self.log_path)
                    cached = self._cpu_weight_cache[cache_key]
                    # 缓存命中：直接返回 sglang 格式（{'w13': ..., 'w2': ...}）
                    if device == "cpu":
                        return cached  # 直接返回 CPU pinned 引用（零拷贝）
                    else:
                        # 从 CPU pinned → GPU（享受 DMA 加速）
                        return {
                            'w13': cached['w13'].to(device),
                            'w2': cached['w2'].to(device),
                        }
        
        # ========== 2. 缓存未命中：从磁盘加载 ==========
        try:
            # Determine weight key patterns based on model architecture
            # Mixtral pattern: model.layers.{layer}.block_sparse_moe.experts.{expert}.{w1/w2/w3}.weight
            key_patterns = [
                # Mixtral
                f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight',
                f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight',
                f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight',
            ]
            
            # Alternative patterns for other architectures
            alt_patterns = [
                # Qwen2-MoE
                (f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight',
                 f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight',
                 f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight'),
            ]
            
            # ===== 关键修复：始终先加载到 CPU =====
            # 这样才能正确地缓存在 CPU pinned memory 中
            shard_index = self._load_shard_index()
            if shard_index is not None:
                weights = self._batch_load_weights_with_index(key_patterns, shard_index, "cpu")
            else:
                weights = self._try_load_weights(key_patterns, "cpu")
            
            if weights is None:
                # Try alternative patterns
                for alt in alt_patterns:
                    if shard_index is not None:
                        weights = self._batch_load_weights_with_index(list(alt), shard_index, "cpu")
                    else:
                        weights = self._try_load_weights(list(alt), "cpu")
                    if weights is not None:
                        break
            
            if weights is None:
                if self.log_path:
                    append_log(
                        f'ExpertResolver: could not find weights for expert[{layer_idx}][{expert_idx}]',
                        self.log_path
                    )
                return None
            
            # ========== 3. 缓存到 CPU pinned memory ==========
            cpu_weights = None  # 用于最终返回的 CPU weights
            
            if cache_on_cpu and self._cpu_cache_enabled:
                with self._cpu_cache_lock:
                    # 如果缓存已满，执行LRU淘汰 (简单实现：清除一半)
                    if len(self._cpu_weight_cache) >= self._cpu_cache_max_experts:
                        # 清除前一半的缓存
                        keys_to_remove = list(self._cpu_weight_cache.keys())[:len(self._cpu_weight_cache)//2]
                        for key in keys_to_remove:
                            del self._cpu_weight_cache[key]
                        if self.log_path:
                            append_log(
                                f'ExpertResolver: CPU cache full, evicted {len(keys_to_remove)} entries',
                                self.log_path
                            )
                    
                    # ===== 使用 native_gpu_cache 的统一 pinned memory pool =====
                    # 通过 native_gpu_cache 获取 pinned memory tensor
                    # 避免重复分配，统一管理
                    if self._use_pinned_memory:
                        try:
                            from ..native.native_gpu_cache import get_native_cache
                            native_cache = get_native_cache()
                            
                            if native_cache is not None:
                                # 构造 sglang 格式的权重
                                intermediate = weights[0].shape[0]
                                hidden = weights[0].shape[1]
                                w13_shape = torch.Size([intermediate * 2, hidden])
                                
                                # 先拼接成 sglang 格式
                                w13_cpu = torch.empty(w13_shape, dtype=weights[0].dtype, device='cpu')
                                w13_cpu[:intermediate].copy_(weights[0])
                                w13_cpu[intermediate:].copy_(weights[2])
                                w2_cpu = weights[1].clone()
                                
                                # 通过 native_cache 获取 pinned memory
                                pinned_w13, pinned_w2 = native_cache.get_cpu_pinned_tensors(
                                    layer_idx=layer_idx,
                                    expert_idx=expert_idx,
                                    w13_weight=w13_cpu,
                                    w2_weight=w2_cpu,
                                )
                                
                                # 缓存 pinned 版本
                                cpu_weights = {
                                    'w13': pinned_w13,  # [intermediate*2, hidden] pinned
                                    'w2': pinned_w2,    # [hidden, intermediate] pinned
                                }
                                self._cpu_weight_cache[cache_key] = cpu_weights
                                
                                if self.log_path:
                                    append_log(
                                        f'ExpertResolver: cached expert[{layer_idx}][{expert_idx}] in unified pinned pool (via native_cache)',
                                        self.log_path
                                    )
                            else:
                                # native_cache not available, use non-pinned cache
                                intermediate = weights[0].shape[0]
                                w13_cpu = torch.empty([intermediate * 2, weights[0].shape[1]], dtype=weights[0].dtype, device='cpu')
                                w13_cpu[:intermediate].copy_(weights[0])
                                w13_cpu[intermediate:].copy_(weights[2])
                                
                                cpu_weights = {
                                    'w13': w13_cpu,
                                    'w2': weights[1].clone(),
                                }
                                self._cpu_weight_cache[cache_key] = cpu_weights
                        except Exception as e:
                            if self.log_path:
                                append_log(
                                    f'ExpertResolver: failed to use native_cache pinned pool: {e}, using non-pinned',
                                    self.log_path
                                )
                            # Fallback to non-pinned
                            intermediate = weights[0].shape[0]
                            w13_cpu = torch.empty([intermediate * 2, weights[0].shape[1]], dtype=weights[0].dtype, device='cpu')
                            w13_cpu[:intermediate].copy_(weights[0])
                            w13_cpu[intermediate:].copy_(weights[2])
                            
                            cpu_weights = {
                                'w13': w13_cpu,
                                'w2': weights[1].clone(),
                            }
                            self._cpu_weight_cache[cache_key] = cpu_weights
                    else:
                        # pinned memory disabled, use regular CPU cache
                        intermediate = weights[0].shape[0]
                        w13_cpu = torch.empty([intermediate * 2, weights[0].shape[1]], dtype=weights[0].dtype, device='cpu')
                        w13_cpu[:intermediate].copy_(weights[0])
                        w13_cpu[intermediate:].copy_(weights[2])
                        
                        cpu_weights = {
                            'w13': w13_cpu,
                            'w2': weights[1].clone(),
                        }
                        self._cpu_weight_cache[cache_key] = cpu_weights

            
            # ========== 3.5. 处理未缓存情况 ==========
            if cpu_weights is None:
                # 不缓存，直接转换为 sglang 格式返回
                w13 = torch.cat([weights[0], weights[2]], dim=0)
                cpu_weights = {
                    'w13': w13,
                    'w2': weights[1],
                }
            
            # ========== 4. 根据 device 参数返回 ==========
            if device == "cpu":
                # 返回 CPU weights（可能是 pinned）
                return cpu_weights
            else:
                # 传输到目标 GPU 设备（sglang 格式）
                return {
                    'w13': cpu_weights['w13'].to(device),
                    'w2': cpu_weights['w2'].to(device),
                }
            
        except Exception as e:
            if self.log_path:
                append_log(
                    f'ExpertResolver: failed to load HF weights for [{layer_idx}][{expert_idx}]: {e}',
                    self.log_path
                )
            return None
    
    def _batch_load_weights_with_index(
        self,
        key_patterns: list,
        shard_index: Dict[str, Any],
        device: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Batch load weights using shard index for efficiency.
        
        Groups keys by shard file to minimize file opens.
        """
        weight_map = shard_index.get('weight_map', {})
        
        # Group keys by shard file
        shard_to_keys: Dict[str, list] = {}
        key_to_idx = {}
        
        for idx, key in enumerate(key_patterns):
            if key in weight_map:
                shard_file = weight_map[key]
                if shard_file not in shard_to_keys:
                    shard_to_keys[shard_file] = []
                shard_to_keys[shard_file].append(key)
                key_to_idx[key] = idx
        
        if len(key_to_idx) != len(key_patterns):
            # Some keys not found in index
            return None
        
        weights = [None, None, None]
        
        # Load from each shard file
        for shard_file, keys in shard_to_keys.items():
            shard_path = os.path.join(self._hf_model_path, shard_file)
            
            for key in keys:
                tensor = self._load_tensor_from_file(shard_path, key, device)
                if tensor is None:
                    return None
                weights[key_to_idx[key]] = tensor
        
        if any(w is None for w in weights):
            return None
        
        return tuple(weights)
    
    def _try_load_weights(
        self,
        key_patterns: list,
        device: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Try to load weights matching the given key patterns.
        
        Args:
            key_patterns: List of 3 weight key patterns [w1, w2, w3]
            device: Device to load weights to
            
        Returns:
            Tuple of (w1, w2, w3) tensors or None
        """
        shard_index = self._load_shard_index()
        
        # Check if using safetensors or pytorch format
        safetensors_files = [f for f in os.listdir(self._hf_model_path) if f.endswith('.safetensors')]
        use_safetensors = len(safetensors_files) > 0
        
        weights = []
        
        for key in key_patterns:
            weight = None
            
            if shard_index is not None:
                # Find which shard contains this weight
                weight_map = shard_index.get('weight_map', {})
                if key in weight_map:
                    shard_file = weight_map[key]
                    shard_path = os.path.join(self._hf_model_path, shard_file)
                    weight = self._load_tensor_from_file(shard_path, key, device)
            else:
                # Try loading from single file or iterate through shards
                if use_safetensors:
                    for sf in safetensors_files:
                        shard_path = os.path.join(self._hf_model_path, sf)
                        weight = self._load_tensor_from_file(shard_path, key, device)
                        if weight is not None:
                            break
                else:
                    # Try pytorch format
                    for pf in os.listdir(self._hf_model_path):
                        if pf.endswith('.bin'):
                            shard_path = os.path.join(self._hf_model_path, pf)
                            weight = self._load_tensor_from_file(shard_path, key, device)
                            if weight is not None:
                                break
            
            if weight is None:
                return None
            
            weights.append(weight)
        
        return tuple(weights) if len(weights) == 3 else None
    
    def _load_tensor_from_file(
        self,
        file_path: str,
        key: str,
        device: str
    ) -> Optional[torch.Tensor]:
        """
        Load a single tensor from a checkpoint file.
        
        Uses cached file handles for safetensors to avoid repeated file opens.
        
        Args:
            file_path: Path to checkpoint file
            key: Tensor key name
            device: Device to load to
            
        Returns:
            Tensor or None if not found
        """
        try:
            if file_path.endswith('.safetensors'):
                from safetensors import safe_open
                
                # Use cached file handle if available
                with self._handles_lock:
                    if file_path not in self._safetensors_handles:
                        self._safetensors_handles[file_path] = safe_open(
                            file_path, framework="pt", device=device
                        )
                    handle = self._safetensors_handles[file_path]
                
                if key in handle.keys():
                    return handle.get_tensor(key)
                    
            elif file_path.endswith('.bin'):
                # Load pytorch checkpoint (memory-mapped for efficiency)
                state_dict = torch.load(file_path, map_location=device, weights_only=True)
                if key in state_dict:
                    return state_dict[key]
        except Exception as e:
            if self.log_path:
                append_log(
                    f'ExpertResolver: error loading {key} from {file_path}: {e}',
                    self.log_path
                )
        
        return None
    
    def close_file_handles(self) -> None:
        """Close all cached safetensors file handles."""
        with self._handles_lock:
            self._safetensors_handles.clear()
            if self.log_path:
                append_log('ExpertResolver: closed all file handles', self.log_path)
    
    def clear_cache(self) -> None:
        """Clear the expert weights cache."""
        with self._lock:
            self._expert_cache.clear()
            if self.log_path:
                append_log('ExpertResolver: cache cleared', self.log_path)
    
    def clear_cpu_weight_cache(self) -> None:
        """Clear the CPU weight cache to free memory."""
        with self._cpu_cache_lock:
            num_entries = len(self._cpu_weight_cache)
            self._cpu_weight_cache.clear()
            if self.log_path:
                append_log(f'ExpertResolver: cleared {num_entries} entries from CPU weight cache', self.log_path)
    
    def clear_pinned_pool(self) -> None:
        """清理 pinned memory pool."""
        with self._cpu_cache_lock:
            self._pinned_w1_storage = None
            self._pinned_w2_storage = None
            self._pinned_w3_storage = None
            self._pinned_pool_initialized = False
            if self.log_path:
                append_log('ExpertResolver: cleared pinned memory pool', self.log_path)
    
    def preload_experts_to_cpu_cache(
        self,
        expert_list: list,  # List of (layer_idx, expert_idx) tuples
        num_workers: int = 8
    ) -> int:
        """
        预热: 批量加载专家权重到CPU内存缓存.
        
        这个方法可以在推理开始前调用，将常用专家预加载到内存中，
        避免推理时的磁盘IO延迟。
        
        Args:
            expert_list: List of (layer_idx, expert_idx) tuples to preload
            num_workers: Number of parallel workers for loading
            
        Returns:
            Number of experts successfully loaded
        """
        import concurrent.futures
        
        if not self._cpu_cache_enabled:
            log_once('preload_disabled', 'CPU cache disabled, skipping preload')
            return 0
        
        def load_one(args):
            layer_idx, expert_idx = args
            try:
                # 加载到CPU并缓存
                result = self.load_expert_weights_from_hf(
                    layer_idx, expert_idx, 
                    device="cpu", 
                    use_cache=True, 
                    cache_on_cpu=True
                )
                return result is not None
            except Exception as e:
                return False
        
        loaded = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_one, args): args for args in expert_list}
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    loaded += 1
        
        if self.log_path:
            append_log(
                f'ExpertResolver: preloaded {loaded}/{len(expert_list)} experts to CPU cache',
                self.log_path
            )
        
        return loaded
    
    def get_cpu_cache_info(self) -> Dict[str, Any]:
        """获取CPU缓存信息."""
        with self._cpu_cache_lock:
            # 估算内存使用
            total_bytes = 0
            num_pinned = 0
            num_unpinned = 0
            
            for weights in self._cpu_weight_cache.values():
                for tensor in weights.values():
                    total_bytes += tensor.numel() * tensor.element_size()
                    if tensor.is_pinned():
                        num_pinned += 1
                    else:
                        num_unpinned += 1
            
            return {
                'enabled': self._cpu_cache_enabled,
                'num_cached': len(self._cpu_weight_cache),
                'max_entries': self._cpu_cache_max_experts,
                'cached_keys': list(self._cpu_weight_cache.keys()),
                'memory_usage_mb': total_bytes / (1024 * 1024),
                'use_pinned_memory': self._use_pinned_memory,
                'pinned_pool_initialized': self._pinned_pool_initialized,
                'pinned_pool_size': self._pinned_pool_size,
                'pinned_slots_occupied': len(self._pinned_slot_occupied),
                'num_pinned_tensors': num_pinned,
                'num_unpinned_tensors': num_unpinned,
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached experts.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            cpu_cache_info = self.get_cpu_cache_info()
            return {
                'num_cached_experts': len(self._expert_cache),
                'cached_keys': list(self._expert_cache.keys()),
                'has_model_ref': self._model_ref is not None,
                'cpu_weight_cache': cpu_cache_info,
                'num_file_handles': len(self._safetensors_handles),
            }
