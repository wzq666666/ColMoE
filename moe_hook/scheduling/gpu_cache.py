"""
GPU Expert Cache Manager for dynamic expert scheduling.

This module manages GPU memory slots for dynamically loaded experts,
allowing runtime migration of experts between CPU and GPU.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import threading
import torch

from ..logger import log_once, append_log


@dataclass
class GPUExpertSlot:
    """Represents a single GPU expert slot."""
    slot_idx: int
    layer_idx: int
    expert_idx: Optional[int] = None  # None means empty slot
    w13_weight: Optional[torch.Tensor] = None  # [intermediate*2, hidden]
    w2_weight: Optional[torch.Tensor] = None   # [hidden, intermediate]
    is_occupied: bool = False
    

@dataclass
class LayerGPUCache:
    """GPU cache for a single layer."""
    layer_idx: int
    max_slots: int
    slots: List[GPUExpertSlot] = field(default_factory=list)
    # expert_idx -> slot_idx mapping
    expert_to_slot: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize empty slots
        self.slots = [
            GPUExpertSlot(slot_idx=i, layer_idx=self.layer_idx) 
            for i in range(self.max_slots)
        ]


class GPUExpertCache:
    """
    Manages GPU expert cache across all layers.
    
    Provides dynamic expert loading/unloading to GPU memory slots,
    with support for runtime expert migration.
    """
    
    def __init__(
        self, 
        num_layers: int,
        num_experts_per_layer: int,
        max_gpu_experts_per_layer: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        log_path: Optional[str] = None,
        lazy_init: bool = True  # 延迟初始化
    ):
        """
        Initialize GPU expert cache.
        
        Args:
            num_layers: Total number of MOE layers
            num_experts_per_layer: Total experts per layer (e.g., 8 for Mixtral)
            max_gpu_experts_per_layer: Maximum experts to cache on GPU per layer
            hidden_size: Model hidden dimension
            intermediate_size: Expert intermediate dimension
            dtype: Weight data type
            device: CUDA device
            log_path: Optional logging path
            lazy_init: If True, delay GPU memory allocation until first use
        """
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.max_gpu_experts = max_gpu_experts_per_layer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.device = device
        self.log_path = log_path
        self.lazy_init = lazy_init
        
        self._lock = threading.Lock()
        self._layer_caches: Dict[int, LayerGPUCache] = {}
        self._initialized = False
        
        if not lazy_init:
            # Immediately allocate GPU memory
            self._preallocate_gpu_memory()
        
        log_once('gpu_cache_init', 
                 f'GPUExpertCache initialized: {num_layers} layers, '
                 f'{max_gpu_experts_per_layer} GPU slots/layer, '
                 f'hidden={hidden_size}, intermediate={intermediate_size}, '
                 f'lazy={lazy_init}')
    
    def initialize(self):
        """
        Explicitly initialize GPU memory allocation.
        Call this after the model has been loaded to avoid memory conflicts.
        """
        self._ensure_initialized()
    
    @property
    def is_initialized(self) -> bool:
        """Check if GPU memory has been allocated."""
        return self._initialized
    
    def _ensure_initialized(self):
        """Ensure GPU memory is allocated (for lazy init mode)."""
        if self._initialized:
            return
        
        with self._lock:
            if not self._initialized:
                self._preallocate_gpu_memory()
                self._initialized = True
    
    def _preallocate_gpu_memory(self):
        """Pre-allocate GPU memory slots for all layers."""
        for layer_idx in range(self.num_layers):
            cache = LayerGPUCache(
                layer_idx=layer_idx,
                max_slots=self.max_gpu_experts
            )
            
            # Pre-allocate tensor memory for each slot
            for slot in cache.slots:
                # w13 = [gate_proj, up_proj] concatenated: [intermediate*2, hidden]
                slot.w13_weight = torch.zeros(
                    self.intermediate_size * 2, 
                    self.hidden_size,
                    dtype=self.dtype,
                    device=self.device
                )
                # w2 = down_proj: [hidden, intermediate]
                slot.w2_weight = torch.zeros(
                    self.hidden_size,
                    self.intermediate_size,
                    dtype=self.dtype,
                    device=self.device
                )
            
            self._layer_caches[layer_idx] = cache
        
        if self.log_path:
            total_memory = (
                self.num_layers * self.max_gpu_experts * 
                (self.intermediate_size * 2 * self.hidden_size + 
                 self.hidden_size * self.intermediate_size) *
                (2 if self.dtype == torch.bfloat16 else 4)
            ) / (1024 ** 3)
            append_log(
                f'GPUExpertCache: pre-allocated {total_memory:.2f} GB GPU memory',
                self.log_path
            )
    
    def load_expert(
        self, 
        layer_idx: int, 
        expert_idx: int,
        w1_weight: torch.Tensor,  # gate_proj: [intermediate, hidden]
        w2_weight: torch.Tensor,  # down_proj: [hidden, intermediate]
        w3_weight: torch.Tensor,  # up_proj: [intermediate, hidden]
    ) -> Optional[int]:
        """
        Load an expert to GPU cache.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            w1_weight: gate_proj weights
            w2_weight: down_proj weights
            w3_weight: up_proj weights
            
        Returns:
            GPU slot index if successful, None if no slot available
        """
        # Ensure GPU memory is allocated (lazy init)
        self._ensure_initialized()
        
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                if self.log_path:
                    append_log(f'GPUExpertCache: invalid layer_idx {layer_idx}', self.log_path)
                return None
            
            # Check if already loaded
            if expert_idx in cache.expert_to_slot:
                slot_idx = cache.expert_to_slot[expert_idx]
                if self.log_path:
                    append_log(
                        f'GPUExpertCache: expert[{layer_idx}][{expert_idx}] already in slot {slot_idx}',
                        self.log_path
                    )
                return slot_idx
            
            # Find empty slot
            empty_slot = None
            for slot in cache.slots:
                if not slot.is_occupied:
                    empty_slot = slot
                    break
            
            if empty_slot is None:
                if self.log_path:
                    append_log(
                        f'GPUExpertCache: no empty slot for expert[{layer_idx}][{expert_idx}]',
                        self.log_path
                    )
                return None
            
            # Copy weights to GPU slot
            # w13 = concat(w1, w3) = concat(gate_proj, up_proj)
            w13 = torch.cat([w1_weight, w3_weight], dim=0)
            
            empty_slot.w13_weight.copy_(w13.to(self.dtype).to(self.device))
            empty_slot.w2_weight.copy_(w2_weight.to(self.dtype).to(self.device))
            empty_slot.expert_idx = expert_idx
            empty_slot.is_occupied = True
            
            cache.expert_to_slot[expert_idx] = empty_slot.slot_idx
            
            if self.log_path:
                append_log(
                    f'GPUExpertCache: loaded expert[{layer_idx}][{expert_idx}] to slot {empty_slot.slot_idx}',
                    self.log_path
                )
            
            return empty_slot.slot_idx
    
    def unload_expert(self, layer_idx: int, expert_idx: int) -> bool:
        """
        Unload an expert from GPU cache.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return False
            
            if expert_idx not in cache.expert_to_slot:
                if self.log_path:
                    append_log(
                        f'GPUExpertCache: expert[{layer_idx}][{expert_idx}] not in GPU cache',
                        self.log_path
                    )
                return False
            
            slot_idx = cache.expert_to_slot.pop(expert_idx)
            slot = cache.slots[slot_idx]
            
            # Mark slot as empty (don't need to zero out memory)
            slot.expert_idx = None
            slot.is_occupied = False
            
            if self.log_path:
                append_log(
                    f'GPUExpertCache: unloaded expert[{layer_idx}][{expert_idx}] from slot {slot_idx}',
                    self.log_path
                )
            
            return True
    
    def get_gpu_experts(self, layer_idx: int) -> Set[int]:
        """
        Get set of expert indices currently on GPU for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Set of expert indices on GPU
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return set()
            return set(cache.expert_to_slot.keys())
    
    def get_slot_for_expert(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        """
        Get GPU slot index for an expert.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            Slot index if expert is on GPU, None otherwise
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return None
            return cache.expert_to_slot.get(expert_idx)
    
    def get_slot_weights(
        self, 
        layer_idx: int, 
        slot_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get weight tensors for a GPU slot.
        
        Args:
            layer_idx: Layer index
            slot_idx: Slot index
            
        Returns:
            Tuple of (w13_weight, w2_weight) or None
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None or slot_idx >= len(cache.slots):
                return None
            slot = cache.slots[slot_idx]
            if not slot.is_occupied:
                return None
            return (slot.w13_weight, slot.w2_weight)
    
    def get_all_slot_weights(
        self, 
        layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get stacked weight tensors for all occupied slots in a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tuple of (w13_weights, w2_weights) with shape [num_occupied, ...]
            or (None, None) if no slots occupied
        """
        with self._lock:
            cache = self._layer_caches.get(layer_idx)
            if cache is None:
                return (None, None)
            
            w13_list = []
            w2_list = []
            
            for slot in cache.slots:
                if slot.is_occupied:
                    w13_list.append(slot.w13_weight)
                    w2_list.append(slot.w2_weight)
            
            if not w13_list:
                return (None, None)
            
            return (torch.stack(w13_list), torch.stack(w2_list))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = {
                'num_layers': self.num_layers,
                'max_gpu_experts_per_layer': self.max_gpu_experts,
                'layer_stats': {}
            }
            
            for layer_idx, cache in self._layer_caches.items():
                occupied = sum(1 for s in cache.slots if s.is_occupied)
                stats['layer_stats'][layer_idx] = {
                    'occupied_slots': occupied,
                    'total_slots': self.max_gpu_experts,
                    'experts_on_gpu': list(cache.expert_to_slot.keys())
                }
            
            return stats


# Global GPU cache instance
_gpu_cache: Optional[GPUExpertCache] = None
_gpu_cache_lock = threading.Lock()


def get_gpu_cache() -> Optional[GPUExpertCache]:
    """Get the global GPU expert cache instance."""
    return _gpu_cache


def init_gpu_cache(
    num_layers: int,
    num_experts_per_layer: int,
    max_gpu_experts_per_layer: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    log_path: Optional[str] = None,
    lazy_init: bool = True
) -> GPUExpertCache:
    """
    Initialize the global GPU expert cache.
    
    Should be called once during model initialization.
    
    Args:
        lazy_init: If True, defer GPU memory allocation until first use or
                   explicit initialize() call. This avoids memory conflicts
                   during model loading.
    """
    global _gpu_cache
    
    with _gpu_cache_lock:
        if _gpu_cache is not None:
            log_once('gpu_cache_reinit', 'GPUExpertCache already initialized, returning existing')
            return _gpu_cache
        
        _gpu_cache = GPUExpertCache(
            num_layers=num_layers,
            num_experts_per_layer=num_experts_per_layer,
            max_gpu_experts_per_layer=max_gpu_experts_per_layer,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            device=device,
            log_path=log_path,
            lazy_init=lazy_init
        )
        
        return _gpu_cache
