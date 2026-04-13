"""
Expert Location Map for dynamic expert routing.

Tracks which experts are on CPU vs GPU and their corresponding slot indices.
Used at runtime to dynamically route tokens to the correct compute location.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import threading

from ..logger import log_once, append_log


class ExpertLocation(Enum):
    """Location of an expert."""
    CPU = "cpu"      # Expert is on CPU (computed by KTMoEWrapper)
    GPU = "gpu"      # Expert is on GPU cache


@dataclass
class ExpertInfo:
    """Information about a single expert's location."""
    layer_idx: int
    expert_idx: int
    location: ExpertLocation = ExpertLocation.CPU
    gpu_slot_idx: Optional[int] = None  # Only valid when location == GPU


class ExpertLocationMap:
    """
    Maps expert indices to their compute locations (CPU or GPU).
    
    This map is updated when experts are migrated between CPU and GPU,
    and is queried during inference to route tokens correctly.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_experts_per_layer: int,
        log_path: Optional[str] = None
    ):
        """
        Initialize expert location map.
        
        Args:
            num_layers: Total number of MOE layers
            num_experts_per_layer: Total experts per layer
            log_path: Optional logging path
        """
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.log_path = log_path
        
        self._lock = threading.RLock()
        
        # Initialize all experts as CPU
        # _map[layer_idx][expert_idx] = ExpertInfo
        self._map: Dict[int, Dict[int, ExpertInfo]] = {}
        for layer_idx in range(num_layers):
            self._map[layer_idx] = {}
            for expert_idx in range(num_experts_per_layer):
                self._map[layer_idx][expert_idx] = ExpertInfo(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    location=ExpertLocation.CPU,
                    gpu_slot_idx=None
                )
        
        log_once('location_map_init', 
                 f'ExpertLocationMap initialized: {num_layers} layers, '
                 f'{num_experts_per_layer} experts/layer')
    
    def get_location(self, layer_idx: int, expert_idx: int) -> ExpertInfo:
        """
        Get location info for an expert.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            ExpertInfo with location details
        """
        with self._lock:
            if layer_idx not in self._map:
                return ExpertInfo(layer_idx, expert_idx, ExpertLocation.CPU, None)
            if expert_idx not in self._map[layer_idx]:
                raise ValueError(f'Invalid expert_idx {expert_idx} for layer {layer_idx}')
            return self._map[layer_idx][expert_idx]
    
    def set_gpu_location(
        self, 
        layer_idx: int, 
        expert_idx: int, 
        gpu_slot_idx: int
    ) -> None:
        """
        Mark an expert as being on GPU.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            gpu_slot_idx: GPU slot index where expert is loaded
        """
        with self._lock:
            if layer_idx not in self._map:
                self._map[layer_idx] = {}
            
            self._map[layer_idx][expert_idx] = ExpertInfo(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                location=ExpertLocation.GPU,
                gpu_slot_idx=gpu_slot_idx
            )
        
        if self.log_path:
            append_log(
                f'LocationMap: expert[{layer_idx}][{expert_idx}] -> GPU slot {gpu_slot_idx}',
                self.log_path
            )
    
    def set_cpu_location(self, layer_idx: int, expert_idx: int) -> None:
        """
        Mark an expert as being on CPU.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
        """
        with self._lock:
            if layer_idx not in self._map:
                self._map[layer_idx] = {}
            
            self._map[layer_idx][expert_idx] = ExpertInfo(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                location=ExpertLocation.CPU,
                gpu_slot_idx=None
            )
        
        if self.log_path:
            append_log(
                f'LocationMap: expert[{layer_idx}][{expert_idx}] -> CPU',
                self.log_path
            )
    
    def get_gpu_experts(self, layer_idx: int) -> Dict[int, int]:
        """
        Get all GPU experts for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Dict mapping expert_idx -> gpu_slot_idx for all GPU experts
        """
        with self._lock:
            if layer_idx not in self._map:
                return {}
            
            result = {}
            for expert_idx, info in self._map[layer_idx].items():
                if info.location == ExpertLocation.GPU and info.gpu_slot_idx is not None:
                    result[expert_idx] = info.gpu_slot_idx
            
            return result
    
    def get_cpu_experts(self, layer_idx: int) -> Set[int]:
        """
        Get all CPU experts for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Set of expert indices on CPU
        """
        with self._lock:
            if layer_idx not in self._map:
                return set(range(self.num_experts_per_layer))
            
            return {
                expert_idx 
                for expert_idx, info in self._map[layer_idx].items()
                if info.location == ExpertLocation.CPU
            }
    
    def partition_expert_ids(
        self, 
        layer_idx: int, 
        topk_ids: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", Dict[int, int]]:
        """
        Partition expert IDs into GPU and CPU sets based on current locations.
        
        Args:
            layer_idx: Layer index
            topk_ids: Tensor of selected expert IDs [num_tokens, top_k]
            
        Returns:
            Tuple of:
            - gpu_mask: Boolean mask where True = expert on GPU
            - cpu_mask: Boolean mask where True = expert on CPU
            - expert_to_slot: Mapping of expert_idx -> gpu_slot_idx
        """
        import torch
        
        with self._lock:
            gpu_experts = self.get_gpu_experts(layer_idx)
            
            # Create masks
            gpu_mask = torch.zeros_like(topk_ids, dtype=torch.bool)
            
            for expert_idx, slot_idx in gpu_experts.items():
                gpu_mask |= (topk_ids == expert_idx)
            
            cpu_mask = ~gpu_mask
            
            return gpu_mask, cpu_mask, gpu_experts
    
    def remap_gpu_expert_ids(
        self,
        layer_idx: int,
        topk_ids: "torch.Tensor",
        gpu_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Remap expert IDs to GPU slot indices for GPU computation.
        
        Args:
            layer_idx: Layer index
            topk_ids: Original expert IDs
            gpu_mask: Mask indicating which positions are GPU experts
            
        Returns:
            Remapped IDs where GPU experts use slot indices
        """
        import torch
        
        with self._lock:
            gpu_experts = self.get_gpu_experts(layer_idx)
            
            # Clone to avoid modifying original
            remapped = topk_ids.clone()
            
            # Remap GPU experts to their slot indices
            for expert_idx, slot_idx in gpu_experts.items():
                expert_mask = (topk_ids == expert_idx) & gpu_mask
                remapped[expert_mask] = slot_idx
            
            # Mask CPU experts as -1 (will be skipped by GPU kernel)
            remapped[~gpu_mask] = -1
            
            return remapped
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about expert locations."""
        with self._lock:
            stats = {
                'num_layers': self.num_layers,
                'num_experts_per_layer': self.num_experts_per_layer,
                'layer_stats': {}
            }
            
            for layer_idx in range(self.num_layers):
                gpu_experts = self.get_gpu_experts(layer_idx)
                cpu_experts = self.get_cpu_experts(layer_idx)
                stats['layer_stats'][layer_idx] = {
                    'gpu_experts': list(gpu_experts.keys()),
                    'cpu_experts': list(cpu_experts),
                    'num_on_gpu': len(gpu_experts),
                    'num_on_cpu': len(cpu_experts)
                }
            
            return stats


# Global location map instance
_location_map: Optional[ExpertLocationMap] = None
_location_map_lock = threading.Lock()


def get_location_map() -> Optional[ExpertLocationMap]:
    """Get the global expert location map instance."""
    return _location_map


def init_location_map(
    num_layers: int,
    num_experts_per_layer: int,
    log_path: Optional[str] = None
) -> ExpertLocationMap:
    """
    Initialize the global expert location map.
    
    Should be called once during model initialization.
    """
    global _location_map
    
    with _location_map_lock:
        if _location_map is not None:
            log_once('location_map_reinit', 'ExpertLocationMap already initialized, returning existing')
            return _location_map
        
        _location_map = ExpertLocationMap(
            num_layers=num_layers,
            num_experts_per_layer=num_experts_per_layer,
            log_path=log_path
        )
        
        return _location_map
