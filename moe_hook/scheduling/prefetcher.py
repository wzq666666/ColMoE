"""
Expert Prefetcher - loads predicted experts to GPU cache.

This module handles the prefetching of experts based on prediction results.
It interfaces with the GPU cache and migration manager to load/unload experts.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import threading

from ..logger import log_once, append_log


class ExpertPrefetcher:
    """
    Prefetches experts to GPU cache based on prediction results.
    
    This class coordinates with the migration manager to:
    1. Load predicted experts to GPU cache (if not already there)
    2. Optionally evict less-used experts to make room
    """
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._lock = threading.Lock()
        self._call_count = 0
        self._prefetch_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'loads_triggered': 0,
            'loads_failed': 0,
        }
    
    def prefetch(
        self,
        predicted_experts: List[Tuple[int, int]],
        async_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Prefetch predicted experts to GPU cache.
        
        Args:
            predicted_experts: List of (layer_idx, expert_idx) tuples to prefetch
            async_mode: If True, submit async migration tasks (non-blocking)
                       If False, wait for migration to complete
        
        Returns:
            Dict with prefetch statistics:
            - requested: number of experts requested
            - already_cached: number already in GPU cache
            - submitted: number of load tasks submitted
            - failed: number of failed submissions
        """
        from .expert_migration import get_migration_manager
        from .gpu_cache import get_gpu_cache
        from .expert_location import get_location_map, ExpertLocation
        
        result = {
            'requested': len(predicted_experts) if predicted_experts else 0,
            'already_cached': 0,
            'submitted': 0,
            'failed': 0,
        }
        
        if not predicted_experts:
            return result
        
        migration_manager = get_migration_manager()
        gpu_cache = get_gpu_cache()
        location_map = get_location_map()
        
        if migration_manager is None:
            if self.log_path:
                append_log('Prefetcher: migration_manager not initialized', self.log_path)
            return result
        
        with self._lock:
            self._prefetch_stats['total_requests'] += len(predicted_experts)
        
        # Check which experts need to be loaded
        to_load: List[Tuple[int, int]] = []
        
        # Track pending slots per layer to avoid over-submitting
        # This tracks how many slots are already used + how many we're about to submit
        pending_slots_per_layer: Dict[int, int] = {}
        
        for layer_idx, expert_idx in predicted_experts:
            # Check if already on GPU
            if location_map is not None:
                expert_info = location_map.get_location(layer_idx, expert_idx)

                if expert_info.location == ExpertLocation.GPU:
                    result['already_cached'] += 1
                    continue
            
            # Check if GPU cache has space (considering already submitted tasks)
            if gpu_cache is not None:
                # Get current GPU experts (already loaded)
                current_gpu_experts = gpu_cache.get_gpu_experts(layer_idx)
                max_slots = gpu_cache.max_gpu_experts
                
                # Initialize pending count for this layer
                if layer_idx not in pending_slots_per_layer:
                    pending_slots_per_layer[layer_idx] = len(current_gpu_experts)
                
                # Check if we have room (current + pending submissions)
                if pending_slots_per_layer[layer_idx] >= max_slots:
                    if self.log_path and self._call_count % 100 == 0:
                        append_log(
                            f'Prefetcher: layer {layer_idx} GPU cache full '
                            f'({pending_slots_per_layer[layer_idx]}/{max_slots}), skipping expert {expert_idx}',
                            self.log_path
                        )
                    continue
                
                # Reserve a slot for this expert
                pending_slots_per_layer[layer_idx] += 1
            
            to_load.append((layer_idx, expert_idx))
        
        if self.log_path and to_load:
            append_log(
                f'Prefetcher: will load {len(to_load)} experts: {to_load[:5]}{"..." if len(to_load) > 5 else ""}',
                self.log_path
            )
        
        # Submit load tasks
        for layer_idx, expert_idx in to_load:
            try:
                if async_mode:
                    task_id = migration_manager.async_load_expert(layer_idx, expert_idx)
                    if task_id is not None:
                        result['submitted'] += 1
                    else:
                        result['failed'] += 1
                else:
                    success = migration_manager.load_expert_to_gpu(layer_idx, expert_idx)
                    if success:
                        result['submitted'] += 1
                    else:
                        result['failed'] += 1
            except Exception as e:
                result['failed'] += 1
                if self.log_path:
                    append_log(f'Prefetcher: failed to load expert[{layer_idx}][{expert_idx}]: {e}', self.log_path)
        
        # Update stats
        with self._lock:
            self._call_count += 1
            self._prefetch_stats['cache_hits'] += result['already_cached']
            self._prefetch_stats['loads_triggered'] += result['submitted']
            self._prefetch_stats['loads_failed'] += result['failed']
        
        # Occasional logging
        if self.log_path and self._call_count % 50 == 1:
            append_log(
                f'Prefetcher: requested={result["requested"]}, '
                f'cached={result["already_cached"]}, '
                f'submitted={result["submitted"]}, '
                f'failed={result["failed"]}',
                self.log_path
            )
        
        return result
    
    def prefetch_for_layer(
        self,
        layer_idx: int,
        expert_indices: List[int],
        async_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Prefetch specific experts for a single layer.
        
        Args:
            layer_idx: Target layer index
            expert_indices: List of expert indices to prefetch
            async_mode: If True, use async migration
            
        Returns:
            Prefetch statistics dict
        """
        predicted_experts = [(layer_idx, exp_idx) for exp_idx in expert_indices]
        return self.prefetch(predicted_experts, async_mode=async_mode)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetch statistics."""
        with self._lock:
            return dict(self._prefetch_stats)
    
    def reset_stats(self):
        """Reset prefetch statistics."""
        with self._lock:
            self._prefetch_stats = {
                'total_requests': 0,
                'cache_hits': 0,
                'loads_triggered': 0,
                'loads_failed': 0,
            }


# ============================================================
# Global prefetcher instance
# ============================================================

_prefetcher: Optional[ExpertPrefetcher] = None
_prefetcher_lock = threading.Lock()


def init_prefetcher(log_path: Optional[str] = None) -> ExpertPrefetcher:
    """Initialize the global prefetcher instance."""
    global _prefetcher
    
    with _prefetcher_lock:
        if _prefetcher is not None:
            log_once('prefetcher_reinit', 'Prefetcher already initialized')
            return _prefetcher
        
        _prefetcher = ExpertPrefetcher(log_path=log_path)
        log_once('prefetcher_init', 'ExpertPrefetcher initialized')
        return _prefetcher


def get_prefetcher() -> Optional[ExpertPrefetcher]:
    """Get the global prefetcher instance."""
    return _prefetcher


def prefetch_experts(
    predicted_experts: List[Tuple[int, int]],
    async_mode: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to prefetch experts using global prefetcher.
    
    Args:
        predicted_experts: List of (layer_idx, expert_idx) to prefetch
        async_mode: If True, use async migration
        
    Returns:
        Prefetch statistics dict
    """
    if _prefetcher is None:
        log_once('prefetch_no_instance', 'Prefetcher not initialized, skipping prefetch')
        return {'requested': 0, 'already_cached': 0, 'submitted': 0, 'failed': 0}
    
    return _prefetcher.prefetch(predicted_experts, async_mode=async_mode)
