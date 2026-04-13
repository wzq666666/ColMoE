"""
Expert Scheduler - decides expert placement based on predictions.

This module provides the scheduling logic that:
1. Receives prediction results for next layer
2. Decides which experts should be on GPU vs CPU
3. Updates the location map with the execution plan
4. Triggers async prefetch for experts that need migration

The scheduler runs BEFORE inference of each layer, preparing
the next layer's expert distribution.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from ..logger import log_once, append_log
from .expert_location import get_location_map, ExpertLocation
from .gpu_cache import get_gpu_cache
from .prefetcher import get_prefetcher


class ScheduleStatus(Enum):
    """Status of a layer's execution plan."""
    NOT_SCHEDULED = "not_scheduled"
    SCHEDULED = "scheduled"        # Plan created, prefetch started
    READY = "ready"                # All GPU experts loaded
    EXECUTING = "executing"        # Currently being processed
    COMPLETED = "completed"        # Layer inference done


@dataclass
class LayerExecutionPlan:
    """Execution plan for a single layer."""
    layer_idx: int
    gpu_experts: Set[int] = field(default_factory=set)  # Experts planned for GPU
    cpu_experts: Set[int] = field(default_factory=set)  # Experts planned for CPU
    status: ScheduleStatus = ScheduleStatus.NOT_SCHEDULED
    created_at: float = 0.0
    ready_at: Optional[float] = None
    
    # Track which GPU experts are actually loaded
    loaded_experts: Set[int] = field(default_factory=set)
    pending_experts: Set[int] = field(default_factory=set)  # Being prefetched
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def is_expert_ready(self, expert_idx: int) -> bool:
        """Check if a specific expert is ready on GPU."""
        return expert_idx in self.loaded_experts
    
    def mark_expert_loaded(self, expert_idx: int):
        """Mark an expert as loaded to GPU."""
        if expert_idx in self.pending_experts:
            self.pending_experts.remove(expert_idx)
        self.loaded_experts.add(expert_idx)
        
        # Check if all GPU experts are ready
        if self.pending_experts == set() and self.loaded_experts >= self.gpu_experts:
            self.status = ScheduleStatus.READY
            self.ready_at = time.time()
    
    def get_ready_gpu_experts(self) -> Set[int]:
        """Get experts that are planned for GPU and already loaded."""
        return self.gpu_experts & self.loaded_experts
    
    def get_pending_gpu_experts(self) -> Set[int]:
        """Get experts that are planned for GPU but still loading."""
        return self.gpu_experts - self.loaded_experts


class ExpertScheduler:
    """
    Schedules expert placement across layers.
    
    The scheduler maintains execution plans for each layer and
    coordinates with the prefetcher to load experts asynchronously.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        max_gpu_experts_per_layer: int,
        log_path: Optional[str] = None
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.max_gpu_experts = max_gpu_experts_per_layer
        self.log_path = log_path
        
        self._lock = threading.Lock()
        self._plans: Dict[int, LayerExecutionPlan] = {}
        self._current_layer = -1
        
        log_once('scheduler_init', 
                 f'ExpertScheduler initialized: {num_layers} layers, '
                 f'{num_experts} experts, {max_gpu_experts_per_layer} GPU slots')
    
    def schedule_layer(
        self,
        layer_idx: int,
        predicted_experts: Set[int],
        priority_experts: Optional[Set[int]] = None
    ) -> LayerExecutionPlan:
        """
        Create execution plan for a layer based on predictions.
        
        Args:
            layer_idx: Target layer index
            predicted_experts: Set of expert indices predicted to be activated
            priority_experts: Optional set of experts with higher priority for GPU
            
        Returns:
            LayerExecutionPlan for the layer
        """
        with self._lock:
            # Check if plan already exists
            existing_plan = self._plans.get(layer_idx)
            if existing_plan is not None and self.log_path:
                append_log(
                    f'Scheduler: layer {layer_idx} already has plan (GPU={sorted(existing_plan.gpu_experts)}), '
                    f'overwriting with new prediction',
                    self.log_path
                )
            
            # Select which experts go to GPU
            gpu_experts = self._select_gpu_experts(
                layer_idx, predicted_experts, priority_experts
            )
            cpu_experts = predicted_experts - gpu_experts
            
            # Create plan
            plan = LayerExecutionPlan(
                layer_idx=layer_idx,
                gpu_experts=gpu_experts,
                cpu_experts=cpu_experts,
                status=ScheduleStatus.SCHEDULED,
                pending_experts=gpu_experts.copy()
            )
            
            self._plans[layer_idx] = plan
            
            # Verify plan was saved correctly
            # saved_plan = self._plans.get(layer_idx)
            # if self.log_path:
            #     all_plan_keys = sorted(self._plans.keys())
            #     append_log(
            #         f'Scheduler: layer {layer_idx} plan created - '
            #         f'GPU: {sorted(gpu_experts)}, CPU: {sorted(cpu_experts)}'
            #         f' | verified: {saved_plan is not None}, all_plans: {all_plan_keys}',
            #         self.log_path
            #     )
            
            return plan
    
    def _select_gpu_experts(
        self,
        layer_idx: int,
        predicted_experts: Set[int],
        priority_experts: Optional[Set[int]] = None
    ) -> Set[int]:
        """
        Select which experts should be placed on GPU.
        
        Current strategy: Simple priority-based selection
        TODO: Implement more sophisticated strategies
        
        Args:
            layer_idx: Target layer
            predicted_experts: Predicted expert activations
            priority_experts: High priority experts (e.g., from prediction confidence)
            
        Returns:
            Set of expert indices to place on GPU
        """
        # Get current GPU cache state
        gpu_cache = get_gpu_cache()
        location_map = get_location_map()
        
        available_slots = self.max_gpu_experts
        selected = set()
        
        # Strategy 1: Prioritize experts already on GPU (if still predicted)
        if location_map is not None:
            current_gpu = set(location_map.get_gpu_experts(layer_idx).keys())
            already_on_gpu = current_gpu & predicted_experts
            
            for exp_idx in already_on_gpu:
                if len(selected) >= available_slots:
                    break
                selected.add(exp_idx)
        
        # Strategy 2: Add priority experts
        if priority_experts:
            for exp_idx in priority_experts:
                if len(selected) >= available_slots:
                    break
                if exp_idx in predicted_experts and exp_idx not in selected:
                    selected.add(exp_idx)
        
        # Strategy 3: Fill remaining slots with any predicted experts
        # for exp_idx in sorted(predicted_experts):  # Deterministic order
        #     if len(selected) >= available_slots:
        #         break
        #     if exp_idx not in selected:
        #         selected.add(exp_idx)
        
        return selected
    
    def trigger_prefetch(self, layer_idx: int) -> bool:
        """
        Trigger async prefetch for a layer's GPU experts.
        
        Only prefetches experts that are not already in GPU cache.
        
        Args:
            layer_idx: Target layer
            
        Returns:
            True if prefetch was triggered
        """
        with self._lock:
            plan = self._plans.get(layer_idx)
            if plan is None:
                return False
            
            prefetcher = get_prefetcher()
            if prefetcher is None:
                if self.log_path:
                    append_log(f'Scheduler: prefetcher not available for layer {layer_idx}', self.log_path)
                return False
            
            # Check which experts are already in GPU cache
            gpu_cache = get_gpu_cache()
            location_map = get_location_map()
            
            already_loaded = set()
            if gpu_cache is not None and location_map is not None:
                current_gpu_experts = location_map.get_gpu_experts(layer_idx)
                already_loaded = set(current_gpu_experts.keys())
            
            # Filter out experts already on GPU
            need_prefetch = plan.pending_experts - already_loaded
            
            # Mark already loaded experts as ready
            for exp_idx in (plan.pending_experts & already_loaded):
                plan.mark_expert_loaded(exp_idx)
            
            if self.log_path and already_loaded & plan.gpu_experts:
                append_log(
                    f'Scheduler: layer {layer_idx} experts already on GPU: {sorted(already_loaded & plan.gpu_experts)}',
                    self.log_path
                )
            
            # Build prefetch list for experts not yet on GPU
            prefetch_list = [(layer_idx, exp_idx) for exp_idx in need_prefetch]
            
            if prefetch_list:
                result = prefetcher.prefetch(prefetch_list, async_mode=True)
                
                if self.log_path:
                    append_log(
                        f'Scheduler: triggered prefetch for layer {layer_idx} - '
                        f'{len(prefetch_list)} experts, submitted={result.get("submitted", 0)}',
                        self.log_path
                    )
            elif self.log_path:
                append_log(
                    f'Scheduler: layer {layer_idx} all GPU experts already cached, skipping prefetch',
                    self.log_path
                )
            
            return True
    
    def get_plan(self, layer_idx: int) -> Optional[LayerExecutionPlan]:
        """Get execution plan for a layer."""
        with self._lock:
            return self._plans.get(layer_idx)
    
    def has_pending_migrations(self) -> bool:
        """
        Check if there are any pending expert migrations.
        
        Returns:
            True if any layer has pending expert loading, False otherwise
        """
        with self._lock:
            for plan in self._plans.values():
                if plan.pending_experts:
                    return True
            return False
    
    def update_expert_loaded(self, layer_idx: int, expert_idx: int):
        """Callback when an expert finishes loading to GPU."""
        with self._lock:
            plan = self._plans.get(layer_idx)
            if plan:
                plan.mark_expert_loaded(expert_idx)
                
                if self.log_path:
                    append_log(
                        f'Scheduler: expert[{layer_idx}][{expert_idx}] loaded, '
                        f'ready={sorted(plan.loaded_experts)}, pending={sorted(plan.pending_experts)}',
                        self.log_path
                    )
    
    def mark_layer_executing(self, layer_idx: int):
        """Mark a layer as currently executing."""
        with self._lock:
            plan = self._plans.get(layer_idx)
            if plan:
                plan.status = ScheduleStatus.EXECUTING
                self._current_layer = layer_idx
    
    def mark_layer_completed(self, layer_idx: int):
        """Mark a layer as completed."""
        with self._lock:
            plan = self._plans.get(layer_idx)
            if plan:
                plan.status = ScheduleStatus.COMPLETED
    
    def cleanup_old_plans(self, keep_layers: int = 5):
        """Clean up old execution plans to save memory.
        
        Args:
            keep_layers: Number of layers' plans to keep after current layer.
                        Default is 5 to handle async execution and potential delays.
        
        Note:
            This method handles layer wrap-around correctly. When current_layer
            is near the end (e.g., 31) and new plans are created for early layers
            (e.g., 0, 1, 2), those new plans won't be deleted.
        """
        with self._lock:
            if self._current_layer < 0:
                return
            
            to_remove = []
            layers_to_keep = [(self._current_layer + i) % self.num_layers for i in range(keep_layers)]
            for layer_idx in self._plans:
                if layer_idx in layers_to_keep:
                    continue

                to_remove.append(layer_idx)
            
            if to_remove and self.log_path:
                append_log(
                    f'Scheduler: cleanup removing plans for layers {sorted(to_remove)} '
                    f'(current_layer={self._current_layer}, keep={keep_layers})',
                    self.log_path
                )
            
            for layer_idx in to_remove:
                del self._plans[layer_idx]
    
    def get_execution_guidance(
        self,
        layer_idx: int,
        requested_experts: Set[int]
    ) -> Dict[str, Any]:
        """
        Get execution guidance for dynamic_apply.
        
        Returns info about which experts are ready on GPU,
        which are still loading, and which should go to CPU.
        
        Args:
            layer_idx: Current layer being executed
            requested_experts: Experts actually requested by routing
            
        Returns:
            Dict with execution guidance:
            - ready_gpu_experts: Set of experts ready on GPU
            - pending_gpu_experts: Set of experts still loading (wait for them)
            - cpu_experts: Set of experts to run on CPU
            - should_wait: Whether to wait for pending experts
        """
        with self._lock:
            plan = self._plans.get(layer_idx)
            
            # Debug: log current plans state
            if self.log_path:
                available_plans = sorted(self._plans.keys())
                append_log(
                    f'Scheduler: get_execution_guidance layer={layer_idx}, '
                    f'available_plans={available_plans}, '
                    f'found_plan={plan is not None}',
                    self.log_path
                )
            
            if plan is None:
                # No plan for this layer - check location_map for preloaded experts
                location_map = get_location_map()
                if location_map is not None:
                    # Get experts already on GPU from preloading
                    current_gpu_experts = set(location_map.get_gpu_experts(layer_idx).keys())
                    ready_gpu = current_gpu_experts & requested_experts
                    cpu_experts = requested_experts - ready_gpu
                    
                    if self.log_path:
                        append_log(
                            f'Scheduler: layer {layer_idx} no plan, using preloaded GPU experts: {sorted(ready_gpu)}',
                            self.log_path
                        )
                    
                    return {
                        'ready_gpu_experts': ready_gpu,
                        'pending_gpu_experts': set(),
                        'cpu_experts': cpu_experts,
                        'should_wait': False,
                    }
                else:
                    # No location map, all to CPU
                    return {
                        'ready_gpu_experts': set(),
                        'pending_gpu_experts': set(),
                        'cpu_experts': requested_experts,
                        'should_wait': False,
                    }
            
            # Intersect with actually requested experts
            planned_gpu = plan.gpu_experts & requested_experts
            ready_gpu = plan.loaded_experts & requested_experts
            pending_gpu = (planned_gpu - ready_gpu) & plan.pending_experts
            cpu_experts = requested_experts - planned_gpu
            
            # Should we wait for pending GPU experts?
            # wzq todo: another strategy could be based on timeouts
            # Policy: Wait if there are pending experts and they're few enough
            # should_wait = len(pending_gpu) > 0 and len(pending_gpu) <= 2
            should_wait = True  # Always wait for now

            return {
                'ready_gpu_experts': ready_gpu,
                'pending_gpu_experts': pending_gpu,
                'cpu_experts': cpu_experts,
                'should_wait': should_wait,
            }


# ============================================================
# Global scheduler instance
# ============================================================

_scheduler: Optional[ExpertScheduler] = None
_scheduler_lock = threading.Lock()


def init_scheduler(
    num_layers: int,
    num_experts: int,
    max_gpu_experts_per_layer: int,
    log_path: Optional[str] = None
) -> ExpertScheduler:
    """Initialize the global scheduler instance."""
    global _scheduler
    
    with _scheduler_lock:
        if _scheduler is not None:
            log_once('scheduler_reinit', 'Scheduler already initialized')
            return _scheduler
        
        _scheduler = ExpertScheduler(
            num_layers=num_layers,
            num_experts=num_experts,
            max_gpu_experts_per_layer=max_gpu_experts_per_layer,
            log_path=log_path
        )
        return _scheduler


def get_scheduler() -> Optional[ExpertScheduler]:
    """Get the global scheduler instance."""
    return _scheduler


def schedule_next_layer(
    current_layer_idx: int,
    predicted_experts: Set[int]
) -> Optional[LayerExecutionPlan]:
    """
    Convenience function to schedule the next layer.
    
    Args:
        current_layer_idx: Current layer being processed
        predicted_experts: Predicted experts for next layer
        
    Returns:
        Execution plan for next layer
    """
    if _scheduler is None:
        return None
    
    next_layer = (current_layer_idx + 1) % _scheduler.num_layers
    plan = _scheduler.schedule_layer(next_layer, predicted_experts)
    _scheduler.trigger_prefetch(next_layer)
    
    return plan
