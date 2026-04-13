"""
Dynamic Expert Router - the core of dynamic expert scheduling.

This module provides the patched apply() method that routes tokens 
to CPU or GPU based on the current expert location map.

Execution flow:
1. Before inference: Scheduler creates execution plan based on predictions
2. During inference: dynamic_apply checks the plan and executes accordingly
   - Ready GPU experts: Execute immediately on GPU
   - Pending GPU experts: Wait for IO or fall back to CPU
   - CPU experts: Execute on CPU via KTMoEWrapper

Note on deferral mechanism:
When kt_max_deferred_experts_per_token > 0, ktransformers splits CPU experts into:
- Immediate experts: executed right away
- Deferred experts: executed in pipeline with next layer
Our dynamic routing needs to be aware of this to work correctly.
"""

import inspect
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple
import threading
import time
import torch
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor

from ..logger import log_once, append_log
from .expert_location import get_location_map, ExpertLocation
from .gpu_cache import get_gpu_cache

if TYPE_CHECKING:
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput, CombineInput


# 全局线程池用于异步predict
_predict_executor: Optional[ThreadPoolExecutor] = None
_predict_executor_lock = threading.Lock()


def _get_predict_executor() -> ThreadPoolExecutor:
    """获取或创建predict线程池."""
    global _predict_executor
    with _predict_executor_lock:
        if _predict_executor is None:
            _predict_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")
        return _predict_executor


class DynamicExpertRouter:
    """
    Routes tokens dynamically between CPU and GPU experts.
    
    This replaces the static num_gpu_experts threshold with a dynamic
    per-expert location lookup, allowing runtime expert migration.
    
    Important: When deferral is enabled (max_deferred_experts_per_token > 0),
    ktransformers will automatically split CPU experts into immediate and deferred.
    We don't interfere with this mechanism - we just control which experts go to CPU.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._call_count = 0
        self._lock = threading.Lock()
        self._predict_fn = None  # Optional: prediction/prefetch function
        
        # Track deferral state for debugging
        self._deferral_enabled = False
        self._warned_deferral = False
        
        # Performance statistics
        self._stats = {
            'total_calls': 0,
            'total_gpu_time_ms': 0.0,
            'total_cpu_time_ms': 0.0,
            'total_io_wait_time_ms': 0.0,
            'total_predict_time_ms': 0.0,
            'gpu_expert_count': 0,
            'cpu_expert_count': 0,
            'cancelled_experts': 0,
        }
    
    def set_predict_fn(self, fn):
        """Set the prediction/prefetch function to call before routing."""
        self._predict_fn = fn
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            stats = dict(self._stats)
            if stats['total_calls'] > 0:
                stats['avg_gpu_time_ms'] = stats['total_gpu_time_ms'] / stats['total_calls']
                stats['avg_cpu_time_ms'] = stats['total_cpu_time_ms'] / stats['total_calls']
                stats['avg_io_wait_time_ms'] = stats['total_io_wait_time_ms'] / stats['total_calls']
            return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self._lock:
            self._stats = {
                'total_calls': 0,
                'total_gpu_time_ms': 0.0,
                'total_cpu_time_ms': 0.0,
                'total_io_wait_time_ms': 0.0,
                'total_predict_time_ms': 0.0,
                'gpu_expert_count': 0,
                'cpu_expert_count': 0,
                'cancelled_experts': 0,
            }
    
    def dynamic_apply(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """
        Dynamic expert routing based on current location map.
        
        This is the patched apply() method that:
        1. Checks which selected experts are on GPU vs CPU
        2. Routes GPU experts to our GPU cache
        3. Routes CPU experts to KTMoEWrapper
        4. Combines results
        
        Note: When deferral is enabled, ktransformers will internally split
        CPU experts into immediate and deferred. We pass all CPU experts to
        the wrapper and let it handle deferral internally.
        
        Args:
            wrapper: The KTEPWrapperMethod instance
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
            
        Returns:
            Combined computation results
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from .scheduler import get_scheduler
        
        layer_start_time = time.time()
        io_wait_time_ms = 0.0
        gpu_time_ms = 0.0
        cancelled_count = 0
        predict_future = None  # 异步predict的future
        
        # Get layer index
        layer_idx = getattr(wrapper.kt_config, 'layer_idx', -1)
        append_log(f"=============== {layer_idx} ===============", self.log_path)
        
        # Check and warn about deferral mechanism
        max_deferred = getattr(wrapper.kt_config, 'max_deferred_experts_per_token', 0) or 0
        if max_deferred > 0 and not self._warned_deferral:
            self._deferral_enabled = True
            self._warned_deferral = True
            append_log(
                f'DynamicRouter: WARNING - deferral mechanism enabled '
                f'(max_deferred_experts_per_token={max_deferred}). '
                f'CPU experts will be split into immediate/deferred internally by ktransformers. '
                f'Consider setting --kt-max-deferred-experts-per-token 0 for simpler debugging.',
                self.log_path
            )
        
        # ========== 异步执行predict_fn (不阻塞当前层) ==========
        # predict_fn 为下一层做预测和prefetch，可以在当前层计算时并行执行
        if self._predict_fn is not None:
            executor = _get_predict_executor()
            predict_future = executor.submit(
                self._safe_predict, wrapper, dispatch_output, layer_idx
            )
        
        # ========== 快速路径: 使用原生 ktransformers apply ==========
        # 当 num_gpu_experts > 0 且没有动态调度需求时，直接使用原生实现
        # 这样可以获得 FusedMoE Triton kernel 的最佳性能
        num_native_gpu_experts = getattr(wrapper, 'num_gpu_experts', 0)
        location_map = get_location_map()
        scheduler = get_scheduler()
        
        use_native_fast_path = (
            num_native_gpu_experts > 0 and
            (location_map is None or scheduler is None or not scheduler.has_pending_migrations())
        )
        
        if use_native_fast_path:
            # 直接调用原生 apply，只保留我们的预测逻辑
            result = self._native_apply_with_logging(
                wrapper, layer, dispatch_output, layer_idx, 
                layer_start_time, predict_future
            )
            return result
        
        # ========== 动态路由路径 ==========
        # Get location map and GPU cache
        gpu_cache = get_gpu_cache()
        
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        
        # If no location map, fall back to all-CPU (original behavior)
        if location_map is None:
            return self._fallback_cpu_only(wrapper, layer, dispatch_output)
        
        # Get requested experts from topk_ids
        requested_experts = set(topk_ids.flatten().tolist())
        append_log(f"DynamicRouter[{layer_idx}]: requested experts: {sorted(requested_experts)}", self.log_path)
        
        # Get execution guidance from scheduler
        if scheduler is not None:
            scheduler.mark_layer_executing(layer_idx)
            guidance = scheduler.get_execution_guidance(layer_idx, requested_experts)            
            ready_gpu_experts = guidance['ready_gpu_experts']
            pending_gpu_experts = guidance['pending_gpu_experts']
            cpu_experts = guidance['cpu_experts']
            should_wait = guidance['should_wait']
            
            # Always log guidance for debugging
            if self.log_path:
                append_log(
                    f'DynamicRouter[{layer_idx}]: guidance - '
                    f'ready_gpu={sorted(ready_gpu_experts)}, '
                    f'pending_gpu={sorted(pending_gpu_experts)}, '
                    f'cpu={sorted(cpu_experts)}, wait={should_wait}',
                    self.log_path
                )
            
            # Wait for pending GPU experts if needed
            if should_wait and pending_gpu_experts:
                io_wait_start = time.time()
                ready_gpu_experts, still_pending, cancelled = self._wait_for_pending_experts(
                    layer_idx, pending_gpu_experts, ready_gpu_experts,
                    max_wait_ms=100  # Max 100ms wait
                )
                io_wait_time_ms = (time.time() - io_wait_start) * 1000
                cancelled_count = cancelled
                # Any still pending go to CPU
                cpu_experts = cpu_experts | still_pending
            else:
                # Don't wait, pending experts go to CPU
                cpu_experts = cpu_experts | pending_gpu_experts
            
            # Build masks based on our decision
            gpu_mask, cpu_mask, expert_to_slot = self._build_execution_masks(
                topk_ids, ready_gpu_experts, cpu_experts, layer_idx, location_map
            )
        else:
            # No scheduler, use location map directly
            gpu_mask, cpu_mask, expert_to_slot = location_map.partition_expert_ids(
                layer_idx, topk_ids
            )
        
        # Check if any experts are on GPU
        has_gpu_experts = gpu_mask.any().item() if gpu_mask is not None else False
        has_cpu_experts = cpu_mask.any().item() if cpu_mask is not None else True
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # ========== 计算阶段开始 ==========
        compute_start_time = time.time()
        
        # Step 1: Submit CPU computation (non-blocking, starts CPU async execution)
        if has_cpu_experts and wrapper.tp_rank == 0:
            self._submit_cpu_experts(wrapper, layer, dispatch_output, cpu_mask)
        
        # Step 2: Compute GPU experts (in parallel with CPU)
        gpu_start = time.time()
        if has_gpu_experts and gpu_cache is not None and expert_to_slot:
            gpu_output = self._compute_gpu_experts(
                wrapper, layer, dispatch_output,
                gpu_mask, expert_to_slot, layer_idx
            )
            output = output + gpu_output
        gpu_time_ms = (time.time() - gpu_start) * 1000
        
        # Step 3: Sync CPU results (blocks until CPU computation done)
        cpu_sync_start = time.time()
        if has_cpu_experts and wrapper.tp_rank == 0:
            cpu_output = self._sync_cpu_experts(wrapper, x, cpu_mask, topk_weights)
            output = output + cpu_output
        cpu_sync_time_ms = (time.time() - cpu_sync_start) * 1000
        
        # 计算阶段结束
        compute_time_ms = (time.time() - compute_start_time) * 1000
        
        # ========== CPU时间推断 ==========
        # 时间线分析:
        # |------ compute_time ------|
        # |-- GPU time --|-- sync ---|
        # |-------- CPU time --------|  (CPU和GPU并行)

        # 关键洞察: sync_time 反映了 GPU 等待 CPU 的时间
        # 如果 sync_time 很小，说明 CPU 在 GPU 期间已完成
        # 如果 sync_time 较大，说明 CPU 比 GPU 慢
        
        if has_cpu_experts:
            # 额外等待时间 = compute_time - gpu_time
            extra_wait_ms = max(0, compute_time_ms - gpu_time_ms)
            
            if extra_wait_ms > 0.1:  # >0.1ms 认为 CPU 有额外等待
                # CPU 是瓶颈：CPU 总时间 ≈ 整个 compute 时间
                # 因为 GPU 在 0~gpu_time 执行，CPU 在 0~compute_time 执行
                cpu_time_ms = compute_time_ms
                bottleneck = "cpu"
            else:
                # GPU 是瓶颈：CPU 在 GPU 时间内完成
                # CPU 实际时间 ≤ gpu_time，但我们无法精确测量
                # 用 sync_time 作为下界估计
                cpu_time_ms = cpu_sync_time_ms
                bottleneck = "gpu"
        else:
            cpu_time_ms = 0.0
            bottleneck = "gpu" if has_gpu_experts else "none"
        
        # Mark layer completed
        if scheduler is not None:
            scheduler.mark_layer_completed(layer_idx)
            scheduler.cleanup_old_plans()
        
        # ========== 检查predict是否完成 (不等待，只检查) ==========
        predict_time_ms = 0.0
        predict_status = "none"
        if predict_future is not None:
            if predict_future.done():
                predict_time_ms = predict_future.result() or 0.0
                predict_status = "done"
            else:
                # predict还在后台执行，不阻塞
                predict_status = "running"
        
        # Update and log statistics
        num_gpu = gpu_mask.sum().item() if gpu_mask is not None else 0
        num_cpu = cpu_mask.sum().item() if cpu_mask is not None else 0
        
        with self._lock:
            self._call_count += 1
            self._stats['total_calls'] += 1
            self._stats['total_gpu_time_ms'] += gpu_time_ms
            self._stats['total_cpu_time_ms'] += cpu_time_ms
            self._stats['total_io_wait_time_ms'] += io_wait_time_ms
            self._stats['total_predict_time_ms'] += predict_time_ms
            self._stats['gpu_expert_count'] += num_gpu
            self._stats['cpu_expert_count'] += num_cpu
            self._stats['cancelled_experts'] += cancelled_count
        
        # Log timing for every layer
        total_layer_time = (time.time() - layer_start_time) * 1000
        
        # 构建详细的timing日志
        timing_parts = [
            f'total={total_layer_time:.2f}ms',
            f'compute={compute_time_ms:.2f}ms',
            f'gpu={gpu_time_ms:.2f}ms',
            f'cpu={cpu_time_ms:.2f}ms[{bottleneck}]',
        ]
        if io_wait_time_ms > 0:
            timing_parts.append(f'io_wait={io_wait_time_ms:.2f}ms')
        timing_parts.append(f'predict={predict_status}')
        if predict_status == "done":
            timing_parts[-1] = f'predict={predict_time_ms:.2f}ms'
        if cancelled_count > 0:
            timing_parts.append(f'cancelled={cancelled_count}')
        
        append_log(
            f'DynamicRouter[{layer_idx}]: GPU={num_gpu} CPU={num_cpu} | '
            f'{", ".join(timing_parts)}',
            self.log_path
        )
        
        return StandardCombineInput(hidden_states=output)
    
    def _safe_predict(
        self, 
        wrapper: "KTEPWrapperMethod", 
        dispatch_output: "StandardDispatchOutput",
        layer_idx: int
    ) -> float:
        """
        安全执行predict_fn，捕获异常并返回执行时间.
        
        Args:
            wrapper: KTEPWrapperMethod instance
            dispatch_output: Dispatch output
            layer_idx: Current layer index (for logging)
            
        Returns:
            Execution time in milliseconds
        """
        start_time = time.time()
        try:
            self._predict_fn(wrapper, dispatch_output)
        except Exception as e:
            if self.log_path:
                append_log(f'DynamicRouter[{layer_idx}]: predict_fn error: {e}', self.log_path)
        return (time.time() - start_time) * 1000
    
    def _native_apply_with_logging(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        layer_idx: int,
        layer_start_time: float,
        predict_future
    ) -> "CombineInput":
        """
        使用原生 ktransformers apply，同时保留日志和统计.
        
        这个路径使用 FusedMoE Triton kernel，比朴素实现快 4-8 倍。
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        x = dispatch_output.hidden_states
        
        # 获取请求的专家
        requested_experts = set(topk_ids.flatten().tolist())
        num_native_gpu = getattr(wrapper, 'num_gpu_experts', 0)
        
        # 计算 GPU 和 CPU 专家数量
        gpu_experts = {e for e in requested_experts if e < num_native_gpu}
        cpu_experts = {e for e in requested_experts if e >= num_native_gpu}
        
        append_log(
            f"DynamicRouter[{layer_idx}]: [NATIVE] requested={sorted(requested_experts)}, "
            f"gpu(<{num_native_gpu})={sorted(gpu_experts)}, cpu={sorted(cpu_experts)}",
            self.log_path
        )
        
        # 计算阶段
        compute_start = time.time()
        
        # 调用原生 apply (使用 FusedMoE kernel)
        # 这里直接调用原始的 _orig_apply，绕过我们的 patch
        from ..hooks import _orig_apply
        result = _orig_apply(wrapper, layer, dispatch_output)
        
        compute_time_ms = (time.time() - compute_start) * 1000
        
        # 检查 predict 状态
        predict_time_ms = 0.0
        predict_status = "none"
        if predict_future is not None:
            if predict_future.done():
                predict_time_ms = predict_future.result() or 0.0
                predict_status = "done"
            else:
                predict_status = "running"
        
        # 更新统计
        num_gpu = len(gpu_experts) * topk_ids.shape[0]  # 估算
        num_cpu = len(cpu_experts) * topk_ids.shape[0]
        
        with self._lock:
            self._call_count += 1
            self._stats['total_calls'] += 1
            self._stats['total_gpu_time_ms'] += compute_time_ms
            self._stats['gpu_expert_count'] += num_gpu
            self._stats['cpu_expert_count'] += num_cpu
            self._stats['total_predict_time_ms'] += predict_time_ms
        
        total_time_ms = (time.time() - layer_start_time) * 1000
        
        append_log(
            f'DynamicRouter[{layer_idx}]: [NATIVE] GPU={len(gpu_experts)} CPU={len(cpu_experts)} | '
            f'total={total_time_ms:.2f}ms, compute={compute_time_ms:.2f}ms, '
            f'predict={predict_status if predict_status != "done" else f"{predict_time_ms:.2f}ms"}',
            self.log_path
        )
        
        return result
    
    def _wait_for_pending_experts(
        self,
        layer_idx: int,
        pending_experts: Set[int],
        ready_experts: Set[int],
        max_wait_ms: int = 100
    ) -> Tuple[Set[int], Set[int], int]:
        """
        Wait for pending GPU experts to finish loading.
        If timeout, cancel remaining pending tasks to free PCIe bandwidth.
        
        Args:
            layer_idx: Current layer
            pending_experts: Experts still being loaded
            ready_experts: Experts already ready
            max_wait_ms: Maximum time to wait in milliseconds
            
        Returns:
            Tuple of (newly_ready_experts, still_pending_experts, cancelled_count)
        """
        from .scheduler import get_scheduler
        from .gpu_cache import get_gpu_cache
        from .expert_migration import get_migration_manager
        
        scheduler = get_scheduler()
        gpu_cache = get_gpu_cache()
        migration_manager = get_migration_manager()
        
        if scheduler is None or gpu_cache is None:
            return ready_experts, pending_experts, 0
        
        start_time = time.time()
        max_wait_sec = max_wait_ms / 1000.0
        check_interval = 0.005  # 5ms
        
        still_pending = pending_experts.copy()
        newly_ready = ready_experts.copy()
        
        while still_pending and (time.time() - start_time) < max_wait_sec:
            # Check which experts are now loaded
            for expert_idx in list(still_pending):
                # Check if expert is in GPU cache
                slot_idx = gpu_cache.get_slot_for_expert(layer_idx, expert_idx)
                if slot_idx is not None:
                    still_pending.remove(expert_idx)
                    newly_ready.add(expert_idx)
                    scheduler.update_expert_loaded(layer_idx, expert_idx)
            
            if still_pending:
                time.sleep(check_interval)
        
        # Timeout reached - cancel remaining pending tasks
        cancelled_count = 0
        if still_pending and migration_manager is not None:
            cancelled_count = migration_manager.cancel_layer_pending_loads(layer_idx, still_pending)
            if self.log_path:
                append_log(
                    f'DynamicRouter[{layer_idx}]: TIMEOUT after {(time.time()-start_time)*1000:.1f}ms, '
                    f'cancelled {cancelled_count} pending loads: {sorted(still_pending)} -> fallback to CPU',
                    self.log_path
                )
        elif self.log_path and still_pending:
            append_log(
                f'DynamicRouter[{layer_idx}]: waited {(time.time()-start_time)*1000:.1f}ms, '
                f'still pending: {sorted(still_pending)} (no migration manager to cancel)',
                self.log_path
            )
        
        return newly_ready, still_pending, cancelled_count
    
    def _build_execution_masks(
        self,
        topk_ids: torch.Tensor,
        gpu_experts: Set[int],
        cpu_experts: Set[int],
        layer_idx: int,
        location_map
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        """
        Build execution masks based on our scheduling decision.
        
        Args:
            topk_ids: Expert IDs selected by routing
            gpu_experts: Experts to run on GPU
            cpu_experts: Experts to run on CPU
            layer_idx: Current layer index
            location_map: Expert location map
            
        Returns:
            Tuple of (gpu_mask, cpu_mask, expert_to_slot)
        """
        # Build GPU mask
        gpu_mask = torch.zeros_like(topk_ids, dtype=torch.bool)
        for expert_idx in gpu_experts:
            gpu_mask |= (topk_ids == expert_idx)
        
        # Build CPU mask
        cpu_mask = torch.zeros_like(topk_ids, dtype=torch.bool)
        for expert_idx in cpu_experts:
            cpu_mask |= (topk_ids == expert_idx)
        
        # Get expert to slot mapping
        expert_to_slot = {}
        if location_map is not None:
            gpu_expert_slots = location_map.get_gpu_experts(layer_idx)
            for expert_idx in gpu_experts:
                if expert_idx in gpu_expert_slots:
                    expert_to_slot[expert_idx] = gpu_expert_slots[expert_idx]
        
        return gpu_mask, cpu_mask, expert_to_slot
    
    def _fallback_cpu_only(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Fallback to original all-CPU behavior."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        
        x = dispatch_output.hidden_states
        
        # Submit to CPU
        if wrapper.tp_rank == 0 and wrapper.wrapper is not None:
            wrapper.submit(layer, dispatch_output)
        
        # Sync CPU results
        output = torch.zeros_like(x)
        if wrapper.tp_rank == 0:
            cpu_output = wrapper.sync(x)
            output = output + cpu_output
        
        return StandardCombineInput(hidden_states=output)
    
    def _submit_cpu_experts(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        cpu_mask: torch.Tensor
    ) -> None:
        """
        Submit CPU expert computation.
        
        For CPU experts, we use the original KTMoEWrapper.
        We mask out GPU experts by setting their IDs to -1 before submitting.
        
        Important: When deferral is enabled (max_deferred_experts_per_token > 0),
        ktransformers' submit_forward will internally split experts into:
        - Immediate experts: executed right away
        - Deferred experts: pipelined with next layer
        
        We don't interfere with this mechanism - just pass the CPU experts
        and let ktransformers handle the deferral internally.
        """
        if wrapper.wrapper is None:
            return
        
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        
        # Create masked topk_ids where GPU experts are marked as -1
        # This tells KTMoEWrapper to skip those experts
        # ktransformers will then apply its own deferral logic to the remaining CPU experts
        masked_topk_ids = topk_ids.clone()
        masked_topk_ids[~cpu_mask] = -1
        
        # Submit to CPU (uses original wrapper.submit_forward logic)
        # Note: If deferral is enabled, this will internally split into immediate/deferred
        x = dispatch_output.hidden_states
        wrapper.wrapper.submit_forward(
            x, 
            masked_topk_ids, 
            topk_weights,
            torch.cuda.current_stream(x.device).cuda_stream
        )
    
    def _sync_cpu_experts(
        self,
        wrapper: "KTEPWrapperMethod",
        x: torch.Tensor,
        cpu_mask: torch.Tensor,
        topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Sync CPU expert results.
        
        When deferral is enabled (max_deferred_experts_per_token > 0):
        - sync_forward will return immediate experts' results
        - Deferred experts' results will be accumulated in the next layer's sync
        - This is handled internally by ktransformers via _layer_has_pending_deferred
        
        The deferral mechanism works across layers like this:
        Layer N: submit immediate + deferred -> sync returns immediate result
        Layer N+1: submit (incremental=True) -> adds deferred N result to immediate N+1
        
        Our code doesn't need to handle this explicitly - ktransformers does it internally.
        """
        if wrapper.wrapper is None:
            return torch.zeros_like(x)
        
        # Get results from CPU
        # When deferral enabled, this may not include deferred experts' contribution yet
        # but ktransformers handles accumulation across layers automatically
        return wrapper.wrapper.sync_forward(
            x,
            torch.cuda.current_stream(x.device).cuda_stream
        )
    
    def _compute_gpu_experts(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        gpu_mask: torch.Tensor,
        expert_to_slot: Dict[int, int],
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute GPU experts using native FusedMoE kernel when possible.
        
        优化策略:
        1. 如果有 gpu_method 且专家在原生 GPU 权重中，使用 FusedMoE kernel (快)
        2. 否则使用我们的 GPU cache + 朴素实现 (慢，但支持动态加载)
        """
        gpu_cache = get_gpu_cache()
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        
        # ========== 尝试使用原生 FusedMoE kernel ==========
        # 如果专家已经在原生 GPU 权重中（由 sglang 加载），使用 FusedMoE
        gpu_method = getattr(wrapper, 'gpu_method', None)
        num_native_gpu_experts = getattr(wrapper, 'num_gpu_experts', 0)
        
        if gpu_method is not None and num_native_gpu_experts > 0:
            # 检查我们的 GPU 专家是否都在原生范围内
            gpu_experts_in_request = set(expert_to_slot.keys())
            native_experts = set(range(num_native_gpu_experts))
            
            # 如果请求的 GPU 专家都在原生范围内，使用 FusedMoE
            if gpu_experts_in_request.issubset(native_experts):
                return self._compute_gpu_experts_native(
                    wrapper, layer, dispatch_output, gpu_mask, expert_to_slot
                )
        
        # ========== 使用我们的 GPU cache (朴素实现) ==========
        # 这个路径较慢，但支持动态加载的专家
        if gpu_cache is None:
            return torch.zeros_like(x)
        
        return self._compute_gpu_experts_naive(
            layer, dispatch_output, gpu_mask, expert_to_slot, layer_idx, gpu_cache
        )
    
    def _compute_gpu_experts_native(
        self,
        wrapper: "KTEPWrapperMethod",
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        gpu_mask: torch.Tensor,
        expert_to_slot: Dict[int, int],
    ) -> torch.Tensor:
        """
        使用原生 FusedMoE kernel 计算 GPU 专家.
        
        这个实现复用 sglang 的高度优化的 Triton kernel，
        比朴素实现快 4-8 倍。
        """
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        
        # 创建 masked topk_ids：只保留我们要在 GPU 上计算的专家
        # 其他专家（CPU专家）标记为 -1
        masked_topk_ids = topk_ids.clone()
        masked_topk_ids[~gpu_mask] = -1
        
        # 创建修改后的 dispatch_output
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )
        
        # 调用原生 FusedMoE kernel
        gpu_combine_input = wrapper.gpu_method.apply(layer, masked_dispatch_output)
        
        return gpu_combine_input.hidden_states
    
    def _compute_gpu_experts_naive(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        gpu_mask: torch.Tensor,
        expert_to_slot: Dict[int, int],
        layer_idx: int,
        gpu_cache
    ) -> torch.Tensor:
        """
        使用朴素 PyTorch 实现计算 GPU 专家.
        
        这个实现较慢，但支持从我们的 GPU cache 加载的动态专家。
        """
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Get activation function
        moe_runner_config = getattr(layer, 'moe_runner_config', None)
        activation = getattr(moe_runner_config, 'activation', 'silu') if moe_runner_config else 'silu'
        
        # Process each GPU expert
        for expert_idx, slot_idx in expert_to_slot.items():
            # Find tokens routed to this expert
            expert_mask = (topk_ids == expert_idx) & gpu_mask
            
            if not expert_mask.any():
                continue
            
            # Get weights from GPU cache
            weights = gpu_cache.get_slot_weights(layer_idx, slot_idx)
            if weights is None:
                continue
            
            w13_weight, w2_weight = weights
            
            # Get indices of tokens routed to this expert
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue
            
            # Safety check
            if token_indices.max().item() >= x.shape[0]:
                continue
            
            # Get input tokens and routing weights
            tokens = x[token_indices]
            routing_weights = topk_weights[token_indices, k_indices]
            
            # Compute expert forward pass
            gate_up = F.linear(tokens, w13_weight)
            intermediate_size = gate_up.shape[-1] // 2
            gate = gate_up[..., :intermediate_size]
            up = gate_up[..., intermediate_size:]
            
            if activation == 'silu':
                gate = F.silu(gate)
            elif activation == 'gelu':
                gate = F.gelu(gate)
            
            hidden = gate * up
            expert_out = F.linear(hidden, w2_weight)
            expert_out = expert_out * routing_weights.unsqueeze(-1)
            
            output.index_add_(0, token_indices, expert_out.to(output.dtype))
        
        return output


# Global router instance
_dynamic_router: Optional[DynamicExpertRouter] = None
_router_lock = threading.Lock()


def get_dynamic_router() -> Optional[DynamicExpertRouter]:
    """Get the global dynamic router instance."""
    return _dynamic_router


def init_dynamic_router(log_path: Optional[str] = None) -> DynamicExpertRouter:
    """Initialize the global dynamic router."""
    global _dynamic_router
    
    with _router_lock:
        if _dynamic_router is None:
            _dynamic_router = DynamicExpertRouter(log_path=log_path)
            log_once('dynamic_router_init', 'DynamicExpertRouter initialized')
        return _dynamic_router


def disable_deferral_for_wrapper(wrapper: "KTEPWrapperMethod") -> None:
    """
    Disable deferral mechanism for a specific wrapper instance at runtime.
    
    This sets max_deferred_experts_per_token to 0 on the underlying KTMoEWrapper,
    which disables the deferral pipeline.
    
    Args:
        wrapper: The KTEPWrapperMethod instance
        
    Note:
        This should be called before any inference starts.
        For global disable, use --kt-max-deferred-experts-per-token 0 at startup.
    """
    if wrapper.wrapper is not None:
        wrapper.wrapper.max_deferred_experts_per_token = 0
        log_once(
            f'disable_deferral_{wrapper.kt_config.layer_idx}',
            f'Disabled deferral for layer {wrapper.kt_config.layer_idx}'
        )


def check_deferral_compatibility(wrapper: "KTEPWrapperMethod") -> Dict[str, Any]:
    """
    Check if deferral mechanism is enabled and provide compatibility info.
    
    Args:
        wrapper: The KTEPWrapperMethod instance
        
    Returns:
        Dictionary with deferral status and recommendations
    """
    kt_config = wrapper.kt_config
    max_deferred = getattr(kt_config, 'max_deferred_experts_per_token', 0) or 0
    
    result = {
        'deferral_enabled': max_deferred > 0,
        'max_deferred_experts_per_token': max_deferred,
        'layer_idx': kt_config.layer_idx,
        'compatible': True,  # Our implementation is now compatible
        'recommendations': []
    }
    
    if max_deferred > 0:
        result['recommendations'].append(
            'Deferral is enabled. CPU expert computation will be pipelined across layers. '
            'This is compatible with dynamic routing, but may make debugging harder.'
        )
        result['recommendations'].append(
            'To disable deferral, use --kt-max-deferred-experts-per-token 0 at startup, '
            'or call disable_deferral_for_wrapper() at runtime.'
        )
    
    return result


def create_patched_apply(original_apply):
    """
    Create a patched apply method that uses dynamic routing.
    
    Args:
        original_apply: The original KTEPWrapperMethod.apply method
        
    Returns:
        Patched apply method
    """
    def patched_apply(self, layer, dispatch_output):
        router = get_dynamic_router()
        
        if router is None:
            # Fall back to original if router not initialized
            return original_apply(self, layer, dispatch_output)
        
        # Use dynamic routing
        return router.dynamic_apply(self, layer, dispatch_output)
    
    return patched_apply
