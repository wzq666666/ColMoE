"""
Expert Migration API for dynamic expert scheduling.

Provides high-level APIs for migrating experts between CPU and GPU:
- load_expert_to_gpu: Load expert from HF weights to GPU cache
- unload_expert_from_gpu: Remove expert from GPU cache
- batch_migrate_experts: Batch migration operations
- Async migration support for non-blocking operations
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
import threading
import time
import torch

from ..logger import log_once, append_log
from .gpu_cache import get_gpu_cache, GPUExpertCache
from .expert_location import get_location_map, ExpertLocationMap
from ..core.expert_resolver import ExpertResolver


class MigrationStatus(Enum):
    """Status of a migration task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MigrationTask:
    """A single migration task."""
    task_id: int
    layer_idx: int
    expert_idx: int
    action: str  # "load" or "unload"
    status: MigrationStatus = MigrationStatus.PENDING
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None
    cancelled: bool = False  # Flag to cancel the task
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class AsyncMigrationWorker:
    """
    Background worker for async expert migration.
    
    Processes migration tasks in a separate thread to avoid
    blocking inference operations.
    """
    
    def __init__(
        self,
        migration_manager: "ExpertMigrationManager",
        log_path: Optional[str] = None
    ):
        self.migration_manager = migration_manager
        self.log_path = log_path
        
        self._task_queue: Queue[MigrationTask] = Queue()
        self._task_history: Dict[int, MigrationTask] = {}
        self._pending_tasks: Dict[Tuple[int, int], MigrationTask] = {}  # (layer, expert) -> task
        self._task_counter = 0
        self._lock = threading.Lock()
        
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
    
    def start(self):
        """Start the background worker thread."""
        if self._running:
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ExpertMigrationWorker"
        )
        self._worker_thread.start()
        self._running = True
        
        log_once('async_worker_started', 'AsyncMigrationWorker started')
    
    def stop(self, timeout: float = 5.0):
        """Stop the background worker thread."""
        if not self._running:
            return
        
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        self._running = False
        
        log_once('async_worker_stopped', 'AsyncMigrationWorker stopped')
    
    def _worker_loop(self):
        """Main worker loop - processes tasks from queue."""
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.1)
            except Empty:
                continue
            
            self._process_task(task)
    
    def _process_task(self, task: MigrationTask):
        """Process a single migration task."""
        # Check if task was cancelled before starting
        if task.cancelled:
            task.status = MigrationStatus.FAILED
            task.error = "Cancelled before execution"
            task.completed_at = time.time()
            with self._lock:
                self._task_history[task.task_id] = task
                key = (task.layer_idx, task.expert_idx)
                if key in self._pending_tasks:
                    del self._pending_tasks[key]
            if self.log_path:
                append_log(
                    f'AsyncMigration: {task.action} expert[{task.layer_idx}][{task.expert_idx}] '
                    f'CANCELLED before execution',
                    self.log_path
                )
            return
        
        task.status = MigrationStatus.IN_PROGRESS
        
        try:
            if task.action == "load":
                # Pass task to allow cancellation check during load
                success = self.migration_manager._sync_load_expert_with_cancel_check(
                    task.layer_idx, task.expert_idx, task
                )
                if task.cancelled:
                    task.status = MigrationStatus.FAILED
                    task.error = "Cancelled during execution"
                else:
                    task.status = MigrationStatus.COMPLETED if success else MigrationStatus.FAILED
                    if not success:
                        task.error = "Migration failed"
            elif task.action == "unload":
                success = self.migration_manager._sync_unload_expert(
                    task.layer_idx, task.expert_idx
                )
                task.status = MigrationStatus.COMPLETED if success else MigrationStatus.FAILED
                if not success:
                    task.error = "Migration failed"
            else:
                task.status = MigrationStatus.FAILED
                task.error = f"Unknown action: {task.action}"
                
        except Exception as e:
            task.status = MigrationStatus.FAILED
            task.error = str(e)
            if self.log_path:
                append_log(f'AsyncMigration error: {e}', self.log_path)
        
        task.completed_at = time.time()
        
        with self._lock:
            self._task_history[task.task_id] = task
            key = (task.layer_idx, task.expert_idx)
            if key in self._pending_tasks:
                del self._pending_tasks[key]
        
        if self.log_path:
            duration = task.completed_at - task.created_at
            status_str = task.status.value
            if task.cancelled:
                status_str = "cancelled"
            append_log(
                f'AsyncMigration: {task.action} expert[{task.layer_idx}][{task.expert_idx}] '
                f'status={status_str} duration={duration:.3f}s',
                self.log_path
            )
    
    def submit_load(self, layer_idx: int, expert_idx: int) -> int:
        """Submit an async load task. Returns task ID."""
        with self._lock:
            self._task_counter += 1
            task_id = self._task_counter
            
            task = MigrationTask(
                task_id=task_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                action="load"
            )
            
            # Track pending task by (layer, expert)
            key = (layer_idx, expert_idx)
            self._pending_tasks[key] = task
        
        self._task_queue.put(task)
        return task_id
    
    def submit_unload(self, layer_idx: int, expert_idx: int) -> int:
        """Submit an async unload task. Returns task ID."""
        with self._lock:
            self._task_counter += 1
            task_id = self._task_counter
        
        task = MigrationTask(
            task_id=task_id,
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            action="unload"
        )
        
        self._task_queue.put(task)
        return task_id
    
    def submit_batch(
        self, 
        loads: List[Tuple[int, int]], 
        unloads: List[Tuple[int, int]]
    ) -> List[int]:
        """Submit batch of load and unload tasks. Returns list of task IDs."""
        task_ids = []
        
        # Unloads first to free up slots
        for layer_idx, expert_idx in unloads:
            task_ids.append(self.submit_unload(layer_idx, expert_idx))
        
        # Then loads
        for layer_idx, expert_idx in loads:
            task_ids.append(self.submit_load(layer_idx, expert_idx))
        
        return task_ids
    
    def get_task_status(self, task_id: int) -> Optional[MigrationTask]:
        """Get status of a task by ID."""
        with self._lock:
            return self._task_history.get(task_id)
    
    def wait_for_tasks(
        self, 
        task_ids: List[int], 
        timeout: float = 30.0
    ) -> Dict[int, MigrationTask]:
        """Wait for multiple tasks to complete."""
        results = {}
        deadline = time.time() + timeout
        
        for task_id in task_ids:
            while time.time() < deadline:
                task = self.get_task_status(task_id)
                if task and task.status in (MigrationStatus.COMPLETED, MigrationStatus.FAILED):
                    results[task_id] = task
                    break
                time.sleep(0.01)
            else:
                # Timeout - check one more time
                task = self.get_task_status(task_id)
                if task:
                    results[task_id] = task
        
        return results
    
    def get_queue_size(self) -> int:
        """Get number of pending tasks."""
        return self._task_queue.qsize()
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running
    
    def cancel_pending_task(self, layer_idx: int, expert_idx: int) -> bool:
        """
        Cancel a pending load task for a specific expert.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            True if task was found and cancelled, False otherwise
        """
        with self._lock:
            key = (layer_idx, expert_idx)
            if key in self._pending_tasks:
                task = self._pending_tasks[key]
                task.cancelled = True
                if self.log_path:
                    append_log(
                        f'AsyncMigration: marking expert[{layer_idx}][{expert_idx}] for cancellation',
                        self.log_path
                    )
                return True
        return False
    
    def cancel_layer_pending_tasks(self, layer_idx: int, expert_indices: Set[int]) -> int:
        """
        Cancel multiple pending load tasks for a layer.
        
        Args:
            layer_idx: Layer index
            expert_indices: Set of expert indices to cancel
            
        Returns:
            Number of tasks cancelled
        """
        cancelled = 0
        with self._lock:
            for expert_idx in expert_indices:
                key = (layer_idx, expert_idx)
                if key in self._pending_tasks:
                    task = self._pending_tasks[key]
                    task.cancelled = True
                    cancelled += 1
        
        if cancelled > 0 and self.log_path:
            append_log(
                f'AsyncMigration: cancelled {cancelled} pending tasks for layer {layer_idx}',
                self.log_path
            )
        return cancelled


class ExpertMigrationManager:
    """
    Manages expert migration between CPU and GPU.
    
    Supports both sync and async migration modes.
    """
    
    def __init__(
        self,
        expert_resolver: ExpertResolver,
        log_path: Optional[str] = None,
        enable_async: bool = True
    ):
        self.expert_resolver = expert_resolver
        self.log_path = log_path
        self._lock = threading.Lock()
        self._migration_count = 0
        
        # Async worker
        self._async_worker: Optional[AsyncMigrationWorker] = None
        if enable_async:
            self._async_worker = AsyncMigrationWorker(self, log_path)
            self._async_worker.start()
        
        log_once('migration_manager_init', 
                 f'ExpertMigrationManager initialized (async={enable_async})')
    
    def _sync_load_expert(self, layer_idx: int, expert_idx: int, device: str = "cuda") -> bool:
        """Synchronous expert load (internal)."""
        return self._sync_load_expert_with_cancel_check(layer_idx, expert_idx, task=None)
    
    def _sync_load_expert_with_cancel_check(
        self, 
        layer_idx: int, 
        expert_idx: int, 
        task: Optional[MigrationTask] = None,
        device: str = "cuda"
    ) -> bool:
        """
        Synchronous expert load with cancellation support.
        
        Checks the task's cancelled flag at key points to abort early
        if the task was cancelled (e.g., due to timeout).
        """
        gpu_cache = get_gpu_cache()
        location_map = get_location_map()
        
        if gpu_cache is None or location_map is None:
            return False
        
        # Check cancellation before starting
        if task is not None and task.cancelled:
            if self.log_path:
                append_log(
                    f'Migration: expert[{layer_idx}][{expert_idx}] cancelled before start',
                    self.log_path
                )
            return False
        
        with self._lock:
            # Check if already on GPU
            current_info = location_map.get_location(layer_idx, expert_idx)
            if current_info.location.value == "gpu":
                return True
            
            # Check cancellation before heavy IO (loading weights from disk)
            if task is not None and task.cancelled:
                if self.log_path:
                    append_log(
                        f'Migration: expert[{layer_idx}][{expert_idx}] cancelled before weight load',
                        self.log_path
                    )
                return False
            
            # Load weights from HF checkpoint
            # 优化路径:
            # 1. 如果CPU缓存命中: CPU内存 → GPU (~1-2ms)
            # 2. 如果CPU缓存未命中: 磁盘 → CPU缓存 → GPU (~50-100ms)
            weights = self.expert_resolver.load_expert_weights_from_hf(
                layer_idx, expert_idx, 
                device="cpu",      # 先加载到CPU (会被缓存)
                use_cache=True,    # 使用CPU缓存
                cache_on_cpu=True  # 缓存到CPU内存
            )
            
            if weights is None:
                if self.log_path:
                    append_log(f'Migration: failed to load weights for expert[{layer_idx}][{expert_idx}]', self.log_path)
                return False
            
            # Check cancellation after weight load, before GPU transfer
            if task is not None and task.cancelled:
                if self.log_path:
                    append_log(
                        f'Migration: expert[{layer_idx}][{expert_idx}] cancelled after weight load, '
                        f'skipping GPU transfer to save PCIe bandwidth',
                        self.log_path
                    )
                # Don't transfer to GPU - the expert will be computed on CPU instead
                return False
            
            # Load to GPU cache (this transfers data over PCIe)
            slot_idx = gpu_cache.load_expert(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                w1_weight=weights['w1'],
                w2_weight=weights['w2'],
                w3_weight=weights['w3']
            )
            
            if slot_idx is None:
                if self.log_path:
                    append_log(f'Migration: no GPU slot for expert[{layer_idx}][{expert_idx}]', self.log_path)
                return False
            
            location_map.set_gpu_location(layer_idx, expert_idx, slot_idx)
            self._migration_count += 1
            
            if self.log_path:
                append_log(f'Migration: loaded expert[{layer_idx}][{expert_idx}] to slot {slot_idx}', self.log_path)
            
            return True
    
    def _sync_unload_expert(self, layer_idx: int, expert_idx: int) -> bool:
        """Synchronous expert unload (internal)."""
        gpu_cache = get_gpu_cache()
        location_map = get_location_map()
        
        if gpu_cache is None or location_map is None:
            return False
        
        with self._lock:
            current_info = location_map.get_location(layer_idx, expert_idx)
            if current_info.location.value != "gpu":
                return True  # Already on CPU
            
            success = gpu_cache.unload_expert(layer_idx, expert_idx)
            if not success:
                return False
            
            location_map.set_cpu_location(layer_idx, expert_idx)
            
            if self.log_path:
                append_log(f'Migration: unloaded expert[{layer_idx}][{expert_idx}]', self.log_path)
            
            return True
    
    # ========== Sync API ==========
    
    def load_expert_to_gpu(self, layer_idx: int, expert_idx: int, device: str = "cuda") -> bool:
        """Load expert to GPU (sync)."""
        return self._sync_load_expert(layer_idx, expert_idx, device)
    
    def unload_expert_from_gpu(self, layer_idx: int, expert_idx: int) -> bool:
        """Unload expert from GPU (sync)."""
        return self._sync_unload_expert(layer_idx, expert_idx)
    
    # ========== Async API ==========
    
    def async_load_expert(self, layer_idx: int, expert_idx: int) -> int:
        """Load expert (async). Returns task ID."""
        if self._async_worker is None:
            raise RuntimeError("Async worker not enabled")
        return self._async_worker.submit_load(layer_idx, expert_idx)
    
    def async_unload_expert(self, layer_idx: int, expert_idx: int) -> int:
        """Unload expert (async). Returns task ID."""
        if self._async_worker is None:
            raise RuntimeError("Async worker not enabled")
        return self._async_worker.submit_unload(layer_idx, expert_idx)
    
    def async_migrate_batch(self, loads: List[Tuple[int, int]], unloads: List[Tuple[int, int]]) -> List[int]:
        """Batch migration (async). Returns task IDs."""
        if self._async_worker is None:
            raise RuntimeError("Async worker not enabled")
        return self._async_worker.submit_batch(loads, unloads)
    
    def async_migrate_to_target(self, target_gpu_experts: Dict[int, Set[int]]) -> List[int]:
        """Migrate to target (async). Returns task IDs."""
        location_map = get_location_map()
        if location_map is None:
            return []
        
        loads, unloads = [], []
        
        for layer_idx in range(location_map.num_layers):
            target_set = target_gpu_experts.get(layer_idx, set())
            current_gpu = set(location_map.get_gpu_experts(layer_idx).keys())
            
            for expert_idx in (current_gpu - target_set):
                unloads.append((layer_idx, expert_idx))
            for expert_idx in (target_set - current_gpu):
                loads.append((layer_idx, expert_idx))
        
        return self.async_migrate_batch(loads, unloads)
    
    def wait_for_migration(self, task_ids: List[int], timeout: float = 30.0) -> Dict[int, MigrationTask]:
        """Wait for async tasks."""
        if self._async_worker is None:
            return {}
        return self._async_worker.wait_for_tasks(task_ids, timeout)
    
    def get_async_queue_size(self) -> int:
        """Get pending async tasks count."""
        return self._async_worker.get_queue_size() if self._async_worker else 0
    
    # ========== Cancel API ==========
    
    def cancel_pending_load(self, layer_idx: int, expert_idx: int) -> bool:
        """Cancel a pending async load task."""
        if self._async_worker is None:
            return False
        return self._async_worker.cancel_pending_task(layer_idx, expert_idx)
    
    def cancel_layer_pending_loads(self, layer_idx: int, expert_indices: Set[int]) -> int:
        """Cancel multiple pending async load tasks for a layer."""
        if self._async_worker is None:
            return 0
        return self._async_worker.cancel_layer_pending_tasks(layer_idx, expert_indices)
    
    # ========== Batch Sync API ==========
    
    def migrate_to_target_distribution(self, target_gpu_experts: Dict[int, Set[int]], device: str = "cuda") -> bool:
        """Migrate to target (sync)."""
        location_map = get_location_map()
        if location_map is None:
            return False
        
        all_success = True
        
        for layer_idx in range(location_map.num_layers):
            target_set = target_gpu_experts.get(layer_idx, set())
            current_gpu = set(location_map.get_gpu_experts(layer_idx).keys())
            
            for expert_idx in (current_gpu - target_set):
                if not self.unload_expert_from_gpu(layer_idx, expert_idx):
                    all_success = False
            
            for expert_idx in (target_set - current_gpu):
                if not self.load_expert_to_gpu(layer_idx, expert_idx, device):
                    all_success = False
        
        return all_success
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        location_map = get_location_map()
        gpu_cache = get_gpu_cache()
        
        return {
            'total_migrations': self._migration_count,
            'async_queue_size': self.get_async_queue_size(),
            'location_map_stats': location_map.get_stats() if location_map else None,
            'gpu_cache_stats': gpu_cache.get_cache_stats() if gpu_cache else None,
        }
    
    def shutdown(self):
        """Shutdown async worker."""
        if self._async_worker:
            self._async_worker.stop()


# Global instance
_migration_manager: Optional[ExpertMigrationManager] = None
_migration_manager_lock = threading.Lock()


def get_migration_manager() -> Optional[ExpertMigrationManager]:
    return _migration_manager


def init_migration_manager(expert_resolver: ExpertResolver, log_path: Optional[str] = None, enable_async: bool = True) -> ExpertMigrationManager:
    global _migration_manager
    with _migration_manager_lock:
        if _migration_manager is None:
            _migration_manager = ExpertMigrationManager(expert_resolver, log_path, enable_async)
        return _migration_manager


# ========== Convenience Functions ==========

def load_expert_to_gpu(layer_idx: int, expert_idx: int, device: str = "cuda") -> bool:
    manager = get_migration_manager()
    return manager.load_expert_to_gpu(layer_idx, expert_idx, device) if manager else False


def unload_expert_from_gpu(layer_idx: int, expert_idx: int) -> bool:
    manager = get_migration_manager()
    return manager.unload_expert_from_gpu(layer_idx, expert_idx) if manager else False


def async_load_expert(layer_idx: int, expert_idx: int) -> int:
    manager = get_migration_manager()
    if manager is None:
        raise RuntimeError('Migration manager not initialized')
    return manager.async_load_expert(layer_idx, expert_idx)


def async_unload_expert(layer_idx: int, expert_idx: int) -> int:
    manager = get_migration_manager()
    if manager is None:
        raise RuntimeError('Migration manager not initialized')
    return manager.async_unload_expert(layer_idx, expert_idx)


def async_migrate_to_target(target_gpu_experts: Dict[int, Set[int]]) -> List[int]:
    manager = get_migration_manager()
    if manager is None:
        raise RuntimeError('Migration manager not initialized')
    return manager.async_migrate_to_target(target_gpu_experts)


def migrate_experts(target_gpu_experts: Dict[int, Set[int]], device: str = "cuda") -> bool:
    manager = get_migration_manager()
    return manager.migrate_to_target_distribution(target_gpu_experts, device) if manager else False
