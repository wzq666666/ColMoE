"""
Native FusedMoE kernel implementation for dynamic expert scheduling.

This is the high-performance backend that directly manipulates sglang's
FusedMoE weight tensors for 4-8x faster inference compared to naive F.linear.

- native_gpu_cache: Direct manipulation of sglang weight tensors
- native_router: Dynamic routing using FusedMoE Triton kernel
- native_migration: Expert weight loading and format conversion (HF → sglang)
- native_scheduler: Adapter for generic ExpertScheduler
"""

from .native_gpu_cache import NativeGPUCacheManager, init_native_cache, get_native_cache
from .native_router import (
    NativeDynamicRouter,
    init_native_router,
    get_native_router,
)
from .native_migration import (
    NativeExpertMigrationManager,
    init_native_migration_manager,
    get_native_migration_manager,
    convert_hf_to_sglang_format,
    convert_hf_to_sglang_format_contiguous,
)
from .native_scheduler import (
    NativeGPUStateProvider,
    init_native_scheduler,
    get_native_scheduler,
)

__all__ = [
    # GPU cache
    'NativeGPUCacheManager',
    'init_native_cache',
    'get_native_cache',
    # Router
    'NativeDynamicRouter',
    'init_native_router',
    'get_native_router',
    # Scheduler adapter
    'NativeGPUStateProvider',
    'init_native_scheduler',
    'get_native_scheduler',
    # Migration
    'NativeExpertMigrationManager',
    'init_native_migration_manager',
    'get_native_migration_manager',
    'convert_hf_to_sglang_format',
    'convert_hf_to_sglang_format_contiguous',
]
