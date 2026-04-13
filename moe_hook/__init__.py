"""
moe_hook package

Modular hooks for integrating preloading, prediction (FATE), and prefetch
into SGLang+KTransformers MoE execution.

Includes dynamic expert scheduling for runtime CPU<->GPU expert migration.
Also includes native GPU cache mode for high-performance FusedMoE kernel acceleration.

Directory structure:
- core/: Model config, gate resolver, expert resolver
- scheduling/: Dynamic scheduling components (naive F.linear implementation)
- native/: Native FusedMoE kernel implementation (high-performance)
- prediction/: Expert prediction (FATE) and phase detection
- legacy/: Old code kept for reference
"""

from .hooks import (
    install_hooks, 
    get_preload_positions, 
    is_preload_done,
    is_dynamic_scheduling_enabled,
    get_migration_manager,
    load_expert_to_gpu,
    unload_expert_from_gpu,
    migrate_to_target,
    get_expert_locations,
    get_gpu_cache_stats,
    # Native GPU cache API
    is_native_gpu_cache_enabled,
    get_native_cache_stats,
    swap_expert_native,
    get_registered_layers,
)

# Core components
from .core import (
    ExpertResolver,
    HFModelConfig, 
    HFConfigResolver, 
    resolve_hf_config,
    GateResolver,
)

# Scheduling components (naive implementation)
from .scheduling import (
    GPUExpertCache, init_gpu_cache, get_gpu_cache,
    ExpertLocationMap, ExpertLocation, init_location_map, get_location_map,
    ExpertMigrationManager, init_migration_manager, migrate_experts,
    DynamicExpertRouter, init_dynamic_router, get_dynamic_router,
    ExpertScheduler, init_scheduler, get_scheduler, schedule_next_layer,
    ExpertPrefetcher, init_prefetcher, get_prefetcher, prefetch_experts,
)

# Native implementation (high-performance)
from .native import (
    NativeGPUCacheManager, init_native_cache, get_native_cache,
    NativeDynamicRouter, init_native_router, get_native_router,
)

# Prediction
from .prediction import (
    predict_experts, predict_and_prefetch, call_preloader,
    infer_phase,
)

__all__ = [
    # Core hooks
    'install_hooks', 
    'get_preload_positions',
    'is_preload_done',
    
    # Dynamic scheduling status
    'is_dynamic_scheduling_enabled',
    'get_migration_manager',
    
    # Expert migration API
    'load_expert_to_gpu',
    'unload_expert_from_gpu',
    'migrate_to_target',
    'migrate_experts',
    
    # Stats/monitoring
    'get_expert_locations',
    'get_gpu_cache_stats',
    
    # Expert resolver
    'ExpertResolver',
    
    # Model config
    'HFModelConfig',
    'HFConfigResolver', 
    'resolve_hf_config',
    
    # GPU cache
    'GPUExpertCache',
    'init_gpu_cache',
    'get_gpu_cache',
    
    # Location map
    'ExpertLocationMap',
    'ExpertLocation',
    'init_location_map',
    'get_location_map',
    
    # Migration manager
    'ExpertMigrationManager',
    'init_migration_manager',
    
    # Dynamic router
    'DynamicExpertRouter',
    'init_dynamic_router',
    'get_dynamic_router',
    
    # Native GPU cache (high-performance)
    'is_native_gpu_cache_enabled',
    'get_native_cache_stats',
    'swap_expert_native',
    'get_registered_layers',
    'NativeGPUCacheManager',
    'init_native_cache',
    'get_native_cache',
    'NativeDynamicRouter',
    'init_native_router',
    'get_native_router',
    
    # Prediction (separate from prefetch)
    'predict_experts',
    'predict_and_prefetch',
    'call_preloader',
    
    # Prefetcher
    'ExpertPrefetcher',
    'init_prefetcher',
    'get_prefetcher',
    'prefetch_experts',
    
    # Scheduler
    'ExpertScheduler',
    'init_scheduler',
    'get_scheduler',
    'schedule_next_layer',
]
