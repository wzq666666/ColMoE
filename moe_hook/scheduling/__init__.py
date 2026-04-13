"""
Dynamic expert scheduling components.

Naive implementation using F.linear loops.
For high-performance implementation, see `native` package.

- expert_location: Expert location mapping (GPU/CPU)
- gpu_cache: GPU expert cache management
- expert_migration: CPU <-> GPU expert migration
- dynamic_router: Dynamic expert routing
- scheduler: Expert scheduling decisions
- prefetcher: Expert prefetching
"""

from .expert_location import ExpertLocationMap, ExpertLocation, init_location_map, get_location_map
from .gpu_cache import GPUExpertCache, init_gpu_cache, get_gpu_cache
from .expert_migration import ExpertMigrationManager, init_migration_manager, migrate_experts
from .dynamic_router import DynamicExpertRouter, init_dynamic_router, get_dynamic_router
from .scheduler import ExpertScheduler, init_scheduler, get_scheduler, schedule_next_layer
from .prefetcher import ExpertPrefetcher, init_prefetcher, get_prefetcher, prefetch_experts

__all__ = [
    # Location map
    'ExpertLocationMap',
    'ExpertLocation',
    'init_location_map',
    'get_location_map',
    
    # GPU cache
    'GPUExpertCache',
    'init_gpu_cache',
    'get_gpu_cache',
    
    # Migration
    'ExpertMigrationManager',
    'init_migration_manager',
    'migrate_experts',
    
    # Dynamic router
    'DynamicExpertRouter',
    'init_dynamic_router',
    'get_dynamic_router',
    
    # Scheduler
    'ExpertScheduler',
    'init_scheduler',
    'get_scheduler',
    'schedule_next_layer',
    
    # Prefetcher
    'ExpertPrefetcher',
    'init_prefetcher',
    'get_prefetcher',
    'prefetch_experts',
]
