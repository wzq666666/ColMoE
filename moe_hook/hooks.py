"""
Hook installation for KTEPWrapperMethod patches.

This module provides:
1. Original prediction/prefetch hooks
2. Dynamic expert scheduling hooks (GPU cache + location map)
3. Native GPU cache integration for FusedMoE kernel acceleration
"""

import os
import threading
from typing import Any, Dict, Optional, Set
import torch

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

from .config import load_config
from .logger import log_once, reset_log, append_log
from .core.gate_resolver import GateResolver
from .prediction.predictor import call_preloader, predict_and_prefetch, predict_experts


_cfg: Dict[str, Any] = {}
_gate_resolver: Optional[GateResolver] = None
_inited = False
_init_lock = threading.Lock()
_preload_done = False  # 标记预加载是否已完成
_preload_positions: Optional[list] = None  # 缓存预加载位置

# Dynamic scheduling components
_dynamic_scheduling_enabled = False
_expert_resolver = None
_migration_manager = None

# Native GPU cache mode
_native_gpu_cache_enabled = False
_registered_layers: Dict[int, Any] = {}  # layer_idx -> (wrapper, layer)
_native_migration_manager = None  # Native backend migration manager


def _ensure_init():
    """Initialize configuration and gate resolver."""
    global _inited, _cfg, _gate_resolver, _dynamic_scheduling_enabled
    global _expert_resolver, _migration_manager
    
    if _inited:
        return
    
    with _init_lock:
        if _inited:
            return
        
        _cfg = load_config()
        _gate_resolver = GateResolver(_cfg, _cfg.get('log_path'))
        
        if not _cfg.get('enable', True):
            log_once('disabled', 'MOE hooks disabled by config/env')
        else:
            log_once('enabled', 'MOE hooks enabled')

        # Initialize log file and log level
        log_path = _cfg.get('log_path')
        if log_path:
            from .logger import set_log_level
            log_level = _cfg.get('log_level', 2)  # 默认为 2 (重要事件)
            set_log_level(log_level)
            reset_log(log_path)

        # Optionally set capture batch sizes for KT CPU buffers
        try:
            from kt_kernel import KTMoEWrapper  # type: ignore
            log_once('import_ktmoewrapper', 'KTMoEWrapper loaded from kt_kernel')
        except Exception as e_primary:
            try:
                from ktransformers.kt_kernel.python.experts import KTMoEWrapper  # type: ignore
                log_once('import_ktmoewrapper_fallback', 'KTMoEWrapper loaded from ktransformers.kt_kernel.python')
            except Exception as e_fallback:
                KTMoEWrapper = None  # type: ignore
                log_once('import_ktmoewrapper_err', f"Import KTMoEWrapper failed: primary={e_primary}; fallback={e_fallback}")
        
        if KTMoEWrapper is not None:
            try:
                bs = _cfg.get('capture_bs') or []
                if isinstance(bs, list) and all(isinstance(x, int) for x in bs):
                    KTMoEWrapper.set_capture_batch_sizes(bs)
                    log_once('capture_bs', f"Set KT capture batch sizes: {bs}")
            except Exception as e:
                log_once('capture_bs_err', f"Failed to set capture batch sizes: {e}")

        # Initialize dynamic scheduling if enabled
        _dynamic_scheduling_enabled = _cfg.get('dynamic_scheduling', False)
        if _dynamic_scheduling_enabled:
            # Check which backend to use
            scheduling_backend = _cfg.get('scheduling_backend', 'native')
            
            if scheduling_backend == 'native':
                # Native: use FusedMoE kernel (recommended, 4-8x faster)
                _init_native_gpu_cache()
                log_once('scheduling_backend', 'Dynamic scheduling using native FusedMoE backend')
            else:
                # Naive: use F.linear loops (slower but simpler)
                _init_dynamic_scheduling()
                log_once('scheduling_backend', 'Dynamic scheduling using naive F.linear backend')

        _inited = True


def _initialize_gpu_cache_after_model_load():
    """
    Initialize GPU cache memory allocation after the model has been loaded.
    This defers GPU memory allocation until the model is fully loaded,
    avoiding memory conflicts during model loading.
    """
    log_path = _cfg.get('log_path')
    
    try:
        from .scheduling.gpu_cache import get_gpu_cache
        gpu_cache = get_gpu_cache()
        
        if gpu_cache is not None and not gpu_cache.is_initialized:
            if log_path:
                append_log('Initializing GPU cache after model load...', log_path)
            
            gpu_cache.initialize()
            
            if log_path:
                append_log('GPU cache initialized successfully', log_path)
            
            log_once('gpu_cache_post_init', 'GPU cache initialized after model load')
        
    except Exception as e:
        if log_path:
            append_log(f'GPU cache initialization failed: {e}', log_path)
        log_once('gpu_cache_post_init_err', f'GPU cache post-init failed: {e}')
        import traceback
        traceback.print_exc()


def _sync_prefetch_preload_experts():
    """
    Synchronously load preloader results to GPU cache.
    
    This loads the experts identified by the preloader to GPU cache
    using synchronous migration (blocking until all experts are loaded).
    
    Optimized version: batch load all experts per layer to reduce file IO.
    """
    global _preload_positions, _migration_manager
    
    log_path = _cfg.get('log_path')
    
    if _preload_positions is None or len(_preload_positions) == 0:
        if log_path:
            append_log('Sync prefetch: no preload positions available', log_path)
        return
    
    if _migration_manager is None:
        if log_path:
            append_log('Sync prefetch: migration manager not initialized', log_path)
        return
    
    try:
        from .scheduling.gpu_cache import get_gpu_cache
        gpu_cache = get_gpu_cache()
        
        if gpu_cache is None or not gpu_cache.is_initialized:
            if log_path:
                append_log('Sync prefetch: GPU cache not ready', log_path)
            return
        
        max_gpu_experts = gpu_cache.max_gpu_experts
        
        if log_path:
            append_log(
                f'Sync prefetch: loading {len(_preload_positions)} experts '
                f'(max {max_gpu_experts} per layer)',
                log_path
            )
        
        # Group preload positions by layer
        layer_experts: Dict[int, list] = {}
        for pos in _preload_positions:
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                layer_idx, expert_idx = pos[0], pos[1]
                if layer_idx not in layer_experts:
                    layer_experts[layer_idx] = []
                layer_experts[layer_idx].append(expert_idx)
        
        # Load experts for each layer (up to max_gpu_experts per layer)
        total_loaded = 0
        total_failed = 0
        
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        start_time = time.time()
        
        # Use thread pool for parallel loading across layers
        # This significantly speeds up loading by parallelizing file IO
        def load_layer_experts(layer_idx: int, experts: list):
            """Load experts for a single layer."""
            loaded = 0
            failed = 0
            for expert_idx in experts[:max_gpu_experts]:
                try:
                    success = _migration_manager.load_expert_to_gpu(layer_idx, expert_idx)
                    if success:
                        loaded += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    if log_path:
                        append_log(
                            f'Sync prefetch: failed to load expert[{layer_idx}][{expert_idx}]: {e}',
                            log_path
                        )
            return loaded, failed
        
        # Parallel loading with ThreadPoolExecutor
        # Each layer can be loaded in parallel (different files)
        num_workers = min(8, len(layer_experts))  # Limit concurrent file operations
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for layer_idx in sorted(layer_experts.keys()):
                experts = layer_experts[layer_idx]
                future = executor.submit(load_layer_experts, layer_idx, experts)
                futures[future] = layer_idx
            
            for future in as_completed(futures):
                layer_idx = futures[future]
                try:
                    loaded, failed = future.result()
                    total_loaded += loaded
                    total_failed += failed
                except Exception as e:
                    if log_path:
                        append_log(f'Sync prefetch: layer {layer_idx} failed: {e}', log_path)
        
        elapsed = time.time() - start_time
        
        if log_path:
            append_log(
                f'Sync prefetch completed: loaded={total_loaded}, failed={total_failed}, '
                f'time={elapsed:.2f}s',
                log_path
            )
        
        log_once(
            'sync_prefetch_done',
            f'Sync prefetch: loaded {total_loaded} experts in {elapsed:.2f}s'
        )
        
    except Exception as e:
        if log_path:
            append_log(f'Sync prefetch failed: {e}', log_path)
        log_once('sync_prefetch_err', f'Sync prefetch error: {e}')
        import traceback
        traceback.print_exc()


def _sync_prefetch_preload_experts_native():
    """
    Synchronously preload experts to native GPU cache.
    
    For native backend:
    1. sglang 默认加载 experts 0 ~ num_gpu_experts-1
    2. 如果 preloader 请求不同的专家，我们需要进行替换
    3. 使用 NativeExpertMigrationManager 加载权重（已优化为 pinned memory）
    
    优化：
    - 使用 ExpertResolver 的 pinned memory pool
    - 直接传输，无需格式转换
    - 享受 DMA 加速 (11+ GB/s)
    """
    global _preload_positions, _native_migration_manager
    
    log_path = _cfg.get('log_path')
    
    if _preload_positions is None or len(_preload_positions) == 0:
        if log_path:
            append_log('Native prefetch: no preload positions, using default experts', log_path)
        return
    
    try:
        from .native.native_gpu_cache import get_native_cache
        from .native.native_migration import get_native_migration_manager
        
        native_cache = get_native_cache()
        native_migration = get_native_migration_manager()
        
        if native_cache is None:
            if log_path:
                append_log('Native prefetch: cache not initialized', log_path)
            return
        
        # Group by layer
        layer_experts: Dict[int, list] = {}
        for pos in _preload_positions:
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                layer_idx, expert_idx = pos[0], pos[1]
                if layer_idx not in layer_experts:
                    layer_experts[layer_idx] = []
                if expert_idx not in layer_experts[layer_idx]:
                    layer_experts[layer_idx].append(expert_idx)
        
        num_slots = native_cache.num_gpu_slots
        
        if log_path:
            total_experts = sum(len(exps) for exps in layer_experts.values())
            append_log(
                f'Native prefetch: checking {total_experts} experts across '
                f'{len(layer_experts)} layers (max {num_slots} per layer)',
                log_path
            )
        
        # 检查哪些层需要加载不同的专家
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        total_loaded = 0
        total_skipped = 0
        total_failed = 0
        
        def preload_layer(layer_idx: int, experts: list):
            """为单层预加载专家."""
            loaded = 0
            skipped = 0
            failed = 0
            
            # 获取当前 GPU 上的专家
            current_gpu = native_cache.get_gpu_experts(layer_idx)
            default_experts = current_gpu if current_gpu else set(range(num_slots))
            
            # 确定要加载的专家 (最多 num_slots 个)
            wanted_experts = experts[:num_slots]
            
            for slot_idx, expert_idx in enumerate(wanted_experts):
                # 如果这个专家已经在正确的槽位，跳过
                current_slot = native_cache.get_slot_for_expert(layer_idx, expert_idx)
                if current_slot is not None:
                    skipped += 1
                    continue
                
                # 需要加载这个专家到 slot_idx
                if native_migration is not None:
                    try:
                        success = native_migration.load_expert_to_slot(
                            layer_idx=layer_idx,
                            expert_idx=expert_idx,
                            slot_idx=slot_idx,
                        )
                        if success:
                            loaded += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        if log_path:
                            append_log(
                                f'Native prefetch: failed to load '
                                f'expert[{layer_idx}][{expert_idx}]: {e}',
                                log_path
                            )
                else:
                    # 如果没有 migration manager，只能跳过
                    failed += 1
            
            return loaded, skipped, failed
        
        # 并行加载各层专家
        num_workers = min(8, len(layer_experts))
        
        if num_workers > 0:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for layer_idx in sorted(layer_experts.keys()):
                    experts = layer_experts[layer_idx]
                    future = executor.submit(preload_layer, layer_idx, experts)
                    futures[future] = layer_idx
                
                for future in as_completed(futures):
                    layer_idx = futures[future]
                    try:
                        loaded, skipped, failed = future.result()
                        total_loaded += loaded
                        total_skipped += skipped
                        total_failed += failed
                    except Exception as e:
                        if log_path:
                            append_log(
                                f'Native prefetch: layer {layer_idx} failed: {e}',
                                log_path
                            )
        
        elapsed = time.time() - start_time
        
        if log_path:
            append_log(
                f'Native prefetch completed: loaded={total_loaded}, '
                f'skipped={total_skipped}, failed={total_failed}, '
                f'time={elapsed:.2f}s',
                log_path
            )
        
        log_once(
            'native_prefetch_done',
            f'Native prefetch: loaded {total_loaded} experts, '
            f'skipped {total_skipped} in {elapsed:.2f}s'
        )
        
    except Exception as e:
        if log_path:
            append_log(f'Native prefetch failed: {e}', log_path)
        log_once('native_prefetch_err', f'Native prefetch error: {e}')
        import traceback
        traceback.print_exc()


def _init_dynamic_scheduling():
    """Initialize dynamic expert scheduling components."""
    global _expert_resolver, _migration_manager
    
    log_path = _cfg.get('log_path')
    
    try:
        # Get model configuration
        hf_config = _cfg.get('hf_config')
        if hf_config is None:
            log_once('dynamic_no_hf_config', 'Dynamic scheduling: no hf_config available')
            return
        
        # 直接使用已解析的 hf_config 对象（HFConfigResolver 已处理字段映射）
        num_layers = hf_config.num_hidden_layers
        num_experts = hf_config.num_experts
        hidden_size = hf_config.hidden_size
        intermediate_size = hf_config.intermediate_size or 14336
        
        # Get dynamic scheduling config
        max_gpu_experts = _cfg.get('max_gpu_experts_per_layer', 2)
        model_path = _cfg.get('model_path')
        
        if model_path is None:
            log_once('dynamic_no_model_path', 'Dynamic scheduling: no model_path configured')
            return
        
        # Initialize GPU cache
        from .scheduling.gpu_cache import init_gpu_cache
        gpu_cache = init_gpu_cache(
            num_layers=num_layers,
            num_experts_per_layer=num_experts,
            max_gpu_experts_per_layer=max_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=torch.bfloat16,
            device="cuda",
            log_path=log_path
        )
        
        # Initialize location map
        from .scheduling.expert_location import init_location_map
        location_map = init_location_map(
            num_layers=num_layers,
            num_experts_per_layer=num_experts,
            log_path=log_path
        )
        
        # Initialize expert resolver
        from .core.expert_resolver import ExpertResolver
        _expert_resolver = ExpertResolver(_cfg, log_path)
        _expert_resolver.set_hf_model_path(model_path)
        
        # Initialize migration manager
        from .scheduling.expert_migration import init_migration_manager
        _migration_manager = init_migration_manager(
            expert_resolver=_expert_resolver,
            log_path=log_path,
            enable_async=True
        )
        
        # Initialize prefetcher
        from .scheduling.prefetcher import init_prefetcher
        init_prefetcher(log_path=log_path)
        
        # Initialize scheduler
        from .scheduling.scheduler import init_scheduler
        init_scheduler(
            num_layers=num_layers,
            num_experts=num_experts,
            max_gpu_experts_per_layer=max_gpu_experts,
            log_path=log_path
        )
        
        # Initialize dynamic router
        from .scheduling.dynamic_router import init_dynamic_router, get_dynamic_router
        init_dynamic_router(log_path=log_path)
        
        # Set up prediction callback for dynamic router
        # This callback runs prediction and triggers scheduling for next layer
        router = get_dynamic_router()
        if router is not None:
            def _predict_and_schedule_callback(wrapper, dispatch_output):
                """Predict next layer's experts and schedule prefetch."""
                from .scheduling.scheduler import get_scheduler, schedule_next_layer
                
                # Run prediction
                predicted = predict_experts(wrapper, dispatch_output, _cfg, _gate_resolver)
                
                if predicted:
                    # Get current layer
                    layer_idx = getattr(wrapper.kt_config, 'layer_idx', -1)
                    
                    # Convert prediction result to set of expert indices
                    if isinstance(predicted, set):
                        predicted_experts = predicted
                    elif isinstance(predicted, (list, tuple)):
                        # predicted might be [(layer, expert), ...] or just [expert, ...]
                        if predicted and isinstance(predicted[0], (list, tuple)):
                            # Format: [(layer_idx, expert_idx), ...]
                            # Filter for next layer
                            next_layer = (layer_idx + 1) % num_layers
                            predicted_experts = {exp for lay, exp in predicted if lay == next_layer}
                        else:
                            # Format: [expert_idx, ...]
                            predicted_experts = set(predicted)
                    else:
                        predicted_experts = set()
                    
                    # Schedule next layer
                    if predicted_experts:
                        schedule_next_layer(layer_idx, predicted_experts)
                        
                        if log_path:
                            append_log(
                                f'Scheduled layer {(layer_idx + 1) % num_layers} with '
                                f'{len(predicted_experts)} predicted experts',
                                log_path
                            )
            
            router.set_predict_fn(_predict_and_schedule_callback)
            log_once('dynamic_predict_cb', 'Dynamic router: prediction+scheduling callback installed')
        
        log_once('dynamic_init_done', 
                 f'Dynamic scheduling initialized: {num_layers} layers, '
                 f'{num_experts} experts, {max_gpu_experts} GPU slots/layer')
        
        if log_path:
            append_log(
                f'Dynamic scheduling config: hidden={hidden_size}, '
                f'intermediate={intermediate_size}, model_path={model_path}',
                log_path
            )
            
    except Exception as e:
        log_once('dynamic_init_err', f'Dynamic scheduling init failed: {e}')
        if log_path:
            append_log(f'Dynamic scheduling init error: {e}', log_path)
        import traceback
        traceback.print_exc()


def _init_native_gpu_cache():
    """
    Initialize native GPU cache for direct sglang weight manipulation.
    
    This enables using FusedMoE kernel with dynamic expert scheduling by
    directly operating on sglang's weight tensors.
    """
    global _native_gpu_cache_enabled, _expert_resolver, _native_migration_manager
    
    log_path = _cfg.get('log_path')
    append_log("[DEBUG] _init_native_gpu_cache called", log_path)
    append_log(f"[DEBUG] _cfg keys: {list(_cfg.keys())}", log_path)
    
    try:
        # Get model configuration
        hf_config = _cfg.get('hf_config')
        append_log(f"[DEBUG] hf_config: {hf_config}", log_path)
        if hf_config is None:
            log_once('native_cache_no_hf_config', 'Native GPU cache: no hf_config')
            append_log("[DEBUG] hf_config is None, returning", log_path)
            return
        
        # 直接使用已解析的 hf_config 对象（HFConfigResolver 已处理字段映射）
        num_layers = hf_config.num_hidden_layers
        num_experts = hf_config.num_experts
        hidden_size = hf_config.hidden_size
        intermediate_size = hf_config.intermediate_size or 14336
        
        # Initialize native GPU cache manager
        from .native.native_gpu_cache import init_native_cache
        max_gpu_experts_per_layer = _cfg.get('max_gpu_experts_per_layer', 2)
        
        # IO 优化参数（可在 config 中配置）
        enable_pinned_memory = _cfg.get('enable_pinned_memory', True)
        num_transfer_streams = _cfg.get('num_transfer_streams', 2)  # 默认 2，可配置 2-4
        cpu_pinned_pool_size = _cfg.get('pinned_pool_size', 0)  # CPU 侧 pinned pool（供 ExpertResolver 使用）
        
        native_cache = init_native_cache(
            num_layers=num_layers,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_gpu_slots=max_gpu_experts_per_layer,
            log_path=log_path,
            enable_pinned_memory=enable_pinned_memory,
            num_transfer_streams=num_transfer_streams,
            cpu_pinned_pool_size=cpu_pinned_pool_size,
        )
        
        if native_cache is not None:
            _native_gpu_cache_enabled = True
            log_once('native_cache_init', 
                     f'Native GPU cache initialized: {num_layers} layers, '
                     f'{num_experts} experts')
            
            # Initialize native scheduler (must be before router)
            from .native.native_scheduler import init_native_scheduler
            from .core.expert_scheduler import RerouteConfig
            num_gpu_slots = native_cache.num_gpu_slots
            
            # Build reroute config from yaml
            max_dup_per_token = _cfg.get('reroute_max_duplicates_per_expert')
            if max_dup_per_token is None:
                max_dup_per_token = _cfg.get('max_gpu_duplicates', 2)
            reroute_config = RerouteConfig(
                strategy=_cfg.get('reroute_strategy', 'token_reroute'),
                alpha=_cfg.get('reroute_alpha', 0.35),
                allow_duplicate=_cfg.get('reroute_allow_duplicate', False),
                use_limited_reroute=_cfg.get('reroute_use_limited', True),
                max_duplicates_per_expert=max_dup_per_token,
                min_unique_experts=_cfg.get('reroute_min_unique_experts', None),
                score_threshold_ratio=_cfg.get('reroute_score_threshold_ratio', 0.8),
                max_gpu_duplicates=_cfg.get('max_gpu_duplicates', 2),
                dominance_threshold=_cfg.get('dominance_threshold', 0.5),
            )
            if log_path:
                append_log(
                    "Native scheduler config: "
                    f"strategy={reroute_config.strategy}, "
                    f"alpha={reroute_config.alpha}, "
                    f"allow_duplicate={reroute_config.allow_duplicate}, "
                    f"use_limited={reroute_config.use_limited_reroute}, "
                    f"max_dup_per_token={reroute_config.max_duplicates_per_expert}, "
                    f"min_unique={reroute_config.min_unique_experts}, "
                    f"score_threshold={reroute_config.score_threshold_ratio}, "
                    f"max_gpu_duplicates={reroute_config.max_gpu_duplicates}, "
                    f"dominance_threshold={reroute_config.dominance_threshold}",
                    log_path,
                    level=1,
                )
            
            scheduler = init_native_scheduler(
                num_layers=num_layers,
                num_experts=num_experts,
                num_gpu_slots=num_gpu_slots,
                log_path=log_path,
                cache=native_cache,
                reroute_config=reroute_config,
            )
            
            # Initialize native router with scheduler (always initialize for statistics collection)
            from .native.native_router import init_native_router, get_native_router
            prefetch_cfg = _cfg.get('prefetch', {})
            enable_prefetch = prefetch_cfg.get('enable', False)
            prefetch_mode = prefetch_cfg.get('mode', 'layer')
            
            # Always initialize native router for statistics collection, regardless of prefetch settings
            native_router = init_native_router(
                log_path=log_path, 
                scheduler=scheduler, 
                enable_prefetch=enable_prefetch, 
                prefetch_mode=prefetch_mode,
            )
            
            # Set up prediction callback for native router (only if prediction is enabled)
            def _native_predict_callback(next_layer_idx, cur_hidden_states):
                """Predict next layer's experts and notify scheduler."""
                # Run prediction
                predicted = predict_experts(next_layer_idx, cur_hidden_states, _cfg, _gate_resolver)
                
                if predicted: 
                    return predicted
                return None
            
            native_router.set_predict_fn(_native_predict_callback)
            log_once('native_predict_cb', 'Native router: prediction callback installed')
            
            # Initialize expert resolver for native backend (needed for weight loading)
            from .core.expert_resolver import ExpertResolver
            hf_model_path = _cfg.get('hf_model_path')
            if hf_model_path:
                # Create a copy of config for ExpertResolver
                # Note: ExpertResolver will use native_cache's unified pinned pool (no independent allocation)
                resolver_cfg = dict(_cfg)
                resolver_cfg['use_pinned_memory'] = True  # Enable pinned memory (from native_cache)
                resolver_cfg['allocate_own_pinned_pool'] = False  # Disable independent pool allocation
                
                _expert_resolver = ExpertResolver(
                    cfg=resolver_cfg,
                    log_path=log_path,
                )
                
                if log_path:
                    append_log(
                        f'Native ExpertResolver initialized with HF path: {hf_model_path} '
                        f'(using native cache pinned memory, not ExpertResolver pool)',
                        log_path
                    )
                
                # Initialize native migration manager
                from .native.native_migration import init_native_migration_manager
                _native_migration_manager = init_native_migration_manager(
                    expert_resolver=_expert_resolver,
                    native_cache=native_cache,
                    log_path=log_path,
                )
                
                if log_path:
                    append_log(
                        f'NativeExpertMigrationManager initialized',
                        log_path
                    )
            else:
                if log_path:
                    append_log(
                        'Native GPU cache: no hf_model_path, expert swapping disabled',
                        log_path
                    )
            
            if log_path:
                append_log(
                    f'Native GPU cache config: hidden={hidden_size}, '
                    f'intermediate={intermediate_size}',
                    log_path
                )
        
    except Exception as e:
        log_once('native_cache_init_err', f'Native GPU cache init failed: {e}')
        if log_path:
            append_log(f'Native GPU cache init error: {e}', log_path)
        import traceback
        traceback.print_exc()


def _register_layer_to_native_cache(
    wrapper: KTEPWrapperMethod,
    layer: torch.nn.Module,
    layer_idx: int
):
    """
    Register a sglang MoE layer to native GPU cache.
    
    This allows the cache to directly manipulate the layer's weight tensors.
    """
    global _registered_layers
    
    if not _native_gpu_cache_enabled:
        return
    
    log_path = _cfg.get('log_path')
    
    try:
        from .native.native_gpu_cache import get_native_cache
        native_cache = get_native_cache()
        
        if native_cache is None:
            return
        
        # Get number of GPU expert slots
        num_gpu_experts = _cfg.get('max_gpu_experts_per_layer', 0)
        
        # 如果 yaml 没配置，则回退到 kt_config
        if num_gpu_experts <= 0:
            kt_config = getattr(wrapper, 'kt_config', None)
            if kt_config is not None:
                num_gpu_experts = getattr(kt_config, 'num_gpu_experts', 0)
        
        if num_gpu_experts <= 0:
            if log_path:
                append_log(
                    f'Layer {layer_idx}: num_gpu_experts=0, skipping registration',
                    log_path
                )
            return
        
        # Register the layer
        success = native_cache.register_layer(layer_idx, layer, num_gpu_experts)
        
        if success:
            _registered_layers[layer_idx] = (wrapper, layer)
            
            if log_path:
                append_log(
                    f'Registered layer {layer_idx} to native cache: '
                    f'{num_gpu_experts} GPU slots',
                    log_path
                )
            
            log_once(f'native_reg_{layer_idx}', 
                     f'Layer {layer_idx} registered to native cache')
        
    except Exception as e:
        log_once(f'native_reg_err_{layer_idx}', 
                 f'Failed to register layer {layer_idx}: {e}')
        if log_path:
            append_log(f'Failed to register layer {layer_idx}: {e}', log_path)


# Store original methods
_orig_process = KTEPWrapperMethod.process_weights_after_loading
_orig_submit = KTEPWrapperMethod.submit
_orig_apply = KTEPWrapperMethod.apply

# Store original MoE layer methods
_orig_forward_router_experts = None
_orig_mixtral_forward = None


def _patched_process(self: KTEPWrapperMethod, layer) -> None:
    """Patched process_weights_after_loading: runs preloader after weight loading."""
    global _preload_done, _preload_positions
    
    _ensure_init()
    _orig_process(self, layer)
    
    # 获取 layer_idx
    layer_idx: Optional[int] = None
    try:
        kt_config = getattr(self, 'kt_config', None)
        if kt_config is not None:
            layer_idx = getattr(kt_config, 'layer_idx', None)
    except Exception as e:
        log_path = _cfg.get('log_path')
        if log_path:
            from .logger import append_log
            append_log(f'Fetch layer_idx in patched_process failed: {e}', log_path)
        log_once('patched_process_layer_idx_err', f'Fetch layer_idx failed: {e}')
    
    # Register layer to native GPU cache (if enabled)
    if layer_idx is not None:
        _register_layer_to_native_cache(self, layer, layer_idx)
    
    # 从我们自己的 hf_config 获取 total_layers
    total_layers: Optional[int] = None
    hf_config = _cfg.get('hf_config')
    if hf_config is not None:
        total_layers = getattr(hf_config, 'num_hidden_layers', None)
        # kt_config.num_layers = total_layers  # 注入回 kt_config，供后续使用

    # 只在最后一层加载完成后执行一次预加载计算
    is_last_layer = (
        total_layers is not None 
        and layer_idx is not None 
        and layer_idx == total_layers - 1
    )

    if not _preload_done and is_last_layer:
        _preload_positions = call_preloader(
            getattr(self, 'wrapper', None), 
            layer_idx if isinstance(layer_idx, int) else -1, 
            _cfg
        )
        _preload_done = True
        
        log_path = _cfg.get('log_path')
        if log_path:
            from .logger import append_log
            append_log(f'Preload calculation done at layer {layer_idx}, positions: {len(_preload_positions) if _preload_positions else 0}', log_path)
        
        # Initialize cache and preload experts based on backend
        if _dynamic_scheduling_enabled:
            scheduling_backend = _cfg.get('scheduling_backend', 'native')
            
            if scheduling_backend == 'native':
                # Native backend: preload to native GPU cache
                _sync_prefetch_preload_experts_native()
            else:
                # Naive backend: preload to custom GPU cache
                _initialize_gpu_cache_after_model_load()
                _sync_prefetch_preload_experts()
            
    
    # Gate 预加载也只做一次
    try:
        if _gate_resolver is not None and is_last_layer:
            _gate_resolver.preload_all_gates()
    except Exception as e:
        log_path = _cfg.get('log_path')
        if log_path:
            from .logger import append_log
            append_log(f'Preload-all during process failed: {e}', log_path)
        log_once('gate_preload_process_err', f'Preload-all during process failed: {e}')


def _patched_submit(self: KTEPWrapperMethod, layer, dispatch_output):
    """Patched submit: runs prediction/prefetch before original submit."""
    _ensure_init()
    predict_and_prefetch(self, dispatch_output, _cfg, _gate_resolver)
    return _orig_submit(self, layer, dispatch_output)


def _patched_apply(self: KTEPWrapperMethod, layer, dispatch_output):
    """
    Patched apply: uses dynamic routing if enabled, otherwise original.
    
    When dynamic_scheduling is enabled:
    - scheduling_backend='native': uses FusedMoE kernel (fast)
    - scheduling_backend='naive': uses F.linear loops (slow)
    """
    _ensure_init()
    
    log_path = _cfg.get('log_path')
    append_log(f"[DEBUG] _patched_apply called, dynamic_scheduling_enabled: {_dynamic_scheduling_enabled}, native_gpu_cache_enabled: {_native_gpu_cache_enabled}", log_path)
    
    if _dynamic_scheduling_enabled:
        append_log(f"[DEBUG] Dynamic scheduling enabled", log_path)
        # Try native router first (if initialized)
        if _native_gpu_cache_enabled:
            from .native.native_router import get_native_router
            native_router = get_native_router()
            
            if native_router is not None:
                result = native_router.native_dynamic_apply(self, layer, dispatch_output)
                return result
        else:
            append_log(f"[DEBUG] Native GPU cache not enabled", log_path)
        
        # Try naive router
        append_log(f"[DEBUG] Trying naive router", log_path)
        from .scheduling.dynamic_router import get_dynamic_router
        router = get_dynamic_router()
        
        if router is not None:
            append_log(f"[DEBUG] Naive router found, calling dynamic_apply", log_path)
            return router.dynamic_apply(self, layer, dispatch_output)
        else:
            append_log(f"[DEBUG] Naive router is None", log_path)
    else:
        append_log(f"[DEBUG] Dynamic scheduling disabled", log_path)
    
    # Fall back to original apply
    append_log(f"[DEBUG] Falling back to original apply", log_path)
    return _orig_apply(self, layer, dispatch_output)

# Qwen2-57B-A14B
def _patched_forward_router_experts(self, hidden_states: torch.Tensor):
    """
    Patched _forward_router_experts: runs prediction logic before gate computation.
    
    This allows inserting prediction logic right before the gate calculation,
    which is the earliest point where we have the input hidden_states.
    """
    _ensure_init()
    log_path = _cfg.get('log_path')
    layer_idx = getattr(self, 'layer_id', -1)
    append_log(f"Entering patched _forward_router_experts for layer {layer_idx}", log_path, level=3)
    append_log(f"shape of hidden states: {hidden_states.shape}", log_path, level=3)
    from .native.native_router import get_native_router, _get_predict_executor, _get_prefetch_executor
    from .native.native_migration import get_native_migration_manager

    try:
        native_router = get_native_router()
        if native_router._predict_fn is not None:
            executor = _get_predict_executor()
            hf_config = _cfg.get('hf_config')
            num_hidden_layers = getattr(hf_config, 'num_hidden_layers', None)
            if num_hidden_layers is not None:
                executor.submit(
                    native_router._predict_and_schedule,
                    num_hidden_layers, hidden_states, layer_idx
                )
            else:
                append_log(f"[Pre-Gate Prediction] Layer {layer_idx}: hf_config missing num_hidden_layers", log_path, level=1)

    except Exception as e:
        log_path = _cfg.get('log_path')
        if log_path:
            append_log(f"Pre-gate prediction failed at layer {layer_idx}: {e}", log_path)
    
    # 计算 gate/topk，获取 topk ids 后再调用 experts，这样我们可以访问 topk_output
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    try:
        # 与原始实现一致：先计算 router logits -> topk -> experts
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)

        # 停止上一层的预取（如果有迁移管理器可用）
        # 将 GPU->CPU 的转换放到后台线程，避免主线程被同步阻塞
        migration_manager = get_native_migration_manager()
        if migration_manager is not None:
            executor = _get_prefetch_executor()

            if TopKOutputChecker.format_is_standard(topk_output):
                topk_ids_tensor = topk_output.topk_ids

                def _cancel_from_tensor(lidx, topk_ids_t):
                    try:
                        # 过滤掉填充值 -1
                        valid = topk_ids_t[topk_ids_t >= 0]
                        if valid.numel() == 0:
                            migration_manager.cancel_prefetch(lidx, None)
                            return
                        uniq = torch.unique(valid)
                        migration_manager.cancel_prefetch(lidx, set(uniq.cpu().tolist()))
                    except Exception:
                        # 回退为不带具体 expert 列表的取消
                        migration_manager.cancel_prefetch(lidx, None)

                executor.submit(_cancel_from_tensor, layer_idx, topk_ids_tensor)
            else:
                # 非 standard 格式时，保守处理：不传具体集合
                executor.submit(migration_manager.cancel_prefetch, layer_idx, None)

        # 调用 experts（与原始行为一致）
        _ori_return = self.experts(hidden_states, topk_output)

    except Exception:
        # 出现任何异常时回退到原始实现，保证行为兼容性
        _ori_return = _orig_forward_router_experts(self, hidden_states)

    return _ori_return


def _patched_mixtral_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Mixtral block hook to trigger prediction before gate computation."""
    _ensure_init()
    log_path = _cfg.get('log_path')
    layer_idx = getattr(self, 'layer_id', -1)
    append_log(
        f"Entering MixtralMoE.forward patch for layer {layer_idx}",
        log_path,
        level=3,
    )
    append_log(f"shape of hidden states: {hidden_states.shape}", log_path, level=3)

    try:
        from .native.native_router import get_native_router, _get_predict_executor

        native_router = get_native_router()
        if native_router is not None and native_router._predict_fn is not None:
            hf_config = _cfg.get('hf_config')
            num_hidden_layers = getattr(hf_config, 'num_hidden_layers', None)

            if num_hidden_layers is not None:
                try:
                    flat_hidden_states = hidden_states.view(-1, self.hidden_size)
                except Exception:
                    flat_hidden_states = hidden_states

                _get_predict_executor().submit(
                    native_router._predict_and_schedule,
                    num_hidden_layers,
                    flat_hidden_states,
                    layer_idx,
                )
            elif log_path:
                append_log(
                    f"[Pre-Gate Prediction] Layer {layer_idx}: hf_config missing num_hidden_layers",
                    log_path,
                    level=1,
                )
    except Exception as e:
        if log_path:
            append_log(
                f"Mixtral pre-gate prediction failed at layer {layer_idx}: {e}",
                log_path,
            )

    return _orig_mixtral_forward(self, hidden_states)


# =================== hook 入口 =======================
def install_hooks() -> None:
    """Install hook patches to KTEPWrapperMethod and MoE layers if enabled."""
    global _orig_forward_router_experts, _orig_mixtral_forward
    
    if not bool(int(os.environ.get('MOE_HOOK_ENABLE', '1'))):
        log_once('skip_install', 'MOE_HOOK_ENABLE=0; skipping hook installation')
        return
    
    # Install KTEPWrapperMethod hooks
    KTEPWrapperMethod.process_weights_after_loading = _patched_process  # type: ignore
    KTEPWrapperMethod.submit = _patched_submit  # type: ignore
    KTEPWrapperMethod.apply = _patched_apply  # type: ignore
    
    # Install MoE layer hooks for pre-gate prediction
    try:
        # wzq todo: 根据模型，选择不同的 MoE 层类进行 patch
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
        if _orig_forward_router_experts is None:
            _orig_forward_router_experts = Qwen2MoeSparseMoeBlock._forward_router_experts
        Qwen2MoeSparseMoeBlock._forward_router_experts = _patched_forward_router_experts  # type: ignore
        
        log_once('install_moe_layer', 'Installed Qwen2MoeSparseMoeBlock._forward_router_experts hook')
    except ImportError as e:
        log_once('install_moe_layer_err', f'Failed to install MoE layer hook: {e}')
    except Exception as e:
        log_once('install_moe_layer_err', f'Unexpected error installing MoE layer hook: {e}')

    # try:
    #     from sglang.srt.models.mixtral import MixtralMoE
    #     if _orig_mixtral_forward is None:
    #         _orig_mixtral_forward = MixtralMoE.forward
    #     MixtralMoE.forward = _patched_mixtral_forward  # type: ignore

    #     log_once('install_mixtral_layer', 'Installed MixtralMoE.forward hook')
    # except ImportError as e:
    #     log_once('install_mixtral_layer_err', f'Failed to install MixtralMoE hook: {e}')
    # except Exception as e:
    #     log_once('install_mixtral_layer_err', f'Unexpected error installing MixtralMoE hook: {e}')
    
    log_once('install', 'Installed hooks (KTEPWrapperMethod + MoE layers)')


def get_preload_positions() -> Optional[list]:
    """
    获取预加载的专家位置列表。
    
    Returns:
        List of (layer_idx, expert_idx) tuples, or None if not yet computed
    """
    return _preload_positions


def is_preload_done() -> bool:
    """检查预加载计算是否已完成。"""
    return _preload_done


def is_dynamic_scheduling_enabled() -> bool:
    """检查动态调度是否启用。"""
    return _dynamic_scheduling_enabled


def get_migration_manager():
    """获取专家迁移管理器实例。"""
    return _migration_manager


def load_expert_to_gpu(layer_idx: int, expert_idx: int) -> bool:
    """
    将专家加载到 GPU 缓存。
    
    Args:
        layer_idx: 层索引
        expert_idx: 专家索引
        
    Returns:
        是否成功
    """
    if _migration_manager is None:
        log_once('api_no_manager', 'load_expert_to_gpu: migration manager not initialized')
        return False
    return _migration_manager.load_expert_to_gpu(layer_idx, expert_idx)


def unload_expert_from_gpu(layer_idx: int, expert_idx: int) -> bool:
    """
    从 GPU 缓存卸载专家。
    
    Args:
        layer_idx: 层索引
        expert_idx: 专家索引
        
    Returns:
        是否成功
    """
    if _migration_manager is None:
        log_once('api_no_manager', 'unload_expert_from_gpu: migration manager not initialized')
        return False
    return _migration_manager.unload_expert_from_gpu(layer_idx, expert_idx)


def migrate_to_target(target_gpu_experts: Dict[int, Set[int]]) -> bool:
    """
    将专家迁移到目标分布。
    
    Args:
        target_gpu_experts: Dict[layer_idx, Set[expert_idx]] 目标 GPU 专家分布
        
    Returns:
        是否全部成功
    """
    if _migration_manager is None:
        log_once('api_no_manager', 'migrate_to_target: migration manager not initialized')
        return False
    return _migration_manager.migrate_to_target_distribution(target_gpu_experts)


def get_expert_locations() -> Optional[Dict[str, Any]]:
    """
    获取当前专家位置分布统计。
    
    Returns:
        专家位置统计信息
    """
    from .scheduling.expert_location import get_location_map
    location_map = get_location_map()
    if location_map is None:
        return None
    return location_map.get_stats()


def get_gpu_cache_stats() -> Optional[Dict[str, Any]]:
    """
    获取 GPU 缓存统计信息。
    
    Returns:
        GPU 缓存统计
    """
    from .scheduling.gpu_cache import get_gpu_cache
    gpu_cache = get_gpu_cache()
    if gpu_cache is None:
        return None
    return gpu_cache.get_cache_stats()


# ==================================================
# Native GPU Cache API
# ==================================================

def is_native_gpu_cache_enabled() -> bool:
    """检查原生 GPU 缓存是否启用。"""
    return _native_gpu_cache_enabled


def get_native_cache_stats() -> Optional[Dict[str, Any]]:
    """
    获取原生 GPU 缓存统计信息。
    
    Returns:
        原生缓存统计
    """
    from .native.native_gpu_cache import get_native_cache
    native_cache = get_native_cache()
    if native_cache is None:
        return None
    return native_cache.get_cache_stats()


def swap_expert_native(
    layer_idx: int,
    slot_idx: int,
    new_expert_idx: int,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor
) -> bool:
    """
    在原生 GPU 缓存中替换专家。
    
    Args:
        layer_idx: 层索引
        slot_idx: 目标槽位 (0 ~ num_gpu_slots-1)
        new_expert_idx: 新专家的ID
        w13_weight: 新专家的 gate_up 权重
        w2_weight: 新专家的 down 权重
        
    Returns:
        是否成功
    """
    from .native.native_gpu_cache import get_native_cache
    native_cache = get_native_cache()
    if native_cache is None:
        log_once('api_no_native_cache', 'swap_expert_native: native cache not initialized')
        return False
    return native_cache.swap_expert(layer_idx, slot_idx, new_expert_idx, w13_weight, w2_weight)


def get_registered_layers() -> Dict[int, Any]:
    """获取已注册到原生缓存的层。"""
    return dict(_registered_layers)





