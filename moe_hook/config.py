"""
Configuration loading and management for MOE hooks.
"""

import os
from typing import Any, Dict

from .logger import log_once


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or environment."""
    cfg_path = config_path or os.environ.get("MOE_HOOK_CONFIG")
    cfg: Dict[str, Any] = {}
    log_once('cfg_path', f"MOE_HOOK_CONFIG={cfg_path}")
    
    if cfg_path and os.path.exists(cfg_path):
        try:
            if cfg_path.endswith(('.yml', '.yaml')):
                import yaml  # type: ignore
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
            elif cfg_path.endswith('.json'):
                import json
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f) or {}
            else:
                log_once('cfg_ext', f"Unsupported config extension: {cfg_path}")
        except Exception as e:
            log_once('cfg_err', f"Failed to load config {cfg_path}: {e}")
    
    # Set defaults
    cfg.setdefault('enable', bool(int(os.environ.get('MOE_HOOK_ENABLE', '1'))))
    cfg.setdefault('capture_bs', [1, 2, 4, 8])
    cfg.setdefault('preload', {'enable': True, 'mode': 'fate_shallow'})
    cfg.setdefault('predict', {'enable': True, 'mode': 'fate'})
    cfg.setdefault('prefetch', {'enable': True, 'mode': 'hook'})
    cfg.setdefault('gate', {
        'enable': True,
        'format': 'auto',
        'model_path': os.environ.get('MOE_GATE_MODEL_PATH'),
        'next_layer_offset': 1,
        'total_layers': None,
        'device': None,
        'patterns': None,
    })
    cfg.setdefault('log_path', os.environ.get('MOE_HOOK_LOG_PATH'))
    
    # Dynamic scheduling settings
    cfg.setdefault('dynamic_scheduling', bool(int(os.environ.get('MOE_DYNAMIC_SCHEDULING', '0'))))
    cfg.setdefault('max_gpu_experts_per_layer', int(os.environ.get('MOE_MAX_GPU_EXPERTS', '2')))
    cfg.setdefault('model_path', os.environ.get('MOE_MODEL_PATH'))
    
    # Reroute strategy: none/static, io_free, token_reroute (default)
    cfg.setdefault('reroute_strategy', os.environ.get('MOE_REROUTE_STRATEGY', 'token_reroute'))
    
    # Deferral control: set to True to disable ktransformers deferral mechanism at runtime
    # This is recommended when using dynamic expert scheduling
    # Alternatively, use --kt-max-deferred-experts-per-token 0 at startup
    cfg.setdefault('disable_deferral', bool(int(os.environ.get('MOE_DISABLE_DEFERRAL', '0'))))
    
    # Auto-resolve and inject hf_config
    try:
        from .core.model_config import inject_hf_config_to_cfg
        inject_hf_config_to_cfg(cfg)
    except Exception as e:
        log_once('hf_config_inject_err', f"Failed to inject hf_config: {e}")
    
    return cfg
