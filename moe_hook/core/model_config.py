"""
HuggingFace model configuration auto-resolver.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..logger import log_once, append_log


@dataclass
class HFModelConfig:
    """Parsed HuggingFace model configuration for MOE models."""
    num_experts: int                    # 每层专家数
    num_hidden_layers: int              # 总层数
    num_experts_per_tok: int            # top_k
    hidden_size: int                    # 隐藏层大小
    intermediate_size: Optional[int] = None  # MLP 中间层大小
    model_type: Optional[str] = None    # 模型类型 (e.g., 'mixtral', 'qwen2_moe')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_experts': self.num_experts,
            'num_hidden_layers': self.num_hidden_layers,
            'num_experts_per_tok': self.num_experts_per_tok,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'model_type': self.model_type,
        }


class HFConfigResolver:
    """Resolves HuggingFace model configuration from model path."""
    
    # 常见的配置字段映射
    FIELD_MAPPINGS = {
        'num_experts': ['num_local_experts', 'num_experts', 'n_experts'],
        'num_hidden_layers': ['num_hidden_layers', 'n_layers', 'num_layers'],
        'num_experts_per_tok': ['num_experts_per_tok', 'num_experts_per_token', 'top_k'],
        'hidden_size': ['hidden_size', 'd_model', 'n_embd'],
        'intermediate_size': ['intermediate_size', 'moe_intermediate_size', 'd_ff'],
        'model_type': ['model_type'],
    }
    
    def __init__(self, model_path: Optional[str] = None, log_path: Optional[str] = None):
        """
        Initialize resolver.
        
        Args:
            model_path: Path to HF model directory
            log_path: Optional path for logging
        """
        self.model_path = model_path
        self.log_path = log_path
        self._raw_config: Optional[Dict[str, Any]] = None
        self._parsed_config: Optional[HFModelConfig] = None
    
    def _log(self, msg: str) -> None:
        """Helper to log."""
        if self.log_path:
            append_log(f'[HFConfigResolver] {msg}', self.log_path)
    
    def _find_config_file(self) -> Optional[str]:
        """Find config.json file in model path."""
        if not self.model_path:
            return None
        
        config_file = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_file):
            return config_file
        
        # 尝试其他可能的位置
        for subdir in ['', 'model', 'models']:
            path = os.path.join(self.model_path, subdir, 'config.json')
            if os.path.exists(path):
                return path
        
        return None
    
    def _get_field(self, raw_config: Dict[str, Any], field_name: str) -> Optional[Any]:
        """Get field value using multiple possible keys."""
        possible_keys = self.FIELD_MAPPINGS.get(field_name, [field_name])
        for key in possible_keys:
            if key in raw_config:
                return raw_config[key]
        return None
    
    def load_raw_config(self) -> Optional[Dict[str, Any]]:
        """Load raw config.json as dictionary."""
        if self._raw_config is not None:
            return self._raw_config
        
        config_file = self._find_config_file()
        if config_file is None:
            self._log(f'No config.json found in {self.model_path}')
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._raw_config = json.load(f)
            self._log(f'Loaded raw config from {config_file}')
            return self._raw_config
        except Exception as e:
            self._log(f'Failed to load config.json: {e}')
            log_once('hf_config_load_err', f'Failed to load {config_file}: {e}')
            return None
    
    def resolve(self) -> Optional[HFModelConfig]:
        """
        Resolve and parse HF model configuration.
        
        Returns:
            HFModelConfig if successful, None otherwise
        """
        if self._parsed_config is not None:
            return self._parsed_config
        
        raw_config = self.load_raw_config()
        if raw_config is None:
            return None
        
        try:
            # 提取必要字段
            num_experts = self._get_field(raw_config, 'num_experts')
            num_hidden_layers = self._get_field(raw_config, 'num_hidden_layers')
            num_experts_per_tok = self._get_field(raw_config, 'num_experts_per_tok')
            hidden_size = self._get_field(raw_config, 'hidden_size')
            
            # 验证必要字段
            if num_experts is None:
                self._log('num_experts not found in config (not a MOE model?)')
                return None
            if num_hidden_layers is None:
                self._log('num_hidden_layers not found in config')
                return None
            if num_experts_per_tok is None:
                # 某些模型可能没有这个字段，使用默认值
                num_experts_per_tok = 2
                self._log(f'num_experts_per_tok not found, using default: {num_experts_per_tok}')
            if hidden_size is None:
                self._log('hidden_size not found in config')
                return None
            
            # 提取可选字段
            intermediate_size = self._get_field(raw_config, 'intermediate_size')
            model_type = self._get_field(raw_config, 'model_type')
            
            self._parsed_config = HFModelConfig(
                num_experts=int(num_experts),
                num_hidden_layers=int(num_hidden_layers),
                num_experts_per_tok=int(num_experts_per_tok),
                hidden_size=int(hidden_size),
                intermediate_size=int(intermediate_size) if intermediate_size else None,
                model_type=model_type,
            )
            
            log_once('hf_config_resolved', 
                     f'Resolved HF config: experts={num_experts}, layers={num_hidden_layers}, '
                     f'top_k={num_experts_per_tok}, hidden={hidden_size}, type={model_type}')
            self._log(f'Resolved config: {self._parsed_config}')
            
            return self._parsed_config
            
        except Exception as e:
            self._log(f'Failed to parse HF config: {e}')
            log_once('hf_config_parse_err', f'Failed to parse HF config: {e}')
            return None
    
    def get_raw_value(self, key: str, default: Any = None) -> Any:
        """Get raw value from config.json by key."""
        raw_config = self.load_raw_config()
        if raw_config is None:
            return default
        return raw_config.get(key, default)


def resolve_hf_config(
    model_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    log_path: Optional[str] = None
) -> Optional[HFModelConfig]:
    """
    Convenience function to resolve HF config.
    
    Args:
        model_path: Direct path to model directory
        cfg: Config dict that may contain model_path or gate.model_path
        log_path: Optional log path
    
    Returns:
        HFModelConfig if successful, None otherwise
    """
    # 优先使用直接传入的 model_path
    if model_path is None and cfg is not None:
        # 尝试从多个位置获取 model_path
        model_path = (
            cfg.get('model_path')
            or cfg.get('gate', {}).get('model_path')
            or cfg.get('preload', {}).get('model_path')
            or os.environ.get('MOE_MODEL_PATH')
            or os.environ.get('MOE_GATE_MODEL_PATH')
        )
    
    if log_path is None and cfg is not None:
        log_path = cfg.get('log_path')
    
    if model_path is None:
        log_once('hf_config_no_path', 'No model_path provided for HF config resolution')
        return None
    
    resolver = HFConfigResolver(model_path, log_path)
    return resolver.resolve()


def inject_hf_config_to_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve HF config and inject into cfg dictionary.
    
    This function modifies cfg in-place and also returns it.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        Modified cfg with hf_config added
    """
    if 'hf_config' in cfg and cfg['hf_config'] is not None:
        # 已经有 hf_config，不覆盖
        return cfg
    
    log_path = cfg.get('log_path')
    hf_config = resolve_hf_config(cfg=cfg, log_path=log_path)
    
    if hf_config is not None:
        cfg['hf_config'] = hf_config
        if log_path:
            append_log(f'Injected hf_config into cfg: {hf_config}', log_path)
    
    return cfg
