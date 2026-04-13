"""
Gate weight resolver for MOE models.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

from ..logger import log_once, append_log
from .model_config import HFConfigResolver


class GateResolver:
    """Automatically resolves and loads gate weights from model checkpoints."""

    def __init__(self, cfg: Dict[str, Any], log_path: Optional[str] = None):
        gate_cfg = cfg.get('gate', {}) if isinstance(cfg, dict) else {}
        self.log_path = log_path
        self.enable = gate_cfg.get('enable', True)
        self.model_path = gate_cfg.get('model_path') or os.environ.get('MOE_GATE_MODEL_PATH')
        self.format = gate_cfg.get('format', 'auto')
        self.next_layer_offset = gate_cfg.get('next_layer_offset', 1)
        self.patterns = gate_cfg.get('patterns')
        self.device = gate_cfg.get('device') or os.environ.get('MOE_GATE_DEVICE')
        self._cache: Dict[int, Any] = {}
        self._preloaded_all = False
        
        # 使用 HFConfigResolver 获取模型配置
        self._hf_resolver = HFConfigResolver(self.model_path, log_path)
        hf_config = cfg.get('hf_config')  # 可能已被注入
        
        # 优先使用配置中的值，否则从 HF config 自动解析
        self.total_layers = (
            gate_cfg.get('total_layers')
            or os.environ.get('MOE_GATE_TOTAL_LAYERS')
            or (hf_config.num_hidden_layers if hf_config else None)
            or self._hf_resolver.get_raw_value('num_hidden_layers')
        )
        self.top_k = (
            os.environ.get('MOE_GATE_TOP_K')
            or (hf_config.num_experts_per_tok if hf_config else None)
            or self._hf_resolver.get_raw_value('num_experts_per_tok')
        )

    def _log(self, msg: str) -> None:
        """Helper to log with path."""
        if self.log_path:
            append_log(msg, self.log_path)

    def _iter_weight_files(self):
        """Iterate over weight files in model path."""
        if not self.model_path or not os.path.isdir(self.model_path):
            return []
        files = []
        for name in os.listdir(self.model_path):
            if name.endswith('.safetensors') or name.endswith('.bin'):
                files.append(os.path.join(self.model_path, name))
        return files

    def _resolve_from_index(self, layer_idx: int) -> Optional[Tuple[str, str]]:
        """Resolve gate weight key and file from index.json."""
        if not self.model_path or not os.path.isdir(self.model_path):
            return None
        patterns = self.patterns or [
            "model.layers.{idx}.mlp.gate.weight",
            "layers.{idx}.mlp.gate.weight",
            "model.layers.{idx}.gate.weight",
            "layers.{idx}.gate.weight",
        ]
        patterns = [p.format(idx=layer_idx) for p in patterns]
        self._log(f"Resolving gate for layer {layer_idx} using patterns: {patterns}")
        
        for idx_file in (
            os.path.join(self.model_path, 'pytorch_model.bin.index.json'),
            os.path.join(self.model_path, 'model.safetensors.index.json'),
        ):
            if not os.path.exists(idx_file):
                continue
            try:
                with open(idx_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                weight_map = meta.get('weight_map', {})
                for pat in patterns:
                    if pat in weight_map:
                        return pat, os.path.join(self.model_path, weight_map[pat])
            except Exception as e:
                self._log(f"Resolve gate index failed for {idx_file} layer={layer_idx}: {e}")
                log_once('gate_resolve_err', f"Resolve gate index failed for {idx_file}: {e}")
                continue
        return None

    def _target_device(self):
        """Determine target device for gate weights."""
        try:
            import torch
            dev = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.device(dev)
        except Exception as e:
            self._log(f"Resolve target device failed: {e}")
            log_once('target_device_err', f"Resolve target device failed: {e}")
            return None

    def _load_linear(self, weight_file: str, weight_key: str, device=None):
        """Load gate linear layer from weight file."""
        try:
            if weight_file.endswith('.safetensors'):
                from safetensors import safe_open  # type: ignore
                with safe_open(weight_file, framework='pt', device='cpu') as f:
                    if weight_key not in f.keys():
                        return None
                    w = f.get_tensor(weight_key)
            else:
                import torch
                state = torch.load(weight_file, map_location='cpu')
                if weight_key not in state:
                    return None
                w = state[weight_key]
            
            import torch.nn as nn
            num_experts, hidden_size = w.shape[0], w.shape[1]
            gate = nn.Linear(hidden_size, num_experts, bias=False)
            gate.weight.data.copy_(w)  # type: ignore[arg-type]
            gate.eval()
            
            if device is not None:
                try:
                    gate.to(device)
                except Exception as e:
                    self._log(f'Failed to move gate to device {device}: {e}')
                    log_once('gate_to_device_err', f'Failed to move gate to device {device}: {e}')
            return gate
        except Exception as e:
            self._log(f"Failed to load gate {weight_key} from {weight_file}: {e}")
            log_once(f'gate_load_{weight_key}', f"Failed to load gate {weight_key} from {weight_file}: {e}")
            return None

    def _load_gate(self, layer_idx: int, device=None):
        """Load gate for specific layer index."""
        if not self.enable:
            return None
        if layer_idx in self._cache:
            return self._cache[layer_idx]

        if self.format not in ('auto', 'hf'):
            log_once('gate_format', f"Gate resolver format '{self.format}' not supported yet")
            return None

        resolved = self._resolve_from_index(layer_idx)
        if resolved is None:
            for weight_file in self._iter_weight_files():
                resolved = (None, weight_file)
                break
        if resolved is None:
            log_once('gate_missing', 'No gate weight file found; set gate.model_path')
            return None

        weight_key, weight_file = resolved
        if weight_key is None:
            patterns = self.patterns or [
                f"model.layers.{layer_idx}.mlp.gate.weight",
                f"layers.{layer_idx}.mlp.gate.weight",
            ]
            if weight_file.endswith('.safetensors'):
                from safetensors import safe_open  # type: ignore
                with safe_open(weight_file, framework='pt', device='cpu') as f:
                    for pat in patterns:
                        if pat in f.keys():
                            weight_key = pat
                            break
            else:
                import torch
                state = torch.load(weight_file, map_location='cpu')
                for pat in patterns:
                    if pat in state:
                        weight_key = pat
                        break
                if weight_key is None:
                    return None
        
        gate = self._load_linear(weight_file, weight_key, device=device)
        if gate is not None:
            self._cache[layer_idx] = gate
        return gate

    def preload_all_gates(self):
        """Preload all gates into memory."""
        if self._preloaded_all:
            return
        total_layers = None
        try:
            total_layers = int(self.total_layers) if self.total_layers is not None else None
        except Exception as e:
            self._log(f'Parse total_layers failed during preload: {e}')
            log_once('gate_total_layers_parse_err', f'Parse total_layers failed during preload: {e}')
            total_layers = None
        
        if not total_layers or total_layers <= 0:
            return
        
        device = self._target_device()
        log_once('gate_preload_start', f'Preloading {total_layers} gates to device {device}')
        for idx in range(total_layers):
            if idx in self._cache:
                continue
            try:
                self._load_gate(idx, device=device)
            except Exception as e:
                self._log(f'Preload gate {idx} failed: {e}')
                log_once(f'gate_preload_err_{idx}', f'Preload gate {idx} failed: {e}')
        self._preloaded_all = True
        log_once('gate_preload_done', f'Preloaded gates for {total_layers} layers')

    def resolve(self, tar_layer_idx: int):
        """Resolve gate for given layer index (with offset applied)."""
        if not self.enable:
            return None
        if not isinstance(tar_layer_idx, int) or tar_layer_idx < 0:
            return None
        
        
        if tar_layer_idx in self._cache:
            append_log(f'Gate for layer {tar_layer_idx} retrieved from cache', self.log_path, level=3)
            return self._cache[tar_layer_idx]
        
        gate = self._load_gate(tar_layer_idx, device=self._target_device())
        if gate is not None:
            self._cache[tar_layer_idx] = gate
            log_once(f'gate_{tar_layer_idx}', f'Loaded gate for layer {tar_layer_idx} from {self.model_path}')
        return gate
