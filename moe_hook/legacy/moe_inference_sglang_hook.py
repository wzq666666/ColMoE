"""
moe_inference_sglang_hook.py

Minimal-intrusion hooks to integrate user's preloading, prediction (FATE), and prefetch
into SGLang+KTransformers MoE execution without modifying upstream sources.

Enable via:
  PYTHONPATH=/home/ecnu/disk/wzq/moe-inference/src:$PYTHONPATH \
  MOE_HOOK_ENABLE=1 MOE_HOOK_CONFIG=/home/ecnu/disk/wzq/moe-inference/moe_hook_config.yaml \
  python -c "import moe_inference_sglang_hook as _; import sglang.launch_server as ls; ls.main()"
"""

import inspect
import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

_cfg: Dict[str, Any] = {}
_once_logged: Dict[str, bool] = {}
_log_cleared = False


def _log_once(key: str, msg: str) -> None:
    if not _once_logged.get(key):
        print(f"[MOE-HOOK] {msg}", flush=True)
        _once_logged[key] = True


def _append_log(msg: str) -> None:
    path = _cfg.get('log_path') if isinstance(_cfg, dict) else None
    if not path:
        return
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception as e:  # pragma: no cover
        _log_once('logfile_err', f"Failed to write log file {path}: {e}")

# Optional imports from user's repo; all guarded for safety
try:
    from component import ModelPreloader  # type: ignore
    _log_once('import_modelpreloader', 'ModelPreloader loaded from component')
except Exception as e_primary:  # pragma: no cover
    try:
        from moe_inference.src.component import ModelPreloader  # type: ignore
        _log_once('import_modelpreloader_alt', 'ModelPreloader loaded from moe_inference.src.component')
    except Exception as e_fallback:  # pragma: no cover
        ModelPreloader = None  # type: ignore
        _append_log(f"Import ModelPreloader failed: primary={e_primary}; fallback={e_fallback}")
        _log_once('import_modelpreloader_err', f"Import ModelPreloader failed: primary={e_primary}; fallback={e_fallback}")

try:
    from component import predictor as _predictor  # type: ignore
    _log_once('import_predictor', 'predictor loaded from component')
except Exception as e_primary:  # pragma: no cover
    try:
        from moe_inference.src.component import predictor as _predictor  # type: ignore
        _log_once('import_predictor_alt', 'predictor loaded from moe_inference.src.component')
    except Exception as e_fallback:  # pragma: no cover
        _predictor = None  # type: ignore
        _append_log(f"Import predictor failed: primary={e_primary}; fallback={e_fallback}")
        _log_once('import_predictor_err', f"Import predictor failed: primary={e_primary}; fallback={e_fallback}")

try:
    from component import preloader as _preloader  # type: ignore
    _log_once('import_preloader', 'preloader loaded from component')
except Exception as e_primary:  # pragma: no cover
    try:
        from moe_inference.src.component import preloader as _preloader  # type: ignore
        _log_once('import_preloader_alt', 'preloader loaded from moe_inference.src.component')
    except Exception as e_fallback:  # pragma: no cover
        _preloader = None  # type: ignore
        _append_log(f"Import preloader failed: primary={e_primary}; fallback={e_fallback}")
        _log_once('import_preloader_err', f"Import preloader failed: primary={e_primary}; fallback={e_fallback}")

# 不再依赖原型系统里的 MoEPrefetcher，hook 内部做轻量预取
_prefetcher = None  # type: ignore


_inited = False
_init_lock = threading.Lock()


def _prefetch_stub(predicted, wrapper=None, cfg=None):
    # 轻量占位：当前不做磁盘/IO迁移，仅用于接口对齐
    _log_once('prefetch_stub', 'Using hook-side prefetch stub (no IO).')
    return predicted


class GateResolver:
    """根据模型路径自动解析gate权重并构建线性层，提供给FATE预测使用。"""

    def __init__(self, cfg: Dict[str, Any]):
        gate_cfg = cfg.get('gate', {}) if isinstance(cfg, dict) else {}
        self.enable = gate_cfg.get('enable', True)
        self.model_path = gate_cfg.get('model_path') or os.environ.get('MOE_GATE_MODEL_PATH')
        self.format = gate_cfg.get('format', 'auto')
        self.next_layer_offset = gate_cfg.get('next_layer_offset', 1)
        self.patterns = gate_cfg.get('patterns')
        self.total_layers = (
            gate_cfg.get('total_layers')
            or os.environ.get('MOE_GATE_TOTAL_LAYERS')
            or self._load_total_layers_from_config()
        )
        self.top_k = (
            gate_cfg.get('top_k')
            or os.environ.get('MOE_GATE_TOP_K')
            or self._load_topk_from_config()
        )
        self.device = gate_cfg.get('device') or os.environ.get('MOE_GATE_DEVICE')
        self._cache: Dict[int, Any] = {}
        self._preloaded_all = False

    def _load_total_layers_from_config(self) -> Optional[int]:
        if not self.model_path:
            return None
        cfg_file = os.path.join(self.model_path, 'config.json')
        if not os.path.exists(cfg_file):
            return None
        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            val = cfg.get('num_hidden_layers') or cfg.get('n_layers')
            if isinstance(val, int):
                _log_once('gate_total_layers', f'Auto-detected total_layers={val} from {cfg_file}')
                return val
        except Exception as e:  # pragma: no cover
            _append_log(f'Failed to read total_layers from {cfg_file}: {e}')
            _log_once('gate_total_layers_err', f'Failed to read total_layers from {cfg_file}: {e}')
        return None

    def _load_topk_from_config(self) -> Optional[int]:
        if not self.model_path:
            return None
        cfg_file = os.path.join(self.model_path, 'config.json')
        if not os.path.exists(cfg_file):
            return None
        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            val = cfg.get('num_experts_per_tok') or cfg.get('num_experts_per_token')
            if isinstance(val, int):
                _log_once('top_k', f'Auto-detected top_k={val} from {cfg_file}')
                return val
        except Exception as e:  # pragma: no cover
            _append_log(f'Failed to read top_k from {cfg_file}: {e}')
            _log_once('top_k_err', f'Failed to read top_k from {cfg_file}: {e}')
        return None

    def _iter_weight_files(self):
        if not self.model_path or not os.path.isdir(self.model_path):
            return []
        files = []
        for name in os.listdir(self.model_path):
            if name.endswith('.safetensors') or name.endswith('.bin'):
                files.append(os.path.join(self.model_path, name))
        return files

    def _resolve_from_index(self, layer_idx: int) -> Optional[Tuple[str, str]]:
        if not self.model_path or not os.path.isdir(self.model_path):
            return None
        patterns = self.patterns or [
            "model.layers.{idx}.mlp.gate.weight",
            "layers.{idx}.mlp.gate.weight",
            "model.layers.{idx}.gate.weight",
            "layers.{idx}.gate.weight",
        ]
        patterns = [p.format(idx=layer_idx) for p in patterns]
        _append_log(f"Resolving gate for layer {layer_idx} using patterns: {patterns}")
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
                _append_log(f"Resolve gate index failed for {idx_file} layer={layer_idx}: {e}")
                _log_once('gate_resolve_err', f"Resolve gate index failed for {idx_file}: {e}")
                continue
        return None

    def _target_device(self):
        try:
            import torch

            dev = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.device(dev)
        except Exception as e:
            _append_log(f"Resolve target device failed: {e}")
            _log_once('target_device_err', f"Resolve target device failed: {e}")
            return None

    def _load_linear(self, weight_file: str, weight_key: str, device=None):
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
                except Exception as e:  # pragma: no cover
                    _append_log(f'Failed to move gate to device {device}: {e}')
                    _log_once('gate_to_device_err', f'Failed to move gate to device {device}: {e}')
            return gate
        except Exception as e:  # pragma: no cover
            _append_log(f"Failed to load gate {weight_key} from {weight_file}: {e}")
            _log_once(f'gate_load_{weight_key}', f"Failed to load gate {weight_key} from {weight_file}: {e}")
            return None

    def _load_gate(self, layer_idx: int, device=None):
        if not self.enable:
            return None
        if layer_idx in self._cache:
            return self._cache[layer_idx]

        if self.format not in ('auto', 'hf'):
            _log_once('gate_format', f"Gate resolver format '{self.format}' not supported yet")
            return None

        resolved = self._resolve_from_index(layer_idx)
        if resolved is None:
            for weight_file in self._iter_weight_files():
                resolved = (None, weight_file)
                break
        if resolved is None:
            _log_once('gate_missing', 'No gate weight file found; set gate.model_path')
            return None

        weight_key, weight_file = resolved
        # 如果没有匹配到特定key，则尝试通配模式
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

    def _preload_all_gates(self):
        if self._preloaded_all:
            return
        total_layers = None
        try:
            total_layers = int(self.total_layers) if self.total_layers is not None else None
        except Exception as e:
            _append_log(f'Parse total_layers failed during preload: {e}')
            _log_once('gate_total_layers_parse_err', f'Parse total_layers failed during preload: {e}')
            total_layers = None
        if not total_layers or total_layers <= 0:
            return
        device = self._target_device()
        _log_once('gate_preload_start', f'Preloading {total_layers} gates to device {device}')
        for idx in range(total_layers):
            if idx in self._cache:
                continue
            try:
                self._load_gate(idx, device=device)
            except Exception as e:  # pragma: no cover
                _append_log(f'Preload gate {idx} failed: {e}')
                _log_once(f'gate_preload_err_{idx}', f'Preload gate {idx} failed: {e}')
        self._preloaded_all = True

    def resolve(self, layer_idx: int):
        if not self.enable:
            return None
        if not isinstance(layer_idx, int) or layer_idx < 0:
            return None
        total_layers = None
        try:
            total_layers = int(self.total_layers) if self.total_layers is not None else None
        except Exception as e:
            _append_log(f'Parse total_layers failed in resolve: {e}')
            _log_once('gate_total_layers_parse_err_resolve', f'Parse total_layers failed in resolve: {e}')
            total_layers = None

        # _append_log(f"total_layers={total_layers} for gate resolution")
        target_idx = layer_idx + self.next_layer_offset
        if total_layers and total_layers > 0:
            target_idx = target_idx % total_layers
        if target_idx in self._cache:
            return self._cache[target_idx]
        gate = self._load_gate(target_idx, device=self._target_device())
        if gate is not None:
            self._cache[target_idx] = gate
            _log_once(f'gate_{target_idx}', f'Loaded gate for layer {target_idx} from {self.model_path}')
        return gate

_gate_resolver: Optional[GateResolver] = None

def _infer_phase() -> str:
    """Best-effort phase inference (prefill/decode/etc.)."""
    phase: Optional[str] = None
    try:
        from sglang.srt.compilation.piecewise_context_manager import get_forward_context  # type: ignore

        ctx = get_forward_context()
        fb = getattr(ctx, 'forward_batch', None) if ctx is not None else None
        fm = None
        if fb is not None:
            fm = getattr(fb, 'global_forward_mode', None) or getattr(fb, 'forward_mode', None)

        if fm is not None:
            # Prefer semantic helpers if available
            if hasattr(fm, 'is_decode') and fm.is_decode():
                phase = 'decode'
            elif hasattr(fm, 'is_extend') and fm.is_extend(include_draft_extend_v2=True):
                phase = 'prefill'
            elif hasattr(fm, 'is_mixed') and fm.is_mixed():
                phase = 'mixed'
            elif hasattr(fm, 'is_target_verify') and fm.is_target_verify():
                phase = 'target_verify'
            elif hasattr(fm, 'is_prebuilt') and fm.is_prebuilt():
                phase = 'prebuilt'
            elif hasattr(fm, 'is_idle') and fm.is_idle():
                phase = 'idle'
            elif hasattr(fm, 'name'):
                phase = str(fm.name).lower()
    except Exception as e:
        _append_log(f'infer_phase forward context failed: {e}')
        _log_once('infer_phase_err', f'infer_phase forward context failed: {e}')
        phase = None

    if phase is None:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():  # type: ignore
                return 'cuda_graph_capture'
        except Exception as e:
            _append_log(f"cuda capture check failed: {e}")
            _log_once('infer_phase_capture_check_err', f"cuda capture check failed: {e}")

    return phase or 'unknown'


def _load_config() -> Dict[str, Any]:
    cfg_path = os.environ.get("MOE_HOOK_CONFIG")
    cfg: Dict[str, Any] = {}
    _log_once('cfg_path', f"MOE_HOOK_CONFIG={cfg_path}")
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
                _log_once('cfg_ext', f"Unsupported config extension: {cfg_path}")
        except Exception as e:  # pragma: no cover
            _append_log(f"Failed to load config {cfg_path}: {e}")
            _log_once('cfg_err', f"Failed to load config {cfg_path}: {e}")
                    
    cfg.setdefault('enable', bool(int(os.environ.get('MOE_HOOK_ENABLE', '1'))))
    cfg.setdefault('capture_bs', [1, 2, 4, 8])
    cfg.setdefault('preload', {'enable': True, 'mode': 'fate_shallow'})
    cfg.setdefault('predict', {'enable': True, 'mode': 'fate'})
    cfg.setdefault('prefetch', {'enable': True, 'mode': 'hook'})
    cfg.setdefault('gate', {
        'enable': True,
        'format': 'auto',  # auto|hf|gguf|custom
        'model_path': os.environ.get('MOE_GATE_MODEL_PATH'),
        'next_layer_offset': 1,  # fate需要下一层gate
        'total_layers': None,  # 可选：指定总层数用于取模
        'device': None,  # 可选：gate预加载目标设备，默认cuda可用则cuda否则cpu
        'patterns': None,  # 可自定义gate权重key匹配
    })
    cfg.setdefault('log_path', os.environ.get('MOE_HOOK_LOG_PATH'))
    log_path = cfg.get('log_path')
    if log_path:
        global _log_cleared
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            if not _log_cleared:
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"[MOE-HOOK] log reset at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                _log_cleared = True
            _log_once('log_path', f"Hook log file: {log_path}")
        except Exception as e:
            _append_log(f"Failed to reset log file {log_path}: {e}")
            _log_once('log_reset_err', f"Failed to reset log file {log_path}: {e}")
    return cfg


def _ensure_init():
    global _inited, _cfg, _gate_resolver
    if _inited:
        return
    with _init_lock:
        if _inited:
            return
        _cfg = _load_config()
        _gate_resolver = GateResolver(_cfg)
        if not _cfg.get('enable', True):
            _log_once('disabled', 'MOE hooks disabled by config/env')
        else:
            _log_once('enabled', 'MOE hooks enabled')

        # Optionally set capture batch sizes for KT CPU buffers (improves perf, no functional impact)
        try:
            from kt_kernel import KTMoEWrapper  # type: ignore
            _log_once('import_ktmoewrapper', 'KTMoEWrapper loaded from kt_kernel')
        except Exception as e_primary:
            try:
                from ktransformers.kt_kernel.python.experts import KTMoEWrapper  # type: ignore
                _log_once('import_ktmoewrapper_fallback', 'KTMoEWrapper loaded from ktransformers.kt_kernel.python')
            except Exception as e_fallback:
                KTMoEWrapper = None  # type: ignore
                _append_log(f"Import KTMoEWrapper failed: primary={e_primary}; fallback={e_fallback}")
                _log_once('import_ktmoewrapper_err', f"Import KTMoEWrapper failed: primary={e_primary}; fallback={e_fallback}")
        if KTMoEWrapper is not None:
            try:
                bs = _cfg.get('capture_bs') or []
                if isinstance(bs, list) and all(isinstance(x, int) for x in bs):
                    KTMoEWrapper.set_capture_batch_sizes(bs)
                    _log_once('capture_bs', f"Set KT capture batch sizes: {bs}")
            except Exception as e:  # pragma: no cover
                _append_log(f"Failed to set capture batch sizes: {e}")
                _log_once('capture_bs_err', f"Failed to set capture batch sizes: {e}")

        _inited = True


def _call_preloader(wrapper: Any, layer_idx: int) -> None:
    if not _cfg.get('preload', {}).get('enable', True):
        return
    if wrapper is None:
        return
    try:
        # Prefer user preloader.preload(wrapper, cfg)
        if _preloader and hasattr(_preloader, 'preload'):
            _preloader.preload(wrapper, _cfg)
            _log_once(f'preload_{layer_idx}', f"Preloaded experts for layer {layer_idx} via preloader.preload")
            return
        # Fallback: ModelPreloader with method warmup() or preload()
        if ModelPreloader is not None:
            if hasattr(ModelPreloader, 'preload'):
                ModelPreloader.preload(wrapper, _cfg)  # type: ignore
            elif hasattr(ModelPreloader, 'warmup'):
                ModelPreloader.warmup(wrapper, _cfg)  # type: ignore
            _log_once(f'preload_fallback_{layer_idx}', f"Preloaded experts for layer {layer_idx} via ModelPreloader")
    except Exception as e:  # pragma: no cover
        _append_log(f"Preloader failed for layer {layer_idx}: {e}")
        _log_once(f'preload_err_{layer_idx}', f"Preloader failed for layer {layer_idx}: {e}")


def _predict_and_prefetch(self: KTEPWrapperMethod, dispatch_output: Any) -> None:
    prefetch_cfg = _cfg.get('prefetch', {}) if isinstance(_cfg, dict) else {}
    if prefetch_cfg.get('mode') == 'external':
        # 外部预取器（例如用户自带 MoEPrefetcher）已包含预测与搬运，跳过hook侧逻辑
        return
    if not _cfg.get('predict', {}).get('enable', True) and not prefetch_cfg.get('enable', True):
        return
    try:
        import torch
        if torch.cuda.is_available():
            in_capture = False
            try:
                in_capture = torch.cuda.is_current_stream_capturing()
            except Exception as e:
                _append_log(f"is_current_stream_capturing failed: {e}")
                _log_once('capture_check_current_err', f"is_current_stream_capturing failed: {e}")
            # 尝试更宽的检测，可能是别的流在捕获
            try:
                in_capture = in_capture or torch.cuda._C._is_any_stream_capturing()  # type: ignore[attr-defined]
            except Exception as e:
                _append_log(f"_is_any_stream_capturing failed: {e}")
                _log_once('capture_check_any_err', f"_is_any_stream_capturing failed: {e}")
            phase = _infer_phase()
            if in_capture or phase == "cuda_graph_capture":
                _append_log(f"Skip predict/prefetch during CUDA graph capture phase={phase}")
                return
    except Exception as e:
        _append_log(f"capture guard failed: {e}")
        _log_once('predict_capture_guard_err', f"capture guard failed: {e}")
    try:
        # _append_log(f"dispatch_output: {dispatch_output}")
        topk_output = dispatch_output.topk_output
        # topk_output may be a namedtuple with fields (topk_weights, topk_ids, ...)
        topk_ids = getattr(topk_output, 'topk_ids', None)
        if topk_ids is None:
            topk_ids = topk_output[1]
        topk_weights = getattr(topk_output, 'topk_weights', None)
        if topk_weights is None:
            topk_weights = topk_output[0]

        # Gate解析：默认拉取下一层gate
        layer_idx: Optional[int] = getattr(self, 'kt_config', None) and getattr(self.kt_config, 'layer_idx', None)  # type: ignore
        gate = None
        if _gate_resolver is not None:
            gate = _gate_resolver.resolve(layer_idx=layer_idx if isinstance(layer_idx, int) else -1)

        log_enabled = bool(_cfg.get('log_path'))

        def _safe_shape(x):
            try:
                return getattr(x, 'shape', None)
            except Exception as e:
                _append_log(f'safe_shape failed: {e}')
                _log_once('safe_shape_err', f'safe_shape failed: {e}')
                return None

        def _safe_len(x):
            try:
                return len(x)  # type: ignore[arg-type]
            except Exception as e:
                _append_log(f'safe_len failed: {e}')
                _log_once('safe_len_err', f'safe_len failed: {e}')
                return None

        def _safe_numel(x):
            try:
                import torch

                if isinstance(x, torch.Tensor):
                    return x.numel()
            except Exception as e:
                _append_log(f'safe_numel failed: {e}')
                _log_once('safe_numel_err', f'safe_numel failed: {e}')
                return None
            return None

        phase = _infer_phase() if log_enabled else 'unknown'

        if log_enabled:
            _append_log(
                "predict_start "
                f"layer={layer_idx} "
                f"mode={_cfg.get('predict', {}).get('mode')} "
                f"phase={phase} "
                f"topk_ids_shape={_safe_shape(topk_ids)} "
                f"topk_ids_len={_safe_len(topk_ids)} "
                f"topk_ids_numel={_safe_numel(topk_ids)} "
                f"gate_loaded={gate is not None}"
            )

        predicted = None
        predict_enable = bool(_cfg.get('predict', {}).get('enable', True))
        predictor_present = _predictor is not None
        predictor_callable = predictor_present and hasattr(_predictor, 'predict')
        if log_enabled:
            _append_log(
                f"predict_block_guard enable={predict_enable} "
                f"predictor_present={predictor_present} predictor_callable={predictor_callable}"
            )

        if predict_enable and predictor_callable:
            def _call_predictor(fn, **kwargs):
                sig = inspect.signature(fn)
                usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return fn(**usable)

            predictor_kwargs = {
                'topk_ids': topk_ids,
                'topk_weights': topk_weights,
                'top_k': _gate_resolver.top_k if _gate_resolver is not None else None,
                'cfg': _cfg,
                'gate': gate,
                'layer_idx': layer_idx,
                'input_gate_i': getattr(dispatch_output, 'hidden_states', None),
                'mode': (_cfg.get('predict', {}).get('mode') if isinstance(_cfg, dict) else None),
                'dispatch_output': dispatch_output,
            }

            try:
                predicted = _call_predictor(_predictor.predict, **predictor_kwargs)
            except Exception as e:  # pragma: no cover
                _append_log(f"Predictor call failed: {e}")
                _log_once('predictor_call_err', f"Predictor call failed: {e}")
                predicted = None
            _append_log(f"predicted result: {predicted}")
        elif log_enabled:
            _append_log(
                f"predict_block_skipped enable={predict_enable} "
                f"predictor_present={predictor_present} predictor_callable={predictor_callable}"
            )
        if prefetch_cfg.get('enable', True):
            # 当前设计：hook 内部轻量预取，不做磁盘搬运
            _prefetch_stub(predicted, getattr(self, 'wrapper', None), _cfg)

        if log_enabled:
            _append_log(
                "predict_end "
                f"layer={layer_idx} "
                f"mode={_cfg.get('predict', {}).get('mode')} "
                f"phase={phase} "
                f"pred_type={(type(predicted).__name__ if predicted is not None else 'None')} "
                f"prefetch={prefetch_cfg.get('enable', True)}"
            )
    except Exception as e:  # pragma: no cover
        _append_log(f"predict/prefetch failed: {e}")
        _log_once('predict_prefetch_err', f"predict/prefetch failed: {e}")


# Patch: after weights loaded, run optional preloader
_orig_process = KTEPWrapperMethod.process_weights_after_loading


def _patched_process(self: KTEPWrapperMethod, layer) -> None:
    _ensure_init()
    _orig_process(self, layer)
    try:
        layer_idx: Optional[int] = getattr(self, 'kt_config', None) and getattr(self.kt_config, 'layer_idx', None)  # type: ignore
    except Exception as e:
        _append_log(f'Fetch layer_idx in patched_process failed: {e}')
        _log_once('patched_process_layer_idx_err', f'Fetch layer_idx failed: {e}')
        layer_idx = None
    _call_preloader(getattr(self, 'wrapper', None), layer_idx if isinstance(layer_idx, int) else -1)
    try:
        if _gate_resolver is not None:
            _gate_resolver._preload_all_gates()
    except Exception as e:  # pragma: no cover
        _append_log(f'Preload-all during process failed: {e}')
        _log_once('gate_preload_process_err', f'Preload-all during process failed: {e}')


# Patch: before delegating to original submit, do prediction/prefetch
_orig_submit = KTEPWrapperMethod.submit


def _patched_submit(self: KTEPWrapperMethod, layer, dispatch_output):
    _ensure_init()
    _predict_and_prefetch(self, dispatch_output)
    return _orig_submit(self, layer, dispatch_output)


# Install patches if enabled
if bool(int(os.environ.get('MOE_HOOK_ENABLE', '1'))):
    KTEPWrapperMethod.process_weights_after_loading = _patched_process  # type: ignore
    KTEPWrapperMethod.submit = _patched_submit  # type: ignore
    _log_once('install', 'Installed KTEPWrapperMethod hooks (process/submit)')
else:
    _log_once('skip_install', 'MOE_HOOK_ENABLE=0; skipping hook installation')
