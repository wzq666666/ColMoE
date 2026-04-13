"""
Predictor module for MOE inference.

This module is responsible ONLY for prediction - determining which experts
will be activated in future layers. The actual prefetching is handled by
the prefetcher module.

Prediction flow:
1. Get current layer's routing information (topk_ids, topk_weights)
2. Get next layer's gate weights (via gate_resolver)
3. Call predictor to predict next layer's expert activations
4. Return prediction result (list of (layer_idx, expert_idx) tuples)
"""

import inspect
from typing import Any, Dict, List, Optional, Tuple

from ..logger import log_once, append_log
from .phase_detector import infer_phase


# Try to import user's predictor and preloader modules
try:
    from ..component import predictor as _predictor  # type: ignore
    log_once('import_predictor', 'predictor loaded from component')
except Exception as e_primary:
    try:
        from moe_inference.src.component import predictor as _predictor  # type: ignore
        log_once('import_predictor_alt', 'predictor loaded from moe_inference.src.component')
    except Exception as e_fallback:
        _predictor = None  # type: ignore
        log_once('import_predictor_err', f"Import predictor failed: primary={e_primary}; fallback={e_fallback}")

try:
    from ..component import preloader as _preloader  # type: ignore
    log_once('import_preloader', 'preloader loaded from component')
except Exception as e_primary:
    try:
        from moe_inference.src.component import preloader as _preloader  # type: ignore
        log_once('import_preloader_alt', 'preloader loaded from moe_inference.src.component')
    except Exception as e_fallback:
        _preloader = None  # type: ignore
        log_once('import_preloader_err', f"Import preloader failed: primary={e_primary}; fallback={e_fallback}")


def call_preloader(wrapper: Any, layer_idx: int, cfg: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
    """
    Call user's preloader to get expert positions for initial preloading.
    
    This is called once during model initialization to warm up the cache.
    
    Args:
        wrapper: KT wrapper instance (can be None)
        layer_idx: Current layer index
        cfg: Configuration dict
        
    Returns:
        List of (layer_idx, expert_idx) tuples to preload, or None if disabled
    """
    if not cfg.get('preload', {}).get('enable', True):
        return None
    
    log_path = cfg.get('log_path')
    expert_positions = None
    
    try:
        # Call preloader.preload to get expert positions
        if _preloader and hasattr(_preloader, 'preload'):
            expert_positions = _preloader.preload(wrapper, cfg)
            if expert_positions:
                log_once(f'preload_{layer_idx}', 
                         f"Got {len(expert_positions)} expert positions for preload")
                if log_path:
                    append_log(f"Preload expert positions: {expert_positions[:10]}... (total: {len(expert_positions)})", log_path)
            return expert_positions
    except Exception as e:
        if log_path:
            append_log(f"Preloader failed for layer {layer_idx}: {e}", log_path)
        log_once(f'preload_err_{layer_idx}', f"Preloader failed for layer {layer_idx}: {e}")
    
    return expert_positions


def predict_experts(
    next_layer_idx: Any,
    cur_hidden_states: Any,
    cfg: Dict[str, Any],
    gate_resolver: Optional[Any] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    Predict which experts will be activated in the next layer.
    
    This function ONLY performs prediction and returns the result.
    
    Args:
        next_layer_idx: Index of the next layer
        cur_hidden_states: Current hidden states tensor
        cfg: Configuration dict
        gate_resolver: Optional GateResolver instance for next layer's gate
        
    Returns:
        List of (layer_idx, expert_idx) tuples predicted to be activated,
        or None if prediction is disabled/failed
    """
    log_path = cfg.get('log_path')
    
    # Check if prediction is enabled
    if not cfg.get('predict', {}).get('enable', True):
        if log_path:
            append_log("Skip prediction", log_path)
        return None

    # Guard against CUDA graph capture
    if _is_cuda_capturing(cfg):
        if log_path:
            append_log("Skip prediction during CUDA graph capture", log_path)
        return None

    try:
        # Resolve gate for next layer
        gate = None
        if gate_resolver is not None:
            gate = gate_resolver.resolve(tar_layer_idx=next_layer_idx)

        if log_path:
            phase = infer_phase(log_path)
            append_log(
                f"predict_start for layer={next_layer_idx} "
                f"mode={cfg.get('predict', {}).get('mode')} "
                f"phase={phase} "
                f"next_gate_loaded={gate is not None}",
                log_path,
                level=3
            )

        # Call predictor
        predicted = None

        if _predictor is not None and hasattr(_predictor, 'predict'):
            predictor_kwargs = {
                'top_k': gate_resolver.top_k if gate_resolver is not None else None,
                'cfg': cfg,
                'gate': gate,
                'input_gate_i': cur_hidden_states,
                'mode': cfg.get('predict', {}).get('mode'),
            }

            try:
                raw_result = _call_predictor(_predictor.predict, **predictor_kwargs)
                
                # if log_path:
                #     append_log(f"Predictor raw result type: {type(raw_result)}, value: {raw_result}", log_path)
                
                # 处理预测结果
                if raw_result is None:
                    predicted = None
                elif isinstance(raw_result, tuple) and len(raw_result) == 3:
                    # (topk_ids, topk_weights, routing_probs)
                    predicted = raw_result
                else:
                    # 其他格式（如 topk mode 直接返回 topk_ids）
                    if log_path:
                        append_log(f"Unexpected predictor result format: {type(raw_result)}", log_path)
                    predicted = None
                    
            except Exception as e:
                if log_path:
                    import traceback
                    append_log(f"Predictor call failed: {e}\n{traceback.format_exc()}", log_path, level=1)
                log_once('predictor_call_err', f"Predictor call failed: {e}")
                predicted = None
        
        return predicted

    except Exception as e:
        if log_path:
            append_log(f"predict_experts failed: {e}", log_path, level=1)
        log_once('predict_experts_err', f"predict_experts failed: {e}")
        return None


def predict_and_prefetch(
    wrapper_method: Any,
    dispatch_output: Any,
    cfg: Dict[str, Any],
    gate_resolver: Optional[Any] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    Predict experts and trigger prefetch.
    
    This is a convenience function that combines prediction and prefetching.
    For more control, use predict_experts() and prefetch_experts() separately.
    
    Args:
        wrapper_method: KTEPWrapperMethod instance
        dispatch_output: Dispatch output containing topk_ids, topk_weights, etc.
        cfg: Configuration dict
        gate_resolver: Optional GateResolver instance
        
    Returns:
        Prediction result (list of (layer_idx, expert_idx) tuples)
    """
    # Step 1: Predict
    predicted = predict_experts(wrapper_method, dispatch_output, cfg, gate_resolver)
    append_log(f"Predicted experts: {predicted}", cfg.get('log_path'))
    # Step 2: Prefetch (if enabled and we have predictions)
    prefetch_cfg = cfg.get('prefetch', {}) if isinstance(cfg, dict) else {}
    
    if prefetch_cfg.get('enable', True) and predicted:
        from ..scheduling.prefetcher import prefetch_experts
        async_mode = prefetch_cfg.get('async', True)
        prefetch_experts(predicted, async_mode=async_mode)
    
    return predicted


# ============================================================
# Helper functions
# ============================================================

def _is_cuda_capturing(cfg: Dict[str, Any]) -> bool:
    """Check if CUDA graph is being captured."""
    try:
        import torch
        if torch.cuda.is_available():
            in_capture = False
            try:
                in_capture = torch.cuda.is_current_stream_capturing()
            except Exception:
                pass
            
            try:
                in_capture = in_capture or torch.cuda._C._is_any_stream_capturing()
            except Exception:
                pass
            
            log_path = cfg.get('log_path')
            phase = infer_phase(log_path) if log_path else 'unknown'
            if in_capture or phase == "cuda_graph_capture":
                return True
    except Exception:
        pass
    
    return False


def _call_predictor(fn, **kwargs):
    """Call predictor with only the parameters it accepts."""
    sig = inspect.signature(fn)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**usable)


def _safe_shape(x):
    """Safely get tensor shape."""
    try:
        return getattr(x, 'shape', None)
    except Exception:
        return None
