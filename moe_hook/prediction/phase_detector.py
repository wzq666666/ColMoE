"""
Phase detection utilities for MOE inference.
"""

from typing import Optional

from ..logger import log_once, append_log


def infer_phase(log_path: Optional[str] = None) -> str:
    """
    Best-effort inference of current execution phase (prefill/decode/etc.).
    
    Returns:
        Phase string: 'prefill', 'decode', 'mixed', 'cuda_graph_capture', 'unknown', etc.
    """
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
        if log_path:
            append_log(f'infer_phase forward context failed: {e}', log_path)
        log_once('infer_phase_err', f'infer_phase forward context failed: {e}')
        phase = None

    # Fallback: check CUDA graph capture
    if phase is None:
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():  # type: ignore
                return 'cuda_graph_capture'
        except Exception as e:
            if log_path:
                append_log(f"cuda capture check failed: {e}", log_path)
            log_once('infer_phase_capture_check_err', f"cuda capture check failed: {e}")

    return phase or 'unknown'
