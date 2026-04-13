"""
Expert prediction components.

- predictor: FATE-style expert access prediction
- phase_detector: Prefill/decode phase detection
"""

from .predictor import predict_experts, predict_and_prefetch, call_preloader
from .phase_detector import infer_phase

__all__ = [
    'predict_experts',
    'predict_and_prefetch',
    'call_preloader',
    'infer_phase',
]
