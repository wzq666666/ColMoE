"""
Core components for MOE hook.

- model_config: HF model configuration parsing
- gate_resolver: Gate weight resolution
- expert_resolver: Expert weight loading from HF checkpoints
- expert_scheduler: Generic expert scheduling (backend-agnostic)
"""

from .model_config import HFModelConfig, HFConfigResolver, resolve_hf_config
from .gate_resolver import GateResolver
from .expert_resolver import ExpertResolver
from .expert_scheduler import (
    ExpertScheduler,
    LayerPlan,
    SchedulePhase,
    GPUStateProvider,
    get_scheduler,
    init_scheduler,
    reset_scheduler,
)
from .interchangeability import expert_interchangeability
from .routing_redirection import (
    token_level_gpu_preferred_reroute,
    merge_duplicate_experts_with_weights,
)

__all__ = [
    'HFModelConfig',
    'HFConfigResolver',
    'resolve_hf_config',
    'GateResolver',
    'ExpertResolver',
    # Scheduler
    'ExpertScheduler',
    'LayerPlan',
    'SchedulePhase',
    'GPUStateProvider',
    'get_scheduler',
    'init_scheduler',
    'reset_scheduler',
    'expert_interchangeability',
    'token_level_gpu_preferred_reroute',
    'merge_duplicate_experts',
]
