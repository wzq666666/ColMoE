"""
Native Scheduler Adapter - native backend 对通用调度器的适配.

这是一个轻量级适配层，将通用的 ExpertScheduler 与 native backend 集成：
1. 实现 GPUStateProvider 接口，连接 NativeGPUCacheManager
2. 提供便捷的初始化函数

核心调度逻辑在 core/expert_scheduler.py 中，本模块只做适配。
"""

from typing import Optional, Set, Callable

from ..core.expert_scheduler import (
    ExpertScheduler,
    GPUStateProvider,
    LayerPlan,
    RerouteConfig,
    SchedulePhase,
    get_scheduler,
    init_scheduler,
    reset_scheduler,
)
from ..logger import append_log
from .native_gpu_cache import get_native_cache, NativeGPUCacheManager


class NativeGPUStateProvider(GPUStateProvider):
    """
    Native backend 的 GPU 状态提供者.
    
    实现 GPUStateProvider 接口，从 NativeGPUCacheManager 查询状态。
    """
    
    def __init__(self, cache: Optional[NativeGPUCacheManager] = None):
        self._cache = cache
    
    def get_gpu_experts(self, layer_idx: int) -> Set[int]:
        """获取指定层当前在 GPU 上的专家."""
        cache = self._cache or get_native_cache()
        if cache is None:
            return set()
        return cache.get_gpu_experts(layer_idx)
    
    def get_num_gpu_slots(self, layer_idx: int) -> int:
        """获取指定层的 GPU 槽位数."""
        cache = self._cache or get_native_cache()
        if cache is None:
            return 0
        return cache.num_gpu_slots


def init_native_scheduler(
    num_layers: int,
    num_experts: int,
    num_gpu_slots: int,
    log_path: Optional[str] = None,
    cache: Optional[NativeGPUCacheManager] = None,
    reroute_config: Optional[RerouteConfig] = None,
) -> ExpertScheduler:
    """
    初始化 native backend 的调度器.
    
    这是一个便捷函数，创建 GPU 状态提供者并初始化通用调度器。
    
    Args:
        num_layers: MoE 层数
        num_experts: 每层专家数  
        num_gpu_slots: 每层 GPU 槽位数
        log_path: 日志路径（可选）
        cache: GPU 缓存管理器（可选，默认使用全局实例）
        reroute_config: 重路由策略配置（可选）
        
    Returns:
        初始化的 ExpertScheduler 实例
    """
    # 创建 GPU 状态提供者
    gpu_state = NativeGPUStateProvider(cache)
    
    # 创建日志函数
    log_fn = None
    if log_path:
        def log_fn(msg: str):
            append_log(msg, log_path, level=3)
    
    # 初始化通用调度器
    return init_scheduler(
        num_layers=num_layers,
        num_experts=num_experts,
        num_gpu_slots=num_gpu_slots,
        gpu_state_provider=gpu_state,
        log_fn=log_fn,
        reroute_config=reroute_config,
    )


def get_native_scheduler() -> Optional[ExpertScheduler]:
    """获取调度器实例（兼容旧接口）."""
    return get_scheduler()


# 重新导出，方便使用
__all__ = [
    # 核心类型（从 core 导出）
    'ExpertScheduler',
    'LayerPlan', 
    'SchedulePhase',
    'GPUStateProvider',
    # native 适配
    'NativeGPUStateProvider',
    'init_native_scheduler',
    'get_native_scheduler',
    # 通用接口
    'get_scheduler',
    'init_scheduler',
    'reset_scheduler',
]

