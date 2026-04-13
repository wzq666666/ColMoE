"""
MoE Hook API endpoints for dynamic strategy switching.

Adds custom routes to sglang FastAPI app for runtime strategy management.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import our scheduler and config types
from .core.expert_scheduler import RerouteConfig, get_scheduler

logger = logging.getLogger(__name__)


class StrategyRequest(BaseModel):
    """Request to switch routing strategy."""
    strategy: str
    alpha: Optional[float] = None
    allow_duplicate: Optional[bool] = None
    use_limited_reroute: Optional[bool] = None
    max_duplicates_per_expert: Optional[int] = None
    min_unique_experts: Optional[int] = None
    score_threshold_ratio: Optional[float] = None
    max_gpu_duplicates: Optional[int] = None
    dominance_threshold: Optional[float] = None


class StrategyResponse(BaseModel):
    """Response after switching strategy."""
    status: str
    message: str
    current_config: Dict[str, Any]


# Create router for moe_hook endpoints
moe_router = APIRouter(prefix="/moe_hook", tags=["moe_hook"])


@moe_router.post("/switch_strategy", response_model=StrategyResponse)
async def switch_strategy(request: StrategyRequest):
    """
    Dynamically switch MoE routing strategy without restarting server.
    
    Args:
        request: Strategy configuration to apply
        
    Returns:
        StrategyResponse with status and current config
        
    Example:
        POST /moe_hook/switch_strategy
        {
            "strategy": "io_free",
            "alpha": 0.05,
            "score_threshold_ratio": 0.5
        }
    """
    try:
        # Get current scheduler instance
        scheduler = get_scheduler()
        if scheduler is None:
            raise HTTPException(status_code=500, detail="MoE scheduler not initialized")
        
        # Get current config as base
        current_config = scheduler.reroute_config
        
        # Create new config with updated values
        new_config = RerouteConfig(
            strategy=request.strategy,
            alpha=request.alpha if request.alpha is not None else current_config.alpha,
            allow_duplicate=request.allow_duplicate if request.allow_duplicate is not None else current_config.allow_duplicate,
            use_limited_reroute=request.use_limited_reroute if request.use_limited_reroute is not None else current_config.use_limited_reroute,
            max_duplicates_per_expert=request.max_duplicates_per_expert if request.max_duplicates_per_expert is not None else current_config.max_duplicates_per_expert,
            min_unique_experts=request.min_unique_experts if request.min_unique_experts is not None else current_config.min_unique_experts,
            score_threshold_ratio=request.score_threshold_ratio if request.score_threshold_ratio is not None else current_config.score_threshold_ratio,
            max_gpu_duplicates=request.max_gpu_duplicates if request.max_gpu_duplicates is not None else current_config.max_gpu_duplicates,
            dominance_threshold=request.dominance_threshold if request.dominance_threshold is not None else current_config.dominance_threshold,
        )
        
        # Apply new config to scheduler
        scheduler.set_reroute_config(new_config)
        
        logger.info(f"MoE strategy switched to: {request.strategy}")
        
        return StrategyResponse(
            status="success",
            message=f"Successfully switched to strategy: {request.strategy}",
            current_config={
                "strategy": new_config.strategy,
                "alpha": new_config.alpha,
                "allow_duplicate": new_config.allow_duplicate,
                "use_limited_reroute": new_config.use_limited_reroute,
                "max_duplicates_per_expert": new_config.max_duplicates_per_expert,
                "min_unique_experts": new_config.min_unique_experts,
                "score_threshold_ratio": new_config.score_threshold_ratio,
                "max_gpu_duplicates": new_config.max_gpu_duplicates,
                "dominance_threshold": new_config.dominance_threshold,
            }
        )
        
    except ValueError as e:
        logger.error(f"Invalid strategy configuration: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to switch strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@moe_router.get("/current_strategy", response_model=StrategyResponse)
async def get_current_strategy():
    """
    Get current MoE routing strategy configuration.
    
    Returns:
        StrategyResponse with current config
    """
    try:
        scheduler = get_scheduler()
        if scheduler is None:
            raise HTTPException(status_code=500, detail="MoE scheduler not initialized")
        
        current_config = scheduler.reroute_config
        
        return StrategyResponse(
            status="success",
            message="Current strategy retrieved",
            current_config={
                "strategy": current_config.strategy,
                "alpha": current_config.alpha,
                "allow_duplicate": current_config.allow_duplicate,
                "use_limited_reroute": current_config.use_limited_reroute,
                "max_duplicates_per_expert": current_config.max_duplicates_per_expert,
                "min_unique_experts": current_config.min_unique_experts,
                "score_threshold_ratio": current_config.score_threshold_ratio,
                "max_gpu_duplicates": current_config.max_gpu_duplicates,
                "dominance_threshold": current_config.dominance_threshold,
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get current strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def install_moe_api_routes(app):
    """Install MoE Hook API routes to FastAPI app."""
    app.include_router(moe_router)
    logger.info("MoE Hook API routes installed: /moe_hook/switch_strategy, /moe_hook/current_strategy")