"""Health check endpoints for RAG Fusion Factory API."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..services.fusion_factory import FusionFactory
from ..adapters.registry import get_adapter_registry
from ..utils.monitoring import get_performance_monitor
from ..utils.logging import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None


class DetailedHealthStatus(BaseModel):
    """Detailed health status response model."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    components: Dict[str, Any]
    engines: Dict[str, Any]
    performance: Dict[str, Any]
    issues: List[str] = []


class EngineHealthStatus(BaseModel):
    """Engine health status response model."""
    engines: Dict[str, Dict[str, Any]]
    healthy_count: int
    total_count: int
    overall_status: str


# Track application start time for uptime calculation
_app_start_time = datetime.now()


def get_fusion_factory() -> FusionFactory:
    """Dependency to get FusionFactory instance."""
    # This would typically be injected from the main application
    # For now, we'll create a basic instance
    from ..services.fusion_factory import FusionFactory
    return FusionFactory()


@router.get("/", response_model=HealthStatus)
async def basic_health_check() -> HealthStatus:
    """Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    try:
        uptime = (datetime.now() - _app_start_time).total_seconds()
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(
    fusion_factory: FusionFactory = Depends(get_fusion_factory)
) -> DetailedHealthStatus:
    """Detailed health check endpoint.
    
    Args:
        fusion_factory: FusionFactory instance
        
    Returns:
        Detailed health status including all components
    """
    try:
        uptime = (datetime.now() - _app_start_time).total_seconds()
        issues = []
        
        # Check performance monitor
        performance_monitor = get_performance_monitor()
        performance_status = performance_monitor.get_health_status()
        
        if performance_status.get("status") != "healthy":
            issues.extend(performance_status.get("issues", []))
        
        # Check adapter registry
        adapter_registry = get_adapter_registry()
        registry_status = adapter_registry.get_registry_status()
        
        # Check engines
        engine_health = await fusion_factory.health_check_engines()
        healthy_engines = sum(1 for status in engine_health.values() if status)
        total_engines = len(engine_health)
        
        if total_engines > 0 and healthy_engines / total_engines < 0.5:
            issues.append("majority_engines_unhealthy")
        
        # Get error statistics
        error_stats = fusion_factory.get_error_statistics()
        
        # Determine overall status
        overall_status = "healthy"
        if issues:
            overall_status = "degraded"
        if not engine_health or healthy_engines == 0:
            overall_status = "unhealthy"
        
        return DetailedHealthStatus(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            issues=issues,
            components={
                "fusion_factory": {
                    "status": "healthy",
                    "error_statistics": error_stats
                },
                "adapter_registry": {
                    "status": "healthy" if registry_status.get("total_adapters", 0) > 0 else "no_adapters",
                    "details": registry_status
                },
                "performance_monitor": performance_status,
                "monitoring_components": {
                    "wandb": performance_monitor.wandb_integration.enabled,
                    "prometheus": performance_monitor.prometheus_integration.enabled,
                    "metrics_collector": True
                }
            },
            engines={
                "health_status": engine_health,
                "healthy_count": healthy_engines,
                "total_count": total_engines,
                "overall_status": "healthy" if healthy_engines == total_engines else "degraded"
            },
            performance=performance_monitor.get_performance_summary(window_minutes=5)
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@router.get("/engines", response_model=EngineHealthStatus)
async def engine_health_check(
    fusion_factory: FusionFactory = Depends(get_fusion_factory)
) -> EngineHealthStatus:
    """Engine-specific health check endpoint.
    
    Args:
        fusion_factory: FusionFactory instance
        
    Returns:
        Engine health status
    """
    try:
        # Get engine health status
        engine_health = await fusion_factory.health_check_engines()
        
        # Get detailed engine information
        adapter_registry = get_adapter_registry()
        all_adapters = adapter_registry.get_all_adapters()
        
        detailed_engines = {}
        for engine_id, is_healthy in engine_health.items():
            adapter = all_adapters.get(engine_id)
            detailed_engines[engine_id] = {
                "healthy": is_healthy,
                "type": adapter.engine_type if adapter else "unknown",
                "last_check": datetime.now().isoformat(),
                "config": adapter.get_config() if adapter and hasattr(adapter, 'get_config') else {}
            }
        
        healthy_count = sum(1 for status in engine_health.values() if status)
        total_count = len(engine_health)
        
        overall_status = "healthy"
        if total_count == 0:
            overall_status = "no_engines"
        elif healthy_count == 0:
            overall_status = "all_unhealthy"
        elif healthy_count < total_count:
            overall_status = "partially_healthy"
        
        return EngineHealthStatus(
            engines=detailed_engines,
            healthy_count=healthy_count,
            total_count=total_count,
            overall_status=overall_status
        )
        
    except Exception as e:
        logger.error(f"Engine health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Engine health check failed: {str(e)}")


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in text format
    """
    try:
        performance_monitor = get_performance_monitor()
        metrics_text = performance_monitor.prometheus_integration.get_metrics()
        
        if not metrics_text:
            raise HTTPException(status_code=503, detail="Prometheus metrics not available")
        
        return metrics_text
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=503, detail=f"Metrics unavailable: {str(e)}")


@router.get("/performance")
async def performance_summary(window_minutes: int = 5) -> Dict[str, Any]:
    """Get performance summary.
    
    Args:
        window_minutes: Time window for metrics (default: 5 minutes)
        
    Returns:
        Performance summary
    """
    try:
        performance_monitor = get_performance_monitor()
        return performance_monitor.get_performance_summary(window_minutes)
        
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {str(e)}")


@router.post("/reset-metrics")
async def reset_metrics() -> Dict[str, str]:
    """Reset performance metrics.
    
    Returns:
        Reset confirmation
    """
    try:
        performance_monitor = get_performance_monitor()
        performance_monitor.reset_metrics()
        
        return {
            "status": "success",
            "message": "Performance metrics reset",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics reset failed: {str(e)}")


@router.get("/readiness")
async def readiness_check(
    fusion_factory: FusionFactory = Depends(get_fusion_factory)
) -> Dict[str, Any]:
    """Readiness check for Kubernetes deployments.
    
    Args:
        fusion_factory: FusionFactory instance
        
    Returns:
        Readiness status
    """
    try:
        # Check if at least one engine is healthy
        engine_health = await fusion_factory.health_check_engines()
        healthy_engines = sum(1 for status in engine_health.values() if status)
        
        if healthy_engines == 0:
            raise HTTPException(status_code=503, detail="No healthy engines available")
        
        # Check if monitoring is working
        performance_monitor = get_performance_monitor()
        monitor_status = performance_monitor.get_health_status()
        
        return {
            "status": "ready",
            "healthy_engines": healthy_engines,
            "total_engines": len(engine_health),
            "monitoring_status": monitor_status.get("status"),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for Kubernetes deployments.
    
    Returns:
        Liveness status
    """
    try:
        # Basic liveness check - just verify the service is responding
        uptime = (datetime.now() - _app_start_time).total_seconds()
        
        return {
            "status": "alive",
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service not alive: {str(e)}")


# Utility function to run health checks periodically
async def periodic_health_check(interval_seconds: int = 30) -> None:
    """Run periodic health checks in the background.
    
    Args:
        interval_seconds: Interval between health checks
    """
    logger.info(f"Starting periodic health checks every {interval_seconds} seconds")
    
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            
            # Perform basic health check
            health_status = await basic_health_check()
            
            # Log health status
            if health_status.status != "healthy":
                logger.warning(f"Health check status: {health_status.status}")
            else:
                logger.debug(f"Health check passed: {health_status.status}")
                
        except asyncio.CancelledError:
            logger.info("Periodic health checks cancelled")
            break
        except Exception as e:
            logger.error(f"Periodic health check failed: {e}")


def setup_health_monitoring(app, interval_seconds: int = 30) -> None:
    """Set up health monitoring for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        interval_seconds: Interval for periodic health checks
    """
    @app.on_event("startup")
    async def start_health_monitoring():
        """Start health monitoring on application startup."""
        # Start periodic health checks
        asyncio.create_task(periodic_health_check(interval_seconds))
        
        # Start performance monitoring
        from ..utils.monitoring import start_monitoring
        await start_monitoring()
        
        logger.info("Health monitoring started")
    
    @app.on_event("shutdown")
    async def stop_health_monitoring():
        """Stop health monitoring on application shutdown."""
        from ..utils.monitoring import stop_monitoring
        await stop_monitoring()
        
        logger.info("Health monitoring stopped")