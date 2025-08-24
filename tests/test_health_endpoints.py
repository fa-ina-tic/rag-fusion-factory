"""Tests for health check endpoints."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.api.health import (
    basic_health_check, detailed_health_check, engine_health_check,
    metrics_endpoint, performance_summary, reset_metrics, readiness_check,
    liveness_check, setup_health_monitoring, periodic_health_check
)


class TestBasicHealthCheck:
    """Test cases for basic health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_basic_health_check_success(self):
        """Test successful basic health check."""
        result = await basic_health_check()
        
        assert result.status == "healthy"
        assert result.version == "1.0.0"
        assert result.uptime_seconds >= 0
        assert result.timestamp is not None
    
    @patch('src.api.health.datetime')
    @pytest.mark.asyncio
    async def test_basic_health_check_with_mocked_time(self, mock_datetime):
        """Test basic health check with mocked time."""
        mock_now = Mock()
        mock_now.isoformat.return_value = "2023-01-01T12:00:00"
        mock_datetime.now.return_value = mock_now
        
        result = await basic_health_check()
        
        assert result.timestamp == "2023-01-01T12:00:00"


class TestDetailedHealthCheck:
    """Test cases for detailed health check endpoint."""
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_performance_monitor')
    @patch('src.api.health.get_adapter_registry')
    @pytest.mark.asyncio
    async def test_detailed_health_check_success(self, mock_registry, mock_monitor, mock_factory):
        """Test successful detailed health check."""
        # Mock fusion factory
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={
            "engine1": True,
            "engine2": True
        })
        mock_fusion_factory.get_error_statistics.return_value = {
            "total_errors": 0,
            "error_rate": 0.0
        }
        mock_factory.return_value = mock_fusion_factory
        
        # Mock performance monitor
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_health_status.return_value = {
            "status": "healthy",
            "issues": [],
            "memory_mb": 256.0,
            "cpu_percent": 25.0
        }
        mock_perf_monitor.get_performance_summary.return_value = {
            "query_latency": {"mean": 0.5},
            "throughput": 100
        }
        mock_perf_monitor.wandb_integration.enabled = True
        mock_perf_monitor.prometheus_integration.enabled = True
        mock_monitor.return_value = mock_perf_monitor
        
        # Mock adapter registry
        mock_adapter_registry = Mock()
        mock_adapter_registry.get_registry_status.return_value = {
            "total_adapters": 2,
            "healthy_adapters": 2
        }
        mock_registry.return_value = mock_adapter_registry
        
        result = await detailed_health_check(mock_fusion_factory)
        
        assert result.status == "healthy"
        assert result.engines["healthy_count"] == 2
        assert result.engines["total_count"] == 2
        assert "components" in result.dict()
        assert "performance" in result.dict()
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_performance_monitor')
    @patch('src.api.health.get_adapter_registry')
    def test_detailed_health_check_degraded(self, mock_registry, mock_monitor, mock_factory):
        """Test detailed health check with degraded status."""
        # Mock fusion factory with some unhealthy engines
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={
            "engine1": True,
            "engine2": False,
            "engine3": False
        })
        mock_fusion_factory.get_error_statistics.return_value = {
            "total_errors": 5,
            "error_rate": 0.1
        }
        mock_factory.return_value = mock_fusion_factory
        
        # Mock performance monitor with issues
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_health_status.return_value = {
            "status": "degraded",
            "issues": ["high_error_rate"],
            "memory_mb": 1200.0,
            "cpu_percent": 85.0
        }
        mock_perf_monitor.get_performance_summary.return_value = {}
        mock_perf_monitor.wandb_integration.enabled = False
        mock_perf_monitor.prometheus_integration.enabled = False
        mock_monitor.return_value = mock_perf_monitor
        
        # Mock adapter registry
        mock_adapter_registry = Mock()
        mock_adapter_registry.get_registry_status.return_value = {
            "total_adapters": 3,
            "healthy_adapters": 1
        }
        mock_registry.return_value = mock_adapter_registry
        
        client = TestClient(app)
        
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "majority_engines_unhealthy" in data["issues"]
        assert data["engines"]["healthy_count"] == 1
        assert data["engines"]["total_count"] == 3
    
    @patch('src.api.health.get_fusion_factory')
    def test_detailed_health_check_error(self, mock_factory):
        """Test detailed health check with error."""
        mock_factory.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.get("/health/detailed")
        
        assert response.status_code == 503
        assert "Health check failed" in response.json()["detail"]


class TestEngineHealthCheck:
    """Test cases for engine health check endpoint."""
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_adapter_registry')
    def test_engine_health_check_success(self, mock_registry, mock_factory):
        """Test successful engine health check."""
        # Mock fusion factory
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={
            "engine1": True,
            "engine2": True,
            "engine3": False
        })
        mock_factory.return_value = mock_fusion_factory
        
        # Mock adapter registry
        mock_adapter1 = Mock()
        mock_adapter1.engine_type = "elasticsearch"
        mock_adapter1.get_config.return_value = {"host": "localhost"}
        
        mock_adapter2 = Mock()
        mock_adapter2.engine_type = "solr"
        mock_adapter2.get_config.return_value = {"url": "http://localhost:8983"}
        
        mock_adapter_registry = Mock()
        mock_adapter_registry.get_all_adapters.return_value = {
            "engine1": mock_adapter1,
            "engine2": mock_adapter2,
            "engine3": None
        }
        mock_registry.return_value = mock_adapter_registry
        
        client = TestClient(app)
        
        response = client.get("/health/engines")
        
        assert response.status_code == 200
        data = response.json()
        assert data["healthy_count"] == 2
        assert data["total_count"] == 3
        assert data["overall_status"] == "partially_healthy"
        assert "engine1" in data["engines"]
        assert data["engines"]["engine1"]["healthy"] is True
        assert data["engines"]["engine1"]["type"] == "elasticsearch"
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_adapter_registry')
    def test_engine_health_check_no_engines(self, mock_registry, mock_factory):
        """Test engine health check with no engines."""
        # Mock fusion factory
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={})
        mock_factory.return_value = mock_fusion_factory
        
        # Mock adapter registry
        mock_adapter_registry = Mock()
        mock_adapter_registry.get_all_adapters.return_value = {}
        mock_registry.return_value = mock_adapter_registry
        
        client = TestClient(app)
        
        response = client.get("/health/engines")
        
        assert response.status_code == 200
        data = response.json()
        assert data["healthy_count"] == 0
        assert data["total_count"] == 0
        assert data["overall_status"] == "no_engines"
    
    @patch('src.api.health.get_fusion_factory')
    def test_engine_health_check_error(self, mock_factory):
        """Test engine health check with error."""
        mock_factory.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.get("/health/engines")
        
        assert response.status_code == 503
        assert "Engine health check failed" in response.json()["detail"]


class TestMetricsEndpoint:
    """Test cases for metrics endpoint."""
    
    @patch('src.api.health.get_performance_monitor')
    def test_metrics_endpoint_success(self, mock_monitor):
        """Test successful metrics endpoint."""
        mock_perf_monitor = Mock()
        mock_perf_monitor.prometheus_integration.get_metrics.return_value = (
            "# HELP test_metric Test metric\n"
            "# TYPE test_metric counter\n"
            "test_metric 42\n"
        )
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.get("/health/metrics")
        
        assert response.status_code == 200
        assert "test_metric 42" in response.text
    
    @patch('src.api.health.get_performance_monitor')
    def test_metrics_endpoint_unavailable(self, mock_monitor):
        """Test metrics endpoint when unavailable."""
        mock_perf_monitor = Mock()
        mock_perf_monitor.prometheus_integration.get_metrics.return_value = ""
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.get("/health/metrics")
        
        assert response.status_code == 503
        assert "Prometheus metrics not available" in response.json()["detail"]
    
    @patch('src.api.health.get_performance_monitor')
    def test_metrics_endpoint_error(self, mock_monitor):
        """Test metrics endpoint with error."""
        mock_monitor.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.get("/health/metrics")
        
        assert response.status_code == 503
        assert "Metrics unavailable" in response.json()["detail"]


class TestPerformanceSummary:
    """Test cases for performance summary endpoint."""
    
    @patch('src.api.health.get_performance_monitor')
    def test_performance_summary_success(self, mock_monitor):
        """Test successful performance summary."""
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_performance_summary.return_value = {
            "query_latency": {"mean": 0.5, "p95": 1.0},
            "throughput": 100,
            "error_rate": 0.01
        }
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.get("/health/performance?window_minutes=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "query_latency" in data
        assert data["throughput"] == 100
        mock_perf_monitor.get_performance_summary.assert_called_with(10)
    
    @patch('src.api.health.get_performance_monitor')
    def test_performance_summary_default_window(self, mock_monitor):
        """Test performance summary with default window."""
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_performance_summary.return_value = {}
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.get("/health/performance")
        
        assert response.status_code == 200
        mock_perf_monitor.get_performance_summary.assert_called_with(5)
    
    @patch('src.api.health.get_performance_monitor')
    def test_performance_summary_error(self, mock_monitor):
        """Test performance summary with error."""
        mock_monitor.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.get("/health/performance")
        
        assert response.status_code == 500
        assert "Performance summary failed" in response.json()["detail"]


class TestResetMetrics:
    """Test cases for reset metrics endpoint."""
    
    @patch('src.api.health.get_performance_monitor')
    def test_reset_metrics_success(self, mock_monitor):
        """Test successful metrics reset."""
        mock_perf_monitor = Mock()
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.post("/health/reset-metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Performance metrics reset" in data["message"]
        assert "timestamp" in data
        mock_perf_monitor.reset_metrics.assert_called_once()
    
    @patch('src.api.health.get_performance_monitor')
    def test_reset_metrics_error(self, mock_monitor):
        """Test metrics reset with error."""
        mock_monitor.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.post("/health/reset-metrics")
        
        assert response.status_code == 500
        assert "Metrics reset failed" in response.json()["detail"]


class TestReadinessCheck:
    """Test cases for readiness check endpoint."""
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_performance_monitor')
    def test_readiness_check_ready(self, mock_monitor, mock_factory):
        """Test readiness check when ready."""
        # Mock fusion factory
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={
            "engine1": True,
            "engine2": True
        })
        mock_factory.return_value = mock_fusion_factory
        
        # Mock performance monitor
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_health_status.return_value = {"status": "healthy"}
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.get("/health/readiness")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["healthy_engines"] == 2
        assert data["total_engines"] == 2
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_performance_monitor')
    def test_readiness_check_not_ready(self, mock_monitor, mock_factory):
        """Test readiness check when not ready."""
        # Mock fusion factory with no healthy engines
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={
            "engine1": False,
            "engine2": False
        })
        mock_factory.return_value = mock_fusion_factory
        
        # Mock performance monitor
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_health_status.return_value = {"status": "degraded"}
        mock_monitor.return_value = mock_perf_monitor
        
        client = TestClient(app)
        
        response = client.get("/health/readiness")
        
        assert response.status_code == 503
        assert "No healthy engines available" in response.json()["detail"]
    
    @patch('src.api.health.get_fusion_factory')
    def test_readiness_check_error(self, mock_factory):
        """Test readiness check with error."""
        mock_factory.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.get("/health/readiness")
        
        assert response.status_code == 503
        assert "Service not ready" in response.json()["detail"]


class TestLivenessCheck:
    """Test cases for liveness check endpoint."""
    
    def test_liveness_check_success(self):
        """Test successful liveness check."""
        client = TestClient(app)
        
        response = client.get("/health/liveness")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert data["uptime_seconds"] >= 0
    
    @patch('src.api.health.datetime')
    def test_liveness_check_error(self, mock_datetime):
        """Test liveness check with error."""
        mock_datetime.now.side_effect = Exception("Test error")
        
        client = TestClient(app)
        
        response = client.get("/health/liveness")
        
        assert response.status_code == 503
        assert "Service not alive" in response.json()["detail"]


class TestHealthMonitoringSetup:
    """Test cases for health monitoring setup."""
    
    @patch('src.utils.monitoring.start_monitoring')
    @patch('src.utils.monitoring.stop_monitoring')
    def test_setup_health_monitoring(self, mock_stop, mock_start):
        """Test health monitoring setup."""
        test_app = FastAPI()
        
        setup_health_monitoring(test_app, interval_seconds=60)
        
        # Test that startup and shutdown events are registered
        assert len(test_app.router.on_startup) > 0
        assert len(test_app.router.on_shutdown) > 0
    
    @pytest.mark.asyncio
    @patch('src.api.health.basic_health_check')
    async def test_periodic_health_check(self, mock_health_check):
        """Test periodic health check function."""
        from src.api.health import periodic_health_check
        
        mock_health_check.return_value = Mock(status="healthy")
        
        # Create a task that will run briefly
        task = asyncio.create_task(periodic_health_check(0.1))
        
        # Let it run for a short time
        await asyncio.sleep(0.25)
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify health check was called
        assert mock_health_check.call_count >= 1


class TestIntegrationScenarios:
    """Integration test scenarios for health endpoints."""
    
    @patch('src.api.health.get_fusion_factory')
    @patch('src.api.health.get_performance_monitor')
    @patch('src.api.health.get_adapter_registry')
    def test_complete_health_check_workflow(self, mock_registry, mock_monitor, mock_factory):
        """Test complete health check workflow."""
        # Setup mocks for healthy system
        mock_fusion_factory = Mock()
        mock_fusion_factory.health_check_engines = AsyncMock(return_value={
            "engine1": True,
            "engine2": True
        })
        mock_fusion_factory.get_error_statistics.return_value = {"total_errors": 0}
        mock_factory.return_value = mock_fusion_factory
        
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_health_status.return_value = {
            "status": "healthy",
            "issues": []
        }
        mock_perf_monitor.get_performance_summary.return_value = {
            "query_latency": {"mean": 0.3}
        }
        mock_perf_monitor.prometheus_integration.get_metrics.return_value = "test_metric 1"
        mock_perf_monitor.wandb_integration.enabled = True
        mock_perf_monitor.prometheus_integration.enabled = True
        mock_monitor.return_value = mock_perf_monitor
        
        mock_adapter_registry = Mock()
        mock_adapter_registry.get_registry_status.return_value = {"total_adapters": 2}
        mock_adapter_registry.get_all_adapters.return_value = {}
        mock_registry.return_value = mock_adapter_registry
        
        client = TestClient(app)
        
        # Test all health endpoints
        basic_response = client.get("/health/")
        assert basic_response.status_code == 200
        
        detailed_response = client.get("/health/detailed")
        assert detailed_response.status_code == 200
        
        engines_response = client.get("/health/engines")
        assert engines_response.status_code == 200
        
        metrics_response = client.get("/health/metrics")
        assert metrics_response.status_code == 200
        
        performance_response = client.get("/health/performance")
        assert performance_response.status_code == 200
        
        readiness_response = client.get("/health/readiness")
        assert readiness_response.status_code == 200
        
        liveness_response = client.get("/health/liveness")
        assert liveness_response.status_code == 200
        
        # Verify all responses are healthy
        assert basic_response.json()["status"] == "healthy"
        assert detailed_response.json()["status"] == "healthy"
        assert readiness_response.json()["status"] == "ready"
        assert liveness_response.json()["status"] == "alive"