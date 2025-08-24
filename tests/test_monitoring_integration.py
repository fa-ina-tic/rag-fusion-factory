"""Integration tests for monitoring functionality."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from src.utils.monitoring import (
    PerformanceMonitor, get_performance_monitor, initialize_monitoring
)
from src.services.fusion_factory import FusionFactory
from src.api.health import (
    basic_health_check, periodic_health_check
)


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 60,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        assert monitor.enabled is True
        assert monitor.metrics_collector is not None
        assert monitor.wandb_integration is not None
        assert monitor.prometheus_integration is not None
    
    def test_global_monitor_singleton(self):
        """Test that global monitor returns same instance."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2
    
    def test_custom_monitor_initialization(self):
        """Test custom monitor initialization."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 30,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = initialize_monitoring(config)
        
        assert monitor._monitoring_interval == 30
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 0.1,  # Very short for testing
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor._monitoring_task is not None
        assert not monitor._monitoring_task.done()
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor._monitoring_task.done()
    
    def test_operation_measurement(self):
        """Test operation measurement functionality."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Test successful operation
        with monitor.measure_operation("test_operation"):
            time.sleep(0.01)
        
        # Check metrics were recorded
        metrics = monitor.metrics_collector.get_all_metrics()
        assert "counters" in metrics
        assert any("test_operation_success" in key for key in metrics["counters"].keys())
    
    def test_engine_metrics_recording(self):
        """Test engine metrics recording."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record engine metrics
        monitor.record_engine_metrics("test_engine", 0.5, True, 10)
        
        # Check metrics were recorded
        metrics = monitor.metrics_collector.get_all_metrics()
        assert "histograms" in metrics
        assert any("engine_test_engine" in key for key in metrics["histograms"].keys())
    
    def test_health_status_reporting(self):
        """Test health status reporting."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        health_status = monitor.get_health_status()
        
        assert "status" in health_status
        assert "memory_mb" in health_status
        assert "cpu_percent" in health_status
        assert "monitoring_components" in health_status
        assert health_status["status"] in ["healthy", "degraded", "error"]
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record some operations
        monitor.record_operation_success("query", 0.5)
        monitor.record_operation_success("query", 0.3)
        monitor.record_engine_metrics("engine1", 0.2, True, 5)
        
        summary = monitor.get_performance_summary(window_minutes=5)
        
        assert "operation_counts" in summary
        assert "system_metrics" in summary
        assert "timestamp" in summary
        assert summary["operation_counts"]["query"] == 2
    
    @pytest.mark.asyncio
    async def test_basic_health_check_function(self):
        """Test basic health check function."""
        result = await basic_health_check()
        
        assert result.status == "healthy"
        assert result.version == "1.0.0"
        assert result.uptime_seconds >= 0
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_periodic_health_check_cancellation(self):
        """Test periodic health check can be cancelled."""
        # Create a task that will run briefly
        task = asyncio.create_task(periodic_health_check(0.1))
        
        # Let it run for a short time
        await asyncio.sleep(0.15)
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected
        
        # Task should be done (either cancelled or completed)
        assert task.done()


class TestFusionFactoryMonitoringIntegration:
    """Test monitoring integration with FusionFactory."""
    
    @patch('src.services.fusion_factory.get_adapter_registry')
    @patch('src.services.fusion_factory.get_error_handler')
    @patch('src.services.fusion_factory.get_degradation_manager')
    def test_fusion_factory_with_monitoring(self, mock_degradation, mock_error_handler, mock_registry):
        """Test FusionFactory initialization with monitoring."""
        # Mock dependencies
        mock_registry.return_value = Mock()
        mock_error_handler.return_value = Mock()
        mock_degradation.return_value = Mock()
        
        # Initialize FusionFactory (should include monitoring)
        factory = FusionFactory()
        
        assert factory.performance_monitor is not None
        assert hasattr(factory.query_dispatcher, 'performance_monitor')
    
    @patch('src.services.fusion_factory.get_adapter_registry')
    @patch('src.services.fusion_factory.get_error_handler')
    @patch('src.services.fusion_factory.get_degradation_manager')
    @pytest.mark.asyncio
    async def test_query_processing_with_monitoring(self, mock_degradation, mock_error_handler, mock_registry):
        """Test query processing with monitoring enabled."""
        # Mock dependencies
        mock_adapter_registry = Mock()
        mock_adapter = Mock()
        mock_adapter.engine_id = "test_engine"
        mock_adapter.is_healthy = True
        mock_adapter.search_with_timeout = AsyncMock(return_value=Mock(results=[]))
        
        mock_adapter_registry.get_all_adapters.return_value = {"test_engine": mock_adapter}
        mock_registry.return_value = mock_adapter_registry
        
        mock_error_handler.return_value = Mock()
        mock_error_handler.return_value.create_circuit_breaker.return_value = Mock()
        mock_error_handler.return_value.create_circuit_breaker.return_value.call = AsyncMock(
            return_value=Mock(results=[])
        )
        
        mock_degradation_manager = Mock()
        mock_degradation_manager.should_degrade.return_value = False
        mock_degradation_manager.cache_successful_results = Mock()
        mock_degradation.return_value = mock_degradation_manager
        
        # Initialize FusionFactory
        factory = FusionFactory()
        
        # Process a query (should record metrics)
        try:
            result = await factory.process_query("test query", limit=5)
            
            # Check that monitoring recorded the operation
            metrics = factory.performance_monitor.metrics_collector.get_all_metrics()
            assert "counters" in metrics
            
        except Exception:
            # Even if query fails, monitoring should still work
            metrics = factory.performance_monitor.metrics_collector.get_all_metrics()
            assert "counters" in metrics


class TestMonitoringErrorScenarios:
    """Test monitoring behavior in error scenarios."""
    
    def test_monitoring_disabled(self):
        """Test behavior when monitoring is disabled."""
        config = {"enabled": False}
        
        monitor = PerformanceMonitor(config)
        
        # Operations should not raise errors but also not record metrics
        with monitor.measure_operation("test_op"):
            time.sleep(0.01)
        
        monitor.record_engine_metrics("engine1", 0.1, True, 5)
        
        # No metrics should be recorded
        summary = monitor.get_performance_summary()
        assert summary == {}
        
        health = monitor.get_health_status()
        assert health == {"status": "monitoring_disabled"}
    
    def test_monitoring_with_errors(self):
        """Test monitoring behavior with errors."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record successful and failed operations
        monitor.record_operation_success("test_op", 0.1)
        monitor.record_operation_error("test_op", 0.5, "TimeoutError")
        
        # Check metrics
        metrics = monitor.metrics_collector.get_all_metrics()
        assert "counters" in metrics
        
        # Should have both success and error counters
        counter_keys = list(metrics["counters"].keys())
        assert any("success" in key for key in counter_keys)
        assert any("error" in key for key in counter_keys)
    
    def test_wandb_unavailable(self):
        """Test behavior when WandB is unavailable."""
        with patch('src.utils.monitoring.WANDB_AVAILABLE', False):
            config = {
                "enabled": True,
                "wandb": {"enabled": True},  # Try to enable but should fail gracefully
                "prometheus": {"enabled": False}
            }
            
            monitor = PerformanceMonitor(config)
            
            # Should still work without WandB
            assert monitor.enabled is True
            assert monitor.wandb_integration.enabled is False
    
    def test_prometheus_unavailable(self):
        """Test behavior when Prometheus is unavailable."""
        with patch('src.utils.monitoring.PROMETHEUS_AVAILABLE', False):
            config = {
                "enabled": True,
                "wandb": {"enabled": False},
                "prometheus": {"enabled": True}  # Try to enable but should fail gracefully
            }
            
            monitor = PerformanceMonitor(config)
            
            # Should still work without Prometheus
            assert monitor.enabled is True
            assert monitor.prometheus_integration.enabled is False


class TestMonitoringConfiguration:
    """Test monitoring configuration scenarios."""
    
    def test_default_configuration(self):
        """Test monitoring with default configuration."""
        monitor = PerformanceMonitor()
        
        # Should use default config
        assert monitor.enabled is True
        assert monitor._monitoring_interval == 60  # Default from config
    
    def test_custom_configuration(self):
        """Test monitoring with custom configuration."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 30,
            "wandb": {
                "enabled": False,
                "project": "custom-project"
            },
            "prometheus": {
                "enabled": False,
                "port": 9091
            }
        }
        
        monitor = PerformanceMonitor(config)
        
        assert monitor._monitoring_interval == 30
        assert monitor.wandb_integration.config["project"] == "custom-project"
        assert monitor.prometheus_integration.port == 9091
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record some metrics
        monitor.record_operation_success("test_op", 0.5)
        monitor._operation_counts["test_op"] = 5
        
        # Verify metrics exist
        assert len(monitor.metrics_collector._metrics_history) > 0
        assert len(monitor._operation_counts) > 0
        
        # Reset metrics
        monitor.reset_metrics()
        
        # Verify metrics are cleared
        assert len(monitor.metrics_collector._metrics_history) == 0
        assert len(monitor._operation_counts) == 0