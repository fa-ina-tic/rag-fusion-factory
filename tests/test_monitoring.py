"""Tests for performance monitoring and metrics collection."""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.utils.monitoring import (
    MetricsCollector, PerformanceMonitor, WandBIntegration, PrometheusIntegration,
    MetricPoint, PerformanceMetrics, get_performance_monitor, initialize_monitoring
)


class TestMetricsCollector:
    """Test cases for MetricsCollector."""
    
    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_history=500)
        assert collector.max_history == 500
        assert len(collector._metrics_history) == 0
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0
    
    def test_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42.5, {"label": "value"}, {"meta": "data"})
        
        assert "test_metric" in collector._metrics_history
        assert len(collector._metrics_history["test_metric"]) == 1
        
        metric_point = collector._metrics_history["test_metric"][0]
        assert metric_point.value == 42.5
        assert metric_point.labels == {"label": "value"}
        assert metric_point.metadata == {"meta": "data"}
    
    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 5)
        collector.increment_counter("test_counter", 3)
        
        assert collector._counters["test_counter"] == 8
        assert "test_counter_total" in collector._metrics_history
    
    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()
        
        collector.set_gauge("test_gauge", 123.45)
        
        assert collector._gauges["test_gauge"] == 123.45
        assert "test_gauge" in collector._metrics_history
    
    def test_record_histogram(self):
        """Test histogram recording."""
        collector = MetricsCollector()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_histogram("test_histogram", value)
        
        assert len(collector._histograms["test_histogram"]) == 5
        assert collector._histograms["test_histogram"] == values
    
    def test_get_metric_summary(self):
        """Test metric summary calculation."""
        collector = MetricsCollector()
        
        # Record some metrics
        values = [10, 20, 30, 40, 50]
        for value in values:
            collector.record_metric("test_metric", value)
        
        summary = collector.get_metric_summary("test_metric", window_minutes=60)
        
        assert summary["count"] == 5
        assert summary["mean"] == 30.0
        assert summary["median"] == 30.0
        assert summary["min"] == 10
        assert summary["max"] == 50
        assert "p95" in summary
        assert "p99" in summary
    
    def test_get_histogram_summary(self):
        """Test histogram summary calculation."""
        collector = MetricsCollector()
        
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in values:
            collector.record_histogram("test_histogram", value)
        
        summary = collector.get_histogram_summary("test_histogram")
        
        assert summary["count"] == 10
        assert summary["mean"] == 5.5
        assert summary["median"] == 5.5
        assert summary["min"] == 1
        assert summary["max"] == 10
        assert "p95" in summary
        assert "p99" in summary
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        
        collector.increment_counter("counter1", 5)
        collector.set_gauge("gauge1", 42.0)
        collector.record_histogram("histogram1", 10.0)
        
        all_metrics = collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "system_metrics" in all_metrics
        assert "timestamp" in all_metrics
        
        assert all_metrics["counters"]["counter1"] == 5
        assert all_metrics["gauges"]["gauge1"] == 42.0
    
    def test_reset_metrics(self):
        """Test metrics reset."""
        collector = MetricsCollector()
        
        collector.increment_counter("counter1", 5)
        collector.set_gauge("gauge1", 42.0)
        collector.record_histogram("histogram1", 10.0)
        
        collector.reset_metrics()
        
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0
        assert len(collector._histograms) == 0
        assert len(collector._metrics_history) == 0


class TestWandBIntegration:
    """Test cases for WandB integration."""
    
    @patch('src.utils.monitoring.WANDB_AVAILABLE', True)
    @patch('src.utils.monitoring.wandb')
    def test_initialization_enabled(self, mock_wandb):
        """Test WandB initialization when enabled."""
        mock_run = Mock()
        mock_run.project = "test-project"
        mock_run.id = "test-run-id"
        mock_wandb.init.return_value = mock_run
        
        config = {
            "enabled": True,
            "project": "test-project",
            "entity": "test-entity",
            "tags": ["test"]
        }
        
        wandb_integration = WandBIntegration(config)
        
        assert wandb_integration.enabled is True
        mock_wandb.init.assert_called_once()
    
    @patch('src.utils.monitoring.WANDB_AVAILABLE', False)
    def test_initialization_unavailable(self):
        """Test WandB initialization when unavailable."""
        config = {"enabled": True}
        
        wandb_integration = WandBIntegration(config)
        
        assert wandb_integration.enabled is False
    
    @patch('src.utils.monitoring.WANDB_AVAILABLE', True)
    @patch('src.utils.monitoring.wandb')
    def test_log_metrics(self, mock_wandb):
        """Test logging metrics to WandB."""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        
        config = {"enabled": True, "project": "test"}
        wandb_integration = WandBIntegration(config)
        
        metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "nested": {"value": 42}
        }
        
        wandb_integration.log_metrics(metrics, step=100)
        
        mock_run.log.assert_called_once()
        logged_metrics = mock_run.log.call_args[0][0]
        assert "accuracy" in logged_metrics
        assert "loss" in logged_metrics
        assert "nested/value" in logged_metrics
    
    @patch('src.utils.monitoring.WANDB_AVAILABLE', True)
    @patch('src.utils.monitoring.wandb')
    def test_log_performance_metrics(self, mock_wandb):
        """Test logging performance metrics to WandB."""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        
        config = {"enabled": True, "project": "test"}
        wandb_integration = WandBIntegration(config)
        
        perf_metrics = PerformanceMetrics(
            query_latency=0.5,
            engine_response_times={"engine1": 0.2, "engine2": 0.3},
            fusion_accuracy=0.85,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0
        )
        
        wandb_integration.log_performance_metrics(perf_metrics)
        
        mock_run.log.assert_called_once()
        logged_metrics = mock_run.log.call_args[0][0]
        assert "performance/query_latency" in logged_metrics
        assert "engines/engine1/response_time" in logged_metrics


class TestPrometheusIntegration:
    """Test cases for Prometheus integration."""
    
    @patch('src.utils.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.utils.monitoring.Counter')
    @patch('src.utils.monitoring.Histogram')
    @patch('src.utils.monitoring.Gauge')
    @patch('src.utils.monitoring.CollectorRegistry')
    def test_initialization_enabled(self, mock_registry, mock_gauge, mock_histogram, mock_counter):
        """Test Prometheus initialization when enabled."""
        config = {"enabled": True, "port": 9090}
        
        prometheus_integration = PrometheusIntegration(config)
        
        assert prometheus_integration.enabled is True
        mock_registry.assert_called_once()
    
    @patch('src.utils.monitoring.PROMETHEUS_AVAILABLE', False)
    def test_initialization_unavailable(self):
        """Test Prometheus initialization when unavailable."""
        config = {"enabled": True}
        
        prometheus_integration = PrometheusIntegration(config)
        
        assert prometheus_integration.enabled is False
    
    @patch('src.utils.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.utils.monitoring.Counter')
    @patch('src.utils.monitoring.Histogram')
    @patch('src.utils.monitoring.Gauge')
    @patch('src.utils.monitoring.CollectorRegistry')
    def test_record_metrics(self, mock_registry, mock_gauge, mock_histogram, mock_counter):
        """Test recording metrics to Prometheus."""
        config = {"enabled": True}
        prometheus_integration = PrometheusIntegration(config)
        
        # Mock the metrics
        prometheus_integration.query_counter = Mock()
        prometheus_integration.query_duration = Mock()
        prometheus_integration.engine_response_time = Mock()
        prometheus_integration.memory_usage = Mock()
        prometheus_integration.cpu_usage = Mock()
        prometheus_integration.active_connections = Mock()
        prometheus_integration.error_counter = Mock()
        
        # Test recording different types of metrics
        prometheus_integration.record_query("success")
        prometheus_integration.record_query_duration(0.5, "test_engine")
        prometheus_integration.record_engine_response_time("engine1", 0.3)
        prometheus_integration.update_system_metrics(1024*1024*256, 45.0, 10)
        prometheus_integration.record_error("TimeoutError", "QueryDispatcher")
        
        # Verify metrics were called
        prometheus_integration.query_counter.labels.assert_called()
        prometheus_integration.query_duration.labels.assert_called()
        prometheus_integration.engine_response_time.labels.assert_called()
        prometheus_integration.memory_usage.set.assert_called_with(1024*1024*256)
        prometheus_integration.cpu_usage.set.assert_called_with(45.0)
        prometheus_integration.active_connections.set.assert_called_with(10)
        prometheus_integration.error_counter.labels.assert_called()


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 30,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        assert monitor.enabled is True
        assert monitor._monitoring_interval == 30
        assert monitor.metrics_collector is not None
    
    def test_initialization_disabled(self):
        """Test performance monitor initialization when disabled."""
        config = {"enabled": False}
        
        monitor = PerformanceMonitor(config)
        
        assert monitor.enabled is False
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
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
    
    def test_measure_operation_success(self):
        """Test measuring successful operation."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        with monitor.measure_operation("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Check that metrics were recorded
        assert "test_operation_success_total" in monitor.metrics_collector._metrics_history
        assert "test_operation_duration_histogram" in monitor.metrics_collector._metrics_history
    
    def test_measure_operation_error(self):
        """Test measuring failed operation."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        with pytest.raises(ValueError):
            with monitor.measure_operation("test_operation"):
                raise ValueError("Test error")
        
        # Check that error metrics were recorded
        assert "test_operation_error_total" in monitor.metrics_collector._metrics_history
        assert "test_operation_duration_histogram" in monitor.metrics_collector._metrics_history
    
    @pytest.mark.asyncio
    async def test_measure_async_operation_success(self):
        """Test measuring successful async operation."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        async with monitor.measure_async_operation("async_test_operation"):
            await asyncio.sleep(0.1)  # Simulate async work
        
        # Check that metrics were recorded
        assert "async_test_operation_success_total" in monitor.metrics_collector._metrics_history
        assert "async_test_operation_duration_histogram" in monitor.metrics_collector._metrics_history
    
    @pytest.mark.asyncio
    async def test_measure_async_operation_error(self):
        """Test measuring failed async operation."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        with pytest.raises(ValueError):
            async with monitor.measure_async_operation("async_test_operation"):
                raise ValueError("Test async error")
        
        # Check that error metrics were recorded
        assert "async_test_operation_error_total" in monitor.metrics_collector._metrics_history
        assert "async_test_operation_duration_histogram" in monitor.metrics_collector._metrics_history
    
    def test_record_engine_metrics(self):
        """Test recording engine-specific metrics."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        monitor.record_engine_metrics("test_engine", 0.5, True, 10)
        
        # Check that engine metrics were recorded
        assert "engine_test_engine_response_time_histogram" in monitor.metrics_collector._metrics_history
        assert "engine_test_engine_result_count" in monitor.metrics_collector._metrics_history
        assert "engine_test_engine_success_total" in monitor.metrics_collector._metrics_history
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record some metrics
        monitor.record_operation_success("test_op", 0.5)
        monitor.record_engine_metrics("engine1", 0.3, True, 5)
        
        summary = monitor.get_performance_summary(window_minutes=5)
        
        assert "operation_counts" in summary
        assert "system_metrics" in summary
        assert "timestamp" in summary
        assert "window_minutes" in summary
    
    def test_get_health_status(self):
        """Test getting health status."""
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
        assert "timestamp" in health_status
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record some metrics
        monitor.record_operation_success("test_op", 0.5)
        monitor._operation_counts["test_op"] = 5
        
        # Reset metrics
        monitor.reset_metrics()
        
        # Check that metrics were reset
        assert len(monitor.metrics_collector._metrics_history) == 0
        assert len(monitor._operation_counts) == 0


class TestGlobalMonitoringFunctions:
    """Test cases for global monitoring functions."""
    
    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should return the same instance
        assert monitor1 is monitor2
    
    def test_initialize_monitoring(self):
        """Test initializing monitoring with custom config."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 120,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = initialize_monitoring(config)
        
        assert monitor.enabled is True
        assert monitor._monitoring_interval == 120


class TestIntegrationScenarios:
    """Integration test scenarios for monitoring."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        config = {
            "enabled": True,
            "metrics_collection_interval": 0.1,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        try:
            # Start monitoring
            await monitor.start_monitoring()
            
            # Simulate some operations
            with monitor.measure_operation("query_processing"):
                time.sleep(0.05)
            
            monitor.record_engine_metrics("engine1", 0.2, True, 5)
            monitor.record_engine_metrics("engine2", 0.3, False, 0)
            
            # Let monitoring collect some data
            await asyncio.sleep(0.2)
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            assert "operation_counts" in summary
            
            # Get health status
            health = monitor.get_health_status()
            assert health["status"] in ["healthy", "degraded"]
            
        finally:
            # Stop monitoring
            await monitor.stop_monitoring()
    
    def test_monitoring_with_errors(self):
        """Test monitoring behavior with errors."""
        config = {
            "enabled": True,
            "wandb": {"enabled": False},
            "prometheus": {"enabled": False}
        }
        
        monitor = PerformanceMonitor(config)
        
        # Record successful operations
        monitor.record_operation_success("test_op", 0.1)
        monitor.record_operation_success("test_op", 0.2)
        
        # Record failed operations
        monitor.record_operation_error("test_op", 0.5, "TimeoutError")
        monitor.record_operation_error("test_op", 0.3, "ConnectionError")
        
        # Check metrics
        assert monitor._operation_counts["test_op"] == 2  # Only successful operations
        assert "test_op_success_total" in monitor.metrics_collector._metrics_history
        assert "test_op_error_total" in monitor.metrics_collector._metrics_history
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        assert "error_rates" in summary
    
    def test_disabled_monitoring(self):
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