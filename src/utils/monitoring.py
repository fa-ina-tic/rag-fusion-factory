"""Performance monitoring and metrics collection for RAG Fusion Factory."""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
import psutil
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..config.settings import get_monitoring_config
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    timestamp: datetime
    value: Union[float, int]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    query_latency: float = 0.0
    engine_response_times: Dict[str, float] = field(default_factory=dict)
    fusion_accuracy: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    success_count: int = 0
    throughput_qps: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metric points to keep in memory
        """
        self.max_history = max_history
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # System metrics
        self._process = psutil.Process()
        
        logger.info(f"MetricsCollector initialized with max_history={max_history}")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            metadata: Optional metadata
        """
        with self._lock:
            metric_point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
                metadata=metadata or {}
            )
            self._metrics_history[name].append(metric_point)
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
        """
        with self._lock:
            self._counters[name] += value
            self.record_metric(f"{name}_total", self._counters[name])
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Gauge value
        """
        with self._lock:
            self._gauges[name] = value
            self.record_metric(name, value)
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram.
        
        Args:
            name: Histogram name
            value: Value to record
        """
        with self._lock:
            self._histograms[name].append(value)
            # Keep only recent values for memory efficiency
            if len(self._histograms[name]) > self.max_history:
                self._histograms[name] = self._histograms[name][-self.max_history:]
            
            self.record_metric(f"{name}_histogram", value)
    
    def get_metric_summary(self, name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name
            window_minutes: Time window for statistics
            
        Returns:
            Dictionary containing metric statistics
        """
        with self._lock:
            if name not in self._metrics_history:
                return {}
            
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_points = [
                point for point in self._metrics_history[name]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                return {}
            
            values = [point.value for point in recent_points]
            
            return {
                "count": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "window_minutes": window_minutes,
                "latest_value": values[-1] if values else None,
                "latest_timestamp": recent_points[-1].timestamp.isoformat()
            }
    
    def get_histogram_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a histogram.
        
        Args:
            name: Histogram name
            
        Returns:
            Dictionary containing histogram statistics
        """
        with self._lock:
            if name not in self._histograms or not self._histograms[name]:
                return {}
            
            values = self._histograms[name]
            
            return {
                "count": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "p999": np.percentile(values, 99.9)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_summary(name)
                    for name in self._histograms.keys()
                },
                "system_metrics": self._get_system_metrics(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        try:
            memory_info = self._process.memory_info()
            cpu_percent = self._process.cpu_percent()
            
            return {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_percent": cpu_percent,
                "num_threads": self._process.num_threads(),
                "num_fds": self._process.num_fds() if hasattr(self._process, 'num_fds') else None,
                "create_time": self._process.create_time()
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {}
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics_history.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            logger.info("All metrics reset")


class WandBIntegration:
    """Integration with Weights & Biases for experiment tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize WandB integration.
        
        Args:
            config: WandB configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False) and WANDB_AVAILABLE
        self.log_frequency = config.get("log_frequency", 100)
        self._operation_count = 0
        self._run = None
        
        if self.enabled:
            self._initialize_wandb()
        else:
            if not WANDB_AVAILABLE:
                logger.warning("WandB not available, monitoring will be limited")
            else:
                logger.info("WandB monitoring disabled in configuration")
    
    def _initialize_wandb(self) -> None:
        """Initialize WandB run."""
        try:
            self._run = wandb.init(
                project=self.config.get("project", "rag-fusion-factory"),
                entity=self.config.get("entity"),
                tags=self.config.get("tags", []),
                config=self.config
            )
            logger.info(f"WandB initialized: project={self._run.project}, id={self._run.id}")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or not self._run:
            return
        
        try:
            # Flatten nested dictionaries for WandB
            flat_metrics = self._flatten_metrics(metrics)
            self._run.log(flat_metrics, step=step)
            
            self._operation_count += 1
            if self._operation_count % self.log_frequency == 0:
                logger.debug(f"Logged {len(flat_metrics)} metrics to WandB (step {step})")
                
        except Exception as e:
            logger.error(f"Failed to log metrics to WandB: {e}")
    
    def log_performance_metrics(self, perf_metrics: PerformanceMetrics) -> None:
        """Log performance metrics to WandB.
        
        Args:
            perf_metrics: Performance metrics to log
        """
        if not self.enabled:
            return
        
        metrics = {
            "performance/query_latency": perf_metrics.query_latency,
            "performance/fusion_accuracy": perf_metrics.fusion_accuracy,
            "performance/memory_usage_mb": perf_metrics.memory_usage_mb,
            "performance/cpu_usage_percent": perf_metrics.cpu_usage_percent,
            "performance/error_count": perf_metrics.error_count,
            "performance/success_count": perf_metrics.success_count,
            "performance/throughput_qps": perf_metrics.throughput_qps,
            "performance/cache_hit_rate": perf_metrics.cache_hit_rate,
            "performance/active_connections": perf_metrics.active_connections
        }
        
        # Add engine response times
        for engine_id, response_time in perf_metrics.engine_response_times.items():
            metrics[f"engines/{engine_id}/response_time"] = response_time
        
        self.log_metrics(metrics)
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metrics dictionary.
        
        Args:
            metrics: Nested metrics dictionary
            prefix: Prefix for metric names
            
        Returns:
            Flattened metrics dictionary
        """
        flat_metrics = {}
        
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_metrics.update(self._flatten_metrics(value, full_key))
            elif isinstance(value, (int, float, bool)):
                flat_metrics[full_key] = value
            elif value is not None:
                flat_metrics[full_key] = str(value)
        
        return flat_metrics
    
    def finish(self) -> None:
        """Finish WandB run."""
        if self.enabled and self._run:
            try:
                self._run.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.error(f"Failed to finish WandB run: {e}")


class PrometheusIntegration:
    """Integration with Prometheus for metrics collection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Prometheus integration.
        
        Args:
            config: Prometheus configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False) and PROMETHEUS_AVAILABLE
        self.port = config.get("port", 9090)
        self.endpoint = config.get("endpoint", "/metrics")
        
        if self.enabled:
            self._setup_metrics()
        else:
            if not PROMETHEUS_AVAILABLE:
                logger.warning("Prometheus client not available")
            else:
                logger.info("Prometheus monitoring disabled in configuration")
    
    def _setup_metrics(self) -> None:
        """Set up Prometheus metrics."""
        try:
            self.registry = CollectorRegistry()
            
            # Define metrics
            self.query_counter = Counter(
                'rag_fusion_queries_total',
                'Total number of queries processed',
                ['status'],
                registry=self.registry
            )
            
            self.query_duration = Histogram(
                'rag_fusion_query_duration_seconds',
                'Query processing duration',
                ['engine'],
                registry=self.registry
            )
            
            self.engine_response_time = Histogram(
                'rag_fusion_engine_response_seconds',
                'Engine response time',
                ['engine_id'],
                registry=self.registry
            )
            
            self.memory_usage = Gauge(
                'rag_fusion_memory_usage_bytes',
                'Memory usage in bytes',
                registry=self.registry
            )
            
            self.cpu_usage = Gauge(
                'rag_fusion_cpu_usage_percent',
                'CPU usage percentage',
                registry=self.registry
            )
            
            self.active_connections = Gauge(
                'rag_fusion_active_connections',
                'Number of active connections',
                registry=self.registry
            )
            
            self.error_counter = Counter(
                'rag_fusion_errors_total',
                'Total number of errors',
                ['error_type', 'component'],
                registry=self.registry
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Prometheus metrics: {e}")
            self.enabled = False
    
    def record_query(self, status: str) -> None:
        """Record a query with status.
        
        Args:
            status: Query status (success, error, timeout)
        """
        if self.enabled:
            self.query_counter.labels(status=status).inc()
    
    def record_query_duration(self, duration: float, engine: str = "fusion") -> None:
        """Record query duration.
        
        Args:
            duration: Duration in seconds
            engine: Engine identifier
        """
        if self.enabled:
            self.query_duration.labels(engine=engine).observe(duration)
    
    def record_engine_response_time(self, engine_id: str, response_time: float) -> None:
        """Record engine response time.
        
        Args:
            engine_id: Engine identifier
            response_time: Response time in seconds
        """
        if self.enabled:
            self.engine_response_time.labels(engine_id=engine_id).observe(response_time)
    
    def update_system_metrics(self, memory_bytes: float, cpu_percent: float, 
                            active_connections: int) -> None:
        """Update system metrics.
        
        Args:
            memory_bytes: Memory usage in bytes
            cpu_percent: CPU usage percentage
            active_connections: Number of active connections
        """
        if self.enabled:
            self.memory_usage.set(memory_bytes)
            self.cpu_usage.set(cpu_percent)
            self.active_connections.set(active_connections)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record an error.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        if self.enabled:
            self.error_counter.labels(error_type=error_type, component=component).inc()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format.
        
        Returns:
            Metrics in Prometheus text format
        """
        if self.enabled:
            return generate_latest(self.registry).decode('utf-8')
        return ""


class PerformanceMonitor:
    """Main performance monitoring class that coordinates all monitoring components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance monitor.
        
        Args:
            config: Optional monitoring configuration
        """
        self.config = config or get_monitoring_config()
        self.enabled = self.config.get("enabled", True)
        
        if not self.enabled:
            logger.info("Performance monitoring disabled")
            return
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.wandb_integration = WandBIntegration(self.config.get("wandb", {}))
        self.prometheus_integration = PrometheusIntegration(self.config.get("prometheus", {}))
        
        # Performance tracking
        self._query_start_times: Dict[str, float] = {}
        self._operation_counts: Dict[str, int] = defaultdict(int)
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = self.config.get("metrics_collection_interval", 60)
        
        logger.info("PerformanceMonitor initialized successfully")
    
    async def start_monitoring(self) -> None:
        """Start background monitoring task."""
        if not self.enabled:
            return
        
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Background monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Background monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Update metrics
            self.metrics_collector.set_gauge("system_memory_rss_mb", memory_info.rss / 1024 / 1024)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            self.metrics_collector.set_gauge("system_num_threads", process.num_threads())
            
            # Update Prometheus
            self.prometheus_integration.update_system_metrics(
                memory_info.rss, cpu_percent, 0  # TODO: Track active connections
            )
            
            # Log to WandB
            metrics = {
                "system/memory_mb": memory_info.rss / 1024 / 1024,
                "system/cpu_percent": cpu_percent,
                "system/num_threads": process.num_threads()
            }
            self.wandb_integration.log_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    @contextmanager
    def measure_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to measure operation duration.
        
        Args:
            operation_name: Name of the operation
            labels: Optional labels for the metric
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
            # Success
            duration = time.time() - start_time
            self.record_operation_success(operation_name, duration, labels)
        except Exception as e:
            # Error
            duration = time.time() - start_time
            self.record_operation_error(operation_name, duration, str(type(e).__name__), labels)
            raise
    
    @asynccontextmanager
    async def measure_async_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Async context manager to measure operation duration.
        
        Args:
            operation_name: Name of the operation
            labels: Optional labels for the metric
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
            # Success
            duration = time.time() - start_time
            self.record_operation_success(operation_name, duration, labels)
        except Exception as e:
            # Error
            duration = time.time() - start_time
            self.record_operation_error(operation_name, duration, str(type(e).__name__), labels)
            raise
    
    def record_operation_success(self, operation_name: str, duration: float, 
                               labels: Optional[Dict[str, str]] = None) -> None:
        """Record successful operation.
        
        Args:
            operation_name: Name of the operation
            duration: Operation duration in seconds
            labels: Optional labels
        """
        if not self.enabled:
            return
        
        # Update metrics collector
        self.metrics_collector.record_histogram(f"{operation_name}_duration", duration)
        self.metrics_collector.increment_counter(f"{operation_name}_success")
        
        # Update Prometheus
        self.prometheus_integration.record_query_duration(duration, operation_name)
        self.prometheus_integration.record_query("success")
        
        # Update operation counts
        self._operation_counts[operation_name] += 1
    
    def record_operation_error(self, operation_name: str, duration: float, 
                             error_type: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Record failed operation.
        
        Args:
            operation_name: Name of the operation
            duration: Operation duration in seconds
            error_type: Type of error
            labels: Optional labels
        """
        if not self.enabled:
            return
        
        # Update metrics collector
        self.metrics_collector.record_histogram(f"{operation_name}_duration", duration)
        self.metrics_collector.increment_counter(f"{operation_name}_error")
        
        # Update Prometheus
        self.prometheus_integration.record_query_duration(duration, operation_name)
        self.prometheus_integration.record_query("error")
        self.prometheus_integration.record_error(error_type, operation_name)
    
    def record_engine_metrics(self, engine_id: str, response_time: float, 
                            success: bool, result_count: int) -> None:
        """Record engine-specific metrics.
        
        Args:
            engine_id: Engine identifier
            response_time: Response time in seconds
            success: Whether the operation was successful
            result_count: Number of results returned
        """
        if not self.enabled:
            return
        
        # Update metrics collector
        self.metrics_collector.record_histogram(f"engine_{engine_id}_response_time", response_time)
        self.metrics_collector.record_metric(f"engine_{engine_id}_result_count", result_count)
        
        if success:
            self.metrics_collector.increment_counter(f"engine_{engine_id}_success")
        else:
            self.metrics_collector.increment_counter(f"engine_{engine_id}_error")
        
        # Update Prometheus
        self.prometheus_integration.record_engine_response_time(engine_id, response_time)
        
        # Log to WandB
        metrics = {
            f"engines/{engine_id}/response_time": response_time,
            f"engines/{engine_id}/result_count": result_count,
            f"engines/{engine_id}/success": 1 if success else 0
        }
        self.wandb_integration.log_metrics(metrics)
    
    def get_performance_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary for the specified time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary containing performance summary
        """
        if not self.enabled:
            return {}
        
        return {
            "query_latency": self.metrics_collector.get_metric_summary("query_duration", window_minutes),
            "engine_metrics": {
                name: self.metrics_collector.get_metric_summary(name, window_minutes)
                for name in self.metrics_collector._metrics_history.keys()
                if name.startswith("engine_") and name.endswith("_response_time")
            },
            "error_rates": {
                name: self.metrics_collector.get_metric_summary(name, window_minutes)
                for name in self.metrics_collector._metrics_history.keys()
                if name.endswith("_error")
            },
            "system_metrics": self.metrics_collector._get_system_metrics(),
            "operation_counts": dict(self._operation_counts),
            "window_minutes": window_minutes,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Dictionary containing health status
        """
        if not self.enabled:
            return {"status": "monitoring_disabled"}
        
        try:
            # Check system resources
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Determine health status
            memory_mb = memory_info.rss / 1024 / 1024
            health_issues = []
            
            if memory_mb > 1000:  # More than 1GB
                health_issues.append("high_memory_usage")
            
            if cpu_percent > 80:
                health_issues.append("high_cpu_usage")
            
            # Check error rates
            recent_errors = sum(
                len([p for p in self.metrics_collector._metrics_history[name] 
                     if p.timestamp > datetime.now() - timedelta(minutes=5)])
                for name in self.metrics_collector._metrics_history.keys()
                if name.endswith("_error")
            )
            
            if recent_errors > 10:
                health_issues.append("high_error_rate")
            
            status = "healthy" if not health_issues else "degraded"
            
            return {
                "status": status,
                "issues": health_issues,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "recent_errors": recent_errors,
                "monitoring_components": {
                    "metrics_collector": True,
                    "wandb": self.wandb_integration.enabled,
                    "prometheus": self.prometheus_integration.enabled
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        if not self.enabled:
            return
        
        self.metrics_collector.reset_metrics()
        self._operation_counts.clear()
        logger.info("Performance metrics reset")
    
    def shutdown(self) -> None:
        """Shutdown monitoring components."""
        if not self.enabled:
            return
        
        try:
            self.wandb_integration.finish()
            logger.info("Performance monitoring shutdown complete")
        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {e}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def initialize_monitoring(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Initialize global performance monitoring.
    
    Args:
        config: Optional monitoring configuration
        
    Returns:
        Initialized PerformanceMonitor instance
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(config)
    return _performance_monitor


async def start_monitoring() -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.start_monitoring()


async def stop_monitoring() -> None:
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.stop_monitoring()