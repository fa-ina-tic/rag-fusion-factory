#!/usr/bin/env python3
"""
Demonstration of RAG Fusion Factory monitoring capabilities.

This script shows how to use the performance monitoring, health checks,
and metrics collection features.
"""

import asyncio
import time
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.monitoring import (
    PerformanceMonitor, initialize_monitoring, start_monitoring, stop_monitoring
)
from src.api.health import basic_health_check, periodic_health_check


async def demonstrate_monitoring():
    """Demonstrate monitoring functionality."""
    print("🚀 RAG Fusion Factory Monitoring Demo")
    print("=" * 50)
    
    # Initialize monitoring with custom configuration
    config = {
        "enabled": True,
        "metrics_collection_interval": 5,  # 5 seconds for demo
        "wandb": {
            "enabled": False,  # Disabled for demo
            "project": "rag-fusion-factory-demo"
        },
        "prometheus": {
            "enabled": True,
            "port": 9090
        }
    }
    
    print("📊 Initializing performance monitor...")
    monitor = initialize_monitoring(config)
    
    # Start background monitoring
    print("🔄 Starting background monitoring...")
    await start_monitoring()
    
    try:
        # Demonstrate basic health check
        print("\n💚 Running basic health check...")
        health = await basic_health_check()
        print(f"   Status: {health.status}")
        print(f"   Uptime: {health.uptime_seconds:.2f} seconds")
        print(f"   Version: {health.version}")
        
        # Demonstrate operation measurement
        print("\n⏱️  Measuring operations...")
        
        # Simulate some operations
        for i in range(5):
            operation_name = f"demo_operation_{i % 3}"  # Create 3 different operation types
            
            with monitor.measure_operation(operation_name):
                # Simulate work
                await asyncio.sleep(0.1 + (i * 0.05))  # Variable duration
                
                if i == 3:  # Simulate an error on the 4th operation
                    try:
                        raise ValueError("Simulated error for demo")
                    except ValueError:
                        pass  # Error will be recorded by context manager
            
            print(f"   ✅ Completed {operation_name}")
        
        # Simulate engine metrics
        print("\n🔍 Recording engine metrics...")
        engines = ["elasticsearch", "solr", "opensearch"]
        
        for i, engine in enumerate(engines):
            response_time = 0.1 + (i * 0.05)  # Variable response times
            success = i != 1  # Make solr fail for demo
            result_count = 10 - (i * 2) if success else 0
            
            monitor.record_engine_metrics(engine, response_time, success, result_count)
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {engine}: {status} ({response_time:.3f}s, {result_count} results)")
        
        # Wait a bit for metrics to be collected
        print("\n⏳ Collecting metrics...")
        await asyncio.sleep(2)
        
        # Show performance summary
        print("\n📈 Performance Summary:")
        summary = monitor.get_performance_summary(window_minutes=1)
        
        if summary.get("operation_counts"):
            print("   Operation Counts:")
            for op, count in summary["operation_counts"].items():
                print(f"     {op}: {count}")
        
        if summary.get("system_metrics"):
            sys_metrics = summary["system_metrics"]
            print("   System Metrics:")
            print(f"     Memory: {sys_metrics.get('memory_rss_mb', 0):.1f} MB")
            print(f"     CPU: {sys_metrics.get('cpu_percent', 0):.1f}%")
            print(f"     Threads: {sys_metrics.get('num_threads', 0)}")
        
        # Show health status
        print("\n🏥 Health Status:")
        health_status = monitor.get_health_status()
        print(f"   Overall Status: {health_status['status']}")
        
        if health_status.get("issues"):
            print("   Issues:")
            for issue in health_status["issues"]:
                print(f"     ⚠️  {issue}")
        
        print("   Monitoring Components:")
        components = health_status.get("monitoring_components", {})
        for component, enabled in components.items():
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"     {component}: {status}")
        
        # Show all metrics
        print("\n📊 All Metrics:")
        all_metrics = monitor.metrics_collector.get_all_metrics()
        
        if all_metrics.get("counters"):
            print("   Counters:")
            for name, value in all_metrics["counters"].items():
                print(f"     {name}: {value}")
        
        if all_metrics.get("histograms"):
            print("   Histograms (summaries):")
            for name, summary in all_metrics["histograms"].items():
                if summary:
                    print(f"     {name}:")
                    print(f"       Count: {summary.get('count', 0)}")
                    print(f"       Mean: {summary.get('mean', 0):.3f}")
                    print(f"       P95: {summary.get('p95', 0):.3f}")
        
        # Demonstrate Prometheus metrics (if enabled)
        if monitor.prometheus_integration.enabled:
            print("\n🔢 Prometheus Metrics Sample:")
            metrics_text = monitor.prometheus_integration.get_metrics()
            if metrics_text:
                # Show first few lines
                lines = metrics_text.split('\n')[:10]
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        print(f"   {line}")
                if len(lines) >= 10:
                    print("   ... (truncated)")
            else:
                print("   No Prometheus metrics available")
        
        # Demonstrate metrics reset
        print("\n🔄 Resetting metrics...")
        monitor.reset_metrics()
        
        # Verify reset
        reset_metrics = monitor.metrics_collector.get_all_metrics()
        counter_count = len(reset_metrics.get("counters", {}))
        histogram_count = len(reset_metrics.get("histograms", {}))
        print(f"   Metrics after reset: {counter_count} counters, {histogram_count} histograms")
        
        print("\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop monitoring
        print("\n🛑 Stopping monitoring...")
        await stop_monitoring()
        monitor.shutdown()


async def demonstrate_periodic_health_checks():
    """Demonstrate periodic health checks."""
    print("\n🔄 Starting periodic health checks (5 seconds)...")
    
    # Start periodic health checks
    task = asyncio.create_task(periodic_health_check(interval_seconds=2))
    
    # Let it run for a few cycles
    await asyncio.sleep(5)
    
    # Stop periodic health checks
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    print("   Periodic health checks stopped")


def main():
    """Main demo function."""
    print("Starting RAG Fusion Factory Monitoring Demo...")
    
    try:
        # Run the main monitoring demo
        asyncio.run(demonstrate_monitoring())
        
        # Run periodic health check demo
        asyncio.run(demonstrate_periodic_health_checks())
        
        print("\n🎉 All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())