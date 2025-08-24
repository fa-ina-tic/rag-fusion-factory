"""Performance benchmarking tests for RAG Fusion Factory.

This module contains comprehensive performance benchmarking tests for different
adapter configurations and fusion scenarios.

Requirements covered: 2.4, 5.1, 6.2
"""

import asyncio
import time
import statistics
import pytest
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor

from src.services.fusion_factory import FusionFactory
from src.adapters.registry import AdapterRegistry
from src.adapters.mock_adapter import InMemoryMockAdapter
from src.models.core import SearchResult, SearchResults, RankedResults
from src.utils.monitoring import get_performance_monitor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceBenchmark:
    """Utility class for performance benchmarking."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start_benchmark(self, name: str):
        """Start a benchmark measurement."""
        self.start_time = time.perf_counter()
        self.results[name] = {'start_time': self.start_time}
    
    def end_benchmark(self, name: str):
        """End a benchmark measurement."""
        self.end_time = time.perf_counter()
        if name in self.results:
            self.results[name]['end_time'] = self.end_time
            self.results[name]['duration'] = self.end_time - self.results[name]['start_time']
    
    def get_duration(self, name: str) -> float:
        """Get benchmark duration."""
        return self.results.get(name, {}).get('duration', 0.0)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }


class TestAdapterPerformance:
    """Performance tests for individual adapters."""
    
    @pytest.fixture
    def performance_adapters(self):
        """Create adapters with different performance characteristics."""
        adapters = {}
        
        # Fast adapter - minimal delay, small dataset
        fast_docs = [
            {'id': f'fast_{i}', 'title': f'Fast Document {i}', 'content': f'Fast content {i}'}
            for i in range(50)
        ]
        adapters['fast'] = InMemoryMockAdapter('fast_adapter', {
            'documents': fast_docs,
            'response_delay': 0.001,
            'failure_rate': 0.0,
            'max_results': 50
        })
        
        # Medium adapter - moderate delay, medium dataset
        medium_docs = [
            {'id': f'med_{i}', 'title': f'Medium Document {i}', 'content': f'Medium content {i}'}
            for i in range(200)
        ]
        adapters['medium'] = InMemoryMockAdapter('medium_adapter', {
            'documents': medium_docs,
            'response_delay': 0.05,
            'failure_rate': 0.0,
            'max_results': 100
        })
        
        # Slow adapter - high delay, large dataset
        slow_docs = [
            {'id': f'slow_{i}', 'title': f'Slow Document {i}', 'content': f'Slow content {i}'}
            for i in range(500)
        ]
        adapters['slow'] = InMemoryMockAdapter('slow_adapter', {
            'documents': slow_docs,
            'response_delay': 0.1,
            'failure_rate': 0.0,
            'max_results': 200
        })
        
        # Variable adapter - random delays
        variable_docs = [
            {'id': f'var_{i}', 'title': f'Variable Document {i}', 'content': f'Variable content {i}'}
            for i in range(100)
        ]
        adapters['variable'] = InMemoryMockAdapter('variable_adapter', {
            'documents': variable_docs,
            'response_delay': 0.02,  # Base delay, will be randomized
            'failure_rate': 0.1,     # 10% failure rate
            'max_results': 100
        })
        
        return adapters
    
    @pytest.mark.asyncio
    async def test_single_adapter_latency(self, performance_adapters):
        """Benchmark latency for individual adapters."""
        benchmark = PerformanceBenchmark()
        latency_results = {}
        
        for adapter_name, adapter in performance_adapters.items():
            # Warm up (handle potential failures)
            try:
                await adapter.search("warmup", limit=5)
            except Exception:
                # Some adapters may fail during warmup due to failure rates
                pass
            
            # Measure latency over multiple queries
            latencies = []
            successful_queries = 0
            for i in range(10):
                start_time = time.perf_counter()
                try:
                    results = await adapter.search(f"test query {i}", limit=10)
                    end_time = time.perf_counter()
                    
                    latency = end_time - start_time
                    latencies.append(latency)
                    successful_queries += 1
                    
                    # Verify we got results
                    assert isinstance(results, SearchResults)
                except Exception:
                    # Handle failures for adapters with failure rates
                    end_time = time.perf_counter()
                    latency = end_time - start_time
                    latencies.append(latency)  # Include failed query time
            
            latency_results[adapter_name] = {
                'mean_latency': statistics.mean(latencies),
                'median_latency': statistics.median(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'std_latency': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                'success_rate': successful_queries / 10
            }
        
        # Verify performance characteristics
        assert latency_results['fast']['mean_latency'] < latency_results['medium']['mean_latency']
        assert latency_results['medium']['mean_latency'] < latency_results['slow']['mean_latency']
        
        # Fast adapter should be consistently fast
        assert latency_results['fast']['std_latency'] < 0.01
        
        # Variable adapter should have higher variance
        assert latency_results['variable']['std_latency'] > latency_results['fast']['std_latency']
    
    @pytest.mark.asyncio
    async def test_adapter_throughput(self, performance_adapters):
        """Benchmark throughput for individual adapters."""
        throughput_results = {}
        
        for adapter_name, adapter in performance_adapters.items():
            # Measure throughput over 30 seconds
            start_time = time.perf_counter()
            end_time = start_time + 5.0  # 5 second test
            query_count = 0
            total_results = 0
            
            while time.perf_counter() < end_time:
                try:
                    results = await adapter.search(f"throughput test {query_count}", limit=10)
                    total_results += len(results.results)
                except Exception:
                    # Handle failures for adapters with failure rates
                    pass
                query_count += 1
            
            actual_duration = time.perf_counter() - start_time
            
            throughput_results[adapter_name] = {
                'queries_per_second': query_count / actual_duration,
                'results_per_second': total_results / actual_duration,
                'total_queries': query_count,
                'total_results': total_results,
                'duration': actual_duration
            }
        
        # Verify throughput characteristics
        assert throughput_results['fast']['queries_per_second'] > throughput_results['slow']['queries_per_second']
        
        # All adapters should achieve reasonable throughput
        for adapter_name, metrics in throughput_results.items():
            assert metrics['queries_per_second'] > 1.0  # At least 1 query per second
            assert metrics['total_queries'] > 0
    
    @pytest.mark.asyncio
    async def test_adapter_concurrent_performance(self, performance_adapters):
        """Benchmark concurrent performance for adapters."""
        concurrent_results = {}
        
        for adapter_name, adapter in performance_adapters.items():
            # Test with different concurrency levels
            concurrency_levels = [1, 5, 10, 20]
            
            for concurrency in concurrency_levels:
                # Create concurrent tasks
                tasks = [
                    adapter.search(f"concurrent test {i}", limit=5)
                    for i in range(concurrency)
                ]
                
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.perf_counter()
                
                # Count successful results
                successful_results = [r for r in results if isinstance(r, SearchResults)]
                failed_results = [r for r in results if isinstance(r, Exception)]
                
                duration = end_time - start_time
                
                if adapter_name not in concurrent_results:
                    concurrent_results[adapter_name] = {}
                
                concurrent_results[adapter_name][concurrency] = {
                    'duration': duration,
                    'successful_queries': len(successful_results),
                    'failed_queries': len(failed_results),
                    'success_rate': len(successful_results) / concurrency,
                    'queries_per_second': concurrency / duration
                }
        
        # Verify concurrent performance
        for adapter_name, concurrency_data in concurrent_results.items():
            # Higher concurrency should generally maintain reasonable performance
            for concurrency, metrics in concurrency_data.items():
                # Variable adapter has 10% failure rate, so adjust expectations
                expected_success_rate = 0.3 if adapter_name == 'variable' else 0.5
                assert metrics['success_rate'] >= expected_success_rate
                assert metrics['queries_per_second'] > 0


class TestFusionPerformance:
    """Performance tests for fusion operations."""
    
    @pytest.fixture
    def fusion_test_registry(self):
        """Create registry optimized for fusion performance testing."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create multiple adapters with overlapping documents
        base_docs = [
            {'id': f'doc_{i}', 'title': f'Document {i}', 'content': f'Content for document {i}'}
            for i in range(100)
        ]
        
        # Engine 1: High scores, subset of documents
        engine1_docs = base_docs[:60]
        engine1 = InMemoryMockAdapter('fusion_engine_1', {
            'documents': engine1_docs,
            'response_delay': 0.02,
            'failure_rate': 0.0,
            'max_results': 50
        })
        
        # Engine 2: Medium scores, different subset
        engine2_docs = base_docs[30:90]
        engine2 = InMemoryMockAdapter('fusion_engine_2', {
            'documents': engine2_docs,
            'response_delay': 0.03,
            'failure_rate': 0.0,
            'max_results': 40
        })
        
        # Engine 3: Lower scores, another subset
        engine3_docs = base_docs[50:]
        engine3 = InMemoryMockAdapter('fusion_engine_3', {
            'documents': engine3_docs,
            'response_delay': 0.04,
            'failure_rate': 0.0,
            'max_results': 30
        })
        
        registry._instances['fusion_engine_1'] = engine1
        registry._instances['fusion_engine_2'] = engine2
        registry._instances['fusion_engine_3'] = engine3
        
        return registry
    
    @pytest.mark.asyncio
    async def test_fusion_pipeline_performance(self, fusion_test_registry):
        """Benchmark complete fusion pipeline performance."""
        factory = FusionFactory(fusion_test_registry)
        adapters = list(fusion_test_registry.get_all_adapters().values())
        
        benchmark = PerformanceBenchmark()
        
        # Warm up
        await factory.process_query("warmup", adapters=adapters, limit=5)
        
        # Benchmark fusion pipeline
        pipeline_times = []
        memory_usage = []
        
        for i in range(20):
            # Measure memory before
            gc.collect()  # Force garbage collection
            mem_before = benchmark.get_memory_usage()
            
            # Measure pipeline execution
            start_time = time.perf_counter()
            results = await factory.process_query(f"fusion test {i}", adapters=adapters, limit=10)
            end_time = time.perf_counter()
            
            # Measure memory after
            mem_after = benchmark.get_memory_usage()
            
            pipeline_time = end_time - start_time
            pipeline_times.append(pipeline_time)
            memory_usage.append(mem_after['rss_mb'] - mem_before['rss_mb'])
            
            # Verify results
            assert isinstance(results, RankedResults)
            assert results.total_results > 0
        
        # Calculate performance metrics
        performance_metrics = {
            'mean_pipeline_time': statistics.mean(pipeline_times),
            'median_pipeline_time': statistics.median(pipeline_times),
            'min_pipeline_time': min(pipeline_times),
            'max_pipeline_time': max(pipeline_times),
            'std_pipeline_time': statistics.stdev(pipeline_times),
            'mean_memory_delta': statistics.mean(memory_usage),
            'max_memory_delta': max(memory_usage)
        }
        
        # Performance assertions
        assert performance_metrics['mean_pipeline_time'] < 1.0  # Should complete within 1 second
        assert performance_metrics['std_pipeline_time'] < 0.5   # Should be reasonably consistent
        assert performance_metrics['max_memory_delta'] < 50.0   # Should not use excessive memory
    
    @pytest.mark.asyncio
    async def test_fusion_scalability(self, fusion_test_registry):
        """Test fusion performance with increasing numbers of engines."""
        factory = FusionFactory(fusion_test_registry)
        all_adapters = list(fusion_test_registry.get_all_adapters().values())
        
        scalability_results = {}
        
        # Test with 1, 2, and 3 engines
        for num_engines in [1, 2, 3]:
            adapters = all_adapters[:num_engines]
            
            # Measure performance with this number of engines
            times = []
            for i in range(10):
                start_time = time.perf_counter()
                results = await factory.process_query(f"scale test {i}", adapters=adapters, limit=10)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                assert isinstance(results, RankedResults)
            
            scalability_results[num_engines] = {
                'mean_time': statistics.mean(times),
                'median_time': statistics.median(times),
                'std_time': statistics.stdev(times) if len(times) > 1 else 0.0
            }
        
        # Verify scalability characteristics
        # Time should increase with more engines, but not linearly (due to concurrency)
        assert scalability_results[1]['mean_time'] < scalability_results[3]['mean_time']
        
        # The increase should be reasonable (not more than 3x for 3x engines)
        time_ratio = scalability_results[3]['mean_time'] / scalability_results[1]['mean_time']
        assert time_ratio < 2.0  # Should benefit from concurrency
    
    @pytest.mark.asyncio
    async def test_fusion_weight_optimization_performance(self, fusion_test_registry):
        """Benchmark performance of different weight optimization strategies."""
        factory = FusionFactory(fusion_test_registry)
        adapters = list(fusion_test_registry.get_all_adapters().values())
        
        weight_strategies = {
            'equal': np.array([1/3, 1/3, 1/3]),
            'weighted_first': np.array([0.6, 0.3, 0.1]),
            'weighted_last': np.array([0.1, 0.3, 0.6]),
            'extreme': np.array([0.9, 0.05, 0.05])
        }
        
        strategy_performance = {}
        
        for strategy_name, weights in weight_strategies.items():
            times = []
            result_qualities = []
            
            for i in range(10):
                start_time = time.perf_counter()
                results = await factory.process_query(
                    f"weight test {i}",
                    adapters=adapters,
                    weights=weights,
                    limit=10
                )
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                # Measure result quality (average confidence)
                if results.confidence_scores:
                    avg_confidence = statistics.mean(results.confidence_scores)
                    result_qualities.append(avg_confidence)
            
            strategy_performance[strategy_name] = {
                'mean_time': statistics.mean(times),
                'mean_quality': statistics.mean(result_qualities) if result_qualities else 0.0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0.0
            }
        
        # All strategies should have similar performance times (with reasonable tolerance)
        times = [metrics['mean_time'] for metrics in strategy_performance.values()]
        time_variance = statistics.stdev(times) if len(times) > 1 else 0.0
        assert time_variance < 0.2  # Weight strategies shouldn't significantly affect performance (relaxed tolerance)


class TestLoadTesting:
    """Load testing for high-volume scenarios."""
    
    @pytest.fixture
    def load_test_registry(self):
        """Create registry optimized for load testing."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create lightweight adapters for load testing
        for i in range(3):
            docs = [
                {'id': f'load_{i}_{j}', 'content': f'Load test document {i}_{j}'}
                for j in range(20)
            ]
            adapter = InMemoryMockAdapter(f'load_engine_{i}', {
                'documents': docs,
                'response_delay': 0.001,  # Very fast for load testing
                'failure_rate': 0.0,
                'max_results': 20
            })
            registry._instances[f'load_engine_{i}'] = adapter
        
        return registry
    
    @pytest.mark.asyncio
    async def test_high_concurrency_load(self, load_test_registry):
        """Test performance under high concurrency load."""
        factory = FusionFactory(load_test_registry)
        adapters = list(load_test_registry.get_all_adapters().values())
        
        # Test with high concurrency
        concurrency_levels = [10, 25, 50, 100]
        load_results = {}
        
        for concurrency in concurrency_levels:
            # Create many concurrent queries
            tasks = [
                factory.process_query(f"load test {i}", adapters=adapters, limit=5)
                for i in range(concurrency)
            ]
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            # Analyze results
            successful = [r for r in results if isinstance(r, RankedResults)]
            failed = [r for r in results if isinstance(r, Exception)]
            
            duration = end_time - start_time
            
            load_results[concurrency] = {
                'duration': duration,
                'successful_queries': len(successful),
                'failed_queries': len(failed),
                'success_rate': len(successful) / concurrency,
                'queries_per_second': concurrency / duration,
                'avg_results_per_query': statistics.mean([r.total_results for r in successful]) if successful else 0
            }
        
        # Verify load handling
        for concurrency, metrics in load_results.items():
            # Should maintain reasonable success rate even under load
            assert metrics['success_rate'] >= 0.8  # At least 80% success rate
            assert metrics['queries_per_second'] > 5.0  # Reasonable throughput
            assert metrics['avg_results_per_query'] > 0  # Should return results
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, load_test_registry):
        """Test performance under sustained load over time."""
        factory = FusionFactory(load_test_registry)
        adapters = list(load_test_registry.get_all_adapters().values())
        
        # Run sustained load for 30 seconds
        test_duration = 10.0  # 10 seconds for testing
        start_time = time.perf_counter()
        end_time = start_time + test_duration
        
        query_count = 0
        successful_queries = 0
        failed_queries = 0
        response_times = []
        
        while time.perf_counter() < end_time:
            query_start = time.perf_counter()
            
            try:
                result = await factory.process_query(
                    f"sustained load {query_count}",
                    adapters=adapters,
                    limit=5
                )
                successful_queries += 1
                
                query_end = time.perf_counter()
                response_times.append(query_end - query_start)
                
            except Exception:
                failed_queries += 1
            
            query_count += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        actual_duration = time.perf_counter() - start_time
        
        # Calculate sustained load metrics
        sustained_metrics = {
            'total_queries': query_count,
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'success_rate': successful_queries / query_count if query_count > 0 else 0,
            'queries_per_second': query_count / actual_duration,
            'mean_response_time': statistics.mean(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'response_time_p95': sorted(response_times)[int(0.95 * len(response_times))] if response_times else 0
        }
        
        # Verify sustained performance
        assert sustained_metrics['success_rate'] >= 0.9  # High success rate
        assert sustained_metrics['queries_per_second'] > 10.0  # Good throughput
        assert sustained_metrics['mean_response_time'] < 0.5  # Fast responses
        assert sustained_metrics['response_time_p95'] < 1.0  # 95th percentile under 1 second
    
    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self, load_test_registry):
        """Test memory stability under sustained load."""
        factory = FusionFactory(load_test_registry)
        adapters = list(load_test_registry.get_all_adapters().values())
        
        benchmark = PerformanceBenchmark()
        
        # Measure initial memory
        gc.collect()
        initial_memory = benchmark.get_memory_usage()
        
        # Run load test
        memory_samples = []
        
        for i in range(100):  # 100 queries
            await factory.process_query(f"memory test {i}", adapters=adapters, limit=5)
            
            # Sample memory every 10 queries
            if i % 10 == 0:
                gc.collect()
                current_memory = benchmark.get_memory_usage()
                memory_samples.append(current_memory['rss_mb'])
        
        # Final memory measurement
        gc.collect()
        final_memory = benchmark.get_memory_usage()
        
        # Analyze memory usage
        memory_growth = final_memory['rss_mb'] - initial_memory['rss_mb']
        max_memory = max(memory_samples)
        memory_variance = statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0.0
        
        # Memory stability assertions
        assert memory_growth < 100.0  # Should not grow more than 100MB
        assert memory_variance < 50.0  # Should be relatively stable
        assert max_memory < initial_memory['rss_mb'] + 150.0  # Reasonable peak usage


class TestAdvancedPerformanceScenarios:
    """Advanced performance testing scenarios."""
    
    @pytest.fixture
    def mixed_performance_registry(self):
        """Create registry with mixed performance characteristics for advanced testing."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # High-performance engine
        high_perf_docs = [{'id': f'hp_{i}', 'content': f'High performance doc {i}'} for i in range(200)]
        high_perf_engine = InMemoryMockAdapter('high_perf_engine', {
            'documents': high_perf_docs,
            'response_delay': 0.001,  # Very fast
            'failure_rate': 0.0,
            'max_results': 100
        })
        
        # Variable performance engine
        var_perf_docs = [{'id': f'vp_{i}', 'content': f'Variable performance doc {i}'} for i in range(150)]
        var_perf_engine = InMemoryMockAdapter('var_perf_engine', {
            'documents': var_perf_docs,
            'response_delay': 0.02,  # Base delay
            'failure_rate': 0.05,    # 5% failure rate
            'max_results': 75
        })
        
        # Realistic performance engine
        real_perf_docs = [{'id': f'rp_{i}', 'content': f'Realistic performance doc {i}'} for i in range(300)]
        real_perf_engine = InMemoryMockAdapter('real_perf_engine', {
            'documents': real_perf_docs,
            'response_delay': 0.05,  # Realistic delay
            'failure_rate': 0.02,    # 2% failure rate
            'max_results': 50
        })
        
        registry._instances['high_perf_engine'] = high_perf_engine
        registry._instances['var_perf_engine'] = var_perf_engine
        registry._instances['real_perf_engine'] = real_perf_engine
        
        return registry
    
    @pytest.mark.asyncio
    async def test_mixed_engine_performance_optimization(self, mixed_performance_registry):
        """Test performance optimization with mixed engine characteristics."""
        factory = FusionFactory(mixed_performance_registry)
        adapters = list(mixed_performance_registry.get_all_adapters().values())
        
        # Test different weight strategies for performance optimization
        weight_strategies = {
            'favor_fast': np.array([0.6, 0.2, 0.2]),      # Favor high-performance engine
            'balanced': np.array([0.33, 0.33, 0.34]),     # Balanced approach
            'favor_comprehensive': np.array([0.2, 0.3, 0.5])  # Favor realistic engine
        }
        
        performance_results = {}
        
        for strategy_name, weights in weight_strategies.items():
            times = []
            result_qualities = []
            
            for i in range(10):
                start_time = time.perf_counter()
                results = await factory.process_query(
                    f"optimization test {i}",
                    adapters=adapters,
                    weights=weights,
                    limit=20
                )
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                # Quality metric: average confidence and result count
                quality = (
                    statistics.mean(results.confidence_scores) if results.confidence_scores else 0.0
                ) * (results.total_results / 20.0)  # Normalize by expected results
                result_qualities.append(quality)
            
            performance_results[strategy_name] = {
                'mean_time': statistics.mean(times),
                'mean_quality': statistics.mean(result_qualities),
                'efficiency': statistics.mean(result_qualities) / statistics.mean(times)  # Quality per second
            }
        
        # Verify performance characteristics
        # Fast-favoring strategy should be fastest (with some tolerance for timing variations)
        fast_time = performance_results['favor_fast']['mean_time']
        balanced_time = performance_results['balanced']['mean_time']
        assert fast_time <= balanced_time * 1.1  # Allow 10% tolerance
        
        # Comprehensive strategy might be slower but should have good quality
        assert performance_results['favor_comprehensive']['mean_quality'] > 0.0
        
        # All strategies should have reasonable efficiency
        for strategy, metrics in performance_results.items():
            assert metrics['efficiency'] > 0.0
    
    @pytest.mark.asyncio
    async def test_adaptive_timeout_performance(self, mixed_performance_registry):
        """Test performance with adaptive timeout strategies."""
        factory = FusionFactory(mixed_performance_registry)
        adapters = list(mixed_performance_registry.get_all_adapters().values())
        
        # Test different timeout strategies
        timeout_strategies = [0.01, 0.05, 0.1, 0.5, 1.0]  # Various timeout values
        
        timeout_results = {}
        
        for timeout in timeout_strategies:
            successful_queries = 0
            total_time = 0
            total_results = 0
            
            for i in range(10):
                try:
                    start_time = time.perf_counter()
                    results = await factory.process_query(
                        f"timeout test {i}",
                        adapters=adapters,
                        limit=10,
                        timeout=timeout
                    )
                    end_time = time.perf_counter()
                    
                    successful_queries += 1
                    total_time += (end_time - start_time)
                    total_results += results.total_results
                    
                except asyncio.TimeoutError:
                    # Expected for very short timeouts
                    pass
                except Exception:
                    # Other errors
                    pass
            
            timeout_results[timeout] = {
                'success_rate': successful_queries / 10,
                'avg_time': total_time / successful_queries if successful_queries > 0 else 0,
                'avg_results': total_results / successful_queries if successful_queries > 0 else 0
            }
        
        # Verify timeout behavior
        # Longer timeouts should have higher success rates
        success_rates = [metrics['success_rate'] for metrics in timeout_results.values()]
        assert success_rates == sorted(success_rates)  # Should be non-decreasing
        
        # Very short timeouts might have low success rates
        assert timeout_results[0.01]['success_rate'] <= timeout_results[1.0]['success_rate']
    
    @pytest.mark.asyncio
    async def test_query_complexity_performance(self, mixed_performance_registry):
        """Test performance with different query complexity levels."""
        factory = FusionFactory(mixed_performance_registry)
        adapters = list(mixed_performance_registry.get_all_adapters().values())
        
        # Different query complexity scenarios
        query_scenarios = {
            'simple': {
                'queries': ['test', 'document', 'search'],
                'limit': 5
            },
            'moderate': {
                'queries': ['machine learning algorithms', 'data science techniques', 'artificial intelligence'],
                'limit': 10
            },
            'complex': {
                'queries': [
                    'advanced machine learning algorithms for natural language processing',
                    'deep learning neural networks with attention mechanisms',
                    'reinforcement learning agents in multi-agent environments'
                ],
                'limit': 20
            }
        }
        
        complexity_results = {}
        
        for scenario_name, scenario in query_scenarios.items():
            times = []
            result_counts = []
            
            for query in scenario['queries']:
                start_time = time.perf_counter()
                results = await factory.process_query(
                    query,
                    adapters=adapters,
                    limit=scenario['limit']
                )
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                result_counts.append(results.total_results)
            
            complexity_results[scenario_name] = {
                'mean_time': statistics.mean(times),
                'mean_results': statistics.mean(result_counts),
                'throughput': statistics.mean(result_counts) / statistics.mean(times)
            }
        
        # Verify complexity handling
        # All scenarios should complete successfully
        for scenario, metrics in complexity_results.items():
            assert metrics['mean_time'] > 0
            assert metrics['mean_results'] > 0
            assert metrics['throughput'] > 0
        
        # More complex queries might take longer but should still be reasonable
        assert complexity_results['complex']['mean_time'] < 2.0  # Should complete within 2 seconds


class TestScalabilityBenchmarks:
    """Scalability benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_engine_count_scalability(self):
        """Test scalability with increasing number of engines."""
        scalability_results = {}
        
        # Test with different numbers of engines
        for num_engines in [1, 2, 3, 5, 8]:
            registry = AdapterRegistry()
            registry._instances.clear()
            
            # Create specified number of engines
            for i in range(num_engines):
                docs = [{'id': f'scale_{i}_{j}', 'content': f'Scalability doc {i}_{j}'} for j in range(50)]
                engine = InMemoryMockAdapter(f'scale_engine_{i}', {
                    'documents': docs,
                    'response_delay': 0.01,
                    'failure_rate': 0.0,
                    'max_results': 25
                })
                registry._instances[f'scale_engine_{i}'] = engine
            
            factory = FusionFactory(registry)
            adapters = list(registry.get_all_adapters().values())
            
            # Measure performance
            times = []
            for i in range(5):
                start_time = time.perf_counter()
                results = await factory.process_query(f"scalability test {i}", adapters=adapters, limit=10)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                assert isinstance(results, RankedResults)
            
            scalability_results[num_engines] = {
                'mean_time': statistics.mean(times),
                'engines_per_second': num_engines / statistics.mean(times)
            }
        
        # Verify scalability characteristics
        # Performance should not degrade linearly with engine count (due to concurrency)
        single_engine_time = scalability_results[1]['mean_time']
        eight_engine_time = scalability_results[8]['mean_time']
        
        # 8 engines should not take 8x as long (should benefit from concurrency)
        assert eight_engine_time < single_engine_time * 4.0
        
        # Should maintain reasonable throughput even with many engines
        assert scalability_results[8]['engines_per_second'] > 2.0
    
    @pytest.mark.asyncio
    async def test_document_count_scalability(self):
        """Test scalability with increasing document counts."""
        document_counts = [100, 500, 1000, 2000, 5000]
        scalability_results = {}
        
        for doc_count in document_counts:
            registry = AdapterRegistry()
            registry._instances.clear()
            
            # Create engine with specified document count
            docs = [
                {'id': f'doc_{i}', 'content': f'Document {i} content for scalability testing'}
                for i in range(doc_count)
            ]
            
            engine = InMemoryMockAdapter('scalability_engine', {
                'documents': docs,
                'response_delay': 0.001,  # Minimal delay to focus on document processing
                'failure_rate': 0.0,
                'max_results': min(100, doc_count)  # Reasonable result limit
            })
            
            registry._instances['scalability_engine'] = engine
            factory = FusionFactory(registry)
            
            # Measure performance
            times = []
            for i in range(3):  # Fewer iterations for large datasets
                start_time = time.perf_counter()
                results = await factory.process_query(f"document scalability test {i}", adapters=[engine], limit=50)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                assert isinstance(results, RankedResults)
            
            scalability_results[doc_count] = {
                'mean_time': statistics.mean(times),
                'docs_per_second': doc_count / statistics.mean(times)
            }
        
        # Verify document scalability
        # Performance should not degrade linearly with document count
        small_dataset_time = scalability_results[100]['mean_time']
        large_dataset_time = scalability_results[5000]['mean_time']
        
        # 50x more documents should not take 50x as long
        assert large_dataset_time < small_dataset_time * 10.0
        
        # Should maintain reasonable processing rate
        assert scalability_results[5000]['docs_per_second'] > 1000.0


class TestRealWorldPerformanceScenarios:
    """Performance tests for real-world usage scenarios."""
    
    @pytest.fixture
    def realistic_workload_registry(self):
        """Create registry simulating realistic workload patterns."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Simulate different types of search engines with realistic characteristics
        
        # Fast in-memory engine (like Redis/Elasticsearch with good caching)
        fast_docs = [
            {'id': f'fast_{i}', 'title': f'Fast Document {i}', 'content': f'Fast searchable content {i}'}
            for i in range(1000)
        ]
        fast_engine = InMemoryMockAdapter('fast_realistic_engine', {
            'documents': fast_docs,
            'response_delay': 0.005,  # 5ms - very fast
            'failure_rate': 0.001,    # 0.1% failure rate
            'max_results': 100
        })
        
        # Medium database engine (like PostgreSQL full-text search)
        medium_docs = [
            {'id': f'med_{i}', 'title': f'Medium Document {i}', 'content': f'Medium searchable content {i}'}
            for i in range(2000)
        ]
        medium_engine = InMemoryMockAdapter('medium_realistic_engine', {
            'documents': medium_docs,
            'response_delay': 0.050,  # 50ms - typical database query
            'failure_rate': 0.005,    # 0.5% failure rate
            'max_results': 50
        })
        
        # Slow external API engine (like external search service)
        slow_docs = [
            {'id': f'slow_{i}', 'title': f'Slow Document {i}', 'content': f'Slow external content {i}'}
            for i in range(500)
        ]
        slow_engine = InMemoryMockAdapter('slow_realistic_engine', {
            'documents': slow_docs,
            'response_delay': 0.200,  # 200ms - external API call
            'failure_rate': 0.02,     # 2% failure rate
            'max_results': 20
        })
        
        registry._instances['fast_realistic_engine'] = fast_engine
        registry._instances['medium_realistic_engine'] = medium_engine
        registry._instances['slow_realistic_engine'] = slow_engine
        
        return registry
    
    @pytest.mark.asyncio
    async def test_realistic_user_session_simulation(self, realistic_workload_registry):
        """Simulate realistic user session with multiple queries."""
        factory = FusionFactory(realistic_workload_registry)
        adapters = list(realistic_workload_registry.get_all_adapters().values())
        
        # Simulate user session with typical query patterns
        user_queries = [
            "machine learning algorithms",
            "deep learning neural networks",
            "python programming tutorial",
            "data science best practices",
            "artificial intelligence applications",
            "natural language processing",
            "computer vision techniques",
            "reinforcement learning examples"
        ]
        
        session_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'total_time': 0.0,
            'response_times': [],
            'result_counts': []
        }
        
        session_start = time.time()
        
        for query in user_queries:
            query_start = time.time()
            
            try:
                results = await factory.process_query(query, adapters=adapters, limit=10)
                query_end = time.time()
                
                response_time = query_end - query_start
                session_metrics['successful_queries'] += 1
                session_metrics['response_times'].append(response_time)
                session_metrics['result_counts'].append(results.total_results)
                
            except Exception as e:
                logger.warning(f"Query failed: {query} - {e}")
            
            session_metrics['total_queries'] += 1
            
            # Simulate user think time between queries
            await asyncio.sleep(0.1)
        
        session_end = time.time()
        session_metrics['total_time'] = session_end - session_start
        
        # Verify session performance
        success_rate = session_metrics['successful_queries'] / session_metrics['total_queries']
        avg_response_time = statistics.mean(session_metrics['response_times'])
        avg_results = statistics.mean(session_metrics['result_counts'])
        
        assert success_rate >= 0.9  # At least 90% success rate
        assert avg_response_time < 1.0  # Average response under 1 second
        assert avg_results > 0  # Should get some results
        
        # Log session summary
        logger.info(f"Session summary: {session_metrics['successful_queries']}/{session_metrics['total_queries']} queries successful, "
                   f"avg response time: {avg_response_time:.3f}s, avg results: {avg_results:.1f}")
    
    @pytest.mark.asyncio
    async def test_peak_traffic_simulation(self, realistic_workload_registry):
        """Simulate peak traffic conditions."""
        factory = FusionFactory(realistic_workload_registry)
        adapters = list(realistic_workload_registry.get_all_adapters().values())
        
        # Simulate peak traffic with burst of concurrent users
        concurrent_users = 20
        queries_per_user = 5
        
        async def simulate_user_burst(user_id: int):
            user_metrics = {
                'user_id': user_id,
                'successful_queries': 0,
                'failed_queries': 0,
                'total_response_time': 0.0
            }
            
            for i in range(queries_per_user):
                query_start = time.time()
                
                try:
                    results = await factory.process_query(
                        f"peak traffic query {user_id}_{i}",
                        adapters=adapters,
                        limit=5
                    )
                    query_end = time.time()
                    
                    user_metrics['successful_queries'] += 1
                    user_metrics['total_response_time'] += (query_end - query_start)
                    
                except Exception:
                    user_metrics['failed_queries'] += 1
                
                # Small random delay to simulate realistic user behavior
                await asyncio.sleep(0.01 + (user_id % 3) * 0.01)
            
            return user_metrics
        
        # Execute peak traffic simulation
        peak_start = time.time()
        user_tasks = [simulate_user_burst(user_id) for user_id in range(concurrent_users)]
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        peak_end = time.time()
        
        # Analyze peak traffic results
        successful_users = [r for r in user_results if isinstance(r, dict)]
        total_successful_queries = sum(u['successful_queries'] for u in successful_users)
        total_failed_queries = sum(u['failed_queries'] for u in successful_users)
        total_queries = total_successful_queries + total_failed_queries
        
        peak_duration = peak_end - peak_start
        queries_per_second = total_queries / peak_duration
        success_rate = total_successful_queries / total_queries if total_queries > 0 else 0
        
        # Peak traffic performance assertions
        assert success_rate >= 0.8  # At least 80% success rate under peak load
        assert queries_per_second > 10.0  # Should handle at least 10 QPS
        assert len(successful_users) >= concurrent_users * 0.8  # Most users should complete
        
        logger.info(f"Peak traffic: {queries_per_second:.1f} QPS, {success_rate:.1%} success rate")
    
    @pytest.mark.asyncio
    async def test_degraded_performance_scenarios(self, realistic_workload_registry):
        """Test performance under degraded conditions."""
        factory = FusionFactory(realistic_workload_registry)
        adapters = list(realistic_workload_registry.get_all_adapters().values())
        
        # Test with one engine failing
        failing_adapter = adapters[0]
        original_search = failing_adapter.search
        
        async def failing_search(query, limit=10, **kwargs):
            # Simulate intermittent failures
            if hash(query) % 3 == 0:  # Fail 1/3 of the time
                raise Exception("Simulated engine failure")
            return await original_search(query, limit, **kwargs)
        
        failing_adapter.search = failing_search
        
        try:
            degraded_metrics = {
                'successful_queries': 0,
                'failed_queries': 0,
                'response_times': []
            }
            
            # Run queries under degraded conditions
            for i in range(20):
                query_start = time.time()
                
                try:
                    results = await factory.process_query(
                        f"degraded test {i}",
                        adapters=adapters,
                        limit=5
                    )
                    query_end = time.time()
                    
                    degraded_metrics['successful_queries'] += 1
                    degraded_metrics['response_times'].append(query_end - query_start)
                    
                except Exception:
                    degraded_metrics['failed_queries'] += 1
            
            # Verify graceful degradation
            total_queries = degraded_metrics['successful_queries'] + degraded_metrics['failed_queries']
            success_rate = degraded_metrics['successful_queries'] / total_queries
            
            assert success_rate >= 0.6  # Should still work with degraded performance
            
            if degraded_metrics['response_times']:
                avg_response_time = statistics.mean(degraded_metrics['response_times'])
                assert avg_response_time < 2.0  # Should still be reasonably fast
        
        finally:
            # Restore original search method
            failing_adapter.search = original_search
    
    @pytest.mark.asyncio
    async def test_resource_utilization_monitoring(self, realistic_workload_registry):
        """Test resource utilization during sustained operations."""
        factory = FusionFactory(realistic_workload_registry)
        adapters = list(realistic_workload_registry.get_all_adapters().values())
        
        # Monitor resource utilization
        process = psutil.Process()
        resource_samples = []
        
        async def monitor_resources():
            while True:
                try:
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent()
                    
                    resource_samples.append({
                        'timestamp': time.time(),
                        'memory_mb': memory_info.rss / 1024 / 1024,
                        'cpu_percent': cpu_percent,
                        'num_threads': process.num_threads()
                    })
                    
                    await asyncio.sleep(0.5)  # Sample every 500ms
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(monitor_resources())
        
        try:
            # Run sustained workload
            workload_duration = 10.0  # 10 seconds
            workload_start = time.time()
            
            query_count = 0
            while time.time() - workload_start < workload_duration:
                try:
                    await factory.process_query(
                        f"resource test {query_count}",
                        adapters=adapters,
                        limit=5
                    )
                    query_count += 1
                    await asyncio.sleep(0.1)  # 10 QPS rate
                except Exception:
                    pass
        
        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Analyze resource utilization
        if resource_samples:
            memory_values = [s['memory_mb'] for s in resource_samples]
            cpu_values = [s['cpu_percent'] for s in resource_samples if s['cpu_percent'] > 0]
            
            max_memory = max(memory_values)
            avg_memory = statistics.mean(memory_values)
            memory_growth = max_memory - min(memory_values)
            
            if cpu_values:
                avg_cpu = statistics.mean(cpu_values)
                max_cpu = max(cpu_values)
            else:
                avg_cpu = max_cpu = 0
            
            # Resource utilization assertions
            assert memory_growth < 50.0  # Memory growth should be reasonable
            assert max_memory < 500.0  # Should not use excessive memory
            assert avg_cpu < 80.0  # Average CPU should be reasonable
            
            logger.info(f"Resource utilization - Memory: {avg_memory:.1f}MB avg, {max_memory:.1f}MB max, "
                       f"CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% max")


class TestPerformanceRegression:
    """Performance regression testing to ensure performance doesn't degrade."""
    
    @pytest.fixture
    def baseline_performance_registry(self):
        """Create registry for baseline performance testing."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Standardized test setup for consistent benchmarking
        baseline_docs = [
            {'id': f'baseline_{i}', 'title': f'Baseline Document {i}', 'content': f'Baseline content {i}'}
            for i in range(100)
        ]
        
        baseline_engine = InMemoryMockAdapter('baseline_engine', {
            'documents': baseline_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 50
        })
        
        registry._instances['baseline_engine'] = baseline_engine
        return registry
    
    @pytest.mark.asyncio
    async def test_baseline_query_performance(self, baseline_performance_registry):
        """Establish baseline query performance metrics."""
        factory = FusionFactory(baseline_performance_registry)
        adapters = list(baseline_performance_registry.get_all_adapters().values())
        
        # Run standardized performance test
        num_queries = 50
        response_times = []
        
        for i in range(num_queries):
            query_start = time.time()
            results = await factory.process_query(f"baseline test {i}", adapters=adapters, limit=10)
            query_end = time.time()
            
            response_times.append(query_end - query_start)
            assert isinstance(results, RankedResults)
        
        # Calculate baseline metrics
        avg_response_time = statistics.mean(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        p99_response_time = sorted(response_times)[int(0.99 * len(response_times))]
        
        # Baseline performance expectations (relaxed for test environment)
        assert avg_response_time < 0.2  # Average under 200ms
        assert p95_response_time < 0.4  # 95th percentile under 400ms
        assert p99_response_time < 1.0  # 99th percentile under 1 second
        
        logger.info(f"Baseline performance - Avg: {avg_response_time:.3f}s, "
                   f"P95: {p95_response_time:.3f}s, P99: {p99_response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_performance_consistency(self, baseline_performance_registry):
        """Test performance consistency across multiple runs."""
        factory = FusionFactory(baseline_performance_registry)
        adapters = list(baseline_performance_registry.get_all_adapters().values())
        
        # Run multiple test rounds
        num_rounds = 5
        queries_per_round = 20
        round_metrics = []
        
        for round_num in range(num_rounds):
            round_times = []
            
            for i in range(queries_per_round):
                query_start = time.time()
                results = await factory.process_query(
                    f"consistency test {round_num}_{i}",
                    adapters=adapters,
                    limit=10
                )
                query_end = time.time()
                
                round_times.append(query_end - query_start)
                assert isinstance(results, RankedResults)
            
            round_avg = statistics.mean(round_times)
            round_std = statistics.stdev(round_times) if len(round_times) > 1 else 0
            
            round_metrics.append({
                'round': round_num,
                'avg_time': round_avg,
                'std_time': round_std,
                'min_time': min(round_times),
                'max_time': max(round_times)
            })
        
        # Analyze consistency
        avg_times = [r['avg_time'] for r in round_metrics]
        overall_avg = statistics.mean(avg_times)
        consistency_variance = statistics.stdev(avg_times) if len(avg_times) > 1 else 0
        
        # Consistency assertions (relaxed for test environment)
        assert consistency_variance < 0.1  # Low variance between rounds
        assert overall_avg < 0.2  # Consistent with baseline expectations
        
        # No round should be significantly slower than others
        for metrics in round_metrics:
            assert metrics['avg_time'] < overall_avg * 1.5  # Within 50% of average
        
        logger.info(f"Performance consistency - Overall avg: {overall_avg:.3f}s, "
                   f"Variance: {consistency_variance:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for performance tests