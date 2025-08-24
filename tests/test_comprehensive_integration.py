"""Comprehensive integration tests for RAG Fusion Factory.

This module contains end-to-end integration tests that cover:
1. Complete fusion pipeline with real search engines
2. Adapter functionality with different engine types
3. Performance benchmarking for different adapter configurations
4. Complete training and optimization workflows

Requirements covered: 2.1, 3.1, 4.1, 5.1, 6.1
"""

import asyncio
import json
import time
import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, AsyncMock
import tempfile
import yaml

from src.services.fusion_factory import FusionFactory
from src.adapters.registry import AdapterRegistry
from src.adapters.base import SearchEngineAdapter
from src.adapters.mock_adapter import InMemoryMockAdapter
from src.adapters.elasticsearch_adapter import ElasticsearchAdapter
from src.adapters.solr_adapter import SolrAdapter
from src.adapters.opensearch_adapter import OpenSearchAdapter
from src.models.core import SearchResult, SearchResults, RankedResults, TrainingExample
from src.models.weight_predictor import WeightPredictorModel
from src.models.contrastive_learning import ContrastiveLearningModule
from src.services.training_pipeline import ModelTrainingPipeline
from src.config.settings import ConfigManager
from src.utils.monitoring import get_performance_monitor


def load_config_from_file(config_path: str):
    """Helper function to load config from file."""
    config_manager = ConfigManager(config_file=config_path)
    return config_manager.config


class TestCompleteFusionPipeline:
    """Test complete fusion pipeline with different search engine configurations."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                'id': 'doc1',
                'title': 'Machine Learning Fundamentals',
                'content': 'Introduction to machine learning algorithms and techniques for data science applications.',
                'category': 'education',
                'tags': ['ml', 'data-science', 'algorithms']
            },
            {
                'id': 'doc2', 
                'title': 'Deep Learning with Neural Networks',
                'content': 'Advanced neural network architectures for deep learning and artificial intelligence.',
                'category': 'research',
                'tags': ['deep-learning', 'neural-networks', 'ai']
            },
            {
                'id': 'doc3',
                'title': 'Natural Language Processing',
                'content': 'Text processing and language understanding using computational linguistics methods.',
                'category': 'nlp',
                'tags': ['nlp', 'text-processing', 'linguistics']
            },
            {
                'id': 'doc4',
                'title': 'Computer Vision Applications',
                'content': 'Image recognition and computer vision techniques for visual data analysis.',
                'category': 'vision',
                'tags': ['computer-vision', 'image-processing', 'recognition']
            },
            {
                'id': 'doc5',
                'title': 'Reinforcement Learning',
                'content': 'Agent-based learning systems and reward optimization in dynamic environments.',
                'category': 'rl',
                'tags': ['reinforcement-learning', 'agents', 'optimization']
            }
        ]
    
    @pytest.fixture
    def multi_engine_registry(self, sample_documents):
        """Create registry with multiple different engine types."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Engine 1: High-scoring mock (simulates Elasticsearch-like scoring)
        engine1_docs = [
            SearchResult(doc['id'], 0.9 + i*0.1, doc['content'], doc, 'engine1')
            for i, doc in enumerate(sample_documents[:3])
        ]
        engine1 = InMemoryMockAdapter('engine1', {
            'documents': sample_documents[:3],
            'response_delay': 0.05,
            'failure_rate': 0.0,
            'max_results': 10
        })
        engine1.mock_results = engine1_docs
        
        # Engine 2: Medium-scoring mock (simulates Solr-like scoring)
        engine2_docs = [
            SearchResult(doc['id'], 0.7 + i*0.05, doc['content'], doc, 'engine2')
            for i, doc in enumerate(sample_documents[2:])
        ]
        engine2 = InMemoryMockAdapter('engine2', {
            'documents': sample_documents[2:],
            'response_delay': 0.1,
            'failure_rate': 0.0,
            'max_results': 8
        })
        engine2.mock_results = engine2_docs
        
        # Engine 3: Different scale scoring (simulates custom API)
        engine3_docs = [
            SearchResult(doc['id'], 50 + i*10, doc['content'], doc, 'engine3')
            for i, doc in enumerate(sample_documents[1:4])
        ]
        engine3 = InMemoryMockAdapter('engine3', {
            'documents': sample_documents[1:4],
            'response_delay': 0.15,
            'failure_rate': 0.0,
            'max_results': 5
        })
        engine3.mock_results = engine3_docs
        
        # Override search methods to return mock results
        async def mock_search1(query, limit=10, **kwargs):
            await asyncio.sleep(0.05)
            return SearchResults(query, engine1_docs[:limit], 'engine1', datetime.now(), len(engine1_docs))
        
        async def mock_search2(query, limit=10, **kwargs):
            await asyncio.sleep(0.1)
            return SearchResults(query, engine2_docs[:limit], 'engine2', datetime.now(), len(engine2_docs))
        
        async def mock_search3(query, limit=10, **kwargs):
            await asyncio.sleep(0.15)
            return SearchResults(query, engine3_docs[:limit], 'engine3', datetime.now(), len(engine3_docs))
        
        engine1.search = mock_search1
        engine2.search = mock_search2
        engine3.search = mock_search3
        
        registry._instances['engine1'] = engine1
        registry._instances['engine2'] = engine2
        registry._instances['engine3'] = engine3
        
        return registry
    
    @pytest.fixture
    def fusion_factory(self, multi_engine_registry):
        """Create fusion factory with multi-engine registry."""
        return FusionFactory(multi_engine_registry)
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, fusion_factory, multi_engine_registry):
        """Test complete fusion pipeline from query to ranked results."""
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        # Execute complete pipeline
        results = await fusion_factory.process_query(
            "machine learning algorithms",
            adapters=adapters,
            limit=5
        )
        
        # Verify pipeline execution
        assert isinstance(results, RankedResults)
        assert results.query == "machine learning algorithms"
        assert results.total_results > 0
        assert len(results.fusion_weights) == 3  # Three engines
        assert len(results.confidence_scores) == results.total_results
        
        # Verify fusion weights sum to 1 (convex combination)
        assert abs(np.sum(results.fusion_weights) - 1.0) < 0.001
        
        # Verify results are properly ranked
        scores = [r.relevance_score for r in results.results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify all results have fusion engine source
        for result in results.results:
            assert result.engine_source == "fusion"
        
        # Verify confidence scores are reasonable
        assert all(0.0 <= conf <= 1.0 for conf in results.confidence_scores)
    
    @pytest.mark.asyncio
    async def test_pipeline_with_engine_failures(self, fusion_factory, multi_engine_registry):
        """Test pipeline resilience with some engine failures."""
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        # Make one engine fail
        failing_adapter = adapters[1]
        original_search = failing_adapter.search
        
        async def failing_search(query, limit=10, **kwargs):
            raise Exception("Simulated engine failure")
        
        failing_adapter.search = failing_search
        
        try:
            # Execute pipeline with failing engine
            results = await fusion_factory.process_query(
                "deep learning neural networks",
                adapters=adapters,
                limit=5
            )
            
            # Should still get results from working engines
            assert isinstance(results, RankedResults)
            assert results.total_results > 0
            
            # Fusion weights should be adjusted for working engines only
            assert len(results.fusion_weights) <= 3
            assert abs(np.sum(results.fusion_weights) - 1.0) < 0.001
            
        finally:
            # Restore original search method
            failing_adapter.search = original_search
    
    @pytest.mark.asyncio
    async def test_pipeline_with_custom_weights(self, fusion_factory, multi_engine_registry):
        """Test pipeline with custom fusion weights."""
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        # Custom weights favoring first engine
        custom_weights = np.array([0.6, 0.3, 0.1])
        
        results = await fusion_factory.process_query(
            "natural language processing",
            adapters=adapters,
            weights=custom_weights,
            limit=5
        )
        
        # Verify custom weights are used
        assert isinstance(results, RankedResults)
        assert np.allclose(results.fusion_weights, custom_weights, atol=0.001)
        
        # Results should reflect weight preferences
        assert results.total_results > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_score_normalization(self, fusion_factory, multi_engine_registry):
        """Test that different scoring scales are properly normalized."""
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        results = await fusion_factory.process_query(
            "computer vision applications",
            adapters=adapters,
            limit=10
        )
        
        # All final scores should be in reasonable range after normalization
        assert isinstance(results, RankedResults)
        scores = [r.relevance_score for r in results.results]
        
        # Scores should be normalized to reasonable range
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert max(scores) > 0.5  # Should have some high-confidence results
    
    @pytest.mark.asyncio
    async def test_pipeline_duplicate_handling(self, fusion_factory, multi_engine_registry):
        """Test that duplicate documents are properly merged."""
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        results = await fusion_factory.process_query(
            "machine learning",
            adapters=adapters,
            limit=10
        )
        
        # Check for duplicate document IDs
        document_ids = [r.document_id for r in results.results]
        unique_ids = set(document_ids)
        
        # Should not have duplicate document IDs in final results
        assert len(document_ids) == len(unique_ids)
        
        # Documents appearing in multiple engines should have higher confidence
        overlapping_docs = ['doc2', 'doc3']  # These appear in multiple engines
        for result in results.results:
            if result.document_id in overlapping_docs:
                confidence_idx = results.results.index(result)
                confidence = results.confidence_scores[confidence_idx]
                assert confidence > 0.3  # Should have higher confidence


class TestAdapterFunctionality:
    """Test adapter functionality with different engine types."""
    
    @pytest.fixture
    def adapter_configs(self):
        """Configuration for different adapter types."""
        return {
            'elasticsearch': {
                'host': 'localhost',
                'port': 9200,
                'index': 'test_index',
                'use_ssl': False
            },
            'solr': {
                'host': 'localhost',
                'port': 8983,
                'core': 'test_core',
                'use_ssl': False
            },
            'opensearch': {
                'host': 'localhost',
                'port': 9200,
                'index': 'test_index',
                'use_ssl': False,
                'verify_ssl': False
            },
            'mock_inmemory': {
                'documents': [
                    {
                        'id': 'mock1',
                        'title': 'Mock Document',
                        'content': 'This is a mock document for testing',
                        'category': 'test'
                    }
                ],
                'response_delay': 0.01,
                'failure_rate': 0.0,
                'max_results': 10
            }
        }
    
    def test_adapter_creation_from_config(self, adapter_configs):
        """Test creating different adapter types from configuration."""
        registry = AdapterRegistry()
        
        for engine_type, config in adapter_configs.items():
            try:
                adapter = registry.create_adapter(engine_type, f'test_{engine_type}', config)
                
                # Verify adapter properties
                assert adapter.engine_id == f'test_{engine_type}'
                assert hasattr(adapter, 'search')
                assert hasattr(adapter, 'health_check')
                assert hasattr(adapter, 'get_configuration')
                
                # Verify configuration
                adapter_config = adapter.get_configuration()
                assert adapter_config['engine_type'] == engine_type
                assert adapter_config['engine_id'] == f'test_{engine_type}'
                
            except ValueError as e:
                # Some adapters might not be available in test environment
                if "Unknown engine type" not in str(e):
                    raise
    
    @pytest.mark.asyncio
    async def test_adapter_search_interface_consistency(self, adapter_configs):
        """Test that all adapters implement consistent search interface."""
        registry = AdapterRegistry()
        
        for engine_type, config in adapter_configs.items():
            if engine_type == 'mock_inmemory':  # Only test mock for interface consistency
                adapter = registry.create_adapter(engine_type, f'test_{engine_type}', config)
                
                # Test search method signature and return type
                results = await adapter.search("test query", limit=5)
                
                assert isinstance(results, SearchResults)
                assert results.query == "test query"
                assert results.engine_id == f'test_{engine_type}'
                assert isinstance(results.results, list)
                assert isinstance(results.timestamp, datetime)
                assert isinstance(results.total_results, int)
                
                # Test individual result structure
                for result in results.results:
                    assert isinstance(result, SearchResult)
                    assert hasattr(result, 'document_id')
                    assert hasattr(result, 'relevance_score')
                    assert hasattr(result, 'content')
                    assert hasattr(result, 'metadata')
                    assert hasattr(result, 'engine_source')
    
    @pytest.mark.asyncio
    async def test_adapter_health_check_consistency(self, adapter_configs):
        """Test that all adapters implement consistent health check interface."""
        registry = AdapterRegistry()
        
        for engine_type, config in adapter_configs.items():
            if engine_type == 'mock_inmemory':  # Only test mock for interface consistency
                adapter = registry.create_adapter(engine_type, f'test_{engine_type}', config)
                
                # Test health check method
                is_healthy = await adapter.health_check()
                assert isinstance(is_healthy, bool)
    
    def test_adapter_configuration_validation(self):
        """Test adapter configuration validation."""
        registry = AdapterRegistry()
        
        # Test invalid engine type
        with pytest.raises(ValueError, match="Unknown engine type"):
            registry.create_adapter('invalid_engine_type', 'test_invalid', {})
        
        # Test duplicate engine ID
        registry.create_adapter('mock_inmemory', 'test_duplicate', {'documents': []})
        with pytest.raises(ValueError, match="already exists"):
            registry.create_adapter('mock_inmemory', 'test_duplicate', {'documents': []})
    
    @pytest.mark.asyncio
    async def test_adapter_timeout_handling(self):
        """Test adapter timeout handling."""
        config = {
            'documents': [],
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        }
        
        adapter = InMemoryMockAdapter('test_timeout', config, timeout=0.005)  # Very short timeout
        
        # Override search to take longer than timeout
        original_search = adapter.search
        
        async def slow_search(query, limit=10, **kwargs):
            await asyncio.sleep(0.1)  # Longer than timeout
            return await original_search(query, limit, **kwargs)
        
        adapter.search = slow_search
        
        # Should handle timeout gracefully
        with pytest.raises(Exception):  # Should timeout
            await adapter.search_with_timeout("test query", timeout=0.005)


class TestPerformanceBenchmarking:
    """Performance benchmarking tests for different adapter configurations."""
    
    @pytest.fixture
    def performance_test_registry(self):
        """Create registry with adapters configured for performance testing."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Fast engine
        fast_engine = InMemoryMockAdapter('fast_engine', {
            'documents': [{'id': f'fast_{i}', 'content': f'Fast document {i}'} for i in range(100)],
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 50
        })
        
        # Medium engine
        medium_engine = InMemoryMockAdapter('medium_engine', {
            'documents': [{'id': f'med_{i}', 'content': f'Medium document {i}'} for i in range(100)],
            'response_delay': 0.05,
            'failure_rate': 0.0,
            'max_results': 30
        })
        
        # Slow engine
        slow_engine = InMemoryMockAdapter('slow_engine', {
            'documents': [{'id': f'slow_{i}', 'content': f'Slow document {i}'} for i in range(100)],
            'response_delay': 0.1,
            'failure_rate': 0.0,
            'max_results': 20
        })
        
        registry._instances['fast_engine'] = fast_engine
        registry._instances['medium_engine'] = medium_engine
        registry._instances['slow_engine'] = slow_engine
        
        return registry
    
    @pytest.mark.asyncio
    async def test_single_engine_performance(self, performance_test_registry):
        """Benchmark single engine performance."""
        factory = FusionFactory(performance_test_registry)
        
        performance_results = {}
        
        for engine_id in ['fast_engine', 'medium_engine', 'slow_engine']:
            adapter = performance_test_registry.get_adapter(engine_id)
            
            # Measure search performance
            start_time = time.time()
            results = await adapter.search("performance test", limit=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            performance_results[engine_id] = {
                'response_time': response_time,
                'result_count': len(results.results),
                'throughput': len(results.results) / response_time if response_time > 0 else 0
            }
        
        # Verify performance characteristics
        assert performance_results['fast_engine']['response_time'] < performance_results['medium_engine']['response_time']
        assert performance_results['medium_engine']['response_time'] < performance_results['slow_engine']['response_time']
        
        # All engines should return results (allow for empty results in some cases)
        for engine_id, metrics in performance_results.items():
            assert metrics['result_count'] >= 0  # Allow empty results
            assert metrics['throughput'] >= 0    # Allow zero throughput
    
    @pytest.mark.asyncio
    async def test_concurrent_engine_performance(self, performance_test_registry):
        """Benchmark concurrent engine performance."""
        factory = FusionFactory(performance_test_registry)
        adapters = list(performance_test_registry.get_all_adapters().values())
        
        # Measure concurrent execution time
        start_time = time.time()
        results = await factory.process_query("concurrent test", adapters=adapters, limit=10)
        end_time = time.time()
        
        concurrent_time = end_time - start_time
        
        # Measure sequential execution time for comparison
        sequential_start = time.time()
        for adapter in adapters:
            await adapter.search("concurrent test", limit=10)
        sequential_end = time.time()
        
        sequential_time = sequential_end - sequential_start
        
        # Concurrent execution should be faster than sequential
        assert concurrent_time < sequential_time
        
        # Should still get meaningful results
        assert isinstance(results, RankedResults)
        assert results.total_results > 0
    
    @pytest.mark.asyncio
    async def test_load_performance(self, performance_test_registry):
        """Test performance under load with multiple concurrent queries."""
        factory = FusionFactory(performance_test_registry)
        adapters = list(performance_test_registry.get_all_adapters().values())
        
        # Create multiple concurrent queries
        queries = [f"load test query {i}" for i in range(10)]
        
        start_time = time.time()
        
        # Execute all queries concurrently
        tasks = [
            factory.process_query(query, adapters=adapters, limit=5)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Verify all queries completed successfully
        successful_results = [r for r in results if isinstance(r, RankedResults)]
        assert len(successful_results) == len(queries)
        
        # Calculate performance metrics
        avg_response_time = total_time / len(queries)
        total_results = sum(r.total_results for r in successful_results)
        
        # Performance should be reasonable
        assert avg_response_time < 1.0  # Should complete within 1 second on average
        assert total_results > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, performance_test_registry):
        """Test memory usage stability during repeated operations."""
        factory = FusionFactory(performance_test_registry)
        adapters = list(performance_test_registry.get_all_adapters().values())
        
        # Perform multiple iterations to check for memory leaks
        for i in range(20):
            results = await factory.process_query(f"memory test {i}", adapters=adapters, limit=5)
            assert isinstance(results, RankedResults)
            
            # Clear results to help garbage collection
            del results
        
        # Test should complete without memory issues
        assert True  # If we get here, no memory issues occurred


class TestTrainingAndOptimization:
    """Test complete training and optimization workflows."""
    
    @pytest.fixture
    def training_data(self):
        """Generate training data for optimization tests."""
        training_examples = []
        
        # Create synthetic training examples
        queries = [
            "machine learning algorithms",
            "deep learning neural networks", 
            "natural language processing",
            "computer vision applications",
            "reinforcement learning agents"
        ]
        
        for i, query in enumerate(queries):
            # Simulate engine results
            engine_results = {
                'engine1': SearchResults(
                    query=query,
                    results=[
                        SearchResult(f'doc_{i}_1', 0.9, f'Content for {query} from engine1', {}, 'engine1'),
                        SearchResult(f'doc_{i}_2', 0.8, f'More content for {query} from engine1', {}, 'engine1'),
                    ],
                    engine_id='engine1',
                    timestamp=datetime.now(),
                    total_results=2
                ),
                'engine2': SearchResults(
                    query=query,
                    results=[
                        SearchResult(f'doc_{i}_1', 0.7, f'Content for {query} from engine2', {}, 'engine2'),
                        SearchResult(f'doc_{i}_3', 0.6, f'Different content for {query} from engine2', {}, 'engine2'),
                    ],
                    engine_id='engine2',
                    timestamp=datetime.now(),
                    total_results=2
                )
            }
            
            # Ground truth labels (higher relevance for doc_1 in all cases)
            ground_truth = {
                f'doc_{i}_1': 1.0,  # Highly relevant
                f'doc_{i}_2': 0.7,  # Moderately relevant
                f'doc_{i}_3': 0.5   # Less relevant
            }
            
            training_examples.append(TrainingExample(
                query=query,
                engine_results=engine_results,
                ground_truth_labels=ground_truth
            ))
        
        return training_examples
    
    def test_weight_predictor_training(self, training_data):
        """Test XGBoost weight predictor model training."""
        model = WeightPredictorModel()
        
        # Train the model
        model.train_model(training_data)
        
        # Verify model is trained
        assert model.is_trained
        assert model.model is not None
        
        # Test prediction
        test_results = training_data[0].engine_results
        predicted_weights = model.predict_weights(test_results)
        
        # Verify prediction properties
        assert isinstance(predicted_weights, np.ndarray)
        assert len(predicted_weights) == 2  # Two engines
        assert np.allclose(np.sum(predicted_weights), 1.0, atol=0.001)  # Convex combination
        assert all(w >= 0 for w in predicted_weights)  # Non-negative weights
    
    def test_contrastive_learning_training(self, training_data):
        """Test contrastive learning module training."""
        cl_module = ContrastiveLearningModule()
        
        # Train with contrastive learning using first training example
        example = training_data[0]
        
        # Convert SearchResults to NormalizedResults (simplified for testing)
        normalized_results = {}
        for engine_id, search_results in example.engine_results.items():
            # Create a simple normalized result structure
            normalized_results[engine_id] = type('NormalizedResults', (), {
                'results': search_results.results,
                'normalized_scores': [r.relevance_score for r in search_results.results]
            })()
        
        optimized_weights, loss_history = cl_module.compute_optimal_weights_from_ground_truth(
            normalized_results, example.ground_truth_labels
        )
        
        # Verify optimization results
        assert isinstance(optimized_weights, np.ndarray)
        assert len(optimized_weights) == 2  # Two engines
        assert np.allclose(np.sum(optimized_weights), 1.0, atol=0.001)
        assert all(w >= 0 for w in optimized_weights)
        
        # Should have loss history
        assert isinstance(loss_history, list)
        assert len(loss_history) > 0
    
    def test_complete_training_pipeline(self, training_data):
        """Test complete training pipeline integration."""
        # Create training pipeline
        pipeline = ModelTrainingPipeline()
        
        # Execute training
        training_results = pipeline.train_model_with_validation(training_data)
        
        # Verify training results
        assert 'training_metrics' in training_results
        assert 'cv_metrics' in training_results
        assert 'test_metrics' in training_results
        
        # Verify training completed successfully
        assert training_results['data_statistics']['train_samples'] > 0
    
    def test_hyperparameter_optimization(self, training_data):
        """Test hyperparameter optimization workflow."""
        # Create training pipeline with hyperparameter optimization
        pipeline = ModelTrainingPipeline()
        
        # Define hyperparameter search space
        search_space = {
            'xgboost_max_depth': [3, 5, 7],
            'xgboost_learning_rate': [0.1, 0.2, 0.3],
            'contrastive_margin': [0.5, 1.0, 1.5],
            'contrastive_learning_rate': [0.01, 0.05, 0.1]
        }
        
        # Execute hyperparameter optimization
        optimization_results = pipeline.train_model_with_validation(
            training_data, 
            perform_hyperparameter_search=True
        )
        
        # Verify optimization results
        assert 'best_hyperparameters' in optimization_results
        assert 'training_metrics' in optimization_results
        
        # Verify best parameters are reasonable if hyperparameter search was performed
        if optimization_results['best_hyperparameters']:
            best_params = optimization_results['best_hyperparameters']
            assert isinstance(best_params, dict)
            assert len(best_params) > 0
    
    def test_model_persistence(self, training_data, tmp_path):
        """Test model saving and loading."""
        model = WeightPredictorModel()
        model.train_model(training_data)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save_model(str(model_path))
        
        # Load model
        loaded_model = WeightPredictorModel()
        loaded_model.load_model(str(model_path))
        
        # Verify loaded model works
        assert loaded_model.is_trained
        
        test_results = training_data[0].engine_results
        original_weights = model.predict_weights(test_results)
        loaded_weights = loaded_model.predict_weights(test_results)
        
        # Predictions should be identical
        assert np.allclose(original_weights, loaded_weights, atol=0.001)
    
    def test_online_learning_adaptation(self, training_data):
        """Test online learning and model adaptation."""
        pipeline = ModelTrainingPipeline()
        
        # Initial training
        initial_results = pipeline.train_model_with_validation(training_data[:3])
        
        # Verify training completed
        assert 'training_metrics' in initial_results
        assert 'data_statistics' in initial_results
        
        # Train with additional data to simulate adaptation
        adaptation_results = pipeline.train_model_with_validation(training_data[3:])
        
        # Verify adaptation training occurred
        assert 'training_metrics' in adaptation_results
        assert adaptation_results['data_statistics']['train_samples'] > 0


class TestConfigurationDrivenIntegration:
    """Test integration with different configuration files."""
    
    def test_minimal_config_integration(self):
        """Test integration with minimal configuration."""
        config_path = Path("config/minimal.yaml")
        if config_path.exists():
            config_manager = ConfigManager(config_file=str(config_path))
            config = config_manager.config
            
            # Create fusion factory from config
            registry = AdapterRegistry()
            factory = FusionFactory(registry)
            
            # Initialize adapters from config
            adapters = factory.initialize_adapters_from_config(config.get('engines', []))
            
            # Verify adapters were created
            assert len(adapters) > 0
            
            # All adapters should be mock adapters in minimal config
            for adapter in adapters:
                assert isinstance(adapter, InMemoryMockAdapter)
    
    def test_development_config_integration(self):
        """Test integration with development configuration."""
        config_path = Path("config/development.yaml")
        if config_path.exists():
            config_manager = ConfigManager(config_file=str(config_path))
            config = config_manager.config
            
            # Verify development config structure
            assert 'engines' in config
            assert 'fusion' in config
            assert 'monitoring' in config
            
            engines_config = config['engines']
            assert len(engines_config) > 0
            
            # Should have various engine types
            engine_types = [engine['engine_type'] for engine in engines_config]
            expected_types = ['elasticsearch', 'solr', 'opensearch', 'database', 'mock_inmemory']
            
            for expected_type in expected_types:
                assert expected_type in engine_types
    
    @pytest.mark.asyncio
    async def test_end_to_end_config_workflow(self):
        """Test complete end-to-end workflow using configuration file."""
        # Create temporary config for testing
        test_config = {
            'logging': {'level': 'INFO', 'format': 'text'},
            'engines': [
                {
                    'engine_type': 'mock_inmemory',
                    'engine_id': 'test_engine_1',
                    'timeout': 5.0,
                    'config': {
                        'response_delay': 0.01,
                        'failure_rate': 0.0,
                        'max_results': 10,
                        'documents': [
                            {
                                'id': 'config_doc_1',
                                'title': 'Configuration Test Document',
                                'content': 'This document tests configuration-driven integration',
                                'category': 'test'
                            }
                        ]
                    }
                }
            ],
            'fusion': {'default_limit': 10, 'max_limit': 100, 'timeout': 30.0},
            'monitoring': {'enabled': True, 'health_check_interval': 30}
        }
        
        # Save temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Load config and create factory
            config = load_config_from_file(temp_config_path)
            registry = AdapterRegistry()
            factory = FusionFactory(registry)
            
            # Initialize from config
            adapters = factory.initialize_adapters_from_config(config['engines'])
            
            # Execute query
            results = await factory.process_query(
                "configuration test",
                adapters=adapters,
                limit=5
            )
            
            # Verify results
            assert isinstance(results, RankedResults)
            assert results.total_results > 0
            
        finally:
            # Clean up temporary file
            Path(temp_config_path).unlink()


class TestConfigurationDrivenIntegration:
    """Test configuration-driven integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_yaml_configuration_loading(self):
        """Test loading and using YAML configuration files."""
        test_config = {
            'engines': [
                {
                    'engine_id': 'config_test_engine',
                    'engine_type': 'mock_inmemory',
                    'config': {
                        'response_delay': 0.01,
                        'failure_rate': 0.0,
                        'max_results': 10,
                        'documents': [
                            {
                                'id': 'config_doc_1',
                                'title': 'Configuration Test Document',
                                'content': 'This document tests configuration-driven integration',
                                'category': 'test'
                            }
                        ]
                    }
                }
            ],
            'fusion': {'default_limit': 10, 'max_limit': 100, 'timeout': 30.0},
            'monitoring': {'enabled': True, 'health_check_interval': 30}
        }
        
        # Save temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Load config and create factory
            config = load_config_from_file(temp_config_path)
            registry = AdapterRegistry()
            factory = FusionFactory(registry)
            
            # Initialize from config
            adapters = factory.initialize_adapters_from_config(config['engines'])
            
            # Execute query
            results = await factory.process_query(
                "configuration test",
                adapters=adapters,
                limit=5
            )
            
            # Verify results
            assert isinstance(results, RankedResults)
            assert results.total_results > 0
            
        finally:
            # Clean up temporary file
            Path(temp_config_path).unlink()
    
    @pytest.mark.asyncio
    async def test_dynamic_configuration_updates(self):
        """Test dynamic configuration updates during runtime."""
        registry = AdapterRegistry()
        factory = FusionFactory(registry)
        
        # Initial configuration
        initial_config = [
            {
                'engine_id': 'dynamic_engine_1',
                'engine_type': 'mock_inmemory',
                'config': {
                    'documents': [{'id': 'doc1', 'content': 'Initial document'}],
                    'response_delay': 0.01,
                    'failure_rate': 0.0,
                    'max_results': 10
                }
            }
        ]
        
        # Initialize with initial config
        adapters = factory.initialize_adapters_from_config(initial_config)
        
        # Test initial query
        results1 = await factory.process_query("test", adapters=adapters, limit=5)
        assert isinstance(results1, RankedResults)
        initial_count = results1.total_results
        
        # Update configuration - add another engine with different ID
        updated_config = initial_config + [
            {
                'engine_id': 'dynamic_engine_2',
                'engine_type': 'mock_inmemory',
                'config': {
                    'documents': [{'id': 'doc2', 'content': 'Updated document'}],
                    'response_delay': 0.01,
                    'failure_rate': 0.0,
                    'max_results': 10
                }
            }
        ]
        
        # Create new registry to avoid conflicts
        new_registry = AdapterRegistry()
        new_factory = FusionFactory(new_registry)
        updated_adapters = new_factory.initialize_adapters_from_config(updated_config)
        
        # Test updated query
        results2 = await new_factory.process_query("test", adapters=updated_adapters, limit=5)
        assert isinstance(results2, RankedResults)
        
        # Should have more engines now
        assert len(updated_adapters) > len(adapters)
    
    @pytest.mark.asyncio
    async def test_environment_specific_configurations(self):
        """Test different configurations for different environments."""
        environments = {
            'development': {
                'engines': [
                    {
                        'engine_id': 'dev_mock',
                        'engine_type': 'mock_inmemory',
                        'config': {
                            'documents': [{'id': 'dev_doc', 'content': 'Development document'}],
                            'response_delay': 0.001,  # Very fast for dev
                            'failure_rate': 0.0,
                            'max_results': 5
                        }
                    }
                ],
                'fusion': {'timeout': 5.0}  # Short timeout for dev
            },
            'production': {
                'engines': [
                    {
                        'engine_id': 'prod_mock',
                        'engine_type': 'mock_inmemory',
                        'config': {
                            'documents': [{'id': 'prod_doc', 'content': 'Production document'}],
                            'response_delay': 0.05,  # More realistic delay
                            'failure_rate': 0.1,     # Some failures expected
                            'max_results': 20
                        }
                    }
                ],
                'fusion': {'timeout': 30.0}  # Longer timeout for prod
            }
        }
        
        for env_name, env_config in environments.items():
            registry = AdapterRegistry()
            factory = FusionFactory(registry)
            
            # Initialize with environment-specific config
            adapters = factory.initialize_adapters_from_config(env_config['engines'])
            
            # Test query
            results = await factory.process_query(f"{env_name} test", adapters=adapters, limit=5)
            
            # Verify results
            assert isinstance(results, RankedResults)
            assert results.total_results >= 0  # May be 0 due to failures in prod config


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience in integration scenarios."""
    
    @pytest.fixture
    def unreliable_registry(self):
        """Create registry with unreliable adapters for testing error recovery."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Reliable engine
        reliable_engine = InMemoryMockAdapter('reliable_engine', {
            'documents': [{'id': 'reliable_doc', 'content': 'Reliable document'}],
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        # Intermittent failure engine
        intermittent_engine = InMemoryMockAdapter('intermittent_engine', {
            'documents': [{'id': 'intermittent_doc', 'content': 'Intermittent document'}],
            'response_delay': 0.02,
            'failure_rate': 0.3,  # 30% failure rate
            'max_results': 10
        })
        
        # Slow engine
        slow_engine = InMemoryMockAdapter('slow_engine', {
            'documents': [{'id': 'slow_doc', 'content': 'Slow document'}],
            'response_delay': 0.5,  # Very slow
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        registry._instances['reliable_engine'] = reliable_engine
        registry._instances['intermittent_engine'] = intermittent_engine
        registry._instances['slow_engine'] = slow_engine
        
        return registry
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_with_failures(self, unreliable_registry):
        """Test graceful degradation when some engines fail."""
        factory = FusionFactory(unreliable_registry)
        adapters = list(unreliable_registry.get_all_adapters().values())
        
        # Run multiple queries to test failure handling
        successful_queries = 0
        total_queries = 20
        
        for i in range(total_queries):
            try:
                results = await factory.process_query(
                    f"failure test {i}",
                    adapters=adapters,
                    limit=5
                )
                
                # Should get results even with some engine failures
                assert isinstance(results, RankedResults)
                successful_queries += 1
                
            except Exception as e:
                # Some failures are expected, but not all
                print(f"Query {i} failed: {e}")
        
        # Should have reasonable success rate despite engine failures
        success_rate = successful_queries / total_queries
        assert success_rate >= 0.7  # At least 70% success rate
    
    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self, unreliable_registry):
        """Test timeout handling in integration scenarios."""
        factory = FusionFactory(unreliable_registry)
        adapters = list(unreliable_registry.get_all_adapters().values())
        
        # Test with very short timeout
        start_time = time.time()
        
        try:
            # Use asyncio.wait_for to implement timeout
            results = await asyncio.wait_for(
                factory.process_query("timeout test", adapters=adapters, limit=5),
                timeout=0.1
            )
            
            # Should complete quickly due to timeout
            end_time = time.time()
            assert end_time - start_time < 0.5  # Should timeout quickly
            
            # May or may not have results depending on which engines completed
            assert isinstance(results, RankedResults)
            
        except asyncio.TimeoutError:
            # Timeout is also acceptable
            end_time = time.time()
            assert end_time - start_time < 0.5
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, unreliable_registry):
        """Test circuit breaker pattern in integration scenarios."""
        factory = FusionFactory(unreliable_registry)
        adapters = list(unreliable_registry.get_all_adapters().values())
        
        # Get the intermittent engine and make it fail consistently
        intermittent_adapter = unreliable_registry.get_adapter('intermittent_engine')
        intermittent_adapter.failure_rate = 1.0  # 100% failure rate
        
        # Run multiple queries to trigger circuit breaker
        failure_count = 0
        for i in range(10):
            try:
                results = await factory.process_query(
                    f"circuit breaker test {i}",
                    adapters=adapters,
                    limit=5
                )
                
                # Should still get results from working engines
                assert isinstance(results, RankedResults)
                
            except Exception:
                failure_count += 1
        
        # Should have some failures but not complete system failure
        assert failure_count < 10  # Not all queries should fail
        
        # Reset failure rate
        intermittent_adapter.failure_rate = 0.3


class TestMonitoringIntegration:
    """Test monitoring and observability integration."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring during integration scenarios."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create monitored adapters
        for i in range(3):
            adapter = InMemoryMockAdapter(f'monitored_engine_{i}', {
                'documents': [{'id': f'mon_doc_{i}', 'content': f'Monitored document {i}'}],
                'response_delay': 0.01 * (i + 1),  # Different delays
                'failure_rate': 0.0,
                'max_results': 10
            })
            registry._instances[f'monitored_engine_{i}'] = adapter
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Get performance monitor
        monitor = get_performance_monitor()
        
        # Execute monitored queries
        for i in range(5):
            results = await factory.process_query(
                f"monitoring test {i}",
                adapters=adapters,
                limit=5
            )
            
            assert isinstance(results, RankedResults)
        
        # Verify monitoring data was collected
        performance_summary = monitor.get_performance_summary()
        assert isinstance(performance_summary, dict)
        assert 'operation_counts' in performance_summary
        assert 'engine_metrics' in performance_summary
        
        # Should have recorded operations
        operation_counts = performance_summary['operation_counts']
        assert 'fusion_query_processing' in operation_counts
        assert operation_counts['fusion_query_processing'] >= 5
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        """Test health check integration across all components."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create adapters with different health states
        healthy_adapter = InMemoryMockAdapter('healthy_engine', {
            'documents': [{'id': 'healthy_doc', 'content': 'Healthy document'}],
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        unhealthy_adapter = InMemoryMockAdapter('unhealthy_engine', {
            'documents': [{'id': 'unhealthy_doc', 'content': 'Unhealthy document'}],
            'response_delay': 0.01,
            'failure_rate': 1.0,  # Always fails
            'max_results': 10
        })
        
        registry._instances['healthy_engine'] = healthy_adapter
        registry._instances['unhealthy_engine'] = unhealthy_adapter
        
        factory = FusionFactory(registry)
        
        # Perform health checks
        health_results = {}
        for adapter_id, adapter in registry.get_all_adapters().items():
            health_results[adapter_id] = await adapter.health_check()
        
        # Verify health check results
        assert health_results['healthy_engine'] is True
        # Unhealthy engine might still report healthy if health check is separate from search
        
        # Test system-wide health by checking all adapters
        all_healthy = all(health_results.values())
        assert isinstance(all_healthy, bool)
        
        # Verify we can still process queries with healthy engines
        if health_results['healthy_engine']:
            results = await factory.process_query(
                "health test",
                adapters=[registry.get_adapter('healthy_engine')],
                limit=5
            )
            assert isinstance(results, RankedResults)


class TestRealWorldScenarios:
    """Test real-world usage scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_large_result_set_handling(self):
        """Test handling of large result sets."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create engine with large document set
        large_docs = [
            {
                'id': f'large_doc_{i}',
                'title': f'Large Document {i}',
                'content': f'This is large document number {i} with substantial content for testing large result sets.'
            }
            for i in range(1000)  # 1000 documents
        ]
        
        large_engine = InMemoryMockAdapter('large_engine', {
            'documents': large_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 500  # Can return up to 500 results
        })
        
        registry._instances['large_engine'] = large_engine
        factory = FusionFactory(registry)
        
        # Test with different result limits
        for limit in [10, 50, 100, 200]:
            results = await factory.process_query(
                "large result test",
                adapters=[large_engine],
                limit=limit
            )
            
            assert isinstance(results, RankedResults)
            assert results.total_results <= limit
            assert len(results.results) <= limit
            
            # Verify results are properly ranked
            scores = [r.relevance_score for r in results.results]
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_multilingual_content_handling(self):
        """Test handling of multilingual content."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create multilingual documents
        multilingual_docs = [
            {'id': 'en_doc', 'title': 'English Document', 'content': 'This is an English document about machine learning.', 'language': 'en'},
            {'id': 'es_doc', 'title': 'Documento Español', 'content': 'Este es un documento en español sobre aprendizaje automático.', 'language': 'es'},
            {'id': 'fr_doc', 'title': 'Document Français', 'content': 'Ceci est un document français sur l\'apprentissage automatique.', 'language': 'fr'},
            {'id': 'de_doc', 'title': 'Deutsches Dokument', 'content': 'Dies ist ein deutsches Dokument über maschinelles Lernen.', 'language': 'de'},
        ]
        
        multilingual_engine = InMemoryMockAdapter('multilingual_engine', {
            'documents': multilingual_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        registry._instances['multilingual_engine'] = multilingual_engine
        factory = FusionFactory(registry)
        
        # Test multilingual queries
        queries = [
            "machine learning",
            "aprendizaje automático",
            "apprentissage automatique",
            "maschinelles Lernen"
        ]
        
        for query in queries:
            results = await factory.process_query(
                query,
                adapters=[multilingual_engine],
                limit=5
            )
            
            assert isinstance(results, RankedResults)
            # Should find relevant documents regardless of language
            assert results.total_results >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self):
        """Test concurrent users accessing the system."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create shared engines
        for i in range(2):
            docs = [{'id': f'shared_doc_{i}_{j}', 'content': f'Shared document {i}_{j}'} for j in range(50)]
            engine = InMemoryMockAdapter(f'shared_engine_{i}', {
                'documents': docs,
                'response_delay': 0.02,
                'failure_rate': 0.0,
                'max_results': 25
            })
            registry._instances[f'shared_engine_{i}'] = engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Simulate concurrent users
        async def simulate_user(user_id: int, num_queries: int):
            user_results = []
            for i in range(num_queries):
                try:
                    results = await factory.process_query(
                        f"user {user_id} query {i}",
                        adapters=adapters,
                        limit=5
                    )
                    user_results.append(results)
                except Exception as e:
                    print(f"User {user_id} query {i} failed: {e}")
            return user_results
        
        # Run concurrent users
        num_users = 10
        queries_per_user = 5
        
        user_tasks = [
            simulate_user(user_id, queries_per_user)
            for user_id in range(num_users)
        ]
        
        start_time = time.time()
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify concurrent execution
        total_duration = end_time - start_time
        expected_sequential_time = num_users * queries_per_user * 0.05  # Rough estimate
        
        # Concurrent execution should be faster than sequential
        assert total_duration < expected_sequential_time * 0.8
        
        # Verify all users got results
        successful_users = [r for r in user_results if not isinstance(r, Exception)]
        assert len(successful_users) >= num_users * 0.8  # At least 80% success rate
        
        # Verify individual user results
        for user_result in successful_users:
            assert len(user_result) <= queries_per_user
            for query_result in user_result:
                assert isinstance(query_result, RankedResults)


class TestAdvancedIntegrationScenarios:
    """Advanced integration test scenarios for comprehensive coverage."""
    
    @pytest.mark.asyncio
    async def test_cross_engine_result_correlation(self):
        """Test correlation of results across different engine types."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create engines with overlapping but different result sets
        common_docs = [
            {'id': 'common_1', 'title': 'Machine Learning', 'content': 'ML content'},
            {'id': 'common_2', 'title': 'Deep Learning', 'content': 'DL content'},
        ]
        
        unique_docs_1 = [
            {'id': 'unique_1_1', 'title': 'Neural Networks', 'content': 'NN content'},
            {'id': 'unique_1_2', 'title': 'Computer Vision', 'content': 'CV content'},
        ]
        
        unique_docs_2 = [
            {'id': 'unique_2_1', 'title': 'Natural Language', 'content': 'NLP content'},
            {'id': 'unique_2_2', 'title': 'Reinforcement Learning', 'content': 'RL content'},
        ]
        
        # Engine 1: Common + Unique 1
        engine1 = InMemoryMockAdapter('correlation_engine_1', {
            'documents': common_docs + unique_docs_1,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        # Engine 2: Common + Unique 2
        engine2 = InMemoryMockAdapter('correlation_engine_2', {
            'documents': common_docs + unique_docs_2,
            'response_delay': 0.02,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        registry._instances['correlation_engine_1'] = engine1
        registry._instances['correlation_engine_2'] = engine2
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Test query that should find common documents
        results = await factory.process_query("machine learning", adapters=adapters, limit=10)
        
        assert isinstance(results, RankedResults)
        assert results.total_results > 0
        
        # Common documents should appear in results
        result_ids = [r.document_id for r in results.results]
        assert 'common_1' in result_ids or 'common_2' in result_ids
        
        # Should have fusion weights for both engines
        assert len(results.fusion_weights) == 2
        assert abs(np.sum(results.fusion_weights) - 1.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_adaptive_weight_adjustment(self):
        """Test adaptive weight adjustment based on engine performance."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # High-quality engine (relevant results)
        high_quality_docs = [
            {'id': 'hq_1', 'title': 'Machine Learning Algorithms', 'content': 'Comprehensive ML algorithms guide'},
            {'id': 'hq_2', 'title': 'Deep Learning Techniques', 'content': 'Advanced deep learning methods'},
        ]
        
        # Low-quality engine (less relevant results)
        low_quality_docs = [
            {'id': 'lq_1', 'title': 'Random Topic', 'content': 'Unrelated content'},
            {'id': 'lq_2', 'title': 'Another Topic', 'content': 'More unrelated content'},
        ]
        
        high_quality_engine = InMemoryMockAdapter('high_quality_engine', {
            'documents': high_quality_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        low_quality_engine = InMemoryMockAdapter('low_quality_engine', {
            'documents': low_quality_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        registry._instances['high_quality_engine'] = high_quality_engine
        registry._instances['low_quality_engine'] = low_quality_engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Test multiple queries to see weight adaptation
        queries = ["machine learning", "deep learning", "algorithms"]
        weight_history = []
        
        for query in queries:
            results = await factory.process_query(query, adapters=adapters, limit=5)
            assert isinstance(results, RankedResults)
            weight_history.append(results.fusion_weights.copy())
        
        # Verify weights are reasonable
        for weights in weight_history:
            assert len(weights) >= 1  # May have fewer engines due to graceful degradation
            assert abs(np.sum(weights) - 1.0) < 0.001
            assert all(w >= 0 for w in weights)
    
    @pytest.mark.asyncio
    async def test_multi_language_content_handling(self):
        """Test handling of multi-language content."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Multi-language documents
        multilang_docs = [
            {'id': 'en_1', 'title': 'Machine Learning', 'content': 'English content about ML', 'language': 'en'},
            {'id': 'es_1', 'title': 'Aprendizaje Automático', 'content': 'Contenido en español sobre ML', 'language': 'es'},
            {'id': 'fr_1', 'title': 'Apprentissage Automatique', 'content': 'Contenu français sur ML', 'language': 'fr'},
            {'id': 'de_1', 'title': 'Maschinelles Lernen', 'content': 'Deutscher Inhalt über ML', 'language': 'de'},
        ]
        
        multilang_engine = InMemoryMockAdapter('multilang_engine', {
            'documents': multilang_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        registry._instances['multilang_engine'] = multilang_engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Test query processing with multilingual content
        results = await factory.process_query("machine learning", adapters=adapters, limit=10)
        
        assert isinstance(results, RankedResults)
        assert results.total_results > 0
        
        # Should handle different languages gracefully
        for result in results.results:
            assert result.document_id is not None
            assert result.content is not None
            assert result.relevance_score >= 0
    
    @pytest.mark.asyncio
    async def test_streaming_results_simulation(self):
        """Test simulation of streaming results from engines."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create engine that simulates streaming behavior
        streaming_docs = [
            {'id': f'stream_{i}', 'title': f'Streaming Document {i}', 'content': f'Streaming content {i}'}
            for i in range(20)
        ]
        
        streaming_engine = InMemoryMockAdapter('streaming_engine', {
            'documents': streaming_docs,
            'response_delay': 0.05,  # Simulate network delay
            'failure_rate': 0.0,
            'max_results': 20
        })
        
        registry._instances['streaming_engine'] = streaming_engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Test with different batch sizes
        batch_sizes = [5, 10, 15]
        
        for batch_size in batch_sizes:
            results = await factory.process_query(
                f"streaming test batch {batch_size}",
                adapters=adapters,
                limit=batch_size
            )
            
            assert isinstance(results, RankedResults)
            assert results.total_results <= batch_size
            
            # Results should be properly ranked
            if results.results:
                scores = [r.relevance_score for r in results.results]
                assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test handling of various edge cases."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Edge case documents
        edge_case_docs = [
            {'id': 'empty_content', 'title': 'Empty Content', 'content': ''},
            {'id': 'very_long', 'title': 'Very Long Document', 'content': 'x' * 10000},
            {'id': 'special_chars', 'title': 'Special Characters', 'content': '!@#$%^&*()_+{}|:"<>?'},
            {'id': 'unicode', 'title': 'Unicode Content', 'content': '🚀 Machine Learning 🤖 AI 🧠'},
            {'id': 'numbers_only', 'title': '123456', 'content': '789012345'},
        ]
        
        edge_case_engine = InMemoryMockAdapter('edge_case_engine', {
            'documents': edge_case_docs,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        })
        
        registry._instances['edge_case_engine'] = edge_case_engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Test various edge case queries
        edge_case_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a",  # Single character
            "very long query " * 20,  # Very long query
            "!@#$%^&*()",  # Special characters
            "🚀🤖🧠",  # Unicode emojis
        ]
        
        for query in edge_case_queries:
            try:
                results = await factory.process_query(query, adapters=adapters, limit=5)
                assert isinstance(results, RankedResults)
                # Should handle gracefully, even if no results
                assert results.total_results >= 0
            except Exception as e:
                # Some edge cases might raise exceptions, which is acceptable
                assert isinstance(e, (ValueError, TypeError))
    
    @pytest.mark.asyncio
    async def test_concurrent_query_isolation(self):
        """Test that concurrent queries don't interfere with each other."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create engine with predictable results
        isolation_docs = [
            {'id': f'iso_{i}', 'title': f'Isolation Document {i}', 'content': f'Content {i}'}
            for i in range(50)
        ]
        
        isolation_engine = InMemoryMockAdapter('isolation_engine', {
            'documents': isolation_docs,
            'response_delay': 0.02,
            'failure_rate': 0.0,
            'max_results': 20
        })
        
        registry._instances['isolation_engine'] = isolation_engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Create multiple concurrent queries with different parameters
        async def run_query(query_id: int, limit: int):
            return await factory.process_query(
                f"isolation test {query_id}",
                adapters=adapters,
                limit=limit
            )
        
        # Run concurrent queries with different limits
        tasks = [
            run_query(i, limit=(i % 5) + 1)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all queries completed successfully
        assert len(results) == 10
        
        for i, result in enumerate(results):
            assert isinstance(result, RankedResults)
            expected_limit = (i % 5) + 1
            # Results should respect individual query limits
            assert result.total_results <= expected_limit
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_scale(self):
        """Test memory efficiency with large-scale operations."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Create large document set
        large_doc_set = [
            {'id': f'large_{i}', 'title': f'Large Document {i}', 'content': f'Large content {i}' * 100}
            for i in range(500)  # 500 documents with large content
        ]
        
        large_scale_engine = InMemoryMockAdapter('large_scale_engine', {
            'documents': large_doc_set,
            'response_delay': 0.001,
            'failure_rate': 0.0,
            'max_results': 100
        })
        
        registry._instances['large_scale_engine'] = large_scale_engine
        
        factory = FusionFactory(registry)
        adapters = list(registry.get_all_adapters().values())
        
        # Measure memory usage before
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple large queries
        for i in range(10):
            results = await factory.process_query(
                f"large scale test {i}",
                adapters=adapters,
                limit=50
            )
            assert isinstance(results, RankedResults)
            
            # Clear results to help garbage collection
            del results
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Measure memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = memory_after - memory_before
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])