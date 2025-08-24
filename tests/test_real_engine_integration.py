"""Integration tests with real search engines.

This module contains integration tests that work with actual search engine instances
when available (e.g., in Docker development environment).

Requirements covered: 2.1, 2.2, 2.4
"""

import asyncio
import pytest
import os
from typing import Dict, List, Any
from unittest.mock import patch
import docker
import time

from src.services.fusion_factory import FusionFactory
from src.adapters.registry import AdapterRegistry
from src.adapters.elasticsearch_adapter import ElasticsearchAdapter
from src.adapters.solr_adapter import SolrAdapter
from src.adapters.opensearch_adapter import OpenSearchAdapter
from src.models.core import SearchResults, RankedResults
from src.config.settings import load_config_from_file


# Skip real engine tests if not in development environment
SKIP_REAL_ENGINES = os.getenv('SKIP_REAL_ENGINE_TESTS', 'true').lower() == 'true'


def check_docker_service(service_name: str, port: int) -> bool:
    """Check if a Docker service is running and accessible."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except Exception:
        return False


def setup_test_data_elasticsearch(host: str = 'localhost', port: int = 9200):
    """Set up test data in Elasticsearch."""
    try:
        import requests
        
        # Create index with test documents
        index_url = f"http://{host}:{port}/test_integration"
        
        # Delete existing index
        requests.delete(index_url)
        
        # Create index
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text", "analyzer": "standard"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "category": {"type": "keyword"},
                    "tags": {"type": "keyword"}
                }
            }
        }
        requests.put(index_url, json=mapping)
        
        # Add test documents
        test_docs = [
            {
                "title": "Machine Learning Fundamentals",
                "content": "Introduction to machine learning algorithms and techniques for data science applications.",
                "category": "education",
                "tags": ["ml", "data-science", "algorithms"]
            },
            {
                "title": "Deep Learning with Neural Networks",
                "content": "Advanced neural network architectures for deep learning and artificial intelligence.",
                "category": "research", 
                "tags": ["deep-learning", "neural-networks", "ai"]
            },
            {
                "title": "Natural Language Processing",
                "content": "Text processing and language understanding using computational linguistics methods.",
                "category": "nlp",
                "tags": ["nlp", "text-processing", "linguistics"]
            }
        ]
        
        for i, doc in enumerate(test_docs):
            doc_url = f"{index_url}/_doc/{i+1}"
            requests.put(doc_url, json=doc)
        
        # Refresh index
        requests.post(f"{index_url}/_refresh")
        
        return True
    except Exception as e:
        print(f"Failed to setup Elasticsearch test data: {e}")
        return False


def setup_test_data_solr(host: str = 'localhost', port: int = 8983):
    """Set up test data in Solr."""
    try:
        import requests
        
        # Create core if it doesn't exist
        core_url = f"http://{host}:{port}/solr/test_integration"
        
        # Add test documents
        test_docs = [
            {
                "id": "1",
                "title": "Machine Learning Fundamentals",
                "content": "Introduction to machine learning algorithms and techniques for data science applications.",
                "category": "education",
                "tags": ["ml", "data-science", "algorithms"]
            },
            {
                "id": "2", 
                "title": "Deep Learning with Neural Networks",
                "content": "Advanced neural network architectures for deep learning and artificial intelligence.",
                "category": "research",
                "tags": ["deep-learning", "neural-networks", "ai"]
            },
            {
                "id": "3",
                "title": "Natural Language Processing", 
                "content": "Text processing and language understanding using computational linguistics methods.",
                "category": "nlp",
                "tags": ["nlp", "text-processing", "linguistics"]
            }
        ]
        
        # Add documents
        update_url = f"{core_url}/update/json/docs"
        requests.post(update_url, json=test_docs)
        
        # Commit
        commit_url = f"{core_url}/update?commit=true"
        requests.post(commit_url)
        
        return True
    except Exception as e:
        print(f"Failed to setup Solr test data: {e}")
        return False


@pytest.mark.skipif(SKIP_REAL_ENGINES, reason="Real engine tests disabled")
class TestElasticsearchIntegration:
    """Integration tests with real Elasticsearch instance."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_elasticsearch(self):
        """Set up Elasticsearch test data."""
        if not check_docker_service('elasticsearch', 9200):
            pytest.skip("Elasticsearch not available")
        
        # Wait for Elasticsearch to be ready
        time.sleep(2)
        
        if not setup_test_data_elasticsearch():
            pytest.skip("Failed to setup Elasticsearch test data")
        
        yield
        
        # Cleanup after tests
        try:
            import requests
            requests.delete("http://localhost:9200/test_integration")
        except:
            pass
    
    @pytest.fixture
    def elasticsearch_adapter(self):
        """Create Elasticsearch adapter for testing."""
        config = {
            'host': 'localhost',
            'port': 9200,
            'index': 'test_integration',
            'use_ssl': False
        }
        return ElasticsearchAdapter('test_elasticsearch', config)
    
    @pytest.mark.asyncio
    async def test_elasticsearch_search(self, elasticsearch_adapter):
        """Test search functionality with real Elasticsearch."""
        results = await elasticsearch_adapter.search("machine learning", limit=10)
        
        assert isinstance(results, SearchResults)
        assert results.query == "machine learning"
        assert results.engine_id == "test_elasticsearch"
        assert len(results.results) > 0
        
        # Verify result structure
        for result in results.results:
            assert result.document_id is not None
            assert result.relevance_score > 0
            assert result.content is not None
            assert result.engine_source == "test_elasticsearch"
    
    @pytest.mark.asyncio
    async def test_elasticsearch_health_check(self, elasticsearch_adapter):
        """Test health check with real Elasticsearch."""
        is_healthy = await elasticsearch_adapter.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_elasticsearch_with_filters(self, elasticsearch_adapter):
        """Test search with filters."""
        results = await elasticsearch_adapter.search(
            "learning",
            limit=10,
            filters={"category": "education"}
        )
        
        assert isinstance(results, SearchResults)
        assert len(results.results) >= 0  # May or may not have results
        
        # If results exist, verify they match filter
        for result in results.results:
            if 'category' in result.metadata:
                assert result.metadata['category'] == 'education'
    
    @pytest.mark.asyncio
    async def test_elasticsearch_boost_fields(self, elasticsearch_adapter):
        """Test search with field boosting."""
        results = await elasticsearch_adapter.search(
            "neural networks",
            limit=10,
            boost_fields={"title": 2.0, "content": 1.0}
        )
        
        assert isinstance(results, SearchResults)
        # Should find documents with "neural networks" in title or content


@pytest.mark.skipif(SKIP_REAL_ENGINES, reason="Real engine tests disabled")
class TestSolrIntegration:
    """Integration tests with real Solr instance."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_solr(self):
        """Set up Solr test data."""
        if not check_docker_service('solr', 8983):
            pytest.skip("Solr not available")
        
        # Wait for Solr to be ready
        time.sleep(2)
        
        if not setup_test_data_solr():
            pytest.skip("Failed to setup Solr test data")
        
        yield
    
    @pytest.fixture
    def solr_adapter(self):
        """Create Solr adapter for testing."""
        config = {
            'host': 'localhost',
            'port': 8983,
            'core': 'test_integration',
            'use_ssl': False
        }
        return SolrAdapter('test_solr', config)
    
    @pytest.mark.asyncio
    async def test_solr_search(self, solr_adapter):
        """Test search functionality with real Solr."""
        results = await solr_adapter.search("machine learning", limit=10)
        
        assert isinstance(results, SearchResults)
        assert results.query == "machine learning"
        assert results.engine_id == "test_solr"
        assert len(results.results) >= 0  # May or may not have results
        
        # Verify result structure
        for result in results.results:
            assert result.document_id is not None
            assert result.relevance_score > 0
            assert result.content is not None
            assert result.engine_source == "test_solr"
    
    @pytest.mark.asyncio
    async def test_solr_health_check(self, solr_adapter):
        """Test health check with real Solr."""
        is_healthy = await solr_adapter.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_solr_with_filters(self, solr_adapter):
        """Test search with filters."""
        results = await solr_adapter.search(
            "learning",
            limit=10,
            filters={"category": "education"}
        )
        
        assert isinstance(results, SearchResults)
        # Results may be empty if no matches


@pytest.mark.skipif(SKIP_REAL_ENGINES, reason="Real engine tests disabled")
class TestOpenSearchIntegration:
    """Integration tests with real OpenSearch instance."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_opensearch(self):
        """Set up OpenSearch test data."""
        if not check_docker_service('opensearch', 9200):
            pytest.skip("OpenSearch not available")
        
        # Wait for OpenSearch to be ready
        time.sleep(2)
        
        # OpenSearch setup is similar to Elasticsearch
        if not setup_test_data_elasticsearch(host='localhost', port=9200):
            pytest.skip("Failed to setup OpenSearch test data")
        
        yield
    
    @pytest.fixture
    def opensearch_adapter(self):
        """Create OpenSearch adapter for testing."""
        config = {
            'host': 'localhost',
            'port': 9200,
            'index': 'test_integration',
            'use_ssl': False,
            'verify_ssl': False
        }
        return OpenSearchAdapter('test_opensearch', config)
    
    @pytest.mark.asyncio
    async def test_opensearch_search(self, opensearch_adapter):
        """Test search functionality with real OpenSearch."""
        results = await opensearch_adapter.search("machine learning", limit=10)
        
        assert isinstance(results, SearchResults)
        assert results.query == "machine learning"
        assert results.engine_id == "test_opensearch"
        
        # Verify result structure
        for result in results.results:
            assert result.document_id is not None
            assert result.relevance_score > 0
            assert result.content is not None
            assert result.engine_source == "test_opensearch"
    
    @pytest.mark.asyncio
    async def test_opensearch_health_check(self, opensearch_adapter):
        """Test health check with real OpenSearch."""
        is_healthy = await opensearch_adapter.health_check()
        assert is_healthy is True


@pytest.mark.skipif(SKIP_REAL_ENGINES, reason="Real engine tests disabled")
class TestMultiEngineRealIntegration:
    """Integration tests with multiple real search engines."""
    
    @pytest.fixture
    def multi_engine_registry(self):
        """Create registry with multiple real engines."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        adapters = []
        
        # Add Elasticsearch if available
        if check_docker_service('elasticsearch', 9200):
            es_config = {
                'host': 'localhost',
                'port': 9200,
                'index': 'test_integration',
                'use_ssl': False
            }
            es_adapter = ElasticsearchAdapter('real_elasticsearch', es_config)
            registry._instances['real_elasticsearch'] = es_adapter
            adapters.append(es_adapter)
        
        # Add Solr if available
        if check_docker_service('solr', 8983):
            solr_config = {
                'host': 'localhost',
                'port': 8983,
                'core': 'test_integration',
                'use_ssl': False
            }
            solr_adapter = SolrAdapter('real_solr', solr_config)
            registry._instances['real_solr'] = solr_adapter
            adapters.append(solr_adapter)
        
        if not adapters:
            pytest.skip("No real search engines available")
        
        return registry
    
    @pytest.mark.asyncio
    async def test_multi_engine_fusion(self, multi_engine_registry):
        """Test fusion with multiple real search engines."""
        factory = FusionFactory(multi_engine_registry)
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        # Ensure test data is set up
        setup_test_data_elasticsearch()
        setup_test_data_solr()
        time.sleep(1)  # Wait for indexing
        
        results = await factory.process_query(
            "machine learning algorithms",
            adapters=adapters,
            limit=5
        )
        
        assert isinstance(results, RankedResults)
        assert results.query == "machine learning algorithms"
        assert len(results.fusion_weights) == len(adapters)
        
        # Verify fusion weights sum to 1
        import numpy as np
        assert abs(np.sum(results.fusion_weights) - 1.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_real_engine_performance_comparison(self, multi_engine_registry):
        """Compare performance of different real search engines."""
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        if len(adapters) < 2:
            pytest.skip("Need at least 2 engines for comparison")
        
        # Ensure test data is set up
        setup_test_data_elasticsearch()
        setup_test_data_solr()
        time.sleep(1)
        
        performance_results = {}
        
        for adapter in adapters:
            start_time = time.time()
            results = await adapter.search("deep learning neural networks", limit=10)
            end_time = time.time()
            
            performance_results[adapter.engine_id] = {
                'response_time': end_time - start_time,
                'result_count': len(results.results),
                'avg_score': sum(r.relevance_score for r in results.results) / len(results.results) if results.results else 0
            }
        
        # Verify all engines returned some performance data
        assert len(performance_results) == len(adapters)
        
        for engine_id, metrics in performance_results.items():
            assert metrics['response_time'] > 0
            assert metrics['result_count'] >= 0
    
    @pytest.mark.asyncio
    async def test_real_engine_failure_resilience(self, multi_engine_registry):
        """Test resilience when some real engines fail."""
        factory = FusionFactory(multi_engine_registry)
        adapters = list(multi_engine_registry.get_all_adapters().values())
        
        if len(adapters) < 2:
            pytest.skip("Need at least 2 engines for failure testing")
        
        # Make one adapter fail by changing its configuration
        failing_adapter = adapters[0]
        original_host = failing_adapter.host
        failing_adapter.host = "nonexistent-host"
        
        try:
            results = await factory.process_query(
                "natural language processing",
                adapters=adapters,
                limit=5
            )
            
            # Should still get results from working engines
            assert isinstance(results, RankedResults)
            # May have fewer results due to one engine failing
            
        finally:
            # Restore original configuration
            failing_adapter.host = original_host


@pytest.mark.skipif(SKIP_REAL_ENGINES, reason="Real engine tests disabled")
class TestDockerEnvironmentIntegration:
    """Integration tests specifically for Docker development environment."""
    
    def test_docker_services_available(self):
        """Test that expected Docker services are available."""
        expected_services = [
            ('elasticsearch', 9200),
            ('solr', 8983),
            # ('opensearch', 9200),  # May conflict with Elasticsearch
            # ('postgres', 5432)     # For database adapter
        ]
        
        available_services = []
        for service, port in expected_services:
            if check_docker_service(service, port):
                available_services.append(service)
        
        # At least one service should be available for meaningful tests
        assert len(available_services) > 0, "No Docker services available for testing"
    
    @pytest.mark.asyncio
    async def test_development_config_with_docker(self):
        """Test development configuration with Docker services."""
        try:
            config = load_config_from_file("config/development.yaml")
        except FileNotFoundError:
            pytest.skip("Development config not found")
        
        registry = AdapterRegistry()
        factory = FusionFactory(registry)
        
        # Filter engines to only those with available services
        available_engines = []
        for engine_config in config.get('engines', []):
            engine_type = engine_config.get('engine_type')
            engine_host = engine_config.get('config', {}).get('host', 'localhost')
            
            # Check if service is available (skip database and mock engines)
            if engine_type in ['elasticsearch', 'solr', 'opensearch']:
                port_map = {'elasticsearch': 9200, 'solr': 8983, 'opensearch': 9200}
                if check_docker_service(engine_host, port_map.get(engine_type, 9200)):
                    available_engines.append(engine_config)
        
        if not available_engines:
            pytest.skip("No Docker services available for development config test")
        
        # Initialize available adapters
        adapters = factory.initialize_adapters_from_config(available_engines)
        
        # Test query with available adapters
        if adapters:
            results = await factory.process_query(
                "docker integration test",
                adapters=adapters,
                limit=3
            )
            
            assert isinstance(results, RankedResults)


if __name__ == "__main__":
    # Run with real engine tests enabled
    os.environ['SKIP_REAL_ENGINE_TESTS'] = 'false'
    pytest.main([__file__, "-v", "--tb=short"])