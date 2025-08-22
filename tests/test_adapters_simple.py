"""Simplified unit tests for search engine adapters."""

import pytest
from unittest.mock import patch, AsyncMock
from src.adapters import ElasticsearchAdapter, SolrAdapter, AdapterRegistry


class TestElasticsearchAdapterSimple:
    """Simplified test cases for ElasticsearchAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 9200,
            'index': 'test_index'
        }
        self.adapter = ElasticsearchAdapter('test_es', self.config)
    
    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.engine_id == 'test_es'
        assert self.adapter.host == 'localhost'
        assert self.adapter.port == 9200
        assert self.adapter.index == 'test_index'
        assert self.adapter.base_url == 'http://localhost:9200'
    
    def test_get_configuration(self):
        """Test configuration retrieval."""
        config = self.adapter.get_configuration()
        
        expected_config = {
            'engine_type': 'elasticsearch',
            'engine_id': 'test_es',
            'host': 'localhost',
            'port': 9200,
            'index': 'test_index',
            'use_ssl': False,
            'timeout': 30.0,
            'has_auth': False
        }
        
        assert config == expected_config
    
    def test_build_search_body(self):
        """Test search body building."""
        search_body = self.adapter._build_search_body('test query', 10)
        
        assert search_body['size'] == 10
        assert search_body['query']['multi_match']['query'] == 'test query'
        assert search_body['query']['multi_match']['type'] == 'best_fields'
        assert search_body['query']['multi_match']['fuzziness'] == 'AUTO'
    
    def test_build_search_body_with_filters(self):
        """Test search body building with filters."""
        search_body = self.adapter._build_search_body(
            'test query', 5,
            fields=['title', 'content'],
            boost_fields={'title': 2.0, 'content': 1.0},
            filters={'category': 'news'}
        )
        
        assert search_body['size'] == 5
        assert 'bool' in search_body['query']
        assert len(search_body['query']['bool']['filter']) == 1
        assert search_body['query']['bool']['filter'][0] == {'term': {'category': 'news'}}
    
    def test_parse_elasticsearch_response(self):
        """Test parsing Elasticsearch response."""
        mock_response = {
            'hits': {
                'hits': [
                    {
                        '_id': 'doc1',
                        '_score': 1.5,
                        '_source': {
                            'title': 'Test Document 1',
                            'content': 'This is test content 1'
                        },
                        '_index': 'test_index'
                    }
                ],
                'max_score': 1.5
            }
        }
        
        raw_results = self.adapter._parse_elasticsearch_response(mock_response)
        
        assert len(raw_results) == 1
        doc_id, score, content, metadata = raw_results[0]
        assert doc_id == 'doc1'
        assert score == 1.0  # Normalized (1.5/1.5)
        assert content == 'This is test content 1'
        assert metadata['index'] == 'test_index'


class TestSolrAdapterSimple:
    """Simplified test cases for SolrAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 8983,
            'core': 'test_core'
        }
        self.adapter = SolrAdapter('test_solr', self.config)
    
    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.engine_id == 'test_solr'
        assert self.adapter.host == 'localhost'
        assert self.adapter.port == 8983
        assert self.adapter.core == 'test_core'
        assert self.adapter.base_url == 'http://localhost:8983/solr/test_core'
    
    def test_initialization_missing_core(self):
        """Test adapter initialization with missing core."""
        config = self.config.copy()
        del config['core']
        
        with pytest.raises(ValueError, match="Solr core name is required"):
            SolrAdapter('test_solr', config)
    
    def test_get_configuration(self):
        """Test configuration retrieval."""
        config = self.adapter.get_configuration()
        
        expected_config = {
            'engine_type': 'solr',
            'engine_id': 'test_solr',
            'host': 'localhost',
            'port': 8983,
            'core': 'test_core',
            'use_ssl': False,
            'timeout': 30.0,
            'has_auth': False
        }
        
        assert config == expected_config
    
    def test_build_search_params(self):
        """Test search parameters building."""
        params = self.adapter._build_search_params('test query', 10)
        
        assert params['q'] == 'test query'
        assert params['rows'] == '10'
        assert params['wt'] == 'json'
        assert params['defType'] == 'edismax'
        assert params['sort'] == 'score desc'
    
    def test_build_search_params_with_boost_fields(self):
        """Test search parameters with field boosting."""
        params = self.adapter._build_search_params(
            'test query', 10,
            boost_fields={'title': 2.0, 'content': 1.0},
            filters={'category': 'news'}
        )
        
        assert 'qf' in params
        assert 'title^2.0' in params['qf']
        assert 'content^1.0' in params['qf']
        assert 'fq' in params
        assert 'category:"news"' in params['fq']
    
    def test_parse_solr_response(self):
        """Test parsing Solr response."""
        mock_response = {
            'response': {
                'docs': [
                    {
                        'id': 'doc1',
                        'score': 1.5,
                        'title': 'Test Document 1',
                        'content': 'This is test content 1'
                    }
                ],
                'maxScore': 1.5
            }
        }
        
        raw_results = self.adapter._parse_solr_response(mock_response)
        
        assert len(raw_results) == 1
        doc_id, score, content, metadata = raw_results[0]
        assert doc_id == 'doc1'
        assert score == 1.0  # Normalized (1.5/1.5)
        assert content == 'This is test content 1'
        assert metadata['raw_score'] == 1.5
    
    def test_escape_solr_value(self):
        """Test Solr value escaping."""
        test_cases = [
            ('simple text', '"simple text"'),
            ('text with + sign', '"text with \\\\+ sign"'),
            ('query: field', '"query\\\\: field"'),
            ('path/to/file', '"path\\/to\\/file"')
        ]
        
        for input_val, expected in test_cases:
            result = self.adapter._escape_solr_value(input_val)
            assert result == expected


class TestAdapterRegistrySimple:
    """Simplified test cases for AdapterRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = AdapterRegistry()
    
    def test_builtin_adapters_registered(self):
        """Test that built-in adapters are registered."""
        available = self.registry.get_available_adapters()
        assert 'elasticsearch' in available
        assert 'solr' in available
    
    def test_create_elasticsearch_adapter(self):
        """Test creating Elasticsearch adapter."""
        config = {
            'host': 'localhost',
            'port': 9200,
            'index': 'test_index'
        }
        
        adapter = self.registry.create_adapter('elasticsearch', 'test_es', config)
        
        assert isinstance(adapter, ElasticsearchAdapter)
        assert adapter.engine_id == 'test_es'
        assert self.registry.get_adapter('test_es') == adapter
    
    def test_create_solr_adapter(self):
        """Test creating Solr adapter."""
        config = {
            'host': 'localhost',
            'port': 8983,
            'core': 'test_core'
        }
        
        adapter = self.registry.create_adapter('solr', 'test_solr', config)
        
        assert isinstance(adapter, SolrAdapter)
        assert adapter.engine_id == 'test_solr'
        assert self.registry.get_adapter('test_solr') == adapter
    
    def test_create_unknown_adapter_type(self):
        """Test creating adapter with unknown type."""
        with pytest.raises(ValueError, match="Unknown engine type 'unknown'"):
            self.registry.create_adapter('unknown', 'test_id', {})
    
    def test_create_duplicate_adapter_id(self):
        """Test creating adapter with duplicate ID."""
        config = {'host': 'localhost', 'port': 9200, 'index': 'test'}
        
        self.registry.create_adapter('elasticsearch', 'test_id', config)
        
        with pytest.raises(ValueError, match="Adapter instance 'test_id' already exists"):
            self.registry.create_adapter('elasticsearch', 'test_id', config)
    
    def test_remove_adapter(self):
        """Test removing adapter."""
        config = {'host': 'localhost', 'port': 9200, 'index': 'test'}
        adapter = self.registry.create_adapter('elasticsearch', 'test_id', config)
        
        assert self.registry.get_adapter('test_id') == adapter
        
        removed = self.registry.remove_adapter('test_id')
        assert removed is True
        assert self.registry.get_adapter('test_id') is None
        
        # Try removing again
        removed = self.registry.remove_adapter('test_id')
        assert removed is False
    
    def test_get_registry_status(self):
        """Test getting registry status."""
        es_config = {'host': 'localhost', 'port': 9200, 'index': 'test'}
        solr_config = {'host': 'localhost', 'port': 8983, 'core': 'test'}
        
        self.registry.create_adapter('elasticsearch', 'es1', es_config)
        self.registry.create_adapter('solr', 'solr1', solr_config)
        
        status = self.registry.get_registry_status()
        
        assert 'registered_types' in status
        assert 'total_instances' in status
        assert status['total_instances'] == 2
        assert 'elasticsearch' in status['instances_by_type']
        assert 'solr' in status['instances_by_type']