"""Unit tests for search engine adapters."""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import aiohttp

from src.adapters import (
    ElasticsearchAdapter, 
    SolrAdapter, 
    AdapterRegistry,
    SearchEngineError,
    SearchEngineConnectionError,
    SearchEngineTimeoutError
)
from src.models.core import SearchResult, SearchResults


class TestElasticsearchAdapter:
    """Test cases for ElasticsearchAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 9200,
            'index': 'test_index',
            'username': 'test_user',
            'password': 'test_pass'
        }
        self.adapter = ElasticsearchAdapter('test_es', self.config)
    
    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.engine_id == 'test_es'
        assert self.adapter.host == 'localhost'
        assert self.adapter.port == 9200
        assert self.adapter.index == 'test_index'
        assert self.adapter.username == 'test_user'
        assert self.adapter.password == 'test_pass'
        assert not self.adapter.use_ssl
        assert self.adapter.base_url == 'http://localhost:9200'
    
    def test_initialization_with_ssl(self):
        """Test adapter initialization with SSL."""
        config = self.config.copy()
        config['use_ssl'] = True
        adapter = ElasticsearchAdapter('test_es_ssl', config)
        assert adapter.base_url == 'https://localhost:9200'
    
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
            'has_auth': True
        }
        
        assert config == expected_config
    
    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search operation."""
        # Mock Elasticsearch response
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
                        '_index': 'test_index',
                        'highlight': {'content': ['This is <em>test</em> content 1']}
                    },
                    {
                        '_id': 'doc2',
                        '_score': 1.0,
                        '_source': {
                            'title': 'Test Document 2',
                            'content': 'This is test content 2'
                        },
                        '_index': 'test_index'
                    }
                ],
                'max_score': 1.5
            }
        }
        
        # Mock aiohttp session and response
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
        mock_response_obj.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            results = await self.adapter.search('test query', limit=10)
        
        # Verify results
        assert isinstance(results, SearchResults)
        assert results.query == 'test query'
        assert results.engine_id == 'test_es'
        assert len(results.results) == 2
        
        # Check first result
        result1 = results.results[0]
        assert result1.document_id == 'doc1'
        assert result1.relevance_score == 1.0  # Normalized (1.5/1.5)
        assert result1.content == 'This is test content 1'
        assert result1.engine_source == 'test_es'
        assert 'highlight' in result1.metadata
        
        # Check second result
        result2 = results.results[1]
        assert result2.document_id == 'doc2'
        assert result2.relevance_score == pytest.approx(0.667, rel=1e-2)  # Normalized (1.0/1.5)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with additional filters."""
        mock_response = {
            'hits': {
                'hits': [],
                'max_score': None
            }
        }
        
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
        mock_response_obj.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            await self.adapter.search(
                'test query',
                limit=5,
                fields=['title', 'content'],
                boost_fields={'title': 2.0, 'content': 1.0},
                filters={'category': 'news'}
            )
        
        # Verify the search body was built correctly
        call_args = mock_session.post.call_args
        search_body = call_args[1]['json']
        
        assert search_body['size'] == 5
        assert 'bool' in search_body['query']
        assert len(search_body['query']['bool']['filter']) == 1
        assert search_body['query']['bool']['filter'][0] == {'term': {'category': 'news'}}
    
    @pytest.mark.asyncio
    async def test_search_connection_error(self):
        """Test search with connection error."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.side_effect = aiohttp.ClientError("Connection failed")
            
            with pytest.raises(SearchEngineConnectionError):
                await self.adapter.search('test query')
    
    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """Test search timeout handling."""
        with patch.object(self.adapter, 'search', side_effect=asyncio.TimeoutError()):
            with pytest.raises(SearchEngineTimeoutError):
                await self.adapter.search_with_timeout('test query', timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        mock_response = {'status': 'green'}
        
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
        mock_response_obj.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            is_healthy = await self.adapter.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 500
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            is_healthy = await self.adapter.health_check()
        
        assert is_healthy is False


class TestSolrAdapter:
    """Test cases for SolrAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'host': 'localhost',
            'port': 8983,
            'core': 'test_core',
            'username': 'test_user',
            'password': 'test_pass'
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
            'has_auth': True
        }
        
        assert config == expected_config
    
    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search operation."""
        # Mock Solr response
        mock_response = {
            'response': {
                'docs': [
                    {
                        'id': 'doc1',
                        'score': 1.5,
                        'title': 'Test Document 1',
                        'content': 'This is test content 1'
                    },
                    {
                        'id': 'doc2',
                        'score': 1.0,
                        'title': 'Test Document 2',
                        'content': 'This is test content 2'
                    }
                ],
                'maxScore': 1.5
            },
            'highlighting': {
                'doc1': {'content': ['This is <em>test</em> content 1']}
            }
        }
        
        # Mock aiohttp session and response
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
        mock_response_obj.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            results = await self.adapter.search('test query', limit=10)
        
        # Verify results
        assert isinstance(results, SearchResults)
        assert results.query == 'test query'
        assert results.engine_id == 'test_solr'
        assert len(results.results) == 2
        
        # Check first result
        result1 = results.results[0]
        assert result1.document_id == 'doc1'
        assert result1.relevance_score == 1.0  # Normalized (1.5/1.5)
        assert result1.content == 'This is test content 1'
        assert result1.engine_source == 'test_solr'
    
    @pytest.mark.asyncio
    async def test_search_with_boost_fields(self):
        """Test search with field boosting."""
        mock_response = {
            'response': {
                'docs': [],
                'maxScore': None
            }
        }
        
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
        mock_response_obj.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            await self.adapter.search(
                'test query',
                boost_fields={'title': 2.0, 'content': 1.0},
                filters={'category': 'news'}
            )
        
        # Verify the search parameters
        call_args = mock_session.get.call_args
        params = call_args[1]['params']
        
        assert 'qf' in params
        assert 'title^2.0' in params['qf']
        assert 'content^1.0' in params['qf']
        assert 'fq' in params
        assert 'category:"news"' in params['fq']
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        mock_response = {'status': 'OK'}
        
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
        mock_response_obj.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response_obj)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            is_healthy = await self.adapter.health_check()
        
        assert is_healthy is True
    
    def test_escape_solr_value(self):
        """Test Solr value escaping."""
        test_cases = [
            ('simple text', '"simple text"'),
            ('text with + sign', '"text with \\\\+ sign"'),  # Double backslash in result
            ('query: field', '"query\\\\: field"'),
            ('path/to/file', '"path\\/to\\/file"')  # Single backslash in result
        ]
        
        for input_val, expected in test_cases:
            result = self.adapter._escape_solr_value(input_val)
            assert result == expected


class TestAdapterRegistry:
    """Test cases for AdapterRegistry."""
    
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
    
    def test_get_adapters_by_type(self):
        """Test getting adapters by type."""
        es_config = {'host': 'localhost', 'port': 9200, 'index': 'test'}
        solr_config = {'host': 'localhost', 'port': 8983, 'core': 'test'}
        
        es_adapter = self.registry.create_adapter('elasticsearch', 'es1', es_config)
        solr_adapter = self.registry.create_adapter('solr', 'solr1', solr_config)
        
        es_adapters = self.registry.get_adapters_by_type('elasticsearch')
        solr_adapters = self.registry.get_adapters_by_type('solr')
        
        assert len(es_adapters) == 1
        assert es_adapters[0] == es_adapter
        assert len(solr_adapters) == 1
        assert solr_adapters[0] == solr_adapter
    
    def test_create_adapters_from_config(self):
        """Test creating multiple adapters from configuration."""
        adapters_config = [
            {
                'engine_type': 'elasticsearch',
                'engine_id': 'es1',
                'config': {'host': 'localhost', 'port': 9200, 'index': 'test1'}
            },
            {
                'engine_type': 'solr',
                'engine_id': 'solr1',
                'config': {'host': 'localhost', 'port': 8983, 'core': 'test1'},
                'timeout': 60.0
            }
        ]
        
        adapters = self.registry.create_adapters_from_config(adapters_config)
        
        assert len(adapters) == 2
        assert isinstance(adapters[0], ElasticsearchAdapter)
        assert isinstance(adapters[1], SolrAdapter)
        assert adapters[1].timeout == 60.0
    
    def test_get_adapter_info(self):
        """Test getting adapter information."""
        config = {'host': 'localhost', 'port': 9200, 'index': 'test'}
        adapter = self.registry.create_adapter('elasticsearch', 'test_id', config)
        
        info = self.registry.get_adapter_info('test_id')
        
        assert info is not None
        assert info['engine_id'] == 'test_id'
        assert info['engine_type'] == 'elasticsearch'
        assert 'configuration' in info
        assert 'is_healthy' in info
    
    def test_get_registry_status(self):
        """Test getting registry status."""
        es_config = {'host': 'localhost', 'port': 9200, 'index': 'test'}
        solr_config = {'host': 'localhost', 'port': 8983, 'core': 'test'}
        
        self.registry.create_adapter('elasticsearch', 'es1', es_config)
        self.registry.create_adapter('solr', 'solr1', solr_config)
        
        status = self.registry.get_registry_status()
        
        assert 'registered_types' in status
        assert 'total_instances' in status
        assert 'instances_by_type' in status
        assert status['total_instances'] == 2
        assert 'elasticsearch' in status['instances_by_type']
        assert 'solr' in status['instances_by_type']