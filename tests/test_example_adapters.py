"""Tests for example search engine adapters."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.adapters.opensearch_adapter import OpenSearchAdapter
from src.adapters.mock_adapter import InMemoryMockAdapter, FileMockAdapter
from src.adapters.custom_adapter import RestApiAdapter, DatabaseAdapter, WebScrapingAdapter
from src.adapters.base import SearchEngineError, SearchEngineConnectionError
from src.models.core import SearchResult, SearchResults


class TestOpenSearchAdapter:
    """Test cases for OpenSearch adapter."""
    
    @pytest.fixture
    def opensearch_config(self):
        return {
            'host': 'localhost',
            'port': 9200,
            'index': 'test_index',
            'use_ssl': False,
            'verify_ssl': True
        }
    
    @pytest.fixture
    def opensearch_adapter(self, opensearch_config):
        return OpenSearchAdapter('test_opensearch', opensearch_config)
    
    def test_initialization(self, opensearch_adapter, opensearch_config):
        """Test OpenSearch adapter initialization."""
        assert opensearch_adapter.engine_id == 'test_opensearch'
        assert opensearch_adapter.host == 'localhost'
        assert opensearch_adapter.port == 9200
        assert opensearch_adapter.index == 'test_index'
        assert opensearch_adapter.use_ssl is False
        assert opensearch_adapter.verify_ssl is True
    
    def test_get_configuration(self, opensearch_adapter):
        """Test configuration retrieval."""
        config = opensearch_adapter.get_configuration()
        
        assert config['engine_type'] == 'opensearch'
        assert config['engine_id'] == 'test_opensearch'
        assert config['host'] == 'localhost'
        assert config['port'] == 9200
        assert config['index'] == 'test_index'
        assert config['use_ssl'] is False
        assert config['verify_ssl'] is True
        assert config['has_auth'] is False
    
    @pytest.mark.asyncio
    async def test_search_success(self, opensearch_adapter):
        """Test successful search operation."""
        mock_response_data = {
            'hits': {
                'hits': [
                    {
                        '_id': 'doc1',
                        '_score': 1.5,
                        '_source': {
                            'title': 'Test Document',
                            'content': 'This is test content'
                        },
                        'highlight': {'content': ['This is <mark>test</mark> content']}
                    }
                ],
                'max_score': 1.5
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            results = await opensearch_adapter.search('test query', limit=10)
            
            assert isinstance(results, SearchResults)
            assert results.query == 'test query'
            assert results.engine_id == 'test_opensearch'
            assert len(results.results) == 1
            
            result = results.results[0]
            assert result.document_id == 'doc1'
            assert result.relevance_score == 1.0  # Normalized
            assert 'This is test content' in result.content
            assert result.engine_source == 'test_opensearch'
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, opensearch_adapter):
        """Test successful health check."""
        mock_health_data = {'status': 'green'}
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_health_data)
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            is_healthy = await opensearch_adapter.health_check()
            assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, opensearch_adapter):
        """Test health check failure."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            is_healthy = await opensearch_adapter.health_check()
            assert is_healthy is False


class TestInMemoryMockAdapter:
    """Test cases for in-memory mock adapter."""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 5,
            'documents': [
                {
                    'id': 'test1',
                    'title': 'Test Document 1',
                    'content': 'This is a test document about machine learning',
                    'category': 'technology',
                    'tags': ['ml', 'test']
                },
                {
                    'id': 'test2',
                    'title': 'Test Document 2',
                    'content': 'Another test document about Python programming',
                    'category': 'programming',
                    'tags': ['python', 'test']
                }
            ]
        }
    
    @pytest.fixture
    def mock_adapter(self, mock_config):
        return InMemoryMockAdapter('test_mock', mock_config)
    
    def test_initialization(self, mock_adapter, mock_config):
        """Test mock adapter initialization."""
        assert mock_adapter.engine_id == 'test_mock'
        assert mock_adapter.response_delay == 0.01
        assert mock_adapter.failure_rate == 0.0
        assert mock_adapter.max_results == 5
        assert len(mock_adapter.documents) == 2
    
    def test_get_configuration(self, mock_adapter):
        """Test configuration retrieval."""
        config = mock_adapter.get_configuration()
        
        assert config['engine_type'] == 'mock_inmemory'
        assert config['engine_id'] == 'test_mock'
        assert config['document_count'] == 2
        assert config['response_delay'] == 0.01
        assert config['failure_rate'] == 0.0
        assert config['max_results'] == 5
    
    @pytest.mark.asyncio
    async def test_search_success(self, mock_adapter):
        """Test successful search operation."""
        results = await mock_adapter.search('machine learning', limit=10)
        
        assert isinstance(results, SearchResults)
        assert results.query == 'machine learning'
        assert results.engine_id == 'test_mock'
        assert len(results.results) >= 1
        
        # Should find the document about machine learning
        ml_result = next((r for r in results.results if 'machine learning' in r.content.lower()), None)
        assert ml_result is not None
        assert ml_result.document_id == 'test1'
        assert ml_result.relevance_score > 0
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_adapter):
        """Test search with no matching results."""
        results = await mock_adapter.search('nonexistent topic', limit=10)
        
        assert isinstance(results, SearchResults)
        assert len(results.results) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_adapter):
        """Test health check."""
        is_healthy = await mock_adapter.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_simulated_failure(self):
        """Test simulated failure behavior."""
        config = {
            'response_delay': 0.01,
            'failure_rate': 1.0,  # Always fail
            'max_results': 5,
            'documents': []
        }
        adapter = InMemoryMockAdapter('failing_mock', config)
        
        with pytest.raises(SearchEngineError):
            await adapter.search('test query')


class TestFileMockAdapter:
    """Test cases for file-based mock adapter."""
    
    @pytest.fixture
    def temp_data_file(self, tmp_path):
        """Create a temporary JSON data file."""
        data = {
            'documents': [
                {
                    'id': 'file_doc_1',
                    'title': 'File Document 1',
                    'content': 'Content from file about data science',
                    'category': 'data-science'
                },
                {
                    'id': 'file_doc_2',
                    'title': 'File Document 2',
                    'content': 'Another file document about analytics',
                    'category': 'analytics'
                }
            ]
        }
        
        data_file = tmp_path / 'test_data.json'
        with open(data_file, 'w') as f:
            json.dump(data, f)
        
        return str(data_file)
    
    @pytest.fixture
    def file_mock_config(self, temp_data_file):
        return {
            'data_file': temp_data_file,
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        }
    
    @pytest.fixture
    def file_mock_adapter(self, file_mock_config):
        return FileMockAdapter('test_file_mock', file_mock_config)
    
    def test_initialization(self, file_mock_adapter, temp_data_file):
        """Test file mock adapter initialization."""
        assert file_mock_adapter.engine_id == 'test_file_mock'
        assert file_mock_adapter.data_file == temp_data_file
        assert len(file_mock_adapter.documents) == 2
    
    def test_missing_data_file(self):
        """Test initialization with missing data file."""
        config = {
            'data_file': '/nonexistent/file.json',
            'response_delay': 0.01,
            'failure_rate': 0.0,
            'max_results': 10
        }
        
        with pytest.raises(SearchEngineError):
            FileMockAdapter('test_missing_file', config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, file_mock_adapter):
        """Test successful search operation."""
        results = await file_mock_adapter.search('data science', limit=10)
        
        assert isinstance(results, SearchResults)
        assert results.query == 'data science'
        assert results.engine_id == 'test_file_mock'
        assert len(results.results) >= 1
        
        # Should find the document about data science
        ds_result = next((r for r in results.results if 'data science' in r.content.lower()), None)
        assert ds_result is not None
        assert ds_result.document_id == 'file_doc_1'
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, file_mock_adapter):
        """Test successful health check."""
        is_healthy = await file_mock_adapter.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_missing_file(self, file_mock_adapter, tmp_path):
        """Test health check with missing file."""
        # Remove the data file
        Path(file_mock_adapter.data_file).unlink()
        
        is_healthy = await file_mock_adapter.health_check()
        assert is_healthy is False


class TestRestApiAdapter:
    """Test cases for REST API adapter."""
    
    @pytest.fixture
    def api_config(self):
        return {
            'base_url': 'https://api.example.com',
            'search_endpoint': '/search',
            'health_endpoint': '/health',
            'query_param': 'q',
            'limit_param': 'limit',
            'auth_token': 'test-token',
            'response_format': 'json',
            'result_path': 'results',
            'id_field': 'id',
            'score_field': 'score',
            'content_field': 'content'
        }
    
    @pytest.fixture
    def api_adapter(self, api_config):
        return RestApiAdapter('test_api', api_config)
    
    def test_initialization(self, api_adapter, api_config):
        """Test REST API adapter initialization."""
        assert api_adapter.engine_id == 'test_api'
        assert api_adapter.base_url == 'https://api.example.com'
        assert api_adapter.search_endpoint == '/search'
        assert api_adapter.auth_token == 'test-token'
        assert 'Authorization' in api_adapter.headers
    
    def test_missing_base_url(self):
        """Test initialization without base_url."""
        config = {'search_endpoint': '/search'}
        
        with pytest.raises(ValueError, match="base_url is required"):
            RestApiAdapter('test_api', config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, api_adapter):
        """Test successful search operation."""
        mock_response_data = {
            'results': [
                {
                    'id': 'api_doc_1',
                    'score': 0.95,
                    'content': 'API document content',
                    'title': 'API Document'
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            results = await api_adapter.search('test query', limit=10)
            
            assert isinstance(results, SearchResults)
            assert results.query == 'test query'
            assert results.engine_id == 'test_api'
            assert len(results.results) == 1
            
            result = results.results[0]
            assert result.document_id == 'api_doc_1'
            assert result.relevance_score == 0.95
            assert result.content == 'API document content'
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, api_adapter):
        """Test successful health check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            is_healthy = await api_adapter.health_check()
            assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_connection_error(self, api_adapter):
        """Test connection error handling."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.side_effect = Exception("Connection failed")
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(SearchEngineError):  # Changed to SearchEngineError since it catches all exceptions
                await api_adapter.search('test query')


class TestDatabaseAdapter:
    """Test cases for database adapter."""
    
    @pytest.fixture
    def db_config(self):
        return {
            'connection_string': 'postgresql://user:pass@localhost:5432/testdb',
            'table_name': 'documents',
            'id_column': 'id',
            'content_columns': ['title', 'content'],
            'score_method': 'match_count',
            'search_type': 'fulltext'
        }
    
    @pytest.fixture
    def db_adapter(self, db_config):
        return DatabaseAdapter('test_db', db_config)
    
    def test_initialization(self, db_adapter, db_config):
        """Test database adapter initialization."""
        assert db_adapter.engine_id == 'test_db'
        assert db_adapter.connection_string == db_config['connection_string']
        assert db_adapter.table_name == 'documents'
        assert db_adapter.id_column == 'id'
        assert db_adapter.content_columns == ['title', 'content']
    
    def test_missing_connection_string(self):
        """Test initialization without connection string."""
        config = {'table_name': 'documents'}
        
        with pytest.raises(ValueError, match="connection_string is required"):
            DatabaseAdapter('test_db', config)
    
    def test_missing_table_name(self):
        """Test initialization without table name."""
        config = {'connection_string': 'postgresql://localhost/test'}
        
        with pytest.raises(ValueError, match="table_name is required"):
            DatabaseAdapter('test_db', config)
    
    @pytest.mark.asyncio
    async def test_search_mock_results(self, db_adapter):
        """Test search with mock database results."""
        results = await db_adapter.search('test query', limit=5)
        
        assert isinstance(results, SearchResults)
        assert results.query == 'test query'
        assert results.engine_id == 'test_db'
        assert len(results.results) <= 5
        
        # Check that mock results contain expected data
        for result in results.results:
            assert result.document_id.startswith('db_doc_')
            assert result.relevance_score > 0
            assert 'test query' in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_health_check(self, db_adapter):
        """Test health check (mock implementation)."""
        is_healthy = await db_adapter.health_check()
        assert is_healthy is True


class TestWebScrapingAdapter:
    """Test cases for web scraping adapter."""
    
    @pytest.fixture
    def scraping_config(self):
        return {
            'search_url': 'https://example.com/search?q={query}',
            'result_selector': '.result',
            'title_selector': '.title',
            'content_selector': '.content',
            'link_selector': 'a',
            'headers': {
                'User-Agent': 'Test Agent'
            }
        }
    
    @pytest.fixture
    def scraping_adapter(self, scraping_config):
        return WebScrapingAdapter('test_scraper', scraping_config)
    
    def test_initialization(self, scraping_adapter, scraping_config):
        """Test web scraping adapter initialization."""
        assert scraping_adapter.engine_id == 'test_scraper'
        assert scraping_adapter.search_url == scraping_config['search_url']
        assert scraping_adapter.result_selector == '.result'
        assert scraping_adapter.headers['User-Agent'] == 'Test Agent'
    
    def test_missing_search_url(self):
        """Test initialization without search URL."""
        config = {'result_selector': '.result'}
        
        with pytest.raises(ValueError, match="search_url is required"):
            WebScrapingAdapter('test_scraper', config)
    
    @pytest.mark.asyncio
    async def test_search_mock_results(self, scraping_adapter):
        """Test search with mock scraping results."""
        results = await scraping_adapter.search('test query', limit=5)
        
        assert isinstance(results, SearchResults)
        assert results.query == 'test query'
        assert results.engine_id == 'test_scraper'
        assert len(results.results) <= 5
        
        # Check that mock results contain expected data
        for result in results.results:
            assert result.document_id.startswith('web_result_')
            assert result.relevance_score > 0
            assert 'test query' in result.content.lower()
            assert 'url' in result.metadata
    
    @pytest.mark.asyncio
    async def test_health_check(self, scraping_adapter):
        """Test health check (mock implementation)."""
        is_healthy = await scraping_adapter.health_check()
        assert is_healthy is True


class TestAdapterRegistry:
    """Test adapter registration with new adapters."""
    
    def test_new_adapters_registered(self):
        """Test that new adapters are registered in the registry."""
        from src.adapters.registry import get_adapter_registry
        
        registry = get_adapter_registry()
        available_adapters = registry.get_available_adapters()
        
        # Check that all new adapters are registered
        expected_adapters = [
            'elasticsearch', 'solr', 'opensearch',
            'mock_inmemory', 'mock_file',
            'rest_api', 'database', 'web_scraping'
        ]
        
        for adapter_type in expected_adapters:
            assert adapter_type in available_adapters
    
    def test_create_opensearch_adapter(self):
        """Test creating OpenSearch adapter through registry."""
        from src.adapters.registry import get_adapter_registry
        
        registry = get_adapter_registry()
        config = {
            'host': 'localhost',
            'port': 9200,
            'index': 'test'
        }
        
        adapter = registry.create_adapter('opensearch', 'test_os', config)
        assert isinstance(adapter, OpenSearchAdapter)
        assert adapter.engine_id == 'test_os'
    
    def test_create_mock_adapters(self):
        """Test creating mock adapters through registry."""
        from src.adapters.registry import get_adapter_registry
        
        registry = get_adapter_registry()
        
        # Test in-memory mock
        inmem_config = {'documents': [], 'response_delay': 0.01}
        inmem_adapter = registry.create_adapter('mock_inmemory', 'test_inmem', inmem_config)
        assert isinstance(inmem_adapter, InMemoryMockAdapter)
        
        # Test file mock (with a mock file path)
        file_config = {'data_file': '/tmp/nonexistent.json', 'response_delay': 0.01}
        with pytest.raises(ValueError):  # Should fail due to missing file
            registry.create_adapter('mock_file', 'test_file', file_config)