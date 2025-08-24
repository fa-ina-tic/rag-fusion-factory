"""Custom search engine adapter implementations for specialized use cases."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlencode

import aiohttp

from .base import SearchEngineAdapter, SearchEngineError, SearchEngineConnectionError
from ..models.core import SearchResults


logger = logging.getLogger(__name__)


class RestApiAdapter(SearchEngineAdapter):
    """Generic REST API search engine adapter.
    
    Provides a configurable adapter for any REST API that follows
    common search patterns. Useful for integrating with custom
    search services or APIs that don't have dedicated adapters.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the REST API adapter.
        
        Args:
            engine_id: Unique identifier for this API instance
            config: Configuration dictionary containing:
                - base_url: Base URL of the API
                - search_endpoint: Search endpoint path (default: /search)
                - health_endpoint: Health check endpoint path (default: /health)
                - query_param: Query parameter name (default: q)
                - limit_param: Limit parameter name (default: limit)
                - headers: Additional HTTP headers
                - auth_token: Bearer token for authentication
                - response_format: Expected response format (json/xml)
                - result_path: JSONPath to results in response (default: results)
                - id_field: Field name for document ID (default: id)
                - score_field: Field name for relevance score (default: score)
                - content_field: Field name for content (default: content)
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.base_url = config.get('base_url')
        if not self.base_url:
            raise ValueError("base_url is required for RestApiAdapter")
        
        self.search_endpoint = config.get('search_endpoint', '/search')
        self.health_endpoint = config.get('health_endpoint', '/health')
        self.query_param = config.get('query_param', 'q')
        self.limit_param = config.get('limit_param', 'limit')
        self.headers = config.get('headers', {})
        self.auth_token = config.get('auth_token')
        self.response_format = config.get('response_format', 'json')
        self.result_path = config.get('result_path', 'results')
        self.id_field = config.get('id_field', 'id')
        self.score_field = config.get('score_field', 'score')
        self.content_field = config.get('content_field', 'content')
        
        # Setup headers
        if self.auth_token:
            self.headers['Authorization'] = f'Bearer {self.auth_token}'
        
        if self.response_format == 'json':
            self.headers['Accept'] = 'application/json'
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against the REST API.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters passed as query params
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If the search operation fails
        """
        try:
            # Build query parameters
            params = {
                self.query_param: query,
                self.limit_param: limit
            }
            
            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ['index', 'fields', 'boost_fields', 'filters']:
                    params[key] = value
            
            # Execute search
            url = urljoin(self.base_url, self.search_endpoint)
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchEngineError(
                            f"REST API search failed with status {response.status}: {error_text}"
                        )
                    
                    if self.response_format == 'json':
                        result_data = await response.json()
                    else:
                        result_text = await response.text()
                        result_data = {'raw_response': result_text}
            
            # Parse and normalize results
            raw_results = self._parse_api_response(result_data)
            normalized_results = self._normalize_result_format(raw_results)
            
            return self._create_search_results(query, normalized_results)
            
        except aiohttp.ClientError as e:
            raise SearchEngineConnectionError(f"Failed to connect to REST API: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise SearchEngineError(f"Invalid JSON response from REST API: {str(e)}") from e
        except Exception as e:
            raise SearchEngineError(f"REST API search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the REST API is healthy and responsive.
        
        Returns:
            True if the API is healthy, False otherwise
        """
        try:
            url = urljoin(self.base_url, self.health_endpoint)
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"REST API health check failed: {str(e)}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the REST API adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'rest_api',
            'engine_id': self.engine_id,
            'base_url': self.base_url,
            'search_endpoint': self.search_endpoint,
            'health_endpoint': self.health_endpoint,
            'query_param': self.query_param,
            'limit_param': self.limit_param,
            'response_format': self.response_format,
            'timeout': self.timeout,
            'has_auth': bool(self.auth_token)
        }
    
    def _parse_api_response(self, response_data: Dict[str, Any]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Parse REST API response into standardized format.
        
        Args:
            response_data: Raw API response
            
        Returns:
            List of tuples (document_id, score, content, metadata)
        """
        results = []
        
        try:
            # Extract results using configured path
            results_data = self._extract_results_from_response(response_data)
            
            if not isinstance(results_data, list):
                logger.warning(f"Expected list of results, got {type(results_data)}")
                return results
            
            for item in results_data:
                if not isinstance(item, dict):
                    continue
                
                try:
                    # Extract fields using configured field names
                    doc_id = str(item.get(self.id_field, ''))
                    raw_score = float(item.get(self.score_field, 0.0))
                    content = str(item.get(self.content_field, ''))
                    
                    # Build metadata from remaining fields
                    metadata = {
                        'engine_type': 'rest_api',
                        'raw_score': raw_score,
                        'api_fields': list(item.keys())
                    }
                    
                    # Add all fields to metadata (excluding core fields)
                    for key, value in item.items():
                        if key not in [self.id_field, self.score_field, self.content_field]:
                            metadata[f'api_{key}'] = value
                    
                    results.append((doc_id, raw_score, content, metadata))
                    
                except Exception as e:
                    logger.warning(f"Error parsing API result item: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
        
        return results
    
    def _extract_results_from_response(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract results from API response using configured path.
        
        Args:
            response_data: Raw API response
            
        Returns:
            List of result items
        """
        # Simple JSONPath-like extraction
        path_parts = self.result_path.split('.')
        current_data = response_data
        
        for part in path_parts:
            if isinstance(current_data, dict) and part in current_data:
                current_data = current_data[part]
            else:
                logger.warning(f"Could not find path '{self.result_path}' in API response")
                return []
        
        return current_data if isinstance(current_data, list) else []


class DatabaseAdapter(SearchEngineAdapter):
    """Database search engine adapter for SQL-based search.
    
    Provides search functionality against SQL databases using
    full-text search capabilities or simple LIKE queries.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the database adapter.
        
        Args:
            engine_id: Unique identifier for this database instance
            config: Configuration dictionary containing:
                - connection_string: Database connection string
                - table_name: Table to search in
                - id_column: Column name for document ID
                - content_columns: List of columns to search in
                - score_method: Scoring method (rank, match_count, etc.)
                - search_type: Type of search (fulltext, like, regex)
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.connection_string = config.get('connection_string')
        if not self.connection_string:
            raise ValueError("connection_string is required for DatabaseAdapter")
        
        self.table_name = config.get('table_name')
        if not self.table_name:
            raise ValueError("table_name is required for DatabaseAdapter")
        
        self.id_column = config.get('id_column', 'id')
        self.content_columns = config.get('content_columns', ['content'])
        self.score_method = config.get('score_method', 'match_count')
        self.search_type = config.get('search_type', 'like')
        
        # Note: In a real implementation, you would initialize database connection here
        # For this example, we'll simulate database operations
        logger.warning("DatabaseAdapter is a mock implementation for demonstration")
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against the database.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If the search operation fails
        """
        try:
            # Simulate database search
            await asyncio.sleep(0.1)  # Simulate query time
            
            # Mock results for demonstration
            mock_results = self._generate_mock_database_results(query, limit)
            
            # Normalize results
            normalized_results = self._normalize_result_format(mock_results)
            
            return self._create_search_results(query, normalized_results)
            
        except Exception as e:
            raise SearchEngineError(f"Database search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive.
        
        Returns:
            True if the database is healthy, False otherwise
        """
        try:
            # Simulate database health check
            await asyncio.sleep(0.05)
            return True  # Mock implementation always returns healthy
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the database adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'database',
            'engine_id': self.engine_id,
            'table_name': self.table_name,
            'id_column': self.id_column,
            'content_columns': self.content_columns,
            'score_method': self.score_method,
            'search_type': self.search_type,
            'timeout': self.timeout
        }
    
    def _generate_mock_database_results(self, query: str, limit: int) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Generate mock database search results for demonstration.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of mock results
        """
        mock_data = [
            ('db_doc_1', 0.95, f'Database document about {query} with high relevance', 
             {'table': self.table_name, 'match_type': 'exact'}),
            ('db_doc_2', 0.75, f'Another document mentioning {query} concepts', 
             {'table': self.table_name, 'match_type': 'partial'}),
            ('db_doc_3', 0.60, f'Related content discussing {query} topics', 
             {'table': self.table_name, 'match_type': 'related'})
        ]
        
        return mock_data[:limit]


class WebScrapingAdapter(SearchEngineAdapter):
    """Web scraping search engine adapter.
    
    Provides search functionality by scraping web pages or APIs
    that don't have proper search APIs. Useful for integrating
    with websites that only provide HTML search interfaces.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the web scraping adapter.
        
        Args:
            engine_id: Unique identifier for this scraping instance
            config: Configuration dictionary containing:
                - search_url: URL template for search (use {query} placeholder)
                - result_selector: CSS selector for result items
                - title_selector: CSS selector for result titles
                - content_selector: CSS selector for result content
                - link_selector: CSS selector for result links
                - headers: HTTP headers for requests
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.search_url = config.get('search_url')
        if not self.search_url:
            raise ValueError("search_url is required for WebScrapingAdapter")
        
        self.result_selector = config.get('result_selector', '.result')
        self.title_selector = config.get('title_selector', '.title')
        self.content_selector = config.get('content_selector', '.content')
        self.link_selector = config.get('link_selector', 'a')
        self.headers = config.get('headers', {
            'User-Agent': 'Mozilla/5.0 (compatible; RAG-Fusion-Factory/1.0)'
        })
        
        logger.warning("WebScrapingAdapter is a mock implementation for demonstration")
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search by scraping web results.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If the scraping operation fails
        """
        try:
            # Simulate web scraping
            await asyncio.sleep(0.5)  # Simulate scraping time
            
            # Mock results for demonstration
            mock_results = self._generate_mock_scraping_results(query, limit)
            
            # Normalize results
            normalized_results = self._normalize_result_format(mock_results)
            
            return self._create_search_results(query, normalized_results)
            
        except Exception as e:
            raise SearchEngineError(f"Web scraping search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the target website is accessible.
        
        Returns:
            True if the website is accessible, False otherwise
        """
        try:
            # Simulate website accessibility check
            await asyncio.sleep(0.1)
            return True  # Mock implementation
            
        except Exception as e:
            logger.error(f"Web scraping health check failed: {str(e)}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the web scraping adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'web_scraping',
            'engine_id': self.engine_id,
            'search_url': self.search_url,
            'result_selector': self.result_selector,
            'title_selector': self.title_selector,
            'content_selector': self.content_selector,
            'timeout': self.timeout
        }
    
    def _generate_mock_scraping_results(self, query: str, limit: int) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Generate mock web scraping results for demonstration.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of mock results
        """
        mock_data = [
            ('web_result_1', 0.90, f'Web page about {query} - comprehensive guide', 
             {'url': f'https://example.com/page1?q={query}', 'source': 'web_scraping'}),
            ('web_result_2', 0.80, f'Article discussing {query} in detail', 
             {'url': f'https://example.com/page2?q={query}', 'source': 'web_scraping'}),
            ('web_result_3', 0.70, f'Blog post about {query} concepts', 
             {'url': f'https://example.com/page3?q={query}', 'source': 'web_scraping'})
        ]
        
        return mock_data[:limit]