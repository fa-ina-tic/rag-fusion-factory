"""OpenSearch search engine adapter implementation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp

from .base import SearchEngineAdapter, SearchEngineError, SearchEngineConnectionError
from ..models.core import SearchResults


logger = logging.getLogger(__name__)


class OpenSearchAdapter(SearchEngineAdapter):
    """OpenSearch search engine adapter.
    
    Provides standardized access to OpenSearch clusters through the REST API.
    OpenSearch is compatible with Elasticsearch API but has some differences.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the OpenSearch adapter.
        
        Args:
            engine_id: Unique identifier for this OpenSearch instance
            config: Configuration dictionary containing:
                - host: OpenSearch host (default: localhost)
                - port: OpenSearch port (default: 9200)
                - index: Default index to search
                - username: Optional username for authentication
                - password: Optional password for authentication
                - use_ssl: Whether to use HTTPS (default: False)
                - verify_ssl: Whether to verify SSL certificates (default: True)
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 9200)
        self.index = config.get('index', '_all')
        self.username = config.get('username')
        self.password = config.get('password')
        self.use_ssl = config.get('use_ssl', False)
        self.verify_ssl = config.get('verify_ssl', True)
        
        # Build base URL
        scheme = 'https' if self.use_ssl else 'http'
        self.base_url = f"{scheme}://{self.host}:{self.port}"
        
        # Setup authentication
        self.auth = None
        if self.username and self.password:
            self.auth = aiohttp.BasicAuth(self.username, self.password)
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against OpenSearch.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters:
                - index: Override default index
                - fields: List of fields to search in
                - boost_fields: Dict of field -> boost factor
                - filters: Dict of field -> value filters
                - analyzer: Custom analyzer to use
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If the search operation fails
        """
        try:
            # Build search request
            search_body = self._build_search_body(query, limit, **kwargs)
            index = kwargs.get('index', self.index)
            
            # Execute search
            url = urljoin(self.base_url, f"/{index}/_search")
            
            connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)
            async with aiohttp.ClientSession(auth=self.auth, connector=connector) as session:
                async with session.post(
                    url,
                    json=search_body,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchEngineError(
                            f"OpenSearch search failed with status {response.status}: {error_text}"
                        )
                    
                    result_data = await response.json()
            
            # Parse and normalize results
            raw_results = self._parse_opensearch_response(result_data)
            normalized_results = self._normalize_result_format(raw_results)
            
            return self._create_search_results(query, normalized_results)
            
        except aiohttp.ClientError as e:
            raise SearchEngineConnectionError(f"Failed to connect to OpenSearch: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise SearchEngineError(f"Invalid JSON response from OpenSearch: {str(e)}") from e
        except Exception as e:
            raise SearchEngineError(f"OpenSearch search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if OpenSearch is healthy and responsive.
        
        Returns:
            True if OpenSearch is healthy, False otherwise
        """
        try:
            url = urljoin(self.base_url, "/_cluster/health")
            
            connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)
            async with aiohttp.ClientSession(auth=self.auth, connector=connector) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return False
                    
                    health_data = await response.json()
                    status = health_data.get('status', 'red')
                    
                    # Consider yellow and green as healthy
                    return status in ['yellow', 'green']
                    
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {str(e)}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the OpenSearch adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'opensearch',
            'engine_id': self.engine_id,
            'host': self.host,
            'port': self.port,
            'index': self.index,
            'use_ssl': self.use_ssl,
            'verify_ssl': self.verify_ssl,
            'timeout': self.timeout,
            'has_auth': bool(self.username and self.password)
        }
    
    def _build_search_body(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
        """Build OpenSearch search request body.
        
        Args:
            query: The search query string
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            OpenSearch search request body
        """
        # Basic multi-match query with OpenSearch-specific features
        search_body = {
            "size": limit,
            "query": {
                "multi_match": {
                    "query": query,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "or"
                }
            },
            "_source": True,
            "highlight": {
                "fields": {
                    "*": {}
                },
                "fragment_size": 150,
                "number_of_fragments": 1,
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
        }
        
        # Handle field-specific search
        fields = kwargs.get('fields')
        if fields:
            search_body["query"]["multi_match"]["fields"] = fields
        
        # Handle field boosting
        boost_fields = kwargs.get('boost_fields')
        if boost_fields:
            boosted_fields = []
            for field, boost in boost_fields.items():
                boosted_fields.append(f"{field}^{boost}")
            search_body["query"]["multi_match"]["fields"] = boosted_fields
        
        # Handle custom analyzer
        analyzer = kwargs.get('analyzer')
        if analyzer:
            search_body["query"]["multi_match"]["analyzer"] = analyzer
        
        # Handle filters
        filters = kwargs.get('filters')
        if filters:
            bool_query = {
                "bool": {
                    "must": [search_body["query"]],
                    "filter": []
                }
            }
            
            for field, value in filters.items():
                if isinstance(value, list):
                    # Multiple values - use terms query
                    bool_query["bool"]["filter"].append({
                        "terms": {field: value}
                    })
                else:
                    # Single value - use term query
                    bool_query["bool"]["filter"].append({
                        "term": {field: value}
                    })
            
            search_body["query"] = bool_query
        
        return search_body
    
    def _parse_opensearch_response(self, response_data: Dict[str, Any]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Parse OpenSearch response into standardized format.
        
        Args:
            response_data: Raw OpenSearch response
            
        Returns:
            List of tuples (document_id, score, content, metadata)
        """
        results = []
        
        hits = response_data.get('hits', {}).get('hits', [])
        max_score = response_data.get('hits', {}).get('max_score', 1.0) or 1.0
        
        for hit in hits:
            try:
                # Extract document ID
                doc_id = hit.get('_id', '')
                
                # Normalize score (0-1 range)
                raw_score = hit.get('_score', 0.0)
                normalized_score = raw_score / max_score if max_score > 0 else 0.0
                
                # Extract content from source
                source = hit.get('_source', {})
                content = self._extract_content_from_source(source)
                
                # Build metadata
                metadata = {
                    'index': hit.get('_index', ''),
                    'type': hit.get('_type', ''),
                    'raw_score': raw_score,
                    'max_score': max_score,
                    'source_fields': list(source.keys()) if source else [],
                    'highlight': hit.get('highlight', {}),
                    'engine_type': 'opensearch'
                }
                
                # Add source fields to metadata (excluding content fields)
                for key, value in source.items():
                    if key not in ['content', 'text', 'body', 'description']:
                        metadata[f'source_{key}'] = value
                
                results.append((doc_id, normalized_score, content, metadata))
                
            except Exception as e:
                logger.warning(f"Error parsing OpenSearch hit: {str(e)}")
                continue
        
        return results
    
    def _extract_content_from_source(self, source: Dict[str, Any]) -> str:
        """Extract text content from OpenSearch document source.
        
        Args:
            source: Document source from OpenSearch
            
        Returns:
            Extracted text content
        """
        # Common content field names in order of preference
        content_fields = ['content', 'text', 'body', 'description', 'title', 'name', 'summary']
        
        for field in content_fields:
            if field in source:
                value = source[field]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, list) and value:
                    # Join list of strings
                    text_items = [str(item) for item in value if str(item).strip()]
                    if text_items:
                        return ' '.join(text_items)
        
        # Fallback: concatenate all string values
        text_parts = []
        for key, value in source.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value.strip()}")
            elif isinstance(value, (int, float)):
                text_parts.append(f"{key}: {value}")
        
        return ' | '.join(text_parts) if text_parts else f"Document ID: {source.get('id', 'unknown')}"