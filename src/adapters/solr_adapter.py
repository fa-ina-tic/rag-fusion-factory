"""Apache Solr search engine adapter implementation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlencode

import aiohttp

from .base import SearchEngineAdapter, SearchEngineError, SearchEngineConnectionError
from ..models.core import SearchResults


logger = logging.getLogger(__name__)


class SolrAdapter(SearchEngineAdapter):
    """Apache Solr search engine adapter.
    
    Provides standardized access to Solr cores through the REST API.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the Solr adapter.
        
        Args:
            engine_id: Unique identifier for this Solr instance
            config: Configuration dictionary containing:
                - host: Solr host (default: localhost)
                - port: Solr port (default: 8983)
                - core: Solr core name (required)
                - username: Optional username for authentication
                - password: Optional password for authentication
                - use_ssl: Whether to use HTTPS (default: False)
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 8983)
        self.core = config.get('core')
        if not self.core:
            raise ValueError("Solr core name is required in configuration")
        
        self.username = config.get('username')
        self.password = config.get('password')
        self.use_ssl = config.get('use_ssl', False)
        
        # Build base URL
        scheme = 'https' if self.use_ssl else 'http'
        self.base_url = f"{scheme}://{self.host}:{self.port}/solr/{self.core}"
        
        # Setup authentication
        self.auth = None
        if self.username and self.password:
            self.auth = aiohttp.BasicAuth(self.username, self.password)
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against Solr.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters:
                - fields: List of fields to return (fl parameter)
                - query_fields: List of fields to search in (qf parameter)
                - boost_fields: Dict of field -> boost factor
                - filters: Dict of field -> value filters (fq parameter)
                - sort: Sort specification
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If the search operation fails
        """
        try:
            # Build search parameters
            search_params = self._build_search_params(query, limit, **kwargs)
            
            # Execute search
            url = urljoin(self.base_url, "/select")
            
            async with aiohttp.ClientSession(auth=self.auth) as session:
                async with session.get(url, params=search_params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchEngineError(
                            f"Solr search failed with status {response.status}: {error_text}"
                        )
                    
                    result_data = await response.json()
            
            # Parse and normalize results
            raw_results = self._parse_solr_response(result_data)
            normalized_results = self._normalize_result_format(raw_results)
            
            return self._create_search_results(query, normalized_results)
            
        except aiohttp.ClientError as e:
            raise SearchEngineConnectionError(f"Failed to connect to Solr: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise SearchEngineError(f"Invalid JSON response from Solr: {str(e)}") from e
        except Exception as e:
            raise SearchEngineError(f"Solr search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if Solr is healthy and responsive.
        
        Returns:
            True if Solr is healthy, False otherwise
        """
        try:
            url = urljoin(self.base_url, "/admin/ping")
            
            async with aiohttp.ClientSession(auth=self.auth) as session:
                async with session.get(url, params={'wt': 'json'}) as response:
                    if response.status != 200:
                        return False
                    
                    ping_data = await response.json()
                    status = ping_data.get('status', 'ERROR')
                    
                    return status == 'OK'
                    
        except Exception as e:
            logger.error(f"Solr health check failed: {str(e)}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the Solr adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'solr',
            'engine_id': self.engine_id,
            'host': self.host,
            'port': self.port,
            'core': self.core,
            'use_ssl': self.use_ssl,
            'timeout': self.timeout,
            'has_auth': bool(self.username and self.password)
        }
    
    def _build_search_params(self, query: str, limit: int, **kwargs) -> Dict[str, str]:
        """Build Solr search parameters.
        
        Args:
            query: The search query string
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary of Solr search parameters
        """
        # Basic search parameters
        params = {
            'q': query,
            'rows': str(limit),
            'wt': 'json',
            'indent': 'false',
            'defType': 'edismax',  # Extended DisMax query parser
            'hl': 'true',  # Enable highlighting
            'hl.fl': '*',  # Highlight all fields
            'hl.fragsize': '150',
            'hl.maxAnalyzedChars': '51200'
        }
        
        # Handle field selection
        fields = kwargs.get('fields')
        if fields:
            params['fl'] = ','.join(fields)
        else:
            params['fl'] = '*,score'  # Return all fields plus score
        
        # Handle query fields (which fields to search in)
        query_fields = kwargs.get('query_fields')
        boost_fields = kwargs.get('boost_fields')
        
        if boost_fields:
            # Build qf parameter with boosts
            qf_parts = []
            for field, boost in boost_fields.items():
                qf_parts.append(f"{field}^{boost}")
            params['qf'] = ' '.join(qf_parts)
        elif query_fields:
            params['qf'] = ' '.join(query_fields)
        
        # Handle filters
        filters = kwargs.get('filters')
        if filters:
            fq_params = []
            for field, value in filters.items():
                # Escape special characters in filter values
                escaped_value = self._escape_solr_value(str(value))
                fq_params.append(f"{field}:{escaped_value}")
            params['fq'] = fq_params
        
        # Handle sorting
        sort_spec = kwargs.get('sort')
        if sort_spec:
            params['sort'] = sort_spec
        else:
            params['sort'] = 'score desc'  # Default sort by relevance
        
        return params
    
    def _parse_solr_response(self, response_data: Dict[str, Any]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Parse Solr response into standardized format.
        
        Args:
            response_data: Raw Solr response
            
        Returns:
            List of tuples (document_id, score, content, metadata)
        """
        results = []
        
        response = response_data.get('response', {})
        docs = response.get('docs', [])
        max_score = response.get('maxScore', 1.0) or 1.0
        
        # Get highlighting information
        highlighting = response_data.get('highlighting', {})
        
        for doc in docs:
            try:
                # Extract document ID
                doc_id = doc.get('id', str(doc.get('_id', '')))
                if not doc_id:
                    # Generate ID from first available field
                    doc_id = str(next(iter(doc.values()), 'unknown'))
                
                # Normalize score (0-1 range)
                raw_score = doc.get('score', 0.0)
                normalized_score = raw_score / max_score if max_score > 0 else 0.0
                
                # Extract content
                content = self._extract_content_from_doc(doc)
                
                # Build metadata
                metadata = {
                    'raw_score': raw_score,
                    'max_score': max_score,
                    'doc_fields': list(doc.keys()),
                    'highlight': highlighting.get(doc_id, {})
                }
                
                # Add document fields to metadata (excluding content fields)
                for key, value in doc.items():
                    if key not in ['content', 'text', 'body', 'description', 'score']:
                        metadata[f'doc_{key}'] = value
                
                results.append((doc_id, normalized_score, content, metadata))
                
            except Exception as e:
                logger.warning(f"Error parsing Solr document: {str(e)}")
                continue
        
        return results
    
    def _extract_content_from_doc(self, doc: Dict[str, Any]) -> str:
        """Extract text content from Solr document.
        
        Args:
            doc: Solr document
            
        Returns:
            Extracted text content
        """
        # Common content field names in order of preference
        content_fields = ['content', 'text', 'body', 'description', 'title', 'name']
        
        for field in content_fields:
            if field in doc:
                value = doc[field]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, list) and value:
                    # Join list of strings
                    text_items = [str(item) for item in value if str(item).strip()]
                    if text_items:
                        return ' '.join(text_items)
        
        # Fallback: concatenate all string values (excluding system fields)
        text_parts = []
        system_fields = {'_version_', 'score', 'id'}
        
        for key, value in doc.items():
            if key in system_fields:
                continue
                
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value.strip()}")
            elif isinstance(value, list):
                list_text = ', '.join(str(item) for item in value if str(item).strip())
                if list_text:
                    text_parts.append(f"{key}: {list_text}")
            elif isinstance(value, (int, float)):
                text_parts.append(f"{key}: {value}")
        
        return ' | '.join(text_parts) if text_parts else f"Document ID: {doc.get('id', 'unknown')}"
    
    def _escape_solr_value(self, value: str) -> str:
        """Escape special characters in Solr query values.
        
        Args:
            value: Value to escape
            
        Returns:
            Escaped value safe for Solr queries
        """
        # Solr special characters that need escaping
        special_chars = ['+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/']
        
        escaped_value = value
        for char in special_chars:
            escaped_value = escaped_value.replace(char, f'\\{char}')
        
        return f'"{escaped_value}"'  # Wrap in quotes for safety