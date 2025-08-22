"""Abstract base class for search engine adapters."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..models.core import SearchResult, SearchResults
from ..utils.error_handling import ErrorHandler, ErrorContext, get_error_handler


logger = logging.getLogger(__name__)


class SearchEngineError(Exception):
    """Base exception for search engine adapter errors."""
    pass


class SearchEngineTimeoutError(SearchEngineError):
    """Exception raised when search engine operations timeout."""
    pass


class SearchEngineConnectionError(SearchEngineError):
    """Exception raised when connection to search engine fails."""
    pass


class SearchEngineAdapter(ABC):
    """Abstract base class for search engine adapters.
    
    This class defines the interface that all search engine adapters must implement
    to provide standardized access to different search engines.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the search engine adapter.
        
        Args:
            engine_id: Unique identifier for this search engine instance
            config: Configuration dictionary for the search engine
            timeout: Default timeout for search operations in seconds
        """
        self.engine_id = engine_id
        self.config = config
        self.timeout = timeout
        self._is_healthy = True
        self._last_health_check = None
        self.error_handler = get_error_handler()
        
    @abstractmethod
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against the search engine.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters specific to the engine
            
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If the search operation fails
            SearchEngineTimeoutError: If the search operation times out
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the search engine is healthy and responsive.
        
        Returns:
            True if the engine is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the search engine adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        pass
    
    async def search_with_timeout(self, query: str, limit: int = 10, 
                                 timeout: Optional[float] = None, **kwargs) -> SearchResults:
        """Execute a search with timeout handling.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Additional search parameters
            
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineTimeoutError: If the operation times out
            SearchEngineError: If the search operation fails
        """
        timeout = timeout or self.timeout
        
        try:
            return await asyncio.wait_for(
                self.search(query, limit, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            error_context = ErrorContext(
                component="SearchEngineAdapter",
                operation="search_with_timeout",
                engine_id=self.engine_id,
                query=query
            )
            timeout_error = SearchEngineTimeoutError(
                f"Search operation timed out after {timeout} seconds"
            )
            self.error_handler.handle_error(timeout_error, error_context)
            raise timeout_error
        except Exception as e:
            error_context = ErrorContext(
                component="SearchEngineAdapter",
                operation="search_with_timeout",
                engine_id=self.engine_id,
                query=query
            )
            search_error = SearchEngineError(f"Search operation failed: {str(e)}")
            self.error_handler.handle_error(search_error, error_context)
            raise search_error from e
    
    async def health_check_with_timeout(self, timeout: Optional[float] = None) -> bool:
        """Perform health check with timeout handling.
        
        Args:
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            True if healthy, False otherwise
        """
        timeout = timeout or min(self.timeout, 10.0)  # Health checks should be quick
        
        try:
            result = await asyncio.wait_for(
                self.health_check(),
                timeout=timeout
            )
            self._is_healthy = result
            self._last_health_check = datetime.now()
            return result
        except asyncio.TimeoutError:
            error_context = ErrorContext(
                component="SearchEngineAdapter",
                operation="health_check_with_timeout",
                engine_id=self.engine_id
            )
            timeout_error = SearchEngineTimeoutError("Health check timed out")
            self.error_handler.handle_error(timeout_error, error_context)
            self._is_healthy = False
            return False
        except Exception as e:
            error_context = ErrorContext(
                component="SearchEngineAdapter",
                operation="health_check_with_timeout",
                engine_id=self.engine_id
            )
            self.error_handler.handle_error(e, error_context)
            self._is_healthy = False
            return False
    
    def _normalize_result_format(self, raw_results: List[Tuple[str, float, str, Dict[str, Any]]]) -> List[SearchResult]:
        """Convert raw search results to standardized SearchResult format.
        
        Args:
            raw_results: List of tuples (document_id, score, content, metadata)
            
        Returns:
            List of SearchResult objects
        """
        normalized_results = []
        
        for result_data in raw_results:
            try:
                if len(result_data) != 4:
                    logger.warning(f"Invalid result format from engine {self.engine_id}: {result_data}")
                    continue
                    
                document_id, score, content, metadata = result_data
                
                # Validate and normalize score
                if not isinstance(score, (int, float)) or score < 0:
                    logger.warning(f"Invalid score {score} for document {document_id}")
                    score = 0.0
                
                # Ensure metadata is a dictionary
                if not isinstance(metadata, dict):
                    metadata = {}
                
                search_result = SearchResult(
                    document_id=str(document_id),
                    relevance_score=float(score),
                    content=str(content) if content else "",
                    metadata=metadata,
                    engine_source=self.engine_id
                )
                
                normalized_results.append(search_result)
                
            except Exception as e:
                logger.error(f"Error normalizing result from engine {self.engine_id}: {str(e)}")
                continue
        
        return normalized_results
    
    def _create_search_results(self, query: str, normalized_results: List[SearchResult]) -> SearchResults:
        """Create a SearchResults object from normalized results.
        
        Args:
            query: The original search query
            normalized_results: List of normalized SearchResult objects
            
        Returns:
            SearchResults object
        """
        return SearchResults(
            query=query,
            results=normalized_results,
            engine_id=self.engine_id,
            timestamp=datetime.now(),
            total_results=len(normalized_results)
        )
    
    @property
    def is_healthy(self) -> bool:
        """Get the last known health status."""
        return self._is_healthy
    
    @property
    def last_health_check(self) -> Optional[datetime]:
        """Get the timestamp of the last health check."""
        return self._last_health_check
    
    def __str__(self) -> str:
        """String representation of the adapter."""
        return f"{self.__class__.__name__}(engine_id='{self.engine_id}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the adapter."""
        return (f"{self.__class__.__name__}(engine_id='{self.engine_id}', "
                f"healthy={self.is_healthy}, timeout={self.timeout})")