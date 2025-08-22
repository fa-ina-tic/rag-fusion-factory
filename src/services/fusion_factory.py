"""Core fusion factory engine for orchestrating the entire RAG fusion process."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..models.core import SearchResults, NormalizedResults, RankedResults, SearchResult
from ..adapters.base import SearchEngineAdapter, SearchEngineError
from ..adapters.registry import AdapterRegistry, get_adapter_registry
from ..config.settings import get_search_config, get_normalization_config
from ..utils.error_handling import (
    ErrorHandler, ErrorContext, ErrorCategory, CircuitBreakerConfig,
    get_error_handler, FusionFactoryError
)
from ..utils.graceful_degradation import (
    GracefulDegradationManager, DegradationConfig, get_degradation_manager
)


logger = logging.getLogger(__name__)


class QueryDispatcher:
    """Handles concurrent querying to multiple search engines with fault tolerance."""
    
    def __init__(self, max_concurrent_engines: int = 10, timeout: float = 30.0, 
                 error_handler: Optional[ErrorHandler] = None):
        """Initialize the query dispatcher.
        
        Args:
            max_concurrent_engines: Maximum number of engines to query concurrently
            timeout: Default timeout for search operations
            error_handler: Optional error handler instance
        """
        self.max_concurrent_engines = max_concurrent_engines
        self.timeout = timeout
        self.error_handler = error_handler or get_error_handler()
        self._semaphore = None
        
        # Create circuit breakers for individual engines
        self.engine_circuit_breakers: Dict[str, Any] = {}
    
    @property
    def semaphore(self):
        """Get or create the semaphore for the current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_engines)
        return self._semaphore
    
    async def dispatch_query(self, query: str, adapters: List[SearchEngineAdapter], 
                           limit: int = 10) -> Dict[str, SearchResults]:
        """Dispatch query to multiple search engines concurrently.
        
        Args:
            query: Search query string
            adapters: List of search engine adapters to query
            limit: Maximum number of results per engine
            
        Returns:
            Dictionary mapping engine_id to SearchResults
        """
        if not adapters:
            logger.warning("No adapters provided for query dispatch")
            return {}
        
        logger.info(f"Dispatching query '{query}' to {len(adapters)} engines")
        
        # Create tasks for concurrent execution
        tasks = []
        for adapter in adapters:
            task = asyncio.create_task(
                self._query_single_engine(adapter, query, limit)
            )
            tasks.append((adapter.engine_id, task))
        
        # Wait for all tasks to complete
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        # Process results and handle exceptions
        for (engine_id, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Query failed for engine {engine_id}: {result}")
                continue
            
            if result is not None:
                results[engine_id] = result
                logger.debug(f"Query successful for engine {engine_id}: {len(result.results)} results")
        
        logger.info(f"Query dispatch completed: {len(results)}/{len(adapters)} engines responded")
        return results
    
    async def _query_single_engine(self, adapter: SearchEngineAdapter, 
                                 query: str, limit: int) -> Optional[SearchResults]:
        """Query a single search engine with semaphore control and circuit breaker protection.
        
        Args:
            adapter: Search engine adapter to query
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            SearchResults if successful, None if failed
        """
        async with self.semaphore:
            # Get or create circuit breaker for this engine
            circuit_breaker = self._get_engine_circuit_breaker(adapter.engine_id)
            
            try:
                # Use circuit breaker to protect the query
                return await circuit_breaker.call(
                    adapter.search_with_timeout, query, limit, self.timeout
                )
            except SearchEngineError as e:
                error_context = ErrorContext(
                    component="QueryDispatcher",
                    operation="query_single_engine",
                    engine_id=adapter.engine_id,
                    query=query
                )
                self.error_handler.handle_error(e, error_context)
                return None
            except Exception as e:
                error_context = ErrorContext(
                    component="QueryDispatcher",
                    operation="query_single_engine",
                    engine_id=adapter.engine_id,
                    query=query
                )
                self.error_handler.handle_error(e, error_context)
                return None
    
    def _get_engine_circuit_breaker(self, engine_id: str):
        """Get or create circuit breaker for specific engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Circuit breaker instance for the engine
        """
        if engine_id not in self.engine_circuit_breakers:
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=SearchEngineError,
                name=f"engine_{engine_id}"
            )
            self.engine_circuit_breakers[engine_id] = self.error_handler.create_circuit_breaker(
                f"engine_{engine_id}", cb_config
            )
        
        return self.engine_circuit_breakers[engine_id]


class ScoreNormalizer:
    """Normalizes scores from different engines to enable fair combination."""
    
    def __init__(self, method: str = "min_max", outlier_threshold: float = 2.5):
        """Initialize the score normalizer.
        
        Args:
            method: Normalization method (min_max, z_score, quantile)
            outlier_threshold: Threshold for outlier detection
        """
        self.method = method
        self.outlier_threshold = outlier_threshold
        self.engine_params: Dict[str, Dict[str, float]] = {}
    
    def normalize_scores(self, results: Dict[str, SearchResults]) -> Dict[str, NormalizedResults]:
        """Normalize scores from multiple engines.
        
        Args:
            results: Dictionary mapping engine_id to SearchResults
            
        Returns:
            Dictionary mapping engine_id to NormalizedResults
        """
        if not results:
            return {}
        
        logger.info(f"Normalizing scores for {len(results)} engines using {self.method} method")
        
        normalized_results = {}
        for engine_id, search_results in results.items():
            try:
                normalized = self._normalize_engine_scores(engine_id, search_results)
                normalized_results[engine_id] = normalized
                logger.debug(f"Normalized {len(normalized.results)} results for engine {engine_id}")
            except Exception as e:
                logger.error(f"Failed to normalize scores for engine {engine_id}: {e}")
                continue
        
        return normalized_results
    
    def _normalize_engine_scores(self, engine_id: str, 
                               search_results: SearchResults) -> NormalizedResults:
        """Normalize scores for a single engine.
        
        Args:
            engine_id: Engine identifier
            search_results: Original search results
            
        Returns:
            NormalizedResults with normalized scores
        """
        if not search_results.results:
            return NormalizedResults(
                query=search_results.query,
                results=[],
                engine_id=engine_id,
                normalization_method=self.method,
                timestamp=datetime.now()
            )
        
        # Extract scores
        scores = np.array([result.relevance_score for result in search_results.results])
        
        # Apply normalization method
        if self.method == "min_max":
            normalized_scores = self._min_max_normalize(scores)
        elif self.method == "z_score":
            normalized_scores = self._z_score_normalize(scores)
        elif self.method == "quantile":
            normalized_scores = self._quantile_normalize(scores)
        else:
            logger.warning(f"Unknown normalization method {self.method}, using min_max")
            normalized_scores = self._min_max_normalize(scores)
        
        # Create normalized results
        normalized_results = []
        for i, result in enumerate(search_results.results):
            normalized_result = SearchResult(
                document_id=result.document_id,
                relevance_score=float(normalized_scores[i]),
                content=result.content,
                metadata=result.metadata.copy(),
                engine_source=result.engine_source
            )
            normalized_results.append(normalized_result)
        
        return NormalizedResults(
            query=search_results.query,
            results=normalized_results,
            engine_id=engine_id,
            normalization_method=self.method,
            timestamp=datetime.now()
        )
    
    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to scores."""
        if len(scores) <= 1:
            return np.ones_like(scores)
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _z_score_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to scores."""
        if len(scores) <= 1:
            return np.ones_like(scores)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return np.ones_like(scores)
        
        z_scores = (scores - mean_score) / std_score
        
        # Convert to 0-1 range using sigmoid
        return 1 / (1 + np.exp(-z_scores))
    
    def _quantile_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Apply quantile-based normalization to scores."""
        if len(scores) <= 1:
            return np.ones_like(scores)
        
        # Use rank-based normalization
        ranks = np.argsort(np.argsort(scores))
        return ranks / (len(scores) - 1)


class ResultFusionEngine:
    """Combines normalized results using weighted score combination."""
    
    def __init__(self):
        """Initialize the result fusion engine."""
        pass
    
    def fuse_results(self, normalized_results: Dict[str, NormalizedResults], 
                    weights: Optional[np.ndarray] = None) -> RankedResults:
        """Fuse results from multiple engines using weighted combination.
        
        Args:
            normalized_results: Dictionary mapping engine_id to NormalizedResults
            weights: Optional weights for engines (defaults to equal weights)
            
        Returns:
            RankedResults with fused and ranked results
        """
        if not normalized_results:
            logger.warning("No normalized results provided for fusion")
            return RankedResults(
                query="",
                results=[],
                fusion_weights=np.array([]),
                confidence_scores=[],
                timestamp=datetime.now(),
                total_results=0
            )
        
        engine_ids = list(normalized_results.keys())
        num_engines = len(engine_ids)
        
        # Use equal weights if not provided
        if weights is None:
            weights = np.ones(num_engines) / num_engines
            logger.info(f"Using equal weights for {num_engines} engines")
        else:
            # Ensure weights sum to 1 (convex combination)
            weights = weights / np.sum(weights)
            logger.info(f"Using provided weights: {weights}")
        
        # Get query from first result set
        query = next(iter(normalized_results.values())).query
        
        # Collect all unique documents
        document_scores = {}  # document_id -> list of (engine_index, score)
        all_documents = {}    # document_id -> SearchResult (from first occurrence)
        
        for engine_idx, (engine_id, results) in enumerate(normalized_results.items()):
            for result in results.results:
                doc_id = result.document_id
                
                if doc_id not in document_scores:
                    document_scores[doc_id] = [0.0] * num_engines
                    all_documents[doc_id] = result
                
                document_scores[doc_id][engine_idx] = result.relevance_score
        
        # Calculate fused scores
        fused_results = []
        for doc_id, engine_scores in document_scores.items():
            # Calculate weighted score
            fused_score = np.dot(weights, engine_scores)
            
            # Calculate confidence based on agreement between engines
            non_zero_scores = [s for s in engine_scores if s > 0]
            confidence = len(non_zero_scores) / num_engines if num_engines > 0 else 0.0
            
            # Create fused result
            original_result = all_documents[doc_id]
            fused_result = SearchResult(
                document_id=doc_id,
                relevance_score=float(fused_score),
                content=original_result.content,
                metadata=original_result.metadata.copy(),
                engine_source="fusion"
            )
            
            fused_results.append((fused_result, confidence))
        
        # Sort by fused score (descending)
        fused_results.sort(key=lambda x: x[0].relevance_score, reverse=True)
        
        # Extract results and confidence scores
        final_results = [result for result, _ in fused_results]
        confidence_scores = [confidence for _, confidence in fused_results]
        
        logger.info(f"Fused {len(final_results)} unique documents from {num_engines} engines")
        
        return RankedResults(
            query=query,
            results=final_results,
            fusion_weights=weights,
            confidence_scores=confidence_scores,
            timestamp=datetime.now(),
            total_results=len(final_results)
        )


class FusionFactory:
    """Main orchestrator class for the RAG fusion process."""
    
    def __init__(self, adapter_registry: Optional[AdapterRegistry] = None):
        """Initialize the fusion factory.
        
        Args:
            adapter_registry: Optional adapter registry (uses global if None)
        """
        self.adapter_registry = adapter_registry or get_adapter_registry()
        
        # Load configuration
        search_config = get_search_config()
        normalization_config = get_normalization_config()
        
        # Initialize error handling
        self.error_handler = get_error_handler()
        
        # Initialize graceful degradation
        degradation_config = DegradationConfig(
            min_engines_required=search_config.get('min_engines_required', 1),
            min_results_threshold=search_config.get('min_results_threshold', 5),
            enable_fallback_results=search_config.get('enable_fallback_results', True),
            fallback_timeout=search_config.get('fallback_timeout', 10.0),
            quality_threshold=search_config.get('quality_threshold', 0.3),
            enable_cached_results=search_config.get('enable_cached_results', True),
            cache_expiry_minutes=search_config.get('cache_expiry_minutes', 30)
        )
        self.degradation_manager = get_degradation_manager(degradation_config)
        
        # Initialize components
        self.query_dispatcher = QueryDispatcher(
            max_concurrent_engines=search_config.get('max_concurrent_engines', 10),
            timeout=search_config.get('timeout', 30.0),
            error_handler=self.error_handler
        )
        
        self.score_normalizer = ScoreNormalizer(
            method=normalization_config.get('default_method', 'min_max'),
            outlier_threshold=normalization_config.get('outlier_threshold', 2.5)
        )
        
        self.fusion_engine = ResultFusionEngine()
        
        # Create circuit breakers for critical operations
        self._setup_circuit_breakers()
        
        logger.info("FusionFactory initialized successfully with error handling and graceful degradation")
    
    def _setup_circuit_breakers(self) -> None:
        """Set up circuit breakers for critical operations."""
        # Circuit breaker for query processing
        query_cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception,
            name="query_processing"
        )
        self.query_circuit_breaker = self.error_handler.create_circuit_breaker(
            "query_processing", query_cb_config
        )
        
        # Circuit breaker for adapter initialization
        adapter_cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception,
            name="adapter_initialization"
        )
        self.adapter_circuit_breaker = self.error_handler.create_circuit_breaker(
            "adapter_initialization", adapter_cb_config
        )
    
    def initialize_adapters_from_config(self, adapters_config: List[Dict[str, Any]]) -> List[SearchEngineAdapter]:
        """Initialize search engine adapters from configuration.
        
        Args:
            adapters_config: List of adapter configurations
            
        Returns:
            List of initialized adapter instances
        """
        try:
            adapters = self.adapter_registry.create_adapters_from_config(adapters_config)
            logger.info(f"Initialized {len(adapters)} adapters from configuration")
            return adapters
        except Exception as e:
            logger.error(f"Failed to initialize adapters from configuration: {e}")
            raise
    
    async def process_query(self, query: str, adapters: Optional[List[SearchEngineAdapter]] = None,
                          weights: Optional[np.ndarray] = None, limit: int = 10) -> RankedResults:
        """Process a query through the complete fusion pipeline with error handling and graceful degradation.
        
        Args:
            query: Search query string
            adapters: Optional list of adapters (uses all registered if None)
            weights: Optional fusion weights (uses equal weights if None)
            limit: Maximum number of results per engine
            
        Returns:
            RankedResults with fused and ranked results
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Use all registered adapters if none provided
        if adapters is None:
            adapters = list(self.adapter_registry.get_all_adapters().values())
        
        if not adapters:
            raise ValueError("No search engine adapters available")
        
        logger.info(f"Processing query: '{query}' with {len(adapters)} engines")
        
        try:
            # Use circuit breaker for query processing
            return await self.query_circuit_breaker.call(
                self._process_query_internal, query, adapters, weights, limit
            )
            
        except Exception as e:
            error_context = ErrorContext(
                component="FusionFactory",
                operation="process_query",
                query=query,
                additional_data={"num_adapters": len(adapters)}
            )
            self.error_handler.handle_error(e, error_context)
            
            # Try graceful degradation
            try:
                available_engines = [adapter.engine_id for adapter in adapters if adapter.is_healthy]
                return await self.degradation_manager.handle_degraded_query(
                    query, available_engines, fusion_engine=self.fusion_engine
                )
            except Exception as degradation_error:
                logger.error(f"Graceful degradation also failed: {degradation_error}")
                raise FusionFactoryError(
                    f"Query processing failed and graceful degradation unsuccessful: {str(e)}"
                ) from e
    
    async def _process_query_internal(self, query: str, adapters: List[SearchEngineAdapter],
                                    weights: Optional[np.ndarray], limit: int) -> RankedResults:
        """Internal query processing method with error handling.
        
        Args:
            query: Search query string
            adapters: List of search engine adapters
            weights: Optional fusion weights
            limit: Maximum number of results per engine
            
        Returns:
            RankedResults with fused and ranked results
        """
        # Step 1: Check if we should enter degraded mode
        healthy_engines = sum(1 for adapter in adapters if adapter.is_healthy)
        total_engines = len(adapters)
        
        if self.degradation_manager.should_degrade(total_engines, healthy_engines):
            logger.warning("Entering degraded mode for query processing")
            available_engines = [adapter.engine_id for adapter in adapters if adapter.is_healthy]
            return await self.degradation_manager.handle_degraded_query(
                query, available_engines, fusion_engine=self.fusion_engine
            )
        
        # Step 2: Dispatch query to all engines
        search_results = await self.query_dispatcher.dispatch_query(query, adapters, limit)
        
        if not search_results:
            logger.warning("No search results returned from any engine")
            # Try graceful degradation
            available_engines = [adapter.engine_id for adapter in adapters if adapter.is_healthy]
            return await self.degradation_manager.handle_degraded_query(
                query, available_engines, fusion_engine=self.fusion_engine
            )
        
        # Step 3: Check if we have sufficient results for normal processing
        total_results = sum(len(results.results) for results in search_results.values())
        if total_results < self.degradation_manager.config.min_results_threshold:
            logger.warning(f"Insufficient results ({total_results}), attempting graceful degradation")
            available_engines = list(search_results.keys())
            return await self.degradation_manager.handle_degraded_query(
                query, available_engines, partial_results=search_results, fusion_engine=self.fusion_engine
            )
        
        # Step 4: Normalize scores
        normalized_results = self.score_normalizer.normalize_scores(search_results)
        
        # Step 5: Fuse results
        fused_results = self.fusion_engine.fuse_results(normalized_results, weights)
        
        # Step 6: Cache successful results for future fallback
        available_engines = list(search_results.keys())
        self.degradation_manager.cache_successful_results(query, fused_results, available_engines)
        
        logger.info(f"Query processing completed: {fused_results.total_results} results")
        return fused_results
    
    async def health_check_engines(self) -> Dict[str, bool]:
        """Perform health checks on all registered engines.
        
        Returns:
            Dictionary mapping engine_id to health status
        """
        return await self.adapter_registry.health_check_all()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status information for all engines.
        
        Returns:
            Dictionary containing engine status information
        """
        return self.adapter_registry.get_registry_status()
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine types.
        
        Returns:
            List of available engine type names
        """
        return self.adapter_registry.get_available_adapters()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics and circuit breaker states.
        
        Returns:
            Dictionary containing error statistics
        """
        return {
            "error_handler_stats": self.error_handler.get_error_statistics(),
            "degradation_status": self.degradation_manager.get_degradation_status(),
            "query_dispatcher_circuit_breakers": {
                engine_id: cb.get_state()
                for engine_id, cb in self.query_dispatcher.engine_circuit_breakers.items()
            }
        }
    
    def reset_error_statistics(self) -> None:
        """Reset all error statistics and circuit breakers."""
        self.error_handler.reset_statistics()
        
        # Reset circuit breakers
        for cb in self.error_handler.circuit_breakers.values():
            cb.reset()
        
        logger.info("Error statistics and circuit breakers reset")
    
    def configure_degradation(self, config: DegradationConfig) -> None:
        """Update graceful degradation configuration.
        
        Args:
            config: New degradation configuration
        """
        self.degradation_manager = GracefulDegradationManager(config)
        logger.info("Graceful degradation configuration updated")
    
    async def test_engine_resilience(self, engine_id: str, num_failures: int = 3) -> Dict[str, Any]:
        """Test engine resilience by simulating failures.
        
        Args:
            engine_id: Engine to test
            num_failures: Number of failures to simulate
            
        Returns:
            Dictionary containing test results
        """
        adapter = self.adapter_registry.get_adapter(engine_id)
        if not adapter:
            return {"error": f"Engine {engine_id} not found"}
        
        circuit_breaker = self.query_dispatcher._get_engine_circuit_breaker(engine_id)
        initial_state = circuit_breaker.get_state()
        
        # Simulate failures
        for i in range(num_failures):
            try:
                await circuit_breaker.call(self._simulate_failure)
            except Exception:
                pass  # Expected to fail
        
        final_state = circuit_breaker.get_state()
        
        return {
            "engine_id": engine_id,
            "initial_state": initial_state,
            "final_state": final_state,
            "failures_simulated": num_failures,
            "circuit_breaker_opened": final_state["state"] == "open"
        }
    
    async def _simulate_failure(self) -> None:
        """Simulate a failure for testing purposes."""
        raise SearchEngineError("Simulated failure for testing")