"""Graceful degradation and fallback mechanisms for RAG Fusion Factory."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from ..models.core import SearchResults, RankedResults, SearchResult
from .error_handling import ErrorHandler, ErrorContext, ErrorCategory, get_error_handler

logger = logging.getLogger(__name__)


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    min_engines_required: int = 1
    min_results_threshold: int = 5
    enable_fallback_results: bool = True
    fallback_timeout: float = 10.0
    quality_threshold: float = 0.3
    enable_cached_results: bool = True
    cache_expiry_minutes: int = 30


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute fallback strategy.
        
        Args:
            context: Context information for fallback execution
            
        Returns:
            Fallback result
        """
        pass
    
    @abstractmethod
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this fallback strategy is applicable.
        
        Args:
            context: Context information
            
        Returns:
            True if strategy is applicable, False otherwise
        """
        pass


class CachedResultsFallback(FallbackStrategy):
    """Fallback strategy using cached results."""
    
    def __init__(self, cache_manager: Optional[Any] = None):
        """Initialize cached results fallback.
        
        Args:
            cache_manager: Optional cache manager instance
        """
        self.cache_manager = cache_manager
        self.cache: Dict[str, tuple[RankedResults, datetime]] = {}
    
    async def execute(self, context: Dict[str, Any]) -> Optional[RankedResults]:
        """Execute cached results fallback.
        
        Args:
            context: Context containing query and other information
            
        Returns:
            Cached results if available, None otherwise
        """
        query = context.get('query', '')
        if not query:
            return None
        
        # Check internal cache first
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.cache:
            cached_results, timestamp = self.cache[cache_key]
            
            # Check if cache is still valid
            cache_age = (datetime.now() - timestamp).total_seconds() / 60
            if cache_age <= context.get('cache_expiry_minutes', 30):
                logger.info(f"Using cached results for query: {query}")
                return cached_results
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if cached results fallback is applicable."""
        return context.get('enable_cached_results', True) and bool(context.get('query'))
    
    def cache_results(self, query: str, results: RankedResults, context: Dict[str, Any]) -> None:
        """Cache results for future fallback use.
        
        Args:
            query: Search query
            results: Results to cache
            context: Context information
        """
        cache_key = self._generate_cache_key(query, context)
        self.cache[cache_key] = (results, datetime.now())
        logger.debug(f"Cached results for query: {query}")
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query and context.
        
        Args:
            query: Search query
            context: Context information
            
        Returns:
            Cache key string
        """
        # Simple cache key based on query and engine configuration
        engines = context.get('available_engines', [])
        engine_key = '_'.join(sorted(engines)) if engines else 'no_engines'
        return f"{query}:{engine_key}"


class MinimalResultsFallback(FallbackStrategy):
    """Fallback strategy providing minimal default results."""
    
    async def execute(self, context: Dict[str, Any]) -> RankedResults:
        """Execute minimal results fallback.
        
        Args:
            context: Context containing query and other information
            
        Returns:
            Minimal default results
        """
        query = context.get('query', '')
        
        # Create minimal fallback results
        fallback_results = [
            SearchResult(
                document_id="fallback_1",
                relevance_score=0.5,
                content=f"No results found for query: {query}",
                metadata={"fallback": True, "reason": "service_unavailable"},
                engine_source="fallback"
            )
        ]
        
        logger.warning(f"Using minimal fallback results for query: {query}")
        
        return RankedResults(
            query=query,
            results=fallback_results,
            fusion_weights=np.array([1.0]),
            confidence_scores=[0.1],
            timestamp=datetime.now(),
            total_results=1
        )
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Minimal results fallback is always applicable as last resort."""
        return True


class PartialResultsFallback(FallbackStrategy):
    """Fallback strategy using partial results from available engines."""
    
    async def execute(self, context: Dict[str, Any]) -> Optional[RankedResults]:
        """Execute partial results fallback.
        
        Args:
            context: Context containing partial results
            
        Returns:
            Fused partial results if sufficient, None otherwise
        """
        partial_results = context.get('partial_results', {})
        min_engines = context.get('min_engines_required', 1)
        
        if len(partial_results) < min_engines:
            return None
        
        # Use fusion engine to combine partial results
        fusion_engine = context.get('fusion_engine')
        if not fusion_engine:
            return None
        
        try:
            # Create equal weights for available engines
            num_engines = len(partial_results)
            weights = np.ones(num_engines) / num_engines
            
            fused_results = fusion_engine.fuse_results(partial_results, weights)
            
            # Check if results meet quality threshold
            if fused_results.total_results >= context.get('min_results_threshold', 5):
                logger.info(f"Using partial results from {num_engines} engines")
                return fused_results
            
        except Exception as e:
            logger.error(f"Failed to fuse partial results: {e}")
        
        return None
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if partial results fallback is applicable."""
        partial_results = context.get('partial_results', {})
        min_engines = context.get('min_engines_required', 1)
        return len(partial_results) >= min_engines


class GracefulDegradationManager:
    """Manager for graceful degradation and fallback mechanisms."""
    
    def __init__(self, config: DegradationConfig):
        """Initialize graceful degradation manager.
        
        Args:
            config: Degradation configuration
        """
        self.config = config
        self.error_handler = get_error_handler()
        
        # Initialize fallback strategies in order of preference
        self.fallback_strategies: List[FallbackStrategy] = [
            CachedResultsFallback(),
            PartialResultsFallback(),
            MinimalResultsFallback()  # Always last as final fallback
        ]
        
        logger.info("Graceful degradation manager initialized")
    
    async def handle_degraded_query(self, query: str, available_engines: List[str],
                                  partial_results: Dict[str, Any] = None,
                                  fusion_engine: Any = None) -> RankedResults:
        """Handle query processing under degraded conditions.
        
        Args:
            query: Search query
            available_engines: List of available engine IDs
            partial_results: Partial results from available engines
            fusion_engine: Fusion engine instance for combining results
            
        Returns:
            Best available results using fallback strategies
        """
        context = {
            'query': query,
            'available_engines': available_engines,
            'partial_results': partial_results or {},
            'fusion_engine': fusion_engine,
            'min_engines_required': self.config.min_engines_required,
            'min_results_threshold': self.config.min_results_threshold,
            'enable_cached_results': self.config.enable_cached_results,
            'cache_expiry_minutes': self.config.cache_expiry_minutes
        }
        
        logger.info(f"Handling degraded query with {len(available_engines)} available engines")
        
        # Try fallback strategies in order
        for strategy in self.fallback_strategies:
            if strategy.is_applicable(context):
                try:
                    result = await asyncio.wait_for(
                        strategy.execute(context),
                        timeout=self.config.fallback_timeout
                    )
                    
                    if result is not None:
                        logger.info(f"Fallback successful using {strategy.__class__.__name__}")
                        return result
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Fallback strategy {strategy.__class__.__name__} timed out")
                    continue
                except Exception as e:
                    error_context = ErrorContext(
                        component="GracefulDegradationManager",
                        operation="fallback_execution",
                        query=query,
                        additional_data={"strategy": strategy.__class__.__name__}
                    )
                    self.error_handler.handle_error(e, error_context)
                    continue
        
        # If all strategies fail, raise an error
        raise RuntimeError("All fallback strategies failed")
    
    def should_degrade(self, total_engines: int, healthy_engines: int,
                      results_quality: float = None) -> bool:
        """Determine if system should enter degraded mode.
        
        Args:
            total_engines: Total number of configured engines
            healthy_engines: Number of healthy engines
            results_quality: Optional quality score of results
            
        Returns:
            True if system should degrade, False otherwise
        """
        # Check if minimum engines are available
        if healthy_engines < self.config.min_engines_required:
            logger.warning(f"Degrading: only {healthy_engines} engines available, minimum {self.config.min_engines_required}")
            return True
        
        # Check engine availability ratio
        availability_ratio = healthy_engines / total_engines if total_engines > 0 else 0
        if availability_ratio < 0.5:  # Less than 50% engines available
            logger.warning(f"Degrading: only {availability_ratio:.1%} engines available")
            return True
        
        # Check results quality if provided
        if results_quality is not None and results_quality < self.config.quality_threshold:
            logger.warning(f"Degrading: results quality {results_quality:.2f} below threshold {self.config.quality_threshold}")
            return True
        
        return False
    
    def cache_successful_results(self, query: str, results: RankedResults,
                               available_engines: List[str]) -> None:
        """Cache successful results for future fallback use.
        
        Args:
            query: Search query
            results: Successful results to cache
            available_engines: List of engines that provided results
        """
        if not self.config.enable_cached_results:
            return
        
        # Find cached results fallback strategy
        for strategy in self.fallback_strategies:
            if isinstance(strategy, CachedResultsFallback):
                context = {
                    'available_engines': available_engines,
                    'cache_expiry_minutes': self.config.cache_expiry_minutes
                }
                strategy.cache_results(query, results, context)
                break
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status and statistics.
        
        Returns:
            Dictionary containing degradation status information
        """
        return {
            "config": {
                "min_engines_required": self.config.min_engines_required,
                "min_results_threshold": self.config.min_results_threshold,
                "enable_fallback_results": self.config.enable_fallback_results,
                "fallback_timeout": self.config.fallback_timeout,
                "quality_threshold": self.config.quality_threshold,
                "enable_cached_results": self.config.enable_cached_results,
                "cache_expiry_minutes": self.config.cache_expiry_minutes
            },
            "fallback_strategies": [
                {
                    "name": strategy.__class__.__name__,
                    "type": type(strategy).__name__
                }
                for strategy in self.fallback_strategies
            ]
        }
    
    def add_fallback_strategy(self, strategy: FallbackStrategy, position: int = -1) -> None:
        """Add a custom fallback strategy.
        
        Args:
            strategy: Fallback strategy to add
            position: Position to insert strategy (-1 for before last)
        """
        if position == -1:
            # Insert before the last strategy (MinimalResultsFallback)
            self.fallback_strategies.insert(-1, strategy)
        else:
            self.fallback_strategies.insert(position, strategy)
        
        logger.info(f"Added fallback strategy: {strategy.__class__.__name__}")
    
    def remove_fallback_strategy(self, strategy_class: type) -> bool:
        """Remove a fallback strategy by class type.
        
        Args:
            strategy_class: Class type of strategy to remove
            
        Returns:
            True if strategy was removed, False if not found
        """
        for i, strategy in enumerate(self.fallback_strategies):
            if isinstance(strategy, strategy_class):
                # Don't allow removal of MinimalResultsFallback (last resort)
                if not isinstance(strategy, MinimalResultsFallback):
                    del self.fallback_strategies[i]
                    logger.info(f"Removed fallback strategy: {strategy_class.__name__}")
                    return True
                else:
                    logger.warning("Cannot remove MinimalResultsFallback strategy")
                    return False
        
        return False


# Global degradation manager instance
degradation_manager: Optional[GracefulDegradationManager] = None


def get_degradation_manager(config: Optional[DegradationConfig] = None) -> GracefulDegradationManager:
    """Get or create the global degradation manager instance.
    
    Args:
        config: Optional configuration for new instance
        
    Returns:
        Global GracefulDegradationManager instance
    """
    global degradation_manager
    
    if degradation_manager is None:
        if config is None:
            config = DegradationConfig()
        degradation_manager = GracefulDegradationManager(config)
    
    return degradation_manager