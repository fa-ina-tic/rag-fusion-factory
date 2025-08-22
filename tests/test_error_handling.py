"""Tests for comprehensive error handling functionality."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.utils.error_handling import (
    ErrorHandler, ErrorContext, ErrorResponse, ErrorSeverity, ErrorCategory,
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, CircuitBreakerError,
    FusionFactoryError, get_error_handler
)
from src.utils.graceful_degradation import (
    GracefulDegradationManager, DegradationConfig, CachedResultsFallback,
    PartialResultsFallback, MinimalResultsFallback, get_degradation_manager
)
from src.models.core import SearchResult, RankedResults
from src.adapters.base import SearchEngineError, SearchEngineTimeoutError


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_error_classification_network_error(self):
        """Test classification of network errors."""
        error = ConnectionError("Network connection failed")
        context = ErrorContext(component="test", operation="test_op")
        
        response = self.error_handler.handle_error(error, context)
        
        assert response.category == ErrorCategory.NETWORK
        assert response.severity == ErrorSeverity.HIGH
        assert "NETWORK_CONNECTIONERROR" in response.error_code
        assert "Check network connectivity" in response.suggestions
    
    def test_error_classification_timeout_error(self):
        """Test classification of timeout errors."""
        error = SearchEngineTimeoutError("Operation timed out")
        context = ErrorContext(component="test", operation="test_op")
        
        response = self.error_handler.handle_error(error, context)
        
        assert response.category == ErrorCategory.TIMEOUT
        assert response.severity == ErrorSeverity.MEDIUM
        assert "TIMEOUT_SEARCHENGINETIMEOUTERROR" in response.error_code
        assert "Increase timeout configuration" in response.suggestions
    
    def test_error_classification_validation_error(self):
        """Test classification of validation errors."""
        error = ValueError("Invalid input data")
        context = ErrorContext(component="test", operation="test_op")
        
        response = self.error_handler.handle_error(error, context)
        
        assert response.category == ErrorCategory.VALIDATION
        assert response.severity == ErrorSeverity.LOW
        assert "VALIDATION_VALUEERROR" in response.error_code
        assert any("Check input data format" in suggestion for suggestion in response.suggestions)
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        error = ValueError("Test error")
        context = ErrorContext(component="test_component", operation="test_op")
        
        # Handle same error multiple times
        for _ in range(3):
            self.error_handler.handle_error(error, context)
        
        stats = self.error_handler.get_error_statistics()
        assert stats["total_errors"] == 3
        assert "test_component:VALIDATION_VALUEERROR" in stats["error_counts_by_type"]
        assert stats["error_counts_by_type"]["test_component:VALIDATION_VALUEERROR"] == 3
    
    def test_error_response_serialization(self):
        """Test error response serialization to dictionary."""
        error = ValueError("Test error")
        context = ErrorContext(
            component="test", 
            operation="test_op",
            engine_id="test_engine",
            query="test query"
        )
        
        response = self.error_handler.handle_error(error, context)
        response_dict = response.to_dict()
        
        assert isinstance(response_dict, dict)
        assert response_dict["error_code"] == response.error_code
        assert response_dict["message"] == response.message
        assert response_dict["severity"] == response.severity.value
        assert response_dict["category"] == response.category.value
        assert response_dict["context"]["component"] == "test"
        assert response_dict["context"]["engine_id"] == "test_engine"
    
    def test_reset_statistics(self):
        """Test resetting error statistics."""
        error = ValueError("Test error")
        context = ErrorContext(component="test", operation="test_op")
        
        self.error_handler.handle_error(error, context)
        assert self.error_handler.get_error_statistics()["total_errors"] == 1
        
        self.error_handler.reset_statistics()
        assert self.error_handler.get_error_statistics()["total_errors"] == 0


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for testing
            expected_exception=Exception,
            name="test_breaker"
        )
        self.circuit_breaker = CircuitBreaker(self.config)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful operations."""
        async def successful_operation():
            return "success"
        
        result = await self.circuit_breaker.call(successful_operation)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert self.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        async def failing_operation():
            raise Exception("Test failure")
        
        # Fail below threshold
        for _ in range(2):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Fail at threshold - should open circuit
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        assert self.circuit_breaker.failure_count == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker behavior when open."""
        async def failing_operation():
            raise Exception("Test failure")
        
        # Open the circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Should raise CircuitBreakerError when open
        with pytest.raises(CircuitBreakerError):
            await self.circuit_breaker.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        async def failing_operation():
            raise Exception("Test failure")
        
        async def successful_operation():
            return "success"
        
        # Open the circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(failing_operation)
        
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should transition to HALF_OPEN and then CLOSED on success
        result = await self.circuit_breaker.call(successful_operation)
        assert result == "success"
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert self.circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_state_info(self):
        """Test circuit breaker state information."""
        state_info = self.circuit_breaker.get_state()
        
        assert state_info["name"] == "test_breaker"
        assert state_info["state"] == "closed"
        assert state_info["failure_count"] == 0
        assert state_info["failure_threshold"] == 3
        assert state_info["time_until_retry"] == 0
    
    def test_circuit_breaker_manual_reset(self):
        """Test manual circuit breaker reset."""
        # Simulate failures
        self.circuit_breaker.failure_count = 5
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.last_failure_time = datetime.now()
        
        self.circuit_breaker.reset()
        
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.last_failure_time is None


class TestGracefulDegradation:
    """Test cases for graceful degradation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DegradationConfig(
            min_engines_required=2,
            min_results_threshold=5,
            enable_fallback_results=True,
            fallback_timeout=5.0,
            quality_threshold=0.3,
            enable_cached_results=True,
            cache_expiry_minutes=30
        )
        self.degradation_manager = GracefulDegradationManager(self.config)
    
    def test_should_degrade_insufficient_engines(self):
        """Test degradation decision with insufficient engines."""
        should_degrade = self.degradation_manager.should_degrade(
            total_engines=5, healthy_engines=1
        )
        assert should_degrade is True
    
    def test_should_degrade_low_availability(self):
        """Test degradation decision with low engine availability."""
        should_degrade = self.degradation_manager.should_degrade(
            total_engines=10, healthy_engines=4  # 40% availability
        )
        assert should_degrade is True
    
    def test_should_degrade_poor_quality(self):
        """Test degradation decision with poor results quality."""
        should_degrade = self.degradation_manager.should_degrade(
            total_engines=5, healthy_engines=5, results_quality=0.2
        )
        assert should_degrade is True
    
    def test_should_not_degrade_healthy_system(self):
        """Test no degradation with healthy system."""
        should_degrade = self.degradation_manager.should_degrade(
            total_engines=5, healthy_engines=5, results_quality=0.8
        )
        assert should_degrade is False
    
    @pytest.mark.asyncio
    async def test_cached_results_fallback(self):
        """Test cached results fallback strategy."""
        fallback = CachedResultsFallback()
        
        # Create test results to cache
        test_results = RankedResults(
            query="test query",
            results=[
                SearchResult(
                    document_id="doc1",
                    relevance_score=0.9,
                    content="Test content",
                    metadata={},
                    engine_source="test"
                )
            ],
            fusion_weights=[1.0],
            confidence_scores=[0.9],
            timestamp=datetime.now(),
            total_results=1
        )
        
        context = {
            'query': 'test query',
            'available_engines': ['engine1'],
            'cache_expiry_minutes': 30
        }
        
        # Cache results
        fallback.cache_results("test query", test_results, context)
        
        # Test fallback execution
        result = await fallback.execute(context)
        assert result is not None
        assert result.query == "test query"
        assert len(result.results) == 1
    
    @pytest.mark.asyncio
    async def test_minimal_results_fallback(self):
        """Test minimal results fallback strategy."""
        fallback = MinimalResultsFallback()
        
        context = {'query': 'test query'}
        
        result = await fallback.execute(context)
        assert result is not None
        assert result.query == "test query"
        assert len(result.results) == 1
        assert result.results[0].metadata.get("fallback") is True
    
    @pytest.mark.asyncio
    async def test_partial_results_fallback(self):
        """Test partial results fallback strategy."""
        fallback = PartialResultsFallback()
        
        # Mock fusion engine
        mock_fusion_engine = Mock()
        mock_fusion_engine.fuse_results.return_value = RankedResults(
            query="test query",
            results=[
                SearchResult(
                    document_id="doc1",
                    relevance_score=0.8,
                    content="Partial result",
                    metadata={},
                    engine_source="engine1"
                )
            ] * 6,  # Ensure we meet min_results_threshold
            fusion_weights=[1.0],
            confidence_scores=[0.8] * 6,
            timestamp=datetime.now(),
            total_results=6
        )
        
        context = {
            'partial_results': {'engine1': Mock()},
            'min_engines_required': 1,
            'min_results_threshold': 5,
            'fusion_engine': mock_fusion_engine
        }
        
        result = await fallback.execute(context)
        assert result is not None
        assert result.total_results == 6
    
    @pytest.mark.asyncio
    async def test_handle_degraded_query_with_cache(self):
        """Test handling degraded query with cached results."""
        # First, cache some results
        cached_fallback = None
        for strategy in self.degradation_manager.fallback_strategies:
            if isinstance(strategy, CachedResultsFallback):
                cached_fallback = strategy
                break
        
        assert cached_fallback is not None
        
        test_results = RankedResults(
            query="cached query",
            results=[
                SearchResult(
                    document_id="cached_doc",
                    relevance_score=0.9,
                    content="Cached content",
                    metadata={},
                    engine_source="cache"
                )
            ],
            fusion_weights=[1.0],
            confidence_scores=[0.9],
            timestamp=datetime.now(),
            total_results=1
        )
        
        context = {
            'available_engines': ['engine1'],
            'cache_expiry_minutes': 30
        }
        cached_fallback.cache_results("cached query", test_results, context)
        
        # Now test degraded query handling
        result = await self.degradation_manager.handle_degraded_query(
            "cached query", ['engine1']
        )
        
        assert result is not None
        assert result.query == "cached query"
        assert len(result.results) == 1
    
    def test_add_remove_fallback_strategies(self):
        """Test adding and removing custom fallback strategies."""
        class CustomFallback(CachedResultsFallback):
            async def execute(self, context):
                return None
            
            def is_applicable(self, context):
                return True
        
        custom_strategy = CustomFallback()
        initial_count = len(self.degradation_manager.fallback_strategies)
        
        # Add custom strategy
        self.degradation_manager.add_fallback_strategy(custom_strategy)
        assert len(self.degradation_manager.fallback_strategies) == initial_count + 1
        
        # Remove custom strategy
        removed = self.degradation_manager.remove_fallback_strategy(CustomFallback)
        assert removed is True
        assert len(self.degradation_manager.fallback_strategies) == initial_count
        
        # Try to remove MinimalResultsFallback (should fail)
        removed = self.degradation_manager.remove_fallback_strategy(MinimalResultsFallback)
        assert removed is False


class TestIntegratedErrorHandling:
    """Integration tests for error handling with other components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.degradation_config = DegradationConfig()
        self.degradation_manager = GracefulDegradationManager(self.degradation_config)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error handler."""
        cb_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            name="integration_test"
        )
        
        circuit_breaker = self.error_handler.create_circuit_breaker(
            "integration_test", cb_config
        )
        
        async def failing_function():
            raise SearchEngineError("Integration test failure")
        
        # Trigger failures to open circuit breaker
        for _ in range(2):
            with pytest.raises(SearchEngineError):
                await circuit_breaker.call(failing_function)
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(failing_function)
        
        # Check error statistics
        stats = self.error_handler.get_error_statistics()
        assert "integration_test" in stats["circuit_breaker_states"]
        assert stats["circuit_breaker_states"]["integration_test"]["state"] == "open"
    
    def test_error_context_with_degradation(self):
        """Test error context integration with degradation manager."""
        error = SearchEngineTimeoutError("Test timeout")
        context = ErrorContext(
            component="FusionFactory",
            operation="process_query",
            engine_id="test_engine",
            query="test query"
        )
        
        error_response = self.error_handler.handle_error(error, context)
        
        # Verify error response contains degradation-relevant information
        assert error_response.category == ErrorCategory.TIMEOUT
        assert error_response.context.engine_id == "test_engine"
        assert error_response.context.query == "test query"
        
        # Check if degradation should be triggered
        should_degrade = self.degradation_manager.should_degrade(
            total_engines=3, healthy_engines=1
        )
        assert should_degrade is True


if __name__ == "__main__":
    pytest.main([__file__])