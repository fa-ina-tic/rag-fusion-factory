"""Comprehensive error handling utilities for RAG Fusion Factory."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    INTERNAL = "internal"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    engine_id: Optional[str] = None
    query: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorResponse:
    """Standardized error response model."""
    error_code: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    timestamp: datetime = field(default_factory=datetime.now)
    retry_after: Optional[int] = None
    suggestions: List[str] = field(default_factory=list)
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error response to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "retry_after": self.retry_after,
            "suggestions": self.suggestions,
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
                "engine_id": self.context.engine_id,
                "query": self.context.query,
                "timestamp": self.context.timestamp.isoformat(),
                "additional_data": self.context.additional_data
            },
            "details": self.details
        }


class FusionFactoryError(Exception):
    """Base exception for RAG Fusion Factory errors."""
    
    def __init__(self, message: str, error_response: Optional[ErrorResponse] = None):
        super().__init__(message)
        self.error_response = error_response
        self.message = message


class CircuitBreakerError(FusionFactoryError):
    """Exception raised when circuit breaker is open."""
    pass


class GracefulDegradationError(FusionFactoryError):
    """Exception raised during graceful degradation scenarios."""
    pass


class FallbackError(FusionFactoryError):
    """Exception raised when fallback mechanisms fail."""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: Type[Exception] = Exception
    name: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        self.name = config.name or "unnamed"
        
        logger.info(f"Circuit breaker '{self.name}' initialized with threshold {config.failure_threshold}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next retry in {self._time_until_retry()} seconds."
                )
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _time_until_retry(self) -> int:
        """Calculate seconds until next retry attempt."""
        if self.last_failure_time is None:
            return 0
        
        time_since_failure = datetime.now() - self.last_failure_time
        remaining = self.config.recovery_timeout - time_since_failure.total_seconds()
        return max(0, int(remaining))
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
        
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.config.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "time_until_retry": self._time_until_retry() if self.state == CircuitBreakerState.OPEN else 0
        }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class ErrorHandler:
    """Centralized error handler for standardized error processing."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register a circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
            
        Returns:
            Created circuit breaker instance
        """
        config.name = name
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker instance if found, None otherwise
        """
        return self.circuit_breakers.get(name)
    
    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorResponse:
        """Handle and classify an error.
        
        Args:
            error: Exception that occurred
            context: Error context information
            
        Returns:
            Standardized error response
        """
        # Classify error
        error_code, category, severity = self._classify_error(error)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(error, category)
        
        # Create error response
        error_response = ErrorResponse(
            error_code=error_code,
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            suggestions=suggestions,
            details=self._extract_error_details(error)
        )
        
        # Track error for monitoring
        self._track_error(error_code, context)
        
        # Log error
        self._log_error(error_response, error)
        
        return error_response
    
    def _classify_error(self, error: Exception) -> tuple[str, ErrorCategory, ErrorSeverity]:
        """Classify error and determine code, category, and severity.
        
        Args:
            error: Exception to classify
            
        Returns:
            Tuple of (error_code, category, severity)
        """
        error_type = type(error).__name__
        
        # Network and connection errors
        if "Connection" in error_type or "Network" in error_type:
            return f"NETWORK_{error_type.upper()}", ErrorCategory.NETWORK, ErrorSeverity.HIGH
        
        # Timeout errors
        if "Timeout" in error_type or "timeout" in str(error).lower():
            return f"TIMEOUT_{error_type.upper()}", ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        
        # Authentication/Authorization errors
        if "Auth" in error_type or "Permission" in error_type:
            return f"AUTH_{error_type.upper()}", ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        
        # Validation errors
        if "Validation" in error_type or "ValueError" in error_type:
            return f"VALIDATION_{error_type.upper()}", ErrorCategory.VALIDATION, ErrorSeverity.LOW
        
        # Configuration errors
        if "Config" in error_type or "Setting" in error_type:
            return f"CONFIG_{error_type.upper()}", ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM
        
        # Resource errors
        if "Memory" in error_type or "Resource" in error_type:
            return f"RESOURCE_{error_type.upper()}", ErrorCategory.RESOURCE, ErrorSeverity.HIGH
        
        # Circuit breaker errors
        if isinstance(error, CircuitBreakerError):
            return "CIRCUIT_BREAKER_OPEN", ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM
        
        # Default classification
        return f"INTERNAL_{error_type.upper()}", ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM
    
    def _generate_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate helpful suggestions based on error type.
        
        Args:
            error: Exception that occurred
            category: Error category
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Consider increasing timeout values"
            ])
        elif category == ErrorCategory.TIMEOUT:
            suggestions.extend([
                "Increase timeout configuration",
                "Check service response times",
                "Consider implementing retry logic"
            ])
        elif category == ErrorCategory.AUTHENTICATION:
            suggestions.extend([
                "Verify authentication credentials",
                "Check API keys and tokens",
                "Ensure proper permissions are configured"
            ])
        elif category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Check input data format and values",
                "Verify required fields are provided",
                "Review data validation rules"
            ])
        elif category == ErrorCategory.CONFIGURATION:
            suggestions.extend([
                "Review configuration files",
                "Check environment variables",
                "Verify service settings"
            ])
        elif category == ErrorCategory.RESOURCE:
            suggestions.extend([
                "Check available memory and CPU",
                "Monitor system resources",
                "Consider scaling resources"
            ])
        
        return suggestions
    
    def _extract_error_details(self, error: Exception) -> Dict[str, Any]:
        """Extract detailed information from error.
        
        Args:
            error: Exception to extract details from
            
        Returns:
            Dictionary containing error details
        """
        details = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
        
        # Add specific details for known error types
        if hasattr(error, '__dict__'):
            for key, value in error.__dict__.items():
                if not key.startswith('_') and key not in ['args']:
                    try:
                        # Only include serializable values
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            details[key] = value
                    except Exception:
                        continue
        
        return details
    
    def _track_error(self, error_code: str, context: ErrorContext) -> None:
        """Track error occurrence for monitoring.
        
        Args:
            error_code: Error code to track
            context: Error context
        """
        tracking_key = f"{context.component}:{error_code}"
        self.error_counts[tracking_key] = self.error_counts.get(tracking_key, 0) + 1
    
    def _log_error(self, error_response: ErrorResponse, original_error: Exception) -> None:
        """Log error with appropriate level based on severity.
        
        Args:
            error_response: Standardized error response
            original_error: Original exception
        """
        log_message = (
            f"Error in {error_response.context.component}.{error_response.context.operation}: "
            f"{error_response.message}"
        )
        
        if error_response.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=original_error)
        elif error_response.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=original_error)
        elif error_response.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring.
        
        Returns:
            Dictionary containing error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_counts_by_type": dict(self.error_counts),
            "circuit_breaker_states": {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()
        logger.info("Error statistics reset")


# Global error handler instance
error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance.
    
    Returns:
        Global ErrorHandler instance
    """
    return error_handler