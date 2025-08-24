# Custom Search Engine Adapter Development Guide

This guide explains how to create custom search engine adapters for the RAG Fusion Factory system. Custom adapters allow you to integrate any search engine or data source into the fusion pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Base Adapter Interface](#base-adapter-interface)
3. [Implementation Steps](#implementation-steps)
4. [Example Implementation](#example-implementation)
5. [Testing Your Adapter](#testing-your-adapter)
6. [Registration and Configuration](#registration-and-configuration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The RAG Fusion Factory uses a plugin-based architecture for search engines. All adapters must inherit from the `SearchEngineAdapter` base class and implement the required abstract methods. This ensures consistent behavior and integration with the fusion pipeline.

### Key Concepts

- **Adapter**: A class that provides standardized access to a search engine
- **Engine ID**: Unique identifier for a specific adapter instance
- **Configuration**: Dictionary containing engine-specific settings
- **Normalization**: Converting engine-specific results to standard format
- **Health Checks**: Monitoring adapter availability and performance

## Base Adapter Interface

All custom adapters must inherit from `SearchEngineAdapter` and implement these abstract methods:

```python
from src.adapters.base import SearchEngineAdapter
from src.models.core import SearchResults

class CustomAdapter(SearchEngineAdapter):
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query and return standardized results."""
        pass
    
    async def health_check(self) -> bool:
        """Check if the search engine is healthy and responsive."""
        pass
    
    def get_configuration(self) -> Dict[str, Any]:
        """Return the current adapter configuration."""
        pass
```

### Required Methods

#### `search(query, limit, **kwargs)`
- **Purpose**: Execute search queries against your engine
- **Parameters**:
  - `query`: Search query string
  - `limit`: Maximum number of results to return
  - `**kwargs`: Additional engine-specific parameters
- **Returns**: `SearchResults` object with normalized results
- **Raises**: `SearchEngineError` for search failures

#### `health_check()`
- **Purpose**: Verify engine availability and health
- **Returns**: `True` if healthy, `False` otherwise
- **Should be**: Fast and lightweight (< 5 seconds)

#### `get_configuration()`
- **Purpose**: Return current adapter configuration
- **Returns**: Dictionary with configuration details
- **Used for**: Debugging, monitoring, and documentation

## Implementation Steps

### Step 1: Create Your Adapter Class

```python
import asyncio
import logging
from typing import Any, Dict, List, Tuple

from src.adapters.base import SearchEngineAdapter, SearchEngineError
from src.models.core import SearchResults

logger = logging.getLogger(__name__)

class MyCustomAdapter(SearchEngineAdapter):
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        super().__init__(engine_id, config, timeout)
        
        # Extract your configuration
        self.api_url = config.get('api_url')
        self.api_key = config.get('api_key')
        
        # Validate required configuration
        if not self.api_url:
            raise ValueError("api_url is required")
```

### Step 2: Implement the Search Method

```python
async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
    try:
        # 1. Build your search request
        search_params = self._build_search_params(query, limit, **kwargs)
        
        # 2. Execute the search (with timeout handling)
        raw_results = await self._execute_search(search_params)
        
        # 3. Parse and normalize results
        normalized_results = self._parse_results(raw_results)
        
        # 4. Create SearchResults object
        return self._create_search_results(query, normalized_results)
        
    except Exception as e:
        raise SearchEngineError(f"Search failed: {str(e)}") from e

def _build_search_params(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
    """Build engine-specific search parameters."""
    return {
        'q': query,
        'limit': limit,
        'api_key': self.api_key
    }

async def _execute_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the actual search request."""
    # Implement your search logic here
    # This could be HTTP requests, database queries, file operations, etc.
    pass

def _parse_results(self, raw_results: Dict[str, Any]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
    """Parse raw results into standardized format."""
    results = []
    
    for item in raw_results.get('items', []):
        doc_id = item.get('id')
        score = item.get('score', 0.0)
        content = item.get('content', '')
        metadata = {
            'source': 'my_custom_engine',
            'raw_data': item
        }
        
        results.append((doc_id, score, content, metadata))
    
    return results
```

### Step 3: Implement Health Check

```python
async def health_check(self) -> bool:
    try:
        # Implement a lightweight check
        # Examples: ping endpoint, check connection, validate credentials
        response = await self._ping_engine()
        return response.get('status') == 'ok'
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

async def _ping_engine(self) -> Dict[str, Any]:
    """Ping your search engine to check availability."""
    # Implement engine-specific health check
    pass
```

### Step 4: Implement Configuration Method

```python
def get_configuration(self) -> Dict[str, Any]:
    return {
        'engine_type': 'my_custom_engine',
        'engine_id': self.engine_id,
        'api_url': self.api_url,
        'timeout': self.timeout,
        'has_api_key': bool(self.api_key)
    }
```

## Example Implementation

Here's a complete example of a simple HTTP API adapter:

```python
import aiohttp
import asyncio
import json
from typing import Any, Dict, List, Tuple

from src.adapters.base import SearchEngineAdapter, SearchEngineError, SearchEngineConnectionError
from src.models.core import SearchResults

class HttpApiAdapter(SearchEngineAdapter):
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        super().__init__(engine_id, config, timeout)
        
        self.base_url = config.get('base_url')
        self.api_key = config.get('api_key')
        self.search_endpoint = config.get('search_endpoint', '/search')
        
        if not self.base_url:
            raise ValueError("base_url is required")
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        try:
            url = f"{self.base_url}{self.search_endpoint}"
            params = {
                'q': query,
                'limit': limit
            }
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        raise SearchEngineError(f"API returned status {response.status}")
                    
                    data = await response.json()
            
            # Parse results
            raw_results = []
            for item in data.get('results', []):
                doc_id = item.get('id', '')
                score = float(item.get('score', 0.0))
                content = item.get('text', '')
                metadata = {'api_data': item}
                
                raw_results.append((doc_id, score, content, metadata))
            
            # Normalize and create SearchResults
            normalized_results = self._normalize_result_format(raw_results)
            return self._create_search_results(query, normalized_results)
            
        except aiohttp.ClientError as e:
            raise SearchEngineConnectionError(f"Connection failed: {str(e)}") from e
        except Exception as e:
            raise SearchEngineError(f"Search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        return {
            'engine_type': 'http_api',
            'engine_id': self.engine_id,
            'base_url': self.base_url,
            'search_endpoint': self.search_endpoint,
            'has_api_key': bool(self.api_key),
            'timeout': self.timeout
        }
```

## Testing Your Adapter

### Unit Tests

Create unit tests for your adapter:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from src.adapters.my_custom_adapter import MyCustomAdapter

@pytest.fixture
def adapter():
    config = {
        'api_url': 'https://api.example.com',
        'api_key': 'test-key'
    }
    return MyCustomAdapter('test_engine', config)

@pytest.mark.asyncio
async def test_search_success(adapter):
    # Mock the search execution
    with patch.object(adapter, '_execute_search') as mock_search:
        mock_search.return_value = {
            'items': [
                {'id': 'doc1', 'score': 0.9, 'content': 'Test content'}
            ]
        }
        
        results = await adapter.search('test query')
        
        assert len(results.results) == 1
        assert results.results[0].document_id == 'doc1'
        assert results.results[0].relevance_score == 0.9

@pytest.mark.asyncio
async def test_health_check(adapter):
    with patch.object(adapter, '_ping_engine') as mock_ping:
        mock_ping.return_value = {'status': 'ok'}
        
        is_healthy = await adapter.health_check()
        assert is_healthy is True
```

### Integration Tests

Test with real data:

```python
@pytest.mark.asyncio
async def test_real_search():
    config = {
        'api_url': 'https://your-test-api.com',
        'api_key': 'your-test-key'
    }
    adapter = MyCustomAdapter('integration_test', config)
    
    results = await adapter.search('machine learning', limit=5)
    
    assert len(results.results) <= 5
    assert all(result.relevance_score >= 0 for result in results.results)
```

## Registration and Configuration

### Register Your Adapter

Add your adapter to the registry:

```python
# In src/adapters/registry.py
from .my_custom_adapter import MyCustomAdapter

class AdapterRegistry:
    def _register_builtin_adapters(self) -> None:
        # ... existing registrations
        self.register_adapter('my_custom_engine', MyCustomAdapter)
```

### Configuration Example

Create a configuration file:

```yaml
# config/my_engine.yaml
my_custom_engine_example:
  engine_type: my_custom_engine
  engine_id: custom_primary
  timeout: 30.0
  config:
    api_url: "https://api.myengine.com"
    api_key: "${MY_ENGINE_API_KEY}"
    search_endpoint: "/v1/search"
    max_results: 20
```

### Usage Example

```python
from src.adapters.registry import get_adapter_registry

registry = get_adapter_registry()

# Create adapter instance
adapter = registry.create_adapter(
    engine_type='my_custom_engine',
    engine_id='my_instance',
    config={
        'api_url': 'https://api.example.com',
        'api_key': 'your-key'
    }
)

# Use the adapter
results = await adapter.search('test query')
```

## Best Practices

### Error Handling

1. **Use Specific Exceptions**: Raise `SearchEngineError` for search failures, `SearchEngineConnectionError` for connection issues
2. **Preserve Original Errors**: Use `raise ... from e` to maintain error chains
3. **Log Appropriately**: Log errors with context but avoid logging sensitive data

### Performance

1. **Implement Timeouts**: Use the provided timeout parameter
2. **Connection Pooling**: Reuse connections when possible
3. **Async Operations**: Use async/await for I/O operations
4. **Caching**: Consider caching frequently accessed data

### Security

1. **Validate Input**: Sanitize query parameters
2. **Secure Credentials**: Don't log API keys or passwords
3. **Use HTTPS**: Prefer secure connections
4. **Rate Limiting**: Respect API rate limits

### Reliability

1. **Graceful Degradation**: Handle partial failures
2. **Retry Logic**: Implement exponential backoff for transient failures
3. **Circuit Breaker**: Use the built-in error handling
4. **Health Monitoring**: Implement meaningful health checks

## Troubleshooting

### Common Issues

#### "Adapter not found" Error
- Ensure your adapter is registered in the registry
- Check the engine_type matches the registration name

#### Timeout Errors
- Verify your search operations complete within the timeout
- Consider increasing timeout for slow engines
- Implement proper async operations

#### Connection Failures
- Check network connectivity to your search engine
- Verify credentials and authentication
- Ensure proper error handling for connection issues

#### Result Format Errors
- Verify your `_parse_results` method returns the correct tuple format
- Check that document IDs are strings
- Ensure scores are numeric (0.0-1.0 range preferred)

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.getLogger('src.adapters').setLevel(logging.DEBUG)
   ```

2. **Test Individual Methods**:
   ```python
   # Test health check
   is_healthy = await adapter.health_check()
   print(f"Health: {is_healthy}")
   
   # Test configuration
   config = adapter.get_configuration()
   print(f"Config: {config}")
   ```

3. **Use Mock Data**: Start with mock responses to verify your parsing logic

4. **Check Raw Responses**: Log raw responses from your search engine to understand the data format

### Getting Help

- Check existing adapter implementations for patterns
- Review the base adapter class for available helper methods
- Use the built-in error handling and logging utilities
- Test with mock adapters first to verify integration

## Advanced Topics

### Custom Result Scoring

If your engine doesn't provide relevance scores:

```python
def _calculate_custom_score(self, query_terms: List[str], content: str) -> float:
    """Calculate a custom relevance score."""
    score = 0.0
    for term in query_terms:
        if term.lower() in content.lower():
            score += 1.0
    return min(score / len(query_terms), 1.0) if query_terms else 0.0
```

### Batch Operations

For engines that support batch queries:

```python
async def batch_search(self, queries: List[str], limit: int = 10) -> List[SearchResults]:
    """Execute multiple searches in a single request."""
    # Implement batch search logic
    pass
```

### Custom Metadata

Add engine-specific metadata:

```python
def _build_metadata(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
    """Build comprehensive metadata."""
    return {
        'engine_type': 'my_custom_engine',
        'timestamp': raw_item.get('indexed_at'),
        'category': raw_item.get('category'),
        'confidence': raw_item.get('confidence_score'),
        'raw_data': raw_item  # Include original data for debugging
    }
```

This guide should help you create robust, well-integrated custom search engine adapters for the RAG Fusion Factory system.