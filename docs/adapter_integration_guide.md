# Adapter Integration Guide

This guide provides comprehensive information on integrating search engine adapters with the RAG Fusion Factory system, including configuration, best practices, and troubleshooting.

## Table of Contents

1. [Overview](#overview)
2. [Built-in Adapters](#built-in-adapters)
3. [Adapter Configuration](#adapter-configuration)
4. [Integration Patterns](#integration-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling](#error-handling)
7. [Testing Strategies](#testing-strategies)
8. [Production Considerations](#production-considerations)
9. [Troubleshooting](#troubleshooting)

## Overview

The RAG Fusion Factory uses a plugin-based architecture that allows seamless integration of multiple search engines through standardized adapters. Each adapter provides a consistent interface while handling engine-specific implementation details.

### Key Benefits

- **Unified Interface**: All adapters implement the same interface
- **Fault Tolerance**: Individual adapter failures don't affect the system
- **Scalability**: Easy to add new search engines
- **Flexibility**: Mix different engine types in one deployment

### Adapter Lifecycle

1. **Registration**: Adapter types are registered with the system
2. **Configuration**: Instances are configured with engine-specific settings
3. **Initialization**: Adapters establish connections and validate settings
4. **Operation**: Adapters handle search queries and health checks
5. **Monitoring**: System tracks adapter performance and health

## Built-in Adapters

### Production-Ready Adapters

#### Elasticsearch Adapter
**Use Case**: Full-text search with advanced features

```yaml
engines:
  - engine_type: elasticsearch
    engine_id: es_primary
    timeout: 30.0
    config:
      host: elasticsearch.example.com
      port: 9200
      index: documents
      username: elastic_user
      password: secure_password
      use_ssl: true
```

**Features**:
- Full-text search with relevance scoring
- Advanced query capabilities (fuzzy, wildcard, etc.)
- Highlighting and snippets
- Aggregations and faceting
- Cluster health monitoring

**Best For**: Large-scale document search, complex queries, analytics

#### Solr Adapter
**Use Case**: Enterprise search with rich features

```yaml
engines:
  - engine_type: solr
    engine_id: solr_primary
    timeout: 30.0
    config:
      host: solr.example.com
      port: 8983
      core: documents
      username: solr_user
      password: secure_password
      use_ssl: true
```

**Features**:
- Powerful query parsing (DisMax, eDisMax)
- Faceting and grouping
- Spell checking and suggestions
- Rich document processing
- Distributed search

**Best For**: Enterprise search, e-commerce, content management

#### OpenSearch Adapter
**Use Case**: Open-source alternative to Elasticsearch

```yaml
engines:
  - engine_type: opensearch
    engine_id: opensearch_primary
    timeout: 30.0
    config:
      host: opensearch.example.com
      port: 9200
      index: documents
      username: admin
      password: admin_password
      use_ssl: true
      verify_ssl: true
```

**Features**:
- Elasticsearch-compatible API
- Advanced security features
- Machine learning capabilities
- Observability tools
- Multi-tenancy support

**Best For**: Organizations requiring open-source solutions, AWS deployments

### Development and Testing Adapters

#### In-Memory Mock Adapter
**Use Case**: Development and unit testing

```yaml
engines:
  - engine_type: mock_inmemory
    engine_id: dev_mock
    timeout: 5.0
    config:
      response_delay: 0.1
      failure_rate: 0.0
      max_results: 10
      documents:
        - id: doc1
          title: "Sample Document"
          content: "This is sample content for testing"
          category: "test"
          tags: ["sample", "test"]
```

**Features**:
- No external dependencies
- Configurable response times and failure rates
- Custom document sets
- Deterministic results for testing

**Best For**: Unit tests, development, CI/CD pipelines

#### File-Based Mock Adapter
**Use Case**: Testing with larger datasets

```yaml
engines:
  - engine_type: mock_file
    engine_id: file_mock
    timeout: 10.0
    config:
      data_file: "test_data/documents.json"
      response_delay: 0.2
      failure_rate: 0.05
      max_results: 20
```

**Features**:
- Load test data from JSON files
- Simulate realistic response patterns
- Support for large test datasets
- Version-controlled test data

**Best For**: Integration tests, performance testing, demo environments

### Custom Integration Adapters

#### REST API Adapter
**Use Case**: Integration with custom APIs

```yaml
engines:
  - engine_type: rest_api
    engine_id: custom_api
    timeout: 30.0
    config:
      base_url: "https://api.example.com"
      search_endpoint: "/v1/search"
      health_endpoint: "/v1/health"
      query_param: "query"
      limit_param: "max_results"
      headers:
        "X-API-Key": "your-api-key"
        "Content-Type": "application/json"
      auth_token: "bearer-token"
      response_format: "json"
      result_path: "data.results"
      id_field: "document_id"
      score_field: "relevance_score"
      content_field: "text_content"
```

**Features**:
- Configurable API endpoints and parameters
- Flexible authentication methods
- Custom response parsing
- Header customization

**Best For**: Third-party APIs, microservices, custom search services

#### Database Adapter
**Use Case**: SQL-based search

```yaml
engines:
  - engine_type: database
    engine_id: postgres_search
    timeout: 30.0
    config:
      connection_string: "postgresql://user:pass@localhost:5432/searchdb"
      table_name: "documents"
      id_column: "doc_id"
      content_columns: ["title", "content", "description"]
      score_method: "match_count"
      search_type: "fulltext"
```

**Features**:
- Full-text search capabilities
- Custom scoring methods
- Flexible column mapping
- SQL query optimization

**Best For**: Existing database systems, structured data, legacy integrations

#### Web Scraping Adapter
**Use Case**: Websites without APIs

```yaml
engines:
  - engine_type: web_scraping
    engine_id: web_scraper
    timeout: 45.0
    config:
      search_url: "https://example.com/search?q={query}&limit={limit}"
      result_selector: ".search-result"
      title_selector: ".result-title"
      content_selector: ".result-snippet"
      link_selector: ".result-link"
      headers:
        "User-Agent": "RAG-Fusion-Factory/1.0"
```

**Features**:
- CSS selector-based extraction
- Custom HTTP headers
- Rate limiting support
- Error handling for web scraping

**Best For**: Legacy systems, websites without APIs, content aggregation

## Adapter Configuration

### Configuration Structure

All adapters follow a consistent configuration structure:

```yaml
engines:
  - engine_type: <adapter_type>      # Required: Type of adapter
    engine_id: <unique_id>           # Required: Unique identifier
    timeout: <seconds>               # Optional: Operation timeout (default: 30.0)
    config:                          # Required: Adapter-specific configuration
      <adapter_specific_settings>
```

### Common Configuration Patterns

#### Authentication
```yaml
# Basic authentication
config:
  username: "user"
  password: "password"

# Token-based authentication
config:
  auth_token: "bearer-token"
  headers:
    "Authorization": "Bearer ${API_TOKEN}"

# API key authentication
config:
  api_key: "your-api-key"
  headers:
    "X-API-Key": "${API_KEY}"
```

#### SSL/TLS Configuration
```yaml
config:
  use_ssl: true
  verify_ssl: true  # Set to false for self-signed certificates
  ca_cert_path: "/path/to/ca.crt"
  client_cert_path: "/path/to/client.crt"
  client_key_path: "/path/to/client.key"
```

#### Connection Pooling
```yaml
config:
  connection_pool_size: 10
  max_connections: 20
  keep_alive: true
  connection_timeout: 10.0
```

### Environment Variable Substitution

Use environment variables for sensitive data:

```yaml
engines:
  - engine_type: elasticsearch
    engine_id: prod_es
    config:
      host: "${ES_HOST}"
      username: "${ES_USERNAME}"
      password: "${ES_PASSWORD}"
      use_ssl: true
```

Set environment variables:
```bash
export ES_HOST=elasticsearch.prod.internal
export ES_USERNAME=elastic_user
export ES_PASSWORD=secure_password
```

## Integration Patterns

### Single Engine Integration

Simple integration with one search engine:

```yaml
engines:
  - engine_type: elasticsearch
    engine_id: primary_es
    timeout: 30.0
    config:
      host: localhost
      port: 9200
      index: documents
```

### Multi-Engine Integration

Combine multiple search engines for better coverage:

```yaml
engines:
  - engine_type: elasticsearch
    engine_id: es_primary
    timeout: 30.0
    config:
      host: elasticsearch-cluster.internal
      port: 9200
      index: primary_docs

  - engine_type: solr
    engine_id: solr_secondary
    timeout: 30.0
    config:
      host: solr-cluster.internal
      port: 8983
      core: secondary_docs

  - engine_type: opensearch
    engine_id: opensearch_backup
    timeout: 30.0
    config:
      host: opensearch.internal
      port: 9200
      index: backup_docs
```

### Tiered Architecture

Use different engines for different purposes:

```yaml
engines:
  # Primary: Fast, frequently accessed content
  - engine_type: elasticsearch
    engine_id: primary_fast
    timeout: 10.0
    config:
      host: fast-es-cluster.internal
      index: recent_documents

  # Secondary: Comprehensive, slower searches
  - engine_type: solr
    engine_id: secondary_comprehensive
    timeout: 60.0
    config:
      host: comprehensive-solr.internal
      core: all_documents

  # Fallback: Always available mock data
  - engine_type: mock_inmemory
    engine_id: fallback_mock
    timeout: 5.0
    config:
      response_delay: 0.1
      max_results: 5
```

### Development/Staging/Production

Environment-specific configurations:

#### Development
```yaml
# config/development.yaml
engines:
  - engine_type: mock_inmemory
    engine_id: dev_mock
    config:
      response_delay: 0.1
      failure_rate: 0.1  # Simulate failures
      max_results: 5

  - engine_type: elasticsearch
    engine_id: dev_es
    config:
      host: localhost
      port: 9200
      index: dev_documents
```

#### Staging
```yaml
# config/staging.yaml
engines:
  - engine_type: elasticsearch
    engine_id: staging_es
    config:
      host: elasticsearch.staging.internal
      port: 9200
      index: staging_documents
      use_ssl: true

  - engine_type: solr
    engine_id: staging_solr
    config:
      host: solr.staging.internal
      port: 8983
      core: staging_documents
```

#### Production
```yaml
# config/production.yaml
engines:
  - engine_type: elasticsearch
    engine_id: prod_es_primary
    config:
      host: elasticsearch.prod.internal
      port: 443
      index: production_documents
      username: "${ES_USERNAME}"
      password: "${ES_PASSWORD}"
      use_ssl: true

  - engine_type: elasticsearch
    engine_id: prod_es_secondary
    config:
      host: elasticsearch-backup.prod.internal
      port: 443
      index: production_documents
      username: "${ES_USERNAME}"
      password: "${ES_PASSWORD}"
      use_ssl: true
```

## Performance Optimization

### Timeout Configuration

Configure appropriate timeouts for different scenarios:

```yaml
engines:
  # Fast queries: Short timeout
  - engine_type: elasticsearch
    engine_id: fast_es
    timeout: 5.0
    config:
      host: fast-cluster.internal

  # Complex queries: Longer timeout
  - engine_type: solr
    engine_id: complex_solr
    timeout: 60.0
    config:
      host: complex-cluster.internal

  # External APIs: Medium timeout with retries
  - engine_type: rest_api
    engine_id: external_api
    timeout: 30.0
    config:
      base_url: "https://external-api.com"
      retry_count: 3
      retry_delay: 1.0
```

### Connection Optimization

Optimize connection settings:

```yaml
config:
  # Connection pooling
  connection_pool_size: 20
  max_connections: 50
  keep_alive: true
  
  # Timeouts
  connection_timeout: 10.0
  read_timeout: 30.0
  
  # Retries
  max_retries: 3
  retry_backoff: 1.0
```

### Query Optimization

Optimize queries for better performance:

```yaml
config:
  # Limit result size
  default_limit: 10
  max_limit: 100
  
  # Field selection
  return_fields: ["id", "title", "content", "score"]
  
  # Search optimization
  search_type: "dfs_query_then_fetch"  # Elasticsearch
  query_parser: "edismax"              # Solr
```

### Caching Strategies

Implement caching at different levels:

```yaml
fusion:
  caching:
    enabled: true
    ttl: 300  # 5 minutes
    max_size: 1000
    
  # Per-adapter caching
  adapter_caching:
    elasticsearch:
      enabled: true
      ttl: 600
    solr:
      enabled: false
```

## Error Handling

### Graceful Degradation

Configure the system to handle adapter failures gracefully:

```yaml
fusion:
  error_handling:
    # Continue with available adapters if some fail
    fail_fast: false
    
    # Minimum number of healthy adapters required
    min_healthy_adapters: 1
    
    # Retry failed adapters
    retry_failed: true
    retry_interval: 60  # seconds
```

### Circuit Breaker Pattern

Implement circuit breakers for failing adapters:

```yaml
config:
  circuit_breaker:
    enabled: true
    failure_threshold: 5      # Failures before opening circuit
    recovery_timeout: 60      # Seconds before attempting recovery
    success_threshold: 3      # Successes needed to close circuit
```

### Error Monitoring

Monitor and log adapter errors:

```yaml
logging:
  level: INFO
  error_tracking:
    enabled: true
    include_stack_traces: true
    max_error_length: 1000

monitoring:
  error_metrics:
    enabled: true
    alert_threshold: 10  # Errors per minute
```

## Testing Strategies

### Unit Testing

Test individual adapters:

```python
import pytest
from src.adapters.elasticsearch_adapter import ElasticsearchAdapter

@pytest.fixture
def es_adapter():
    config = {
        'host': 'localhost',
        'port': 9200,
        'index': 'test_index'
    }
    return ElasticsearchAdapter('test_es', config)

@pytest.mark.asyncio
async def test_search(es_adapter, mock_elasticsearch):
    # Mock Elasticsearch response
    mock_elasticsearch.return_value = {
        'hits': {
            'hits': [
                {'_id': 'doc1', '_score': 1.0, '_source': {'content': 'test'}}
            ]
        }
    }
    
    results = await es_adapter.search('test query')
    assert len(results.results) == 1
    assert results.results[0].document_id == 'doc1'
```

### Integration Testing

Test adapter integration with real services:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_elasticsearch_integration():
    config = {
        'host': 'localhost',
        'port': 9200,
        'index': 'integration_test'
    }
    adapter = ElasticsearchAdapter('integration_test', config)
    
    # Test health check
    is_healthy = await adapter.health_check()
    assert is_healthy
    
    # Test search
    results = await adapter.search('test query', limit=5)
    assert isinstance(results, SearchResults)
```

### Load Testing

Test adapter performance under load:

```python
import asyncio
import time

async def load_test_adapter(adapter, queries, concurrent_requests=10):
    async def run_query(query):
        start_time = time.time()
        try:
            results = await adapter.search(query)
            return time.time() - start_time, len(results.results), None
        except Exception as e:
            return time.time() - start_time, 0, str(e)
    
    # Run concurrent requests
    tasks = []
    for _ in range(concurrent_requests):
        for query in queries:
            tasks.append(run_query(query))
    
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    response_times = [r[0] for r in results]
    result_counts = [r[1] for r in results]
    errors = [r[2] for r in results if r[2]]
    
    print(f"Average response time: {sum(response_times)/len(response_times):.3f}s")
    print(f"Average results: {sum(result_counts)/len(result_counts):.1f}")
    print(f"Error rate: {len(errors)/len(results)*100:.1f}%")
```

## Production Considerations

### Security

#### Authentication and Authorization
```yaml
config:
  # Use strong authentication
  username: "${SECURE_USERNAME}"
  password: "${SECURE_PASSWORD}"
  
  # Enable SSL/TLS
  use_ssl: true
  verify_ssl: true
  
  # Use certificates
  client_cert: "/etc/ssl/certs/client.crt"
  client_key: "/etc/ssl/private/client.key"
  ca_cert: "/etc/ssl/certs/ca.crt"
```

#### Network Security
```yaml
config:
  # Use private networks
  host: "elasticsearch.private.internal"
  
  # Restrict access
  allowed_ips: ["10.0.0.0/8", "172.16.0.0/12"]
  
  # Use VPN
  proxy: "socks5://vpn.internal:1080"
```

### Monitoring and Alerting

#### Health Monitoring
```yaml
monitoring:
  health_checks:
    enabled: true
    interval: 30  # seconds
    timeout: 10
    
  alerts:
    unhealthy_threshold: 2  # consecutive failures
    notification_channels:
      - email: "ops@example.com"
      - slack: "#alerts"
```

#### Performance Monitoring
```yaml
monitoring:
  metrics:
    enabled: true
    collection_interval: 60
    
  performance_alerts:
    response_time_threshold: 5.0  # seconds
    error_rate_threshold: 0.05    # 5%
    availability_threshold: 0.99  # 99%
```

### Scalability

#### Horizontal Scaling
```yaml
engines:
  # Multiple instances of the same engine type
  - engine_type: elasticsearch
    engine_id: es_shard_1
    config:
      host: "es-shard-1.internal"
      
  - engine_type: elasticsearch
    engine_id: es_shard_2
    config:
      host: "es-shard-2.internal"
```

#### Load Balancing
```yaml
config:
  load_balancing:
    enabled: true
    strategy: "round_robin"  # or "least_connections", "random"
    health_check_interval: 30
```

### Backup and Recovery

#### Configuration Backup
```bash
# Backup configurations
cp -r config/ backup/config-$(date +%Y%m%d)/

# Version control
git add config/
git commit -m "Update production configuration"
```

#### Data Recovery
```yaml
config:
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: 30  # days
    
  recovery:
    automatic: false
    notification: true
```

## Troubleshooting

### Common Issues

#### Connection Failures
```bash
# Test connectivity
curl -v http://elasticsearch:9200/_cluster/health

# Check DNS resolution
nslookup elasticsearch.internal

# Test authentication
curl -u username:password http://elasticsearch:9200/
```

#### Configuration Errors
```bash
# Validate configuration
python scripts/start_fusion_factory.py --validate config/production.yaml

# Test adapter creation
python -c "
from src.adapters.registry import get_adapter_registry
registry = get_adapter_registry()
adapter = registry.create_adapter('elasticsearch', 'test', {
    'host': 'localhost',
    'port': 9200,
    'index': 'test'
})
print('Adapter created successfully')
"
```

#### Performance Issues
```bash
# Monitor resource usage
top -p $(pgrep -f fusion_factory)

# Check network latency
ping elasticsearch.internal

# Analyze slow queries
grep "response_time" logs/fusion_factory.log | sort -k5 -nr | head -10
```

### Debugging Tools

#### Enable Debug Logging
```yaml
logging:
  level: DEBUG
  handlers:
    - type: console
      format: detailed
    - type: file
      filename: logs/debug.log
      format: json
```

#### Health Check Tools
```bash
# Check all adapters
python -c "
import asyncio
from src.adapters.registry import get_adapter_registry

async def check_health():
    registry = get_adapter_registry()
    health = await registry.health_check_all()
    for engine_id, is_healthy in health.items():
        status = '✓' if is_healthy else '✗'
        print(f'{status} {engine_id}: {is_healthy}')

asyncio.run(check_health())
"
```

#### Performance Profiling
```python
import cProfile
import asyncio
from src.services.fusion_factory import FusionFactory

async def profile_search():
    # Setup fusion factory
    factory = FusionFactory(adapters, config)
    
    # Profile search operation
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = await factory.search("test query")
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')

asyncio.run(profile_search())
```

This comprehensive adapter integration guide should help you successfully integrate and manage search engine adapters in your RAG Fusion Factory deployment.