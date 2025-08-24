# RAG Fusion Factory Deployment Guide

This guide covers various deployment options for the RAG Fusion Factory, from local development to production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Configuration Management](#configuration-management)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Access to search engines (Elasticsearch, Solr, OpenSearch, etc.)

### Minimal Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-fusion-factory
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start in development mode**:
   ```bash
   python scripts/start_fusion_factory.py --dev
   ```

This will start the fusion factory with mock adapters for immediate testing.

## Local Development

### Development Environment Setup

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If you have dev requirements
   ```

3. **Configure environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

### Running Different Modes

#### Development Mode (Mock Adapters)
```bash
python scripts/start_fusion_factory.py --dev
```

#### Testing Mode (Minimal Configuration)
```bash
python scripts/start_fusion_factory.py --test
```

#### Custom Configuration
```bash
python scripts/start_fusion_factory.py --config config/development.yaml
```

### Available Commands

```bash
# List available adapter types
python scripts/start_fusion_factory.py --list-adapters

# Validate configuration file
python scripts/start_fusion_factory.py --validate config/my_config.yaml

# Start with specific log level
python scripts/start_fusion_factory.py --dev --log-level DEBUG
```

## Docker Deployment

### Single Container Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t rag-fusion-factory .
   ```

2. **Run with development configuration**:
   ```bash
   docker run -p 8000:8000 \
     -e ENVIRONMENT=development \
     -v $(pwd)/config:/app/config:ro \
     -v $(pwd)/logs:/app/logs \
     rag-fusion-factory
   ```

3. **Run with custom configuration**:
   ```bash
   docker run -p 8000:8000 \
     -e CONFIG_FILE=/app/config/production.yaml \
     -e LOG_LEVEL=INFO \
     -v $(pwd)/config:/app/config:ro \
     -v $(pwd)/logs:/app/logs \
     rag-fusion-factory
   ```

### Docker Compose Deployment

#### Full Stack (Production-like)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-fusion-factory

# Stop all services
docker-compose down
```

#### Development Stack
```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Or simply (override is loaded automatically)
docker-compose up -d
```

#### Services Included

The minimal Docker Compose setup includes:

- **RAG Fusion Factory**: Main application
- **Elasticsearch (Optional)**: Reference search engine for testing (use `--profile reference` to include)

For production deployments, users should connect their own search engines by:
1. Copying `docker-compose.example.yml` as a starting point
2. Configuring their own search engine services
3. Setting appropriate environment variables

### Environment Variables

Key environment variables for Docker deployment:

```bash
# Application settings
ENVIRONMENT=production|development|testing
CONFIG_FILE=/app/config/production.yaml
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR

# Search engines (configure only the ones you use)
ELASTICSEARCH_HOST=your-elasticsearch-host
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USERNAME=your-username
ELASTICSEARCH_PASSWORD=your-password

SOLR_HOST=your-solr-host
SOLR_PORT=8983

OPENSEARCH_HOST=your-opensearch-host
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME=your-username
OPENSEARCH_PASSWORD=your-password

# Database (if using database adapter)
DATABASE_HOST=your-database-host
DATABASE_PORT=5432
DATABASE_USER=your-db-user
DATABASE_PASSWORD=your-db-password
DATABASE_NAME=your-db-name
```

## Production Deployment

### Prerequisites

- **Existing search engines** (Elasticsearch, Solr, OpenSearch, etc.)
- Container orchestration platform (Kubernetes, Docker Swarm, etc.) - optional
- Load balancer - optional for high availability
- Monitoring and logging infrastructure
- SSL certificates for secure connections

### Kubernetes Deployment

1. **Create namespace**:
   ```bash
   kubectl create namespace rag-fusion-factory
   ```

2. **Deploy configuration**:
   ```bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secrets.yaml
   ```

3. **Deploy application**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   ```

### Production Configuration

#### Security Considerations

1. **Use secrets management**:
   ```yaml
   # In production.yaml
   engines:
     - engine_type: elasticsearch
       engine_id: prod_es
       config:
         host: elasticsearch.prod.internal
         username: "${ES_USERNAME}"  # From environment/secrets
         password: "${ES_PASSWORD}"
         use_ssl: true
   ```

2. **Enable authentication**:
   - Configure API authentication
   - Use service accounts for inter-service communication
   - Enable TLS for all connections

3. **Network security**:
   - Use private networks
   - Configure firewalls
   - Enable VPN access for management

#### Performance Optimization

1. **Resource allocation**:
   ```yaml
   # docker-compose.yml
   services:
     rag-fusion-factory:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 4G
           reservations:
             cpus: '1.0'
             memory: 2G
   ```

2. **Scaling configuration**:
   ```yaml
   # docker-compose.yml
   services:
     rag-fusion-factory:
       deploy:
         replicas: 3
         update_config:
           parallelism: 1
           delay: 10s
         restart_policy:
           condition: on-failure
   ```

#### High Availability

1. **Multiple replicas**:
   - Deploy multiple instances
   - Use load balancing
   - Configure health checks

2. **Database clustering**:
   - Use database clusters
   - Configure read replicas
   - Implement backup strategies

3. **Search engine clustering**:
   - Deploy search engines in cluster mode
   - Configure replication
   - Monitor cluster health

## Configuration Management

### Configuration Files

The system supports multiple configuration formats:

- **YAML**: Primary format (recommended)
- **JSON**: Alternative format
- **Environment variables**: For sensitive data

### Configuration Structure

```yaml
# config/production.yaml
logging:
  level: INFO
  format: json

engines:
  - engine_type: elasticsearch
    engine_id: prod_es_primary
    timeout: 30.0
    config:
      host: elasticsearch-cluster.internal
      port: 9200
      index: documents
      use_ssl: true
      username: "${ES_USERNAME}"
      password: "${ES_PASSWORD}"

  - engine_type: solr
    engine_id: prod_solr_primary
    timeout: 30.0
    config:
      host: solr-cluster.internal
      port: 8983
      core: documents
      use_ssl: true

fusion:
  default_limit: 10
  max_limit: 100
  timeout: 60.0
  
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

### Environment-Specific Configurations

#### Development
```yaml
# config/development.yaml
logging:
  level: DEBUG

engines:
  - engine_type: mock_inmemory
    engine_id: dev_mock
    config:
      response_delay: 0.1
      failure_rate: 0.1
```

#### Testing
```yaml
# config/testing.yaml
logging:
  level: WARNING

engines:
  - engine_type: mock_inmemory
    engine_id: test_mock
    config:
      response_delay: 0.01
      failure_rate: 0.0
```

### Configuration Validation

Validate configurations before deployment:

```bash
# Validate configuration file
python scripts/start_fusion_factory.py --validate config/production.yaml

# Test configuration with dry run
python scripts/start_fusion_factory.py --config config/production.yaml --dry-run
```

## Monitoring and Logging

### Logging Configuration

The system uses structured logging with multiple output formats:

```yaml
# In configuration file
logging:
  level: INFO
  format: json  # or 'text'
  output: stdout  # or file path
  rotation:
    max_size: 100MB
    backup_count: 5
```

### Metrics Collection

#### Prometheus Integration

The system exposes metrics for Prometheus:

- **Request metrics**: Response times, error rates
- **Adapter metrics**: Health status, query counts
- **System metrics**: Memory usage, CPU utilization

#### Grafana Dashboards

Pre-configured dashboards are available in `monitoring/grafana/dashboards/`:

- **System Overview**: High-level system metrics
- **Adapter Performance**: Per-adapter performance metrics
- **Error Analysis**: Error rates and patterns

### Health Checks

#### Application Health
```bash
# Check application health
curl http://localhost:8000/health

# Check adapter health
curl http://localhost:8000/adapters/health
```

#### Docker Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "health check script" || exit 1
```

### Log Analysis

#### Centralized Logging

For production deployments, consider:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Fluentd**: Log collection and forwarding
- **Grafana Loki**: Log aggregation system

#### Log Formats

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "component": "FusionFactory",
  "message": "Search completed",
  "query": "machine learning",
  "results_count": 15,
  "response_time_ms": 250,
  "adapters_used": ["elasticsearch", "solr"]
}
```

## Troubleshooting

### Common Issues

#### 1. Adapter Connection Failures

**Symptoms**: Adapters showing as unhealthy, connection timeouts

**Solutions**:
```bash
# Check network connectivity
docker exec rag-fusion-factory nc -zv elasticsearch 9200

# Verify credentials
docker exec rag-fusion-factory curl -u user:pass http://elasticsearch:9200/_cluster/health

# Check logs
docker logs rag-fusion-factory | grep -i error
```

#### 2. Configuration Errors

**Symptoms**: Application fails to start, validation errors

**Solutions**:
```bash
# Validate configuration
python scripts/start_fusion_factory.py --validate config/production.yaml

# Check environment variables
docker exec rag-fusion-factory env | grep -E "(ES_|SOLR_|OPENSEARCH_)"

# Test configuration parsing
python -c "from src.config.settings import load_config; print(load_config('config/production.yaml'))"
```

#### 3. Performance Issues

**Symptoms**: Slow response times, high resource usage

**Solutions**:
```bash
# Monitor resource usage
docker stats rag-fusion-factory

# Check adapter performance
curl http://localhost:8000/metrics | grep adapter_response_time

# Analyze slow queries
docker logs rag-fusion-factory | grep "response_time_ms" | sort -k5 -nr | head -10
```

#### 4. Memory Issues

**Symptoms**: Out of memory errors, container restarts

**Solutions**:
```bash
# Increase memory limits
# In docker-compose.yml:
services:
  rag-fusion-factory:
    deploy:
      resources:
        limits:
          memory: 4G

# Monitor memory usage
docker exec rag-fusion-factory python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.2f}GB')
"
```

### Debugging Tools

#### Container Debugging
```bash
# Access container shell
docker exec -it rag-fusion-factory /bin/bash

# Check running processes
docker exec rag-fusion-factory ps aux

# View system resources
docker exec rag-fusion-factory top
```

#### Application Debugging
```bash
# Enable debug logging
docker exec rag-fusion-factory python scripts/start_fusion_factory.py --config config/debug.yaml --log-level DEBUG

# Test individual adapters
docker exec rag-fusion-factory python -c "
import asyncio
from src.adapters.registry import get_adapter_registry
registry = get_adapter_registry()
adapter = registry.get_adapter('elasticsearch_primary')
result = asyncio.run(adapter.health_check())
print(f'Health: {result}')
"
```

### Getting Help

1. **Check logs**: Always start with application and container logs
2. **Validate configuration**: Use the built-in validation tools
3. **Test components**: Test adapters individually
4. **Monitor metrics**: Use Prometheus and Grafana for insights
5. **Community support**: Check documentation and community forums

### Performance Tuning

#### Application Level
- Adjust timeout values
- Configure connection pooling
- Optimize query patterns
- Use caching strategies

#### Infrastructure Level
- Scale horizontally
- Optimize resource allocation
- Use faster storage
- Implement load balancing

This deployment guide should help you get the RAG Fusion Factory running in various environments, from local development to production deployments.