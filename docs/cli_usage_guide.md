# RAG Fusion Factory CLI Usage Guide

This guide covers how to use the RAG Fusion Factory command-line interface for various tasks including configuration, testing, and operation.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration](#configuration)
5. [Adapter Management](#adapter-management)
6. [Testing and Validation](#testing-and-validation)
7. [Advanced Usage](#advanced-usage)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Overview

The RAG Fusion Factory CLI provides a comprehensive interface for:

- Starting the fusion factory with different configurations
- Managing search engine adapters
- Validating configurations
- Running tests and health checks
- Interactive query testing

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-fusion-factory
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Make scripts executable** (Unix/Linux/macOS):
   ```bash
   chmod +x scripts/start_fusion_factory.py
   ```

## Basic Usage

### Starting the Fusion Factory

#### Quick Start (Development Mode)
```bash
python scripts/start_fusion_factory.py --dev
```

This starts the fusion factory with mock adapters for immediate testing.

#### Production Mode
```bash
python scripts/start_fusion_factory.py --config config/production.yaml
```

#### Testing Mode
```bash
python scripts/start_fusion_factory.py --test
```

### Command Line Options

```bash
python scripts/start_fusion_factory.py --help
```

**Available options**:

- `--config, -c`: Path to configuration file
- `--dev, --development`: Start in development mode with mock adapters
- `--test, --testing`: Start in testing mode with minimal configuration
- `--list-adapters`: List all available adapter types
- `--validate`: Validate a configuration file
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Configuration

### Configuration File Formats

The CLI supports multiple configuration formats:

#### YAML Configuration (Recommended)
```yaml
# config/my_config.yaml
logging:
  level: INFO

engines:
  - engine_type: elasticsearch
    engine_id: es_primary
    timeout: 30.0
    config:
      host: localhost
      port: 9200
      index: documents

  - engine_type: solr
    engine_id: solr_primary
    timeout: 30.0
    config:
      host: localhost
      port: 8983
      core: documents

fusion:
  default_limit: 10
  max_limit: 100
```

#### JSON Configuration
```json
{
  "logging": {
    "level": "INFO"
  },
  "engines": [
    {
      "engine_type": "elasticsearch",
      "engine_id": "es_primary",
      "timeout": 30.0,
      "config": {
        "host": "localhost",
        "port": 9200,
        "index": "documents"
      }
    }
  ]
}
```

### Environment Variables

Use environment variables for sensitive data:

```yaml
# config/production.yaml
engines:
  - engine_type: elasticsearch
    engine_id: prod_es
    config:
      host: "${ES_HOST}"
      username: "${ES_USERNAME}"
      password: "${ES_PASSWORD}"
```

Set environment variables:
```bash
export ES_HOST=elasticsearch.prod.internal
export ES_USERNAME=elastic_user
export ES_PASSWORD=secure_password
```

### Configuration Validation

Validate configuration files before use:

```bash
# Validate configuration
python scripts/start_fusion_factory.py --validate config/production.yaml

# Example output:
# Configuration file 'config/production.yaml' is valid
# Found 3 engine configurations
```

## Adapter Management

### List Available Adapters

```bash
python scripts/start_fusion_factory.py --list-adapters
```

**Output**:
```
Available adapter types:
  - database
  - elasticsearch
  - mock_file
  - mock_inmemory
  - opensearch
  - rest_api
  - solr
  - web_scraping

Total: 8 adapter types available
```

### Adapter Configuration Examples

#### Elasticsearch Adapter
```yaml
engines:
  - engine_type: elasticsearch
    engine_id: es_main
    timeout: 30.0
    config:
      host: localhost
      port: 9200
      index: documents
      username: elastic
      password: changeme
      use_ssl: false
```

#### Mock Adapter (for testing)
```yaml
engines:
  - engine_type: mock_inmemory
    engine_id: test_mock
    timeout: 5.0
    config:
      response_delay: 0.1
      failure_rate: 0.0
      max_results: 10
      documents:
        - id: doc1
          title: "Test Document"
          content: "This is test content"
          category: "test"
```

#### REST API Adapter
```yaml
engines:
  - engine_type: rest_api
    engine_id: custom_api
    timeout: 30.0
    config:
      base_url: "https://api.example.com"
      search_endpoint: "/search"
      query_param: "q"
      limit_param: "limit"
      auth_token: "your-api-token"
```

## Testing and Validation

### Interactive Testing Mode

Start the fusion factory in interactive mode to test queries:

```bash
python scripts/start_fusion_factory.py --dev
```

**Interactive session**:
```
Interactive mode started. Type 'quit' to exit.
Enter search queries to test the fusion factory:

> machine learning
Searching for: 'machine learning'
Found 3 results:
  1. [dev_mock_1] doc1
     Score: 0.850
     Content: Machine learning is a subset of artificial intelligence...
  2. [dev_mock_2] doc2
     Score: 0.720
     Content: Introduction to machine learning algorithms...
  3. [dev_mock_1] doc3
     Score: 0.650
     Content: Machine learning applications in industry...

> python programming
Searching for: 'python programming'
Found 2 results:
  1. [dev_mock_2] doc4
     Score: 0.900
     Content: Python programming best practices...
  2. [dev_mock_1] doc5
     Score: 0.750
     Content: Advanced Python programming techniques...

> quit
Interactive mode ended.
```

### Health Checks

The CLI automatically performs health checks on startup:

```bash
python scripts/start_fusion_factory.py --config config/production.yaml
```

**Output**:
```
Starting RAG Fusion Factory with config: config/production.yaml
Created adapter: es_primary (elasticsearch)
Created adapter: solr_primary (solr)
Performing initial health checks...
Health check results: 2/2 adapters healthy
RAG Fusion Factory started successfully!
Available adapters: ['es_primary', 'solr_primary']
```

### Configuration Testing

Test configurations without starting the full system:

```bash
# Dry run (if supported)
python scripts/start_fusion_factory.py --config config/test.yaml --dry-run

# Validate and test adapter creation
python -c "
from src.config.settings import load_config
from src.adapters.registry import get_adapter_registry

config = load_config('config/test.yaml')
registry = get_adapter_registry()

for engine_config in config['engines']:
    try:
        adapter = registry.create_adapter(
            engine_type=engine_config['engine_type'],
            engine_id=engine_config['engine_id'],
            config=engine_config['config']
        )
        print(f'✓ {adapter.engine_id}: {engine_config[\"engine_type\"]}')
    except Exception as e:
        print(f'✗ {engine_config[\"engine_id\"]}: {str(e)}')
"
```

## Advanced Usage

### Custom Logging Configuration

#### Set Log Level
```bash
python scripts/start_fusion_factory.py --dev --log-level DEBUG
```

#### Custom Log Format
```yaml
# In configuration file
logging:
  level: INFO
  format: json  # or 'text'
  handlers:
    - type: console
    - type: file
      filename: logs/fusion_factory.log
      max_size: 100MB
      backup_count: 5
```

### Environment-Specific Configurations

#### Development Environment
```bash
# Use development configuration
python scripts/start_fusion_factory.py --config config/development.yaml

# Or use development mode
python scripts/start_fusion_factory.py --dev
```

#### Production Environment
```bash
# Production with environment variables
export CONFIG_FILE=config/production.yaml
export LOG_LEVEL=INFO
python scripts/start_fusion_factory.py --config $CONFIG_FILE --log-level $LOG_LEVEL
```

#### Testing Environment
```bash
# Minimal testing setup
python scripts/start_fusion_factory.py --test

# Custom test configuration
python scripts/start_fusion_factory.py --config config/testing.yaml
```

### Programmatic Usage

Use the CLI components programmatically:

```python
import asyncio
from scripts.start_fusion_factory import FusionFactoryLauncher

async def main():
    launcher = FusionFactoryLauncher()
    
    # Start with configuration
    await launcher.start_with_config('config/my_config.yaml')
    
    # Or start in development mode
    # await launcher.start_development_mode()

if __name__ == '__main__':
    asyncio.run(main())
```

## Examples

### Example 1: Multi-Engine Setup

**Configuration** (`config/multi_engine.yaml`):
```yaml
logging:
  level: INFO

engines:
  - engine_type: elasticsearch
    engine_id: es_primary
    timeout: 30.0
    config:
      host: localhost
      port: 9200
      index: documents

  - engine_type: solr
    engine_id: solr_backup
    timeout: 30.0
    config:
      host: localhost
      port: 8983
      core: documents

  - engine_type: mock_inmemory
    engine_id: fallback_mock
    timeout: 5.0
    config:
      response_delay: 0.1
      failure_rate: 0.0
      max_results: 5

fusion:
  default_limit: 10
  timeout: 60.0
```

**Usage**:
```bash
python scripts/start_fusion_factory.py --config config/multi_engine.yaml
```

### Example 2: Development with Custom Mock Data

**Configuration** (`config/dev_custom.yaml`):
```yaml
engines:
  - engine_type: mock_file
    engine_id: custom_data
    timeout: 10.0
    config:
      data_file: "examples/custom_documents.json"
      response_delay: 0.2
      failure_rate: 0.05
      max_results: 15
```

**Usage**:
```bash
python scripts/start_fusion_factory.py --config config/dev_custom.yaml
```

### Example 3: Production with Multiple Environments

**Production script** (`scripts/start_production.sh`):
```bash
#!/bin/bash

# Set environment
export ENVIRONMENT=production
export CONFIG_FILE=config/production.yaml
export LOG_LEVEL=INFO

# Set database credentials
export POSTGRES_HOST=db.prod.internal
export POSTGRES_USER=fusion_prod
export POSTGRES_PASSWORD=$(cat /secrets/db_password)

# Set search engine credentials
export ES_HOST=elasticsearch.prod.internal
export ES_USERNAME=elastic_prod
export ES_PASSWORD=$(cat /secrets/es_password)

# Start fusion factory
python scripts/start_fusion_factory.py \
  --config $CONFIG_FILE \
  --log-level $LOG_LEVEL
```

### Example 4: Batch Configuration Validation

**Validation script** (`scripts/validate_all_configs.sh`):
```bash
#!/bin/bash

echo "Validating all configuration files..."

for config_file in config/*.yaml; do
    echo "Validating $config_file..."
    if python scripts/start_fusion_factory.py --validate "$config_file"; then
        echo "✓ $config_file is valid"
    else
        echo "✗ $config_file has errors"
        exit 1
    fi
done

echo "All configuration files are valid!"
```

## Troubleshooting

### Common Issues

#### 1. Configuration File Not Found

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'config/missing.yaml'
```

**Solution**:
```bash
# Check file exists
ls -la config/

# Use absolute path
python scripts/start_fusion_factory.py --config /full/path/to/config.yaml

# Validate path
python scripts/start_fusion_factory.py --validate config/production.yaml
```

#### 2. Invalid Configuration Format

**Error**:
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/production.yaml'))"

# Use online YAML validator
# Check indentation and syntax
```

#### 3. Adapter Creation Failures

**Error**:
```
ValueError: Unknown engine type 'invalid_engine'
```

**Solution**:
```bash
# List available adapters
python scripts/start_fusion_factory.py --list-adapters

# Check configuration
python scripts/start_fusion_factory.py --validate config/production.yaml

# Test adapter creation
python -c "
from src.adapters.registry import get_adapter_registry
registry = get_adapter_registry()
print(registry.get_available_adapters())
"
```

#### 4. Connection Issues

**Error**:
```
SearchEngineConnectionError: Failed to connect to Elasticsearch
```

**Solution**:
```bash
# Test connectivity
curl http://localhost:9200/_cluster/health

# Check configuration
python scripts/start_fusion_factory.py --validate config/production.yaml

# Use development mode for testing
python scripts/start_fusion_factory.py --dev
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
python scripts/start_fusion_factory.py --config config/production.yaml --log-level DEBUG
```

**Debug output includes**:
- Detailed configuration parsing
- Adapter creation steps
- Connection attempts
- Query execution details
- Error stack traces

### Getting Help

#### Built-in Help
```bash
python scripts/start_fusion_factory.py --help
```

#### Configuration Examples
```bash
# Check example configurations
ls examples/adapter_configurations.yaml

# View sample documents
cat examples/sample_documents.json
```

#### Validation Tools
```bash
# Validate configuration
python scripts/start_fusion_factory.py --validate config/my_config.yaml

# List available adapters
python scripts/start_fusion_factory.py --list-adapters

# Test in development mode
python scripts/start_fusion_factory.py --dev
```

This CLI usage guide should help you effectively use the RAG Fusion Factory command-line interface for various tasks and scenarios.