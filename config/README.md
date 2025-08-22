# RAG Fusion Factory Configuration

This directory contains YAML configuration files for the RAG Fusion Factory system. The configuration system uses OmegaConf for flexible, hierarchical configuration management.

## Configuration Files

### Core Configuration Files

- **`default.yaml`** - Base configuration with all default values
- **`development.yaml`** - Development environment overrides
- **`production.yaml`** - Production environment overrides
- **`user.yaml`** - User-specific overrides (optional, not tracked in git)

### Template Files

- **`user.yaml.template`** - Template for creating user-specific configuration

## Configuration Hierarchy

Configuration values are loaded and merged in the following order (later values override earlier ones):

1. `default.yaml` (base configuration)
2. `{environment}.yaml` (environment-specific overrides)
3. `user.yaml` (user-specific overrides)
4. Environment variables (highest priority)

## Environment Variables

You can override any configuration value using environment variables with the `RAG_` prefix:

```bash
# Override API configuration
export RAG_API_HOST=localhost
export RAG_API_PORT=8080
export RAG_API_DEBUG=true

# Override model configuration
export RAG_MODEL_CACHE_DIR=/custom/path/models

# Override logging
export RAG_LOG_LEVEL=DEBUG
```

## Usage Examples

### Setting Environment

```bash
# Set environment via environment variable
export ENVIRONMENT=production

# Or use the config manager CLI
python scripts/config_manager.py env --set production
```

### Creating User Configuration

```bash
# Initialize user config from template
python scripts/config_manager.py init

# Edit the created user.yaml file
vim config/user.yaml
```

### Validating Configuration

```bash
# Validate current configuration
python scripts/config_manager.py validate
```

### Viewing Configuration

```bash
# Print configuration summary
python scripts/config_manager.py summary

# Export full configuration to file
python scripts/config_manager.py export config_dump.yaml
python scripts/config_manager.py export config_dump.json --format json
```

### Updating Configuration

```bash
# Update user configuration via CLI
python scripts/config_manager.py update api.port 8080
python scripts/config_manager.py update model.xgboost.n_estimators 200
python scripts/config_manager.py update logging.level DEBUG
```

## Configuration Sections

### API Configuration (`api`)
- `host`: API server host
- `port`: API server port  
- `debug`: Enable debug mode
- `workers`: Number of worker processes
- `reload`: Enable auto-reload in development

### Search Configuration (`search`)
- `timeout`: Search request timeout
- `max_concurrent_engines`: Maximum concurrent search engines
- `retry_attempts`: Number of retry attempts for failed requests
- `retry_delay`: Delay between retry attempts

### Model Configuration (`model`)
- `cache_dir`: Directory for model storage
- `auto_save`: Automatically save trained models
- `version_control`: Enable model versioning
- `xgboost`: XGBoost-specific parameters

### Training Configuration (`training`)
- `batch_size`: Training batch size
- `validation_split`: Validation data split ratio
- `max_epochs`: Maximum training epochs
- `early_stopping_patience`: Early stopping patience
- `contrastive_learning`: Contrastive learning parameters

### Normalization Configuration (`normalization`)
- `default_method`: Default normalization method
- `outlier_threshold`: Outlier detection threshold
- `clip_values`: Enable value clipping
- `per_engine_params`: Use per-engine normalization parameters

### Logging Configuration (`logging`)
- `level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `format`: Log message format
- `file_logging`: Enable file logging
- `log_file`: Log file path
- `max_file_size`: Maximum log file size
- `backup_count`: Number of backup log files

### Hyperparameter Optimization (`hyperparameter_optimization`)
- `enabled`: Enable hyperparameter optimization
- `method`: Optimization method (grid, random, bayesian)
- `max_trials`: Maximum number of optimization trials
- `timeout`: Optimization timeout
- `search_space`: Parameter search spaces

### Monitoring Configuration (`monitoring`)
- `enabled`: Enable performance monitoring
- `metrics_collection_interval`: Metrics collection interval
- `performance_tracking`: Enable performance tracking
- `memory_profiling`: Enable memory profiling

### Circuit Breaker Configuration (`circuit_breaker`)
- `failure_threshold`: Number of failures before opening circuit
- `recovery_timeout`: Time before attempting recovery
- `expected_exception_types`: Exception types that trigger circuit breaker

## Best Practices

1. **Never commit `user.yaml`** - This file should contain user-specific settings and may include sensitive information
2. **Use environment-specific files** - Create separate configuration files for different environments
3. **Override with environment variables** - Use environment variables for deployment-specific settings
4. **Validate configuration** - Always validate configuration after changes
5. **Document custom settings** - Add comments to explain custom configuration choices

## Programmatic Access

```python
from src.config.settings import config, config_manager

# Access configuration values
api_host = config.api.host
batch_size = config.training.batch_size

# Get configuration with default
timeout = config_manager.get("search.timeout", 30.0)

# Update configuration
config_manager.set("api.debug", True)

# Save user configuration
config_manager.save_user_config({
    "api": {"port": 8080},
    "logging": {"level": "DEBUG"}
})
```