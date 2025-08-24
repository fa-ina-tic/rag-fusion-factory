#!/bin/bash
"""
Docker startup script for RAG Fusion Factory.

This script handles Docker container initialization and configuration
for different deployment scenarios.
"""

set -e

# Default values
CONFIG_FILE="${CONFIG_FILE:-/app/config/default.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a service is available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    log "Waiting for $service_name at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log "$service_name is available"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not yet available"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log "ERROR: $service_name at $host:$port is not available after $max_attempts attempts"
    return 1
}

# Function to substitute environment variables in config files
substitute_env_vars() {
    local config_file=$1
    
    if [ -f "$config_file" ]; then
        log "Substituting environment variables in $config_file"
        
        # Create a temporary file for substitution
        local temp_file=$(mktemp)
        
        # Substitute environment variables
        envsubst < "$config_file" > "$temp_file"
        
        # Replace original file
        mv "$temp_file" "$config_file"
        
        log "Environment variable substitution completed"
    else
        log "WARNING: Configuration file $config_file not found"
    fi
}

# Function to validate configuration
validate_config() {
    local config_file=$1
    
    log "Validating configuration file: $config_file"
    
    if python3 /app/scripts/start_fusion_factory.py --validate "$config_file"; then
        log "Configuration validation successful"
        return 0
    else
        log "ERROR: Configuration validation failed"
        return 1
    fi
}

# Function to wait for dependent services
wait_for_dependencies() {
    log "Checking for dependent services..."
    
    # Check for any configured search engines
    # Users should set these environment variables if they have external search engines
    if [ -n "$ELASTICSEARCH_HOST" ] && [ -n "$ELASTICSEARCH_PORT" ]; then
        wait_for_service "$ELASTICSEARCH_HOST" "$ELASTICSEARCH_PORT" "Elasticsearch"
    fi
    
    if [ -n "$SOLR_HOST" ] && [ -n "$SOLR_PORT" ]; then
        wait_for_service "$SOLR_HOST" "$SOLR_PORT" "Solr"
    fi
    
    if [ -n "$OPENSEARCH_HOST" ] && [ -n "$OPENSEARCH_PORT" ]; then
        wait_for_service "$OPENSEARCH_HOST" "$OPENSEARCH_PORT" "OpenSearch"
    fi
    
    if [ -n "$DATABASE_HOST" ] && [ -n "$DATABASE_PORT" ]; then
        wait_for_service "$DATABASE_HOST" "$DATABASE_PORT" "Database"
    fi
    
    # If no external services are configured, we'll use mock adapters
    if [ -z "$ELASTICSEARCH_HOST" ] && [ -z "$SOLR_HOST" ] && [ -z "$OPENSEARCH_HOST" ] && [ -z "$DATABASE_HOST" ]; then
        log "No external search engines configured, will use mock adapters"
    fi
    
    log "Dependency checks completed"
}

# Function to setup logging
setup_logging() {
    log "Setting up logging with level: $LOG_LEVEL"
    
    # Create logs directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set log level environment variable
    export PYTHONPATH="/app/src:$PYTHONPATH"
}

# Function to run health checks
run_health_checks() {
    log "Running initial health checks..."
    
    # Run health checks using the fusion factory
    if python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app/src')
from src.adapters.registry import get_adapter_registry
from src.config.settings import load_config

async def health_check():
    try:
        config = load_config('$CONFIG_FILE')
        registry = get_adapter_registry()
        
        # Create adapters from config
        adapters_config = config.get('engines', [])
        for adapter_config in adapters_config:
            adapter = registry.create_adapter(
                engine_type=adapter_config['engine_type'],
                engine_id=adapter_config['engine_id'],
                config=adapter_config['config'],
                timeout=adapter_config.get('timeout', 30.0)
            )
        
        # Run health checks
        health_results = await registry.health_check_all()
        healthy_count = sum(1 for is_healthy in health_results.values() if is_healthy)
        total_count = len(health_results)
        
        print(f'Health check: {healthy_count}/{total_count} adapters healthy')
        
        if healthy_count == 0:
            print('ERROR: No adapters are healthy')
            sys.exit(1)
        
        return True
    except Exception as e:
        print(f'Health check failed: {str(e)}')
        sys.exit(1)

asyncio.run(health_check())
"; then
        log "Health checks passed"
    else
        log "ERROR: Health checks failed"
        exit 1
    fi
}

# Main execution
main() {
    log "Starting RAG Fusion Factory Docker container"
    log "Environment: $ENVIRONMENT"
    log "Configuration file: $CONFIG_FILE"
    log "Log level: $LOG_LEVEL"
    
    # Setup logging
    setup_logging
    
    # Substitute environment variables in configuration
    substitute_env_vars "$CONFIG_FILE"
    
    # Validate configuration
    if ! validate_config "$CONFIG_FILE"; then
        exit 1
    fi
    
    # Wait for dependent services
    wait_for_dependencies
    
    # Run health checks
    run_health_checks
    
    # Start the fusion factory
    log "Starting RAG Fusion Factory..."
    
    case "$ENVIRONMENT" in
        "development")
            log "Starting in development mode"
            exec python3 /app/scripts/start_fusion_factory.py --dev --log-level "$LOG_LEVEL"
            ;;
        "testing")
            log "Starting in testing mode"
            exec python3 /app/scripts/start_fusion_factory.py --test --log-level "$LOG_LEVEL"
            ;;
        "production"|*)
            log "Starting with configuration: $CONFIG_FILE"
            exec python3 /app/scripts/start_fusion_factory.py --config "$CONFIG_FILE" --log-level "$LOG_LEVEL"
            ;;
    esac
}

# Handle signals for graceful shutdown
trap 'log "Received shutdown signal, stopping..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"