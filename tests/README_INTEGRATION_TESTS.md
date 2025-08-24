# RAG Fusion Factory - Comprehensive Integration Tests

This document describes the comprehensive integration test suite for the RAG Fusion Factory system. These tests validate the complete system functionality across different scenarios and configurations.

## Test Structure

### 1. Complete Fusion Pipeline Tests (`TestCompleteFusionPipeline`)

Tests the end-to-end fusion pipeline with multiple search engines:

- **Complete pipeline execution**: Validates query processing from input to ranked results
- **Engine failure resilience**: Tests graceful degradation when some engines fail
- **Custom weight handling**: Verifies custom fusion weights are properly applied
- **Score normalization**: Ensures different scoring scales are normalized correctly
- **Duplicate handling**: Tests merging of duplicate documents from multiple engines

### 2. Adapter Functionality Tests (`TestAdapterFunctionality`)

Tests different search engine adapter types and configurations:

- **Adapter creation from config**: Validates adapter instantiation from configuration
- **Search interface consistency**: Ensures all adapters implement consistent interfaces
- **Health check consistency**: Tests health check functionality across adapter types
- **Configuration validation**: Validates proper error handling for invalid configurations
- **Timeout handling**: Tests adapter timeout behavior

### 3. Performance Benchmarking Tests (`TestPerformanceBenchmarking`)

Comprehensive performance testing for different configurations:

- **Single engine performance**: Benchmarks individual adapter performance
- **Concurrent engine performance**: Tests performance under concurrent load
- **Load performance**: Validates system behavior under high query volume
- **Memory usage stability**: Tests memory stability during sustained operations

### 4. Training and Optimization Tests (`TestTrainingAndOptimization`)

Tests machine learning components and training workflows:

- **Weight predictor training**: Tests XGBoost model training and prediction
- **Contrastive learning training**: Validates contrastive learning optimization
- **Complete training pipeline**: Tests end-to-end training workflow
- **Hyperparameter optimization**: Tests automated hyperparameter tuning

### 5. Configuration-Driven Integration Tests (`TestConfigurationDrivenIntegration`)

Tests configuration-based system initialization and management:

- **YAML configuration loading**: Tests loading and using YAML configuration files
- **Dynamic configuration updates**: Tests runtime configuration changes
- **Environment-specific configurations**: Tests different configurations for dev/prod environments

### 6. Error Recovery Integration Tests (`TestErrorRecoveryIntegration`)

Tests system resilience and error handling:

- **Graceful degradation**: Tests system behavior when engines fail
- **Timeout handling**: Tests timeout behavior in integration scenarios
- **Circuit breaker integration**: Tests circuit breaker pattern implementation

### 7. Monitoring Integration Tests (`TestMonitoringIntegration`)

Tests monitoring and observability features:

- **Performance monitoring**: Tests performance metrics collection
- **Health check integration**: Tests system-wide health monitoring

### 8. Real-World Scenarios Tests (`TestRealWorldScenarios`)

Tests realistic usage patterns and edge cases:

- **Large result set handling**: Tests performance with large document collections
- **Multilingual content handling**: Tests support for multiple languages
- **Concurrent user simulation**: Tests system behavior under concurrent user load

## Running the Tests

### Prerequisites

1. **Python Dependencies**: Install required packages:
   ```bash
   pip install pytest pytest-asyncio numpy psutil pyyaml
   ```

2. **Optional Dependencies** (for real engine tests):
   ```bash
   pip install docker requests aiohttp
   ```

### Running All Integration Tests

```bash
# Run all comprehensive integration tests
python -m pytest tests/test_comprehensive_integration.py -v

# Run with detailed output
python -m pytest tests/test_comprehensive_integration.py -v --tb=long

# Run specific test class
python -m pytest tests/test_comprehensive_integration.py::TestCompleteFusionPipeline -v
```

### Running with Real Search Engines

To run tests with real search engines (requires Docker):

```bash
# Start Docker services first
docker-compose up -d elasticsearch solr

# Run real engine integration tests
SKIP_REAL_ENGINE_TESTS=false python -m pytest tests/test_real_engine_integration.py -v

# Run performance benchmarks
python -m pytest tests/test_performance_benchmarks.py -v
```

### Using the Test Runner

Use the comprehensive test runner for detailed reporting:

```bash
# Run all integration tests with report
python tests/run_integration_tests.py --report-file integration_report.json

# Include real engine tests
python tests/run_integration_tests.py --include-real-engines

# Include performance benchmarks
python tests/run_integration_tests.py --include-performance

# Fast mode (skip slow tests)
python tests/run_integration_tests.py --fast
```

## Test Configuration

### Environment Variables

- `SKIP_REAL_ENGINE_TESTS`: Set to `false` to enable real engine tests (default: `true`)
- `TEST_TIMEOUT`: Override default test timeout (default: 300 seconds)
- `LOG_LEVEL`: Set logging level for tests (default: `INFO`)

### Docker Services

For real engine integration tests, ensure these services are running:

```yaml
# docker-compose.yml excerpt
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false

  solr:
    image: solr:9.2
    ports:
      - "8983:8983"
    command: solr-precreate test_integration

  opensearch:
    image: opensearchproject/opensearch:2.8.0
    ports:
      - "9201:9200"
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
```

## Test Data

### Mock Data

Tests use in-memory mock adapters with synthetic data:

- **Sample documents**: Various document types for testing different scenarios
- **Multilingual content**: Documents in multiple languages
- **Large datasets**: Configurable document collections for performance testing

### Real Engine Data

When testing with real engines, tests automatically set up:

- **Test indices/cores**: Temporary indices with test documents
- **Sample documents**: Standardized test documents across engines
- **Cleanup**: Automatic cleanup after test completion

## Performance Expectations

### Latency Benchmarks

- **Single engine queries**: < 100ms for mock adapters
- **Multi-engine fusion**: < 500ms for 3 engines
- **Large result sets**: < 1s for 1000+ documents

### Throughput Benchmarks

- **Concurrent queries**: > 10 queries/second
- **Sustained load**: > 5 queries/second over 30 seconds
- **Memory stability**: < 100MB growth over 100 queries

### Success Rate Expectations

- **Normal conditions**: > 95% success rate
- **With engine failures**: > 70% success rate
- **High concurrency**: > 80% success rate

## Troubleshooting

### Common Issues

1. **Docker services not available**:
   ```bash
   # Check service status
   docker-compose ps
   
   # Restart services
   docker-compose restart elasticsearch solr
   ```

2. **Memory issues during performance tests**:
   ```bash
   # Run with memory profiling
   python -m pytest tests/test_performance_benchmarks.py --profile-memory
   ```

3. **Timeout issues**:
   ```bash
   # Increase timeout
   TEST_TIMEOUT=600 python -m pytest tests/test_comprehensive_integration.py
   ```

### Debug Mode

Enable debug logging for detailed test execution:

```bash
LOG_LEVEL=DEBUG python -m pytest tests/test_comprehensive_integration.py -v -s
```

## Continuous Integration

### GitHub Actions

Example CI configuration:

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      elasticsearch:
        image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
        ports:
          - 9200:9200
        options: >-
          --health-cmd "curl http://localhost:9200/_cluster/health"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 10

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      
      - name: Run integration tests
        run: |
          python tests/run_integration_tests.py --include-real-engines --report-file ci_report.json
      
      - name: Upload test report
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-report
          path: ci_report.json
```

## Contributing

When adding new integration tests:

1. **Follow naming conventions**: Use descriptive test names with `test_` prefix
2. **Add proper documentation**: Include docstrings explaining test purpose
3. **Use appropriate fixtures**: Leverage existing fixtures or create reusable ones
4. **Handle cleanup**: Ensure tests clean up resources properly
5. **Add performance expectations**: Include assertions for performance requirements
6. **Test error conditions**: Include negative test cases for error handling

## Requirements Coverage

These integration tests cover the following requirements:

- **2.1**: Multi-engine query processing and result collection
- **2.2**: Result format normalization and metadata extraction
- **2.4**: Fault tolerance and timeout handling
- **3.1**: Contrastive learning for weight optimization
- **4.1**: XGBoost-based weight prediction
- **5.1**: Score normalization across engines
- **6.1**: Hyperparameter optimization capabilities

For detailed requirement mapping, see the individual test docstrings and the `_Requirements:_` annotations in each test method.