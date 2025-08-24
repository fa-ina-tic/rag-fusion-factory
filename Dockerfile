# Multi-stage Dockerfile for RAG Fusion Factory

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY examples/ /app/examples/
COPY docs/ /app/docs/

# Copy startup script and make it executable
COPY scripts/docker_start.sh /app/docker_start.sh
RUN chmod +x /app/docker_start.sh

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV CONFIG_FILE="/app/config/minimal.yaml"
ENV LOG_LEVEL="INFO"
ENV ENVIRONMENT="production"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import asyncio; import sys; sys.path.insert(0, '/app/src'); \
    from src.adapters.registry import get_adapter_registry; \
    registry = get_adapter_registry(); \
    result = asyncio.run(registry.health_check_all()); \
    healthy = sum(1 for h in result.values() if h); \
    exit(0 if healthy > 0 else 1)" || exit 1

# Switch to non-root user
USER appuser

# Expose port (if running API server)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/docker_start.sh"]

# Default command
CMD []