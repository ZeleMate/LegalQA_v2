# Optimized Multi-stage Dockerfile for LegalQA

# Build stage - Install dependencies
FROM python:3.10-slim AS builder

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and use virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir .

# Production stage - Minimal runtime image
FROM python:3.10-slim AS production

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser ./src /app/src
COPY --chown=appuser:appuser ./scripts /app/scripts

# Create directories for data and ensure proper permissions
RUN mkdir -p /app/data/processed /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use uvicorn with single worker to avoid startup issues
CMD ["uvicorn", "src.inference.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]

# Development stage - For development with hot reload
FROM production AS development

# Switch back to root for installing development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir jupyter ipython pytest black isort mypy

# Switch back to appuser
USER appuser

# Development command with hot reload
CMD ["uvicorn", "src.inference.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/src", \
     "--log-level", "debug"]