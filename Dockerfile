# Optimized Multi-stage Dockerfile for LegalQA

# Build stage - Install dependencies
FROM python:3.10-slim AS builder

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment and install dependencies in one layer
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with better caching
WORKDIR /build

# Copy dependency files first for better layer caching
COPY pyproject.toml .

# Install all dependencies in one consolidated RUN command
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    pyarrow==14.0.2 \
    python-dotenv==1.0.0 \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    psycopg2-binary==2.9.9 \
    scikit-learn==1.3.2 \
    numpy==1.24.3 \
    faiss-cpu==1.7.4 \
    sentence-transformers==2.2.2 \
    langchain==0.1.0 \
    langchain-openai==0.0.5 \
    langchain-google-genai==0.0.6 \
    langchain-core==0.1.10 \
    langchain-community==0.0.10 \
    google-genai==1.27.0 \
    pgvector==0.2.3 \
    asyncpg==0.29.0 \
    aioredis==2.0.1 \
    "redis[hiredis]==5.0.1" \
    "sqlalchemy[asyncio]==2.0.23" \
    prometheus-client==0.19.0 \
    aioboto3==12.3.0 \
    cachetools==5.3.2

# Production stage - Minimal runtime image
FROM python:3.10-slim AS production

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create user and set up application in one layer
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/data/processed /app/logs \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser ./src /app/src
COPY --chown=appuser:appuser ./scripts /app/scripts

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