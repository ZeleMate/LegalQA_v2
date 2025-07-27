# Optimized Multi-stage Dockerfile for LegalQA

# Build stage - Install dependencies
FROM python:3.10-slim AS builder

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and use virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with better caching
WORKDIR /build

# Copy dependency files first for better layer caching
COPY pyproject.toml .

# Install build tools first
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies first (smaller packages) - Layer 1
RUN pip install --no-cache-dir \
    pandas \
    pyarrow \
    python-dotenv \
    fastapi \
    uvicorn[standard] \
    psycopg2-binary \
    scikit-learn \
    numpy

# Install ML dependencies (larger packages) - Layer 2
RUN pip install --no-cache-dir \
    faiss-cpu

# Install sentence-transformers separately (very large) - Layer 3
RUN pip install --no-cache-dir \
    sentence-transformers

# Install LangChain dependencies - Layer 4
RUN pip install --no-cache-dir \
    langchain \
    langchain-openai \
    langchain-google-genai \
    langchain-core \
    langchain-community \
    google-genai

# Install remaining dependencies - Layer 5
RUN pip install --no-cache-dir \
    pgvector \
    asyncpg \
    aioredis \
    redis[hiredis] \
    sqlalchemy[asyncio] \
    prometheus-client \
    aioboto3 \
    cachetools

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