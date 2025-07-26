# LegalQA: High-Performance RAG System for Legal Documents

[![CI](https://github.com/ZeleMate/LegalQA_v2/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeleMate/LegalQA_v2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LegalQA is a production-ready, high-performance Retrieval-Augmented Generation (RAG) system designed to answer complex legal questions based on a large corpus of documents. It is built with a modern, scalable architecture, fully containerized with Docker, and ready for deployment.

## 🏗️ Architecture

The system is designed for performance and scalability, leveraging asynchronous processing, multi-level caching, and a robust database backend.

```mermaid
graph TD
    subgraph "User Interaction"
        U[User]
    end

    subgraph "Application Layer (Docker)"
        A[FastAPI App]
        C[Redis Cache]
        DB[(PostgreSQL + pgvector)]
        F[FAISS Index]
    end

    subgraph "External Services"
        LLM[OpenAI LLM]
    end
    
    subgraph "Monitoring"
        P[Prometheus]
    end

    U -- "HTTP Request" --> A
    A -- "Caches Queries/Results" --> C
    A -- "Retrieves Text & Vectors" --> DB
    A -- "Finds Similar Chunks" --> F
    A -- "Generates & Reranks" --> LLM
    A -- "Exports Metrics" --> P
```

## ✨ Key Features

- **High-Performance API**: Built with **FastAPI** for asynchronous, high-throughput request handling.
- **Multi-Level Caching**: In-memory and **Redis** cache for rapid responses to repeated queries and reduced API costs.
- **Efficient Document Retrieval**: A combination of **FAISS** for fast vector similarity search and a **PostgreSQL** database (with `pgvector`) for storing and retrieving text chunks.
- **Advanced Reranking**: Utilizes a secondary LLM call to rerank retrieved documents, significantly improving the relevance of the context provided to the final generation model.
- **Containerized & Reproducible**: Fully containerized with **Docker** and managed with **Docker Compose** for consistent development and production environments.
- **Performance Monitoring**: Integrated **Prometheus** endpoint (`/metrics`) for real-time monitoring of application performance, request latency, and cache hit rates.

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) & [Docker Compose](https://docs.docker.com/compose/install/)
- An OpenAI/GEMINI API key
- `make` command-line utility (optional, but recommended)

### Installation

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/ZeleMate/LegalQA_v2.git
    cd LegalQA_v2
    ```

2.  **Configure Environment**
    Create a `.env` file in the project root. You can copy the structure from the example below. This file stores your API keys and other configuration variables. **It is ignored by Git.**

    ```env
    # --- API Keys ---
    OPENAI_API_KEY="sk-..."

    # --- Database Configuration ---
    # These are used by the app to connect to the 'db' service inside Docker.
    # The POSTGRES_HOST must be the service name ('db').
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=admin
    POSTGRES_DB=legalqa
    POSTGRES_HOST=db
    POSTGRES_PORT=5432

    # --- Redis Configuration ---
    REDIS_HOST=redis
    REDIS_PORT=6379
    
    # --- Data File Configuration ---
    # The name of your main parquet file located in the ./data directory
    PARQUET_FILENAME=all_data.parquet
    ```

3.  **Place Your Data**
    -   Place your Parquet data file (e.g., `all_data.parquet`) into the `data/` directory.
    -   The filename must match `PARQUET_FILENAME` in your `.env` file.
    -   Ensure your data conforms to the schema described in the [Data Schema](#-data-schema) section.

## Usage

The `Makefile` provides convenient commands for managing the application lifecycle for both development and production.

### Development Environment

The development environment uses a small sample of your data for a fast startup and enables hot-reloading for code changes.

1.  **Setup Development Data & Services:**
    This command creates a sample dataset, builds a local FAISS index, and starts the services.
    ```sh
    make dev-setup
    ```

2.  **Access the API:**
    The API will be available at `http://localhost:8000`. The interactive Swagger UI documentation can be found at `http://localhost:8000/docs`.

3.  **Stop the Development Environment:**
    ```sh
    make dev-down
    ```

### Production Environment

The production environment uses the entire dataset and is optimized for performance.

1.  **Build and Setup Production:**
    This command builds the production Docker images and populates the database with the full dataset.
    ```sh
    make prod-setup
    ```

2.  **Start Production Services:**
    ```sh
    make prod-up
    ```

3.  **Stop the Production Environment:**
    ```sh
    make prod-down
    ```

### Clean Up

To stop all containers and remove all associated volumes (including database data), run:
```sh
make clean
```

## API Endpoints

-   `POST /ask`: The main endpoint for asking questions.
-   `GET /health`: A detailed health check that reports the status of the application, database, and cache.
-   `GET /stats`: Provides real-time performance statistics.
-   `GET /metrics`: Exposes performance metrics in a format compatible with Prometheus.
-   `POST /clear-cache`: Clears the Redis cache.

**Example `curl` Request:**
```sh
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Milyen ítéleteket hozott a Kúria 2024-ben?",
       "use_cache": true
     }'
```

## 🧪 Testing

The project includes a comprehensive test suite and quality checks.

### Unit and Integration Tests

Run the test suite:

```bash
pytest tests/ -v
```

### Code Quality Checks

The project uses multiple tools to ensure code quality:

```bash
# Format checking (CI mode)
black --check src/ tests/
isort --check-only src/ tests/

# Linting
flake8 src/ tests/
mypy src/
bandit -r src/

# Pre-commit hooks (install with: pre-commit install)
pre-commit run --all-files
```

### Performance Testing

The project includes k6-based smoke performance tests to validate system stability under concurrent load.

#### Local Performance Testing

```bash
# Install k6 (if not already installed)
brew install k6  # macOS
# or download from https://k6.io/docs/getting-started/installation/

# Run smoke test against local development server
k6 run perf/smoke.js

# Run with custom URL
BASE_URL=http://your-app-url:8000 k6 run perf/smoke.js

# Run stress test
k6 run --vus 10 --duration 60s perf/smoke.js
```

#### CI/CD Performance Testing

The GitHub Actions workflow automatically runs performance tests on pull requests and main/develop branches using the integrated CI workflow.

**Requirements:**
- Set `APP_URL_STAGING` secret in GitHub repository settings
- Ensure the staging environment is accessible from GitHub Actions

**Test Coverage:**
- Health check endpoint (`/health`)
- Metrics endpoint (`/metrics`) 
- Stats endpoint (`/stats`)
- QA endpoint (`/ask`)
- Cache management (`/clear-cache`)

**Thresholds:**
- 95% response time < 5 seconds (noise-tolerant for smoke test)
- Error rate < 5% (noise-tolerant for smoke test)
- All endpoints return 200 status codes

### CI/CD Pipeline

The project uses a robust CI/CD pipeline with the following features:

- **Fast feedback**: Lint and tests run on every push to any branch
- **Heavy jobs**: Docker build and performance tests only on PR/main/develop
- **Smart skipping**: Documentation changes and `[skip ci]` commits bypass CI
- **Concurrency control**: New commits cancel previous runs
- **Format checking**: Black and isort run in check mode
- **Type checking**: Strict mypy configuration with specific overrides
- **Security scanning**: Bandit for security vulnerabilities
- **Test coverage**: Minimum 80% coverage requirement
- **Docker build**: Multi-stage builds with layer caching
- **Performance testing**: k6 smoke tests on PR/main/develop

#### CI Policy

- **Every push**: Lint, format check, unit tests (fast feedback)
- **PR/main/develop**: Full pipeline including Docker build and performance tests
- **Documentation**: Automatically skipped (docs/, *.md files)
- **Skip CI**: Use `[skip ci]` in commit message to bypass all checks

### Make Commands

You can also run specific test types using make:

```bash
make test              # Run all tests
make test-functionality # Run functionality tests
make test-performance  # Run performance tests
make lint              # Run all linting tools
```

## 📋 Data Schema

To use your own data, you must provide a Parquet file with the following columns:

-   `chunk_id` (string): A unique identifier for each text chunk.
-   `doc_id` (string): A unique identifier for the parent document.
-   `text` (string): The text content of the chunk.
-   `embedding` (binary/vector): The vector embedding of the `text`.

---

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for more information.
