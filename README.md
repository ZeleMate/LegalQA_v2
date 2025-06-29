# LegalQA: Advanced RAG for Hungarian Legal Documents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, Retrieval-Augmented Generation (RAG) system designed to answer complex legal questions based on a corpus of Hungarian court decisions. This project is containerized using Docker and managed with Docker Compose for robust and reproducible deployment.

## Architecture

The system utilizes a multi-container Docker setup orchestrated by `docker-compose.yml`:

- **`app` service:** A FastAPI application that serves the QA model, handles API requests, and contains all the core logic for the RAG pipeline.
- **`db` service:** A PostgreSQL database with the `pgvector` extension to store document chunks and their vector embeddings efficiently.
- **`legalqa_net` network:** A dedicated bridge network for communication between the `app` and `db` services.
- **`postgres_data` volume:** A named volume to persist the PostgreSQL database data across container restarts.

```mermaid
graph TD
    subgraph "Docker Environment"
        subgraph "legalqa_app container"
            direction LR
            A[FastAPI Server]
        end

        subgraph "legalqa_db container"
            direction LR
            B[PostgreSQL + pgvector]
        end

        A -- "Queries data over legalqa_net" --> B
    end

    C[User] -- "HTTP Request" --> A
    D[Developer] -- "make <command>" --> E{Makefile}
    E -- "docker-compose up/down/exec" --> subgraph "Docker Environment"
    end
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
- A `.parquet` file containing your documents and embeddings. See the `Data Schema` section for details.
- An OpenAI API key.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/LegalQA_v2.git
    cd LegalQA_v2
    ```

2.  **Prepare your data:**
    - Place your processed Parquet file (e.g., `processed_documents_with_embeddings.parquet`) inside the `data/processed/` directory.

3.  **Set up environment variables:**
    - Create a `.env` file in the project root. You can copy the structure from the example below.
    - **This single `.env` file controls both the Docker environment and local notebook execution.**

    ```env
    # --- Data Source & File Paths ---
    # Use 'postgres' for the Dockerized app, 'local_parquet' for notebook testing.
    DATA_SOURCE=local_parquet

    # Paths for local notebook execution
    PARQUET_PATH=data/processed/processed_documents_with_embeddings.parquet
    FAISS_INDEX_PATH=data/processed/faiss_index.bin
    ID_MAPPING_PATH=data/processed/id_mapping.pkl

    # --- API Keys ---
    # Replace with your actual OpenAI API key
    OPENAI_API_KEY="sk-..."

    # --- Database Configuration for Docker Compose ---
    # The 'app' service will use these to connect to the 'db' service.
    # 'POSTGRES_HOST' MUST be the service name ('db') for container networking.
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=admin # Change to a secure password
    POSTGRES_DB=legalqa
    POSTGRES_HOST=db
    POSTGRES_PORT=5432
    ```

## Usage and Workflows

This project supports two primary workflows: full application deployment with Docker and local development/testing in Jupyter notebooks. All Docker-related tasks are managed via the `Makefile`.

### 1. Running the Full Application (Docker)

This is the standard way to run the entire system.

1.  **Set `DATA_SOURCE` for Docker:**
    -   In your `.env` file, ensure the `DATA_SOURCE` is set to `postgres`.
        ```
        DATA_SOURCE=postgres
        ```

2.  **Start all services:**
    -   This command builds the images if they don't exist and starts the `app` and `db` containers in the background.
        ```sh
        make up
        ```

3.  **Build the database:**
    -   With the services running, execute the database build script. This command runs a script inside the `app` container that reads the Parquet file, populates the PostgreSQL database, and builds a FAISS index.
    -   **Warning:** This is a one-time, resource-intensive process.
        ```sh
        make build-db
        ```

4.  **Interact with the API:**
    -   The FastAPI server is now available at `http://localhost:8000`. You can access the interactive documentation at `http://localhost:8000/docs`.
    -   **Example `curl` request:**
        ```sh
        curl -X POST "http://localhost:8000/ask" \
             -H "Content-Type: application/json" \
             -d '{"question": "Mi a bűnszervezet fogalma a Btk. szerint?"}'
        ```

5.  **View logs or stop services:**
    ```sh
    make logs   # Tail the logs from both containers
    make down   # Stop and remove all containers and networks
    ```

### 2. Local Development (Jupyter Notebook)

For experimentation and model development, you can run the notebooks without a live database connection.

1.  **Set `DATA_SOURCE` for local testing:**
    -   In your `.env` file, set the `DATA_SOURCE` to `local_parquet`.
        ```
        DATA_SOURCE=local_parquet
        ```

2.  **Install local dependencies:**
    -   The project uses `pyproject.toml` for dependency management. Create a virtual environment and install the necessary packages.
        ```sh
        python -m venv venv
        source venv/bin/activate
        pip install -e ".[notebook]"
        ```
        *This installs the core dependencies plus the extra packages for notebook development (`matplotlib`, `seaborn`, `jupyter`).*

3.  **Run Jupyter:**
    ```sh
    jupyter notebook
    ```
    -   Now you can open and run the notebooks in the `notebooks/` directory. The retriever will automatically use the local Parquet file specified in your `.env` file, bypassing the need for a database connection.

## Data Schema

To use your own data, you must provide a Parquet file with the following columns:

-   `chunk_id`: A unique identifier for each text chunk.
-   `doc_id`: A unique identifier for the parent document.
-   `text`: The text content of the chunk.
-   `embedding`: The vector embedding of the `text`.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
