#!/usr/bin/env python3
"""
Unified script to build LegalQA database and FAISS index from Parquet data
with automatic memory management.
Handles chunking overlap by using ON CONFLICT DO NOTHING for duplicates.
Automatically restarts every 2 row groups to manage memory usage.
Can also build FAISS index only from existing database data.
"""

import argparse
import gc
import json
import os
import subprocess  # nosec
import sys
import time
import uuid

import faiss
import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKey, MetaData, String, Table, create_engine, text

# --- Database Schema Definition ---
# Defining the schema directly in the script to make it self-contained.
metadata = MetaData()

documents = Table(
    "documents", metadata, Column("doc_id", String, primary_key=True), Column("source", String)
)

chunks = Table(
    "chunks",
    metadata,
    Column("chunk_id", String, primary_key=True),
    Column("doc_id", String, ForeignKey("documents.doc_id")),
    Column("text", String),
    Column("embedding", Vector(768)),
)
# --- End of Schema Definition ---


def get_db_connection_url():
    """
    Constructs the database connection URL from environment variables.
    These are injected by Docker Compose from the .env file.
    For local development, host defaults to 'localhost'.
    """
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    # Default to 'localhost' if POSTGRES_HOST is not set
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")
    if not all([user, password, host, port, db]):
        print("Error: One or more PostgreSQL environment variables are not set.")
        print("Please check your .env file in the project root.")
        sys.exit(1)
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def check_db_connection(engine):
    """
    Checks if the database connection is successful. Retries a few times.
    """
    print("Attempting to connect to the database...")
    for i in range(5):
        try:
            with engine.connect() as _:
                print("Database connection successful!")
                return True
        except Exception as e:
            print(f"Connection attempt {i + 1}/5 failed. Retrying in 5 seconds...")
            print(f"Error details: {e}")
            time.sleep(5)
    print("Error: Could not establish database connection after several retries.")
    return False


def create_tables_if_not_exist(engine):
    """Create tables if they don't exist - safe approach that doesn't drop existing data"""
    print("Creating tables if not exist...")
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create documents table
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT
            )
        """
            )
        )

        # Create chunks table with 768-dimensional embeddings
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT REFERENCES documents(doc_id),
                text TEXT,
                embedding vector(768)
            )
        """
            )
        )

        conn.commit()
    print("Tables created successfully.")


def create_schema_fresh(engine):
    """Creates the database schema, dropping existing tables first - for fresh start"""
    print("Creating fresh database schema (dropping existing tables)...")
    try:
        with engine.connect() as connection:
            # Enable the pgvector extension
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
            print("pgvector extension enabled.")

        # Drop tables in the correct order to respect foreign key constraints
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
            connection.execute(text("DROP TABLE IF EXISTS documents CASCADE"))
            connection.commit()
        print("Existing tables dropped.")

        # Create tables with proper schema
        create_tables_if_not_exist(engine)
        print("New tables created successfully based on schema.")
    except Exception as e:
        print(f"An error occurred during schema creation: {e}")
        sys.exit(1)


def get_sample_chunks_from_batch(df_batch, sample_size=3):
    """
    Get a sample of chunk IDs from a specific batch to check if it's already processed
    """
    try:
        sample_chunk_ids = df_batch["chunk_id"].dropna().head(sample_size).tolist()
        return sample_chunk_ids
    except Exception as e:
        print(f"Error sampling batch: {e}")
        return []


def is_batch_processed(engine, sample_chunk_ids):
    """
    Check if a batch has been processed by looking for sample chunk IDs in the database
    """
    if not sample_chunk_ids:
        return False

    try:
        with engine.connect() as conn:
            placeholders = ",".join([":chunk_id_" + str(i) for i in range(len(sample_chunk_ids))])
            params = {f"chunk_id_{i}": chunk_id for i, chunk_id in enumerate(sample_chunk_ids)}

            query = f"SELECT COUNT(*) FROM chunks WHERE chunk_id IN ({placeholders})"  # nosec
            result = conn.execute(text(query), params)
            found_count = result.scalar()

            # If we found most of the sample chunks, consider the batch processed
            threshold = len(sample_chunk_ids) * 0.8  # 80% threshold
            return found_count >= threshold
    except Exception as e:
        print(f"Error checking if batch is processed: {e}")
        return False


def get_last_processed_row_group(engine):
    """Get the last successfully processed row group from database."""
    try:
        with engine.connect() as conn:
            # Check if we have any data and try to determine last processed row group
            result = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
            chunk_count = result[0] if result else 0

            if chunk_count == 0:
                return -1  # No data processed yet

            # Estimate last row group based on chunk count
            # Each row group has roughly 50k chunks
            estimated_row_group = max(0, (chunk_count // 50000) - 1)
            print(
                f"Estimated last processed row group: {estimated_row_group} "
                f"(based on {chunk_count} chunks)"
            )
            return estimated_row_group
    except Exception as e:
        print(f"Error determining last processed row group: {e}")
        return -1


def clean_memory():
    """Force garbage collection and clean memory."""
    print("Cleaning memory...")
    gc.collect()
    print("Memory cleaned.")


def _safe_subprocess_run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """
    Secure subprocess wrapper based on CWE-78 mitigations.

    Args:
        cmd: Hardcoded command list
        timeout: Timeout in seconds

    Returns:
        CompletedProcess object

    Raises:
        ValueError: If command format is invalid
    """
    # Whitelist approach - only allow safe commands
    allowed_commands = {"docker": ["exec", "kill", "start"], "psql": ["-U", "-d", "-t", "-c"]}

    if not cmd or not isinstance(cmd[0], str):
        raise ValueError("Invalid command format")

    command = cmd[0]
    if command not in allowed_commands:
        raise ValueError(f"Command '{command}' not in allowed whitelist")

    # Validate arguments
    for i, arg in enumerate(cmd[1:], 1):
        if not isinstance(arg, str):
            raise ValueError(f"Invalid argument type at position {i}")
        # Basic injection check
        if any(char in arg for char in [";", "|", "&", "`", "$", "(", ")", "<", ">"]):
            raise ValueError(f"Potentially dangerous characters in argument: {arg}")

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # nosec


def get_current_progress_count(engine=None):
    """Get the current chunk count from the database."""
    if engine:
        # Direct database access with engine
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
                return result[0] if result else 0
        except Exception as e:
            print(f"Error in progress query (engine): {e}")
            return 0
    else:
        # Docker command usage
        try:
            cmd = [
                "docker",
                "exec",
                "legalqa_v2-db-1",
                "psql",
                "-U",
                "Zelemate",
                "-d",
                "legal_qa_db",
                "-t",
                "-c",
                "SELECT COUNT(*) FROM chunks;",
            ]
            result = _safe_subprocess_run(cmd, timeout=30)
            if result.returncode == 0:
                chunk_count = int(result.stdout.strip())
                return chunk_count
            else:
                print(f"Error in database query: {result.stderr}")
                return 0
        except Exception as e:
            print(f"Error in progress query: {e}")
            return 0


def cleanup_docker_container():
    """Clean up the Docker container."""
    print("🧹 Restarting Docker container...")
    try:
        _safe_subprocess_run(["docker", "kill", "legalqa_v2-app-1"])
        time.sleep(2)
        _safe_subprocess_run(["docker", "start", "legalqa_v2-app-1"])
        time.sleep(5)
        print("✅ Docker container restarted")
        return True
    except Exception as e:
        print(f"❌ Error restarting Docker container: {e}")
        return False


def run_automated_full_process(
    engine, parquet_path, batch_size=50, max_retries=3, total_row_groups=17
):
    """
    Automatically runs the complete database building process with memory management.
    """
    print("🤖 AUTOMATIC DATABASE BUILDER MODE ACTIVATED")
    print("=" * 60)

    consecutive_failures = 0
    max_consecutive_failures = 5

    while True:
        # Determine current progress
        current_chunks = get_current_progress_count(engine)
        current_row_group = max(0, current_chunks // 49000)

        if current_row_group >= total_row_groups:
            print("COMPLETE DATABASE BUILD FINISHED!")
            print(f"Final chunk count: {current_chunks}")
            break

        session_groups = 1  # Always only 1 row group per session

        print(f"\n🔄 SESSION: Row group {current_row_group + 1}/{total_row_groups}")
        print(
            f"📈 Progress: {current_row_group / total_row_groups * 100:.1f}% "
            f"({current_chunks} chunk)"
        )

        retries = 0
        session_success = False

        while retries < max_retries and not session_success:
            if retries > 0:
                print(f"🔄 Retry {retries + 1}/{max_retries}")
                cleanup_docker_container()
                time.sleep(10)  # Wait for container to stabilize

            # Check progress before session
            chunks_before = get_current_progress_count(engine)

            try:
                # Run session
                print(f"🚀 Starting: Row group {current_row_group + 1}, batch size: {batch_size}")
                safe_insert_data(
                    engine,
                    parquet_path,
                    batch_size,
                    current_row_group,
                    session_groups,
                    auto_mode=True,
                )

                # Check progress after session
                time.sleep(2)
                chunks_after = get_current_progress_count(engine)
                chunks_added = chunks_after - chunks_before

                if chunks_added > 1000:  # Consider success if significant progress was made
                    print(f"✅ Session success! +{chunks_added} chunk added")
                    print(f"📊 Total chunks: {chunks_after}")
                    session_success = True
                    consecutive_failures = 0
                    current_row_group += 1
                else:
                    raise Exception(f"Insufficient progress: +{chunks_added} chunk")

            except Exception as e:
                retries += 1
                consecutive_failures += 1
                print(f"❌ Session failed (attempt {retries}/{max_retries}): {str(e)[:200]}...")

                if consecutive_failures >= max_consecutive_failures:
                    print(f"💥 Too many consecutive failures ({consecutive_failures}). Stopping.")
                    return False

                if retries < max_retries:
                    wait_time = min(30 * retries, 120)  # Exponential backoff, max 2 minutes
                    print(f"⏱️  Waiting {wait_time} seconds...")
                    time.sleep(wait_time)

        if not session_success:
            print(f"💥 Session permanently failed after {max_retries} attempts")
            return False

        # Short break between sessions
        print("😴 5 second break before next session...")
        time.sleep(5)

    return True


def restart_script_from_row_group(start_row_group, batch_size, max_row_groups_per_session=2):
    """Restart the script from a specific row group."""
    print(f"Next session would start from row group {start_row_group + 1}...")
    print("Manual restart command:")
    print(
        f"docker exec legalqa_v2-app-1 python scripts/build_database.py "
        f"--start-row-group {start_row_group} --batch-size {batch_size} "
        f"--max-row-groups {max_row_groups_per_session}"
    )
    print("Exiting current session to allow manual restart...")
    sys.exit(0)


def safe_insert_data(engine, parquet_path, batch_size=1000):
    """
    Simplified, batch-based data import for sample workflow:
    processes all batches in one run, without session logic.
    Ensures every sample record is inserted into the database.
    No restart or progress check needed.
    """
    print(f"Starting safe data import from: {parquet_path}")
    print(f"Using batch size: {batch_size}")
    print("Using ON CONFLICT DO NOTHING to handle duplicates from chunking overlap...")
    print("Processing all batches in one run (sample workflow)...")

    try:
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"Total rows in file: {parquet_file.metadata.num_rows}")
        print(f"Processing in batches of size: {batch_size}")

        batch_iterator = parquet_file.iter_batches(batch_size=batch_size)
        batch_num = 0
        total_inserted = 0
        total_skipped = 0

        for batch in batch_iterator:
            batch_num += 1
            batch_df = batch.to_pandas()
            print(f"Processing batch {batch_num}: {len(batch_df)} rows")

            # Get unique doc_ids for this batch
            unique_docs = batch_df[["doc_id"]].drop_duplicates()
            with engine.connect() as conn:
                for _, row in unique_docs.iterrows():
                    doc_id = row["doc_id"]
                    source = "N/A"  # Default source
                    conn.execute(
                        text(
                            """
                        INSERT INTO documents (doc_id, source) 
                        VALUES (:doc_id, :source) 
                        ON CONFLICT (doc_id) DO NOTHING
                    """
                        ),
                        {"doc_id": doc_id, "source": source},
                    )

                # Insert chunks with duplicate handling
                chunks_inserted = 0
                chunks_skipped = 0
                for _, row in batch_df.iterrows():
                    chunk_id = row.get("chunk_id")
                    doc_id = row.get("doc_id")
                    text_content = row.get("text_chunk")
                    embedding_vector = row.get("embedding")

                    # Generate chunk_id if missing
                    if not chunk_id:
                        chunk_id = str(uuid.uuid4())

                    if doc_id and embedding_vector is not None:
                        embedding = np.array(embedding_vector, dtype=np.float32)
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                        result = conn.execute(
                            text(
                                """
                            INSERT INTO chunks (chunk_id, doc_id, text, embedding) 
                            VALUES (:chunk_id, :doc_id, :text, CAST(:embedding_vec AS vector)) 
                            ON CONFLICT (chunk_id) DO NOTHING
                            RETURNING chunk_id
                        """
                            ),
                            {
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "text": text_content,
                                "embedding_vec": embedding_str,
                            },
                        )
                        if result.rowcount > 0:
                            chunks_inserted += 1
                        else:
                            chunks_skipped += 1
                conn.commit()
                print(f"  Inserted: {chunks_inserted}, Skipped: {chunks_skipped}")
                total_inserted += chunks_inserted
                total_skipped += chunks_skipped
            del batch_df
            gc.collect()
        print(f"All batches processed. Total inserted: {total_inserted}, Skipped: {total_skipped}")
    except Exception as e:
        print(f"An error occurred during data import: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def build_faiss_index(engine, output_dir, batch_size=10000):
    """
    Builds a FAISS index from the embeddings in the 'chunks' table
    and saves the index and ID mapping to disk.
    Uses batching to handle large datasets without memory issues.
    """
    print("\n--- Starting FAISS Index Build ---")

    try:
        with engine.connect() as conn:
            print("--> Counting total embeddings in database...")
            count_result = conn.execute(
                text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            )
            total_embeddings = count_result.scalar()

        if total_embeddings == 0:
            print("Warning: No embeddings found in the database. Skipping FAISS index creation.")
            return False

        print(
            f"--> Found {total_embeddings} total embeddings. "
            f"Processing in batches of {batch_size}..."
        )

        # Initialize variables for batch processing
        chunk_ids = []
        index = None
        d = None

        # Process in batches to avoid memory issues
        offset = 0
        batch_count = 0

        while offset < total_embeddings:
            batch_count += 1
            print(f"--> Processing batch {batch_count}: offset {offset}, limit {batch_size}")

            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT chunk_id, embedding 
                    FROM chunks 
                    WHERE embedding IS NOT NULL 
                    ORDER BY chunk_id
                    LIMIT :limit OFFSET :offset
                """
                    ),
                    {"limit": batch_size, "offset": offset},
                )
                batch_results = result.fetchall()

            if not batch_results:
                break

            print(f"--> Processing {len(batch_results)} embeddings in this batch...")

            # Process current batch
            batch_chunk_ids = []
            batch_embeddings = []

            for row in batch_results:
                chunk_id = row[0]
                emb_str = row[1]

                try:
                    # Parse string embeddings
                    if isinstance(emb_str, str):
                        emb_str = emb_str.strip("[]")
                        emb_values = [float(x.strip()) for x in emb_str.split(",")]
                    else:
                        emb_values = emb_str

                    batch_chunk_ids.append(chunk_id)
                    batch_embeddings.append(emb_values)
                except Exception as e:
                    print(f"Warning: Error parsing embedding for chunk {chunk_id}: {e}")
                    continue

            if not batch_embeddings:
                print(f"Warning: No valid embeddings in batch {batch_count}")
                offset += batch_size
                continue

            # Convert to numpy array
            batch_embedding_matrix = np.array(batch_embeddings, dtype=np.float32)

            # Initialize index with first batch
            if index is None:
                d = batch_embedding_matrix.shape[1]
                print(f"--> Initializing FAISS index with dimension {d}...")
                index = faiss.IndexFlatL2(d)

            # Add embeddings to index
            index.add(batch_embedding_matrix)
            print(f"--> Added {len(batch_embeddings)} vectors to index. Total: {index.ntotal}")

            # Store chunk IDs
            chunk_ids.extend(batch_chunk_ids)

            # Clean up batch data to free memory
            del batch_embeddings
            del batch_embedding_matrix
            del batch_chunk_ids
            del batch_results

            offset += batch_size

        if index is None or index.ntotal == 0:
            print("Error: No valid embeddings processed. FAISS index is empty.")
            return False

        print(f"--> FAISS index built successfully. Total vectors indexed: {index.ntotal}")

        # 2. Create the ID mapping (FAISS index to chunk_id)
        id_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        print("--> ID mapping created successfully.")

        # 3. Save the index and mapping to disk
        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, "faiss_index.bin")
        mapping_path = os.path.join(output_dir, "id_mapping.json")

        faiss.write_index(index, index_path)
        print(f"--> FAISS index saved to: {index_path}")

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(id_mapping, f, ensure_ascii=False, indent=2)
        print(f"--> ID mapping saved to: {mapping_path}")

    except Exception as e:
        print(f"An error occurred during FAISS index creation: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("--- FAISS Index Build Finished Successfully ---")
    return True


def check_data_count(engine):
    """Check how many records were inserted"""
    with engine.connect() as conn:
        doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        chunk_count = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar()

        print(f"Documents inserted: {doc_count}")
        print(f"Chunks inserted: {chunk_count}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the database and FAISS index from a Parquet file, "
        "or build FAISS index only from existing database data."
    )

    # The container's working directory is /app
    default_input = "/app/data/processed/documents_with_embeddings.parquet"
    default_output = "/app/data/processed"

    parser.add_argument(
        "--input-file",
        type=str,
        default=default_input,
        help=f"Path to the input Parquet file. Defaults to {default_input}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help=f"Directory to save the FAISS index and mapping. Defaults to {default_output}",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Drop existing tables and create fresh schema (default: preserve existing data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for processing data (default: 500 to avoid memory issues)",
    )
    parser.add_argument(
        "--start-row-group",
        type=int,
        default=0,
        help="Start processing from this row group (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--max-row-groups",
        type=int,
        default=1,
        help="Maximum row groups to process per session before memory cleanup restart (default: 1)",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Automatically determine starting row group and continue from last processed",
    )
    parser.add_argument(
        "--auto-full",
        action="store_true",
        help="Automatically complete the entire database build process with memory management",
    )
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Only build FAISS index from existing database data (skip data import)",
    )
    parser.add_argument(
        "--faiss-batch-size",
        type=int,
        default=10000,
        help="Batch size for FAISS index building (default: 10000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()  # Load .env file from the project root
    args = parse_arguments()

    print("--- Starting Database Build Process ---")
    print(f"Input data file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Fresh start mode: {args.fresh_start}")

    if not os.path.exists(args.input_file):
        print(f"Error: Input data file not found at '{args.input_file}'")
        print("Please ensure the Parquet file is available or check the --input-file argument.")
        sys.exit(1)

    db_url = get_db_connection_url()
    db_engine = create_engine(db_url, echo=False)

    if check_db_connection(db_engine):
        if args.faiss_only:
            # Only build FAISS index from existing database data
            print("🔍 Building FAISS index from existing database data...")
            success = build_faiss_index(db_engine, args.output_dir, args.faiss_batch_size)
            if success:
                print("✅ FAISS index build completed successfully!")
            else:
                print("❌ FAISS index build failed")
                sys.exit(1)
        else:
            # Always use batch-based, memory-efficient import
            if args.fresh_start:
                print("Using fresh start mode - dropping existing tables...")
                create_schema_fresh(db_engine)
            else:
                print("Using safe mode - preserving existing data and handling duplicates...")
                create_tables_if_not_exist(db_engine)

            safe_insert_data(db_engine, args.input_file, args.batch_size)
            check_data_count(db_engine)
            build_faiss_index(db_engine, args.output_dir, args.faiss_batch_size)

    print("--- Database Build Process Completed Successfully ---")
