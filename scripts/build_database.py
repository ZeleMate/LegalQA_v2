import os
import sys
import argparse
import time
import numpy as np
import pyarrow.parquet as pq
from sqlalchemy import (
    create_engine, text, MetaData, Table, Column, String, text,
    ForeignKey, Integer
)
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
import uuid
import faiss
import pickle

# --- Database Schema Definition ---
# Defining the schema directly in the script to make it self-contained.
metadata = MetaData()

documents = Table(
    'documents',
    metadata,
    Column('doc_id', String, primary_key=True),
    Column('source', String)
)

chunks = Table(
    'chunks',
    metadata,
    Column('chunk_id', String, primary_key=True),
    Column('doc_id', String, ForeignKey('documents.doc_id')),
    Column('text', String),
    Column('embedding', Vector(1536)) # Corrected embedding dimension
)
# --- End of Schema Definition ---

def get_db_connection_url():
    """
    Constructs the database connection URL from environment variables.
    These are injected by Docker Compose from the .env file.
    """
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST") # Should be 'db'
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
            with engine.connect() as connection:
                print("Database connection successful!")
                return True
        except Exception as e:
            print(f"Connection attempt {i+1}/5 failed. Retrying in 5 seconds...")
            print(f"Error details: {e}")
            time.sleep(5)
    print("Error: Could not establish database connection after several retries.")
    return False

def create_schema(engine):
    """Creates the database schema, dropping existing tables first."""
    print("Creating database schema...")
    try:
        with engine.connect() as connection:
            # Enable the pgvector extension
            connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            connection.commit()
            print("pgvector extension enabled.")

        # Drop tables in the correct order to respect foreign key constraints
        metadata.drop_all(bind=engine, tables=[chunks, documents])
        print("Existing tables dropped.")
        metadata.create_all(engine)
        print("New tables created successfully based on schema.")
    except Exception as e:
        print(f"An error occurred during schema creation: {e}")
        sys.exit(1)

def build_database(engine, parquet_path):
    """
    Reads data from a Parquet file and populates the PostgreSQL database.
    Processes the file in chunks to manage memory usage efficiently.
    """
    print(f"Starting database build from Parquet file: {parquet_path}")
    
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"Total number of row groups to process: {parquet_file.num_row_groups}")

        total_rows_processed = 0
        Session = sessionmaker(bind=engine)
        session = Session()

        for i in range(parquet_file.num_row_groups):
            print(f"Processing row group {i + 1}/{parquet_file.num_row_groups}...")
            df = parquet_file.read_row_group(i).to_pandas()
            
            # Prepare data for bulk insertion
            doc_data = []
            chunk_data = []

            for _, row in df.iterrows():
                doc_id = row.get('doc_id')
                chunk_id = row.get('chunk_id')

                # If chunk_id is missing, generate a new UUID for it.
                if not chunk_id:
                    chunk_id = str(uuid.uuid4())

                # Add to documents table data (avoiding duplicates)
                # A more robust solution might query existing IDs first if docs can span multiple row groups
                if doc_id and not any(d['doc_id'] == doc_id for d in doc_data):
                     doc_data.append({'doc_id': doc_id, 'source': row.get('source', 'N/A')})

                # Add to chunks table data
                embedding_vector = row.get('embedding')
                if embedding_vector is not None:
                    # Convert to numpy array of specific type for pgvector
                    embedding = np.array(embedding_vector, dtype=np.float32)
                else:
                    embedding = None
                
                chunk_data.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'text': row.get('text'),
                    'embedding': embedding
                })

            # Bulk insert into the database using SQLAlchemy Core for performance
            if doc_data:
                session.execute(documents.insert(), doc_data)
            if chunk_data:
                session.execute(chunks.insert(), chunk_data)
            
            session.commit()
            
            total_rows_processed += len(df)
            print(f"Finished row group {i + 1}. Total rows processed: {total_rows_processed}")

    except Exception as e:
        print(f"An error occurred during the database build: {e}")
        session.rollback()
    finally:
        session.close()
        print("Database build process finished.")

def build_faiss_index(engine, output_dir):
    """
    Builds a FAISS index from the embeddings in the 'chunks' table
    and saves the index and ID mapping to disk.
    """
    print("\n--- Starting FAISS Index Build ---")
    
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        print("--> Querying all chunks and embeddings from the database...")
        # Assuming pgvector stores vectors as numpy arrays when using SQLAlchemy
        results = session.query(chunks.c.chunk_id, chunks.c.embedding).all()
        
        if not results:
            print("Warning: No embeddings found in the database. Skipping FAISS index creation.")
            return

        # Filter out entries where embedding is None, which can cause errors
        valid_results = [r for r in results if r.embedding is not None]
        
        if not valid_results:
            print("Warning: All embedding entries were invalid (None). Skipping FAISS index creation.")
            return
            
        print(f"--> Found {len(results)} total entries, {len(valid_results)} have valid embeddings.")
        
        chunk_ids, embeddings = zip(*valid_results)
        
        # Convert embeddings to a format FAISS can use (2D numpy array)
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        
        # Check if the matrix is not empty
        if embedding_matrix.size == 0:
            print("Warning: Embedding matrix is empty. Skipping FAISS index creation.")
            return
            
        d = embedding_matrix.shape[1]
        print(f"--> Building FAISS index for {len(chunk_ids)} vectors of dimension {d}...")
        
        # 1. Build the FAISS index
        index = faiss.IndexFlatL2(d)
        index.add(embedding_matrix)
        print(f"--> FAISS index built successfully. Total vectors indexed: {index.ntotal}")

        # 2. Create the ID mapping (FAISS index to chunk_id)
        id_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        print("--> ID mapping created successfully.")

        # 3. Save the index and mapping to disk
        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, 'faiss_index.bin')
        mapping_path = os.path.join(output_dir, 'id_mapping.pkl')
        
        faiss.write_index(index, index_path)
        print(f"--> FAISS index saved to: {index_path}")
        
        with open(mapping_path, 'wb') as f:
            pickle.dump(id_mapping, f)
        print(f"--> ID mapping saved to: {mapping_path}")

    except Exception as e:
        print(f"An error occurred during FAISS index creation: {e}")
    finally:
        session.close()
        print("--- FAISS Index Build Finished ---")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the database and/or FAISS index from a Parquet file.")
    
    # The container's working directory is /app
    default_input = "/app/data/processed/processed_documents_with_embeddings.parquet"
    default_output = "/app/data/processed"

    parser.add_argument(
        "--input-file",
        type=str,
        default=default_input,
        help=f"Path to the input Parquet file. Defaults to {default_input}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help=f"Directory to save the FAISS index and mapping. Defaults to {default_output}"
    )
    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv() # Load .env file from the project root
    args = parse_arguments()

    print(f"--- Starting Database Build Process ---")
    print(f"Input data file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.input_file):
        print(f"Error: Input data file not found at '{args.input_file}'")
        print("Please ensure the Parquet file is available or check the --input-file argument.")
        sys.exit(1)

    db_url = get_db_connection_url()
    db_engine = create_engine(db_url, echo=False)

    if check_db_connection(db_engine):
        create_schema(db_engine)
        build_database(db_engine, args.input_file)
        build_faiss_index(db_engine, args.output_dir) 