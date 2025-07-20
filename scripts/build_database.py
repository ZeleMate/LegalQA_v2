#!/usr/bin/env python3
"""
Unified script to build LegalQA database and FAISS index from Parquet data with automatic memory management.
Handles chunking overlap by using ON CONFLICT DO NOTHING for duplicates.
Automatically restarts every 2 row groups to manage memory usage.
Can also build FAISS index only from existing database data.
"""

import os
import sys
import gc
import subprocess
import argparse
import time
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sqlalchemy import (
    create_engine, text, MetaData, Table, Column, String, text,
    ForeignKey, Integer, DateTime, func
)
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
import uuid
import faiss
import pickle
import ast

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
    Column('embedding', Vector(768)) # Sentence transformer embedding dimension
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
            with engine.connect() as connection:
                print("Database connection successful!")
                return True
        except Exception as e:
            print(f"Connection attempt {i+1}/5 failed. Retrying in 5 seconds...")
            print(f"Error details: {e}")
            time.sleep(5)
    print("Error: Could not establish database connection after several retries.")
    return False

def create_tables_if_not_exist(engine):
    """Create tables if they don't exist - safe approach that doesn't drop existing data"""
    print("Creating tables if not exist...")
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        
        # Create documents table
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT
            )
        '''))
        
        # Create chunks table with 768-dimensional embeddings
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT REFERENCES documents(doc_id),
                text TEXT,
                embedding vector(768)
            )
        '''))
        
        conn.commit()
    print("Tables created successfully.")

def create_schema_fresh(engine):
    """Creates the database schema, dropping existing tables first - for fresh start"""
    print("Creating fresh database schema (dropping existing tables)...")
    try:
        with engine.connect() as connection:
            # Enable the pgvector extension
            connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            connection.commit()
            print("pgvector extension enabled.")

        # Drop tables in the correct order to respect foreign key constraints
        with engine.connect() as connection:
            connection.execute(text('DROP TABLE IF EXISTS chunks CASCADE'))
            connection.execute(text('DROP TABLE IF EXISTS documents CASCADE'))
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
        sample_chunk_ids = df_batch['chunk_id'].dropna().head(sample_size).tolist()
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
            placeholders = ','.join([':chunk_id_' + str(i) for i in range(len(sample_chunk_ids))])
            params = {f'chunk_id_{i}': chunk_id for i, chunk_id in enumerate(sample_chunk_ids)}
            
            query = f"SELECT COUNT(*) FROM chunks WHERE chunk_id IN ({placeholders})"
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
            print(f"Estimated last processed row group: {estimated_row_group} (based on {chunk_count} chunks)")
            return estimated_row_group
    except Exception as e:
        print(f"Error determining last processed row group: {e}")
        return -1

def clean_memory():
    """Force garbage collection and clean memory."""
    print("Cleaning memory...")
    gc.collect()
    print("Memory cleaned.")

def get_current_progress_count(engine=None):
    """Lekéri a jelenlegi chunk számot az adatbázisból."""
    if engine:
        # Közvetlen adatbázis hozzáférés engine-nel
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
                return result[0] if result else 0
        except Exception as e:
            print(f"Hiba a haladás lekérdezésében (engine): {e}")
            return 0
    else:
        # Docker parancs használata
        try:
            cmd = [
                "docker", "exec", "legalqa_v2-db-1", 
                "psql", "-U", "Zelemate", "-d", "legal_qa_db", 
                "-t", "-c", "SELECT COUNT(*) FROM chunks;"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                chunk_count = int(result.stdout.strip())
                return chunk_count
            else:
                print(f"Hiba az adatbázis lekérdezésben: {result.stderr}")
                return 0
        except Exception as e:
            print(f"Hiba a haladás lekérdezésében: {e}")
            return 0

def cleanup_docker_container():
    """Tisztítja a Docker container-t."""
    print("🧹 Docker container újraindítása...")
    try:
        subprocess.run(["docker", "kill", "legalqa_v2-app-1"], capture_output=True)
        time.sleep(2)
        subprocess.run(["docker", "start", "legalqa_v2-app-1"], capture_output=True)
        time.sleep(5)
        print("✅ Docker container újraindítva")
        return True
    except Exception as e:
        print(f"❌ Hiba a Docker container újraindításakor: {e}")
        return False

def run_automated_full_process(engine, parquet_path, batch_size=50, max_retries=3, total_row_groups=17):
    """
    Automatikusan futtatja a teljes adatbázis építési folyamatot memória-kezeléssel.
    """
    print("🤖 AUTOMATIKUS ADATBÁZIS ÉPÍTŐ MÓD AKTIVÁLVA")
    print("=" * 60)
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        # Jelenlegi haladás meghatározása
        current_chunks = get_current_progress_count(engine)
        current_row_group = max(0, current_chunks // 49000)
        
        if current_row_group >= total_row_groups:
            print(f"\n🎉 TELJES ADATBÁZIS ÉPÍTÉS BEFEJEZVE!")
            print(f"📊 Végső chunk szám: {current_chunks}")
            break
        
        remaining_groups = total_row_groups - current_row_group
        session_groups = 1  # Mindig csak 1 row group per session
        
        print(f"\n🔄 SESSION: Row group {current_row_group + 1}/{total_row_groups}")
        print(f"📈 Haladás: {current_row_group/total_row_groups*100:.1f}% ({current_chunks} chunk)")
        
        retries = 0
        session_success = False
        
        while retries < max_retries and not session_success:
            if retries > 0:
                print(f"🔄 Újrapróbálkozás {retries + 1}/{max_retries}")
                cleanup_docker_container()
                time.sleep(10)  # Várunk hogy a container stabilizálódjon
            
            # Haladás ellenőrzése session előtt
            chunks_before = get_current_progress_count(engine)
            
            try:
                # Session futtatása
                print(f"🚀 Indítás: Row group {current_row_group + 1}, batch size: {batch_size}")
                safe_insert_data(engine, parquet_path, batch_size, current_row_group, session_groups, auto_mode=True)
                
                # Haladás ellenőrzése session után
                time.sleep(2)
                chunks_after = get_current_progress_count(engine)
                chunks_added = chunks_after - chunks_before
                
                if chunks_added > 1000:  # Sikerként értékeljük ha jelentős haladás volt
                    print(f"✅ Session siker! +{chunks_added} chunk hozzáadva")
                    print(f"📊 Összes chunk: {chunks_after}")
                    session_success = True
                    consecutive_failures = 0
                    current_row_group += 1
                else:
                    raise Exception(f"Nem volt elegendő haladás: +{chunks_added} chunk")
                    
            except Exception as e:
                retries += 1
                consecutive_failures += 1
                print(f"❌ Session sikertelen (kísérlet {retries}/{max_retries}): {str(e)[:200]}...")
                
                if consecutive_failures >= max_consecutive_failures:
                    print(f"💥 Túl sok egymás utáni hiba ({consecutive_failures}). Leállás.")
                    return False
                
                if retries < max_retries:
                    wait_time = min(30 * retries, 120)  # Exponential backoff, max 2 perc
                    print(f"⏱️  Várakozás {wait_time} másodperc...")
                    time.sleep(wait_time)
        
        if not session_success:
            print(f"💥 Session véglegesen sikertelen {max_retries} kísérlet után")
            return False
        
        # Rövid szünet session-ök között
        print("😴 5 másodperces szünet a következő session előtt...")
        time.sleep(5)
    
    return True

def restart_script_from_row_group(start_row_group, batch_size, max_row_groups_per_session=2):
    """Restart the script from a specific row group."""
    print(f"Next session would start from row group {start_row_group + 1}...")
    print(f"Manual restart command:")
    print(f"docker exec legalqa_v2-app-1 python scripts/build_database.py --start-row-group {start_row_group} --batch-size {batch_size} --max-row-groups {max_row_groups_per_session}")
    print("Exiting current session to allow manual restart...")
    sys.exit(0)

def safe_insert_data(engine, parquet_path, batch_size=1000, start_row_group=0, max_row_groups_per_session=2, auto_mode=False):
    """
    Safely insert data using ON CONFLICT DO NOTHING to handle duplicates from chunking overlap.
    This approach gracefully handles duplicate chunk_ids that occur due to sliding window overlap.
    Uses smaller batches to avoid memory issues.
    Automatically skips already processed batches within row groups.
    """
    print(f"Starting safe data import from: {parquet_path}")
    print(f"Using batch size: {batch_size}")
    print("Using ON CONFLICT DO NOTHING to handle duplicates from chunking overlap...")
    print("Will automatically skip already processed batches...")
    
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"Total number of row groups to process: {parquet_file.num_row_groups}")
        print(f"Total rows in file: {parquet_file.metadata.num_rows}")

        # Check current progress
        with engine.connect() as conn:
            current_count = conn.execute(text('SELECT COUNT(*) FROM chunks')).scalar()
            print(f"Current chunks in database: {current_count}")

        with engine.connect() as conn:
            end_row_group = min(start_row_group + max_row_groups_per_session, parquet_file.num_row_groups)
            
            for i in range(start_row_group, end_row_group):
                
                print(f"\nProcessing row group {i + 1}/{parquet_file.num_row_groups}... (session: {start_row_group + 1}-{end_row_group})")
                try:
                    df = parquet_file.read_row_group(i).to_pandas()
                    print(f"Row group {i + 1} has {len(df)} rows")
                    
                    # Process in smaller batches within the row group
                    total_batches = (len(df) + batch_size - 1) // batch_size
                    print(f"Will process in {total_batches} batches of size {batch_size}")
                    
                    for batch_start in range(0, len(df), batch_size):
                        batch_end = min(batch_start + batch_size, len(df))
                        batch_df = df.iloc[batch_start:batch_end]
                        
                        batch_num = batch_start // batch_size + 1
                        print(f"  Checking batch {batch_num}/{total_batches}: rows {batch_start}-{batch_end-1}")
                        
                        # Check if this batch is already processed
                        sample_chunks = get_sample_chunks_from_batch(batch_df)
                        if is_batch_processed(engine, sample_chunks):
                            print(f"    Batch {batch_num} already processed - skipping")
                            del batch_df
                            continue
                        
                        print(f"    Processing batch {batch_num} (new data)")
                        
                        # Get unique doc_ids for this batch
                        unique_docs = batch_df[['doc_id']].drop_duplicates()
                        for _, row in unique_docs.iterrows():
                            doc_id = row['doc_id']
                            source = 'N/A'  # Default source
                            
                            # Insert document with conflict handling - handles doc_id duplicates
                            conn.execute(text('''
                                INSERT INTO documents (doc_id, source) 
                                VALUES (:doc_id, :source) 
                                ON CONFLICT (doc_id) DO NOTHING
                            '''), {'doc_id': doc_id, 'source': source})
                        
                        # Insert chunks with duplicate handling
                        chunks_inserted = 0
                        chunks_skipped = 0
                        
                        for _, row in batch_df.iterrows():
                            chunk_id = row.get('chunk_id')
                            doc_id = row.get('doc_id')
                            text_content = row.get('text')
                            embedding_vector = row.get('embedding')
                            
                            # Generate chunk_id if missing
                            if not chunk_id:
                                chunk_id = str(uuid.uuid4())
                            
                            if doc_id and embedding_vector is not None:
                                # Convert embedding to proper format
                                embedding = np.array(embedding_vector, dtype=np.float32)
                                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                                
                                # Insert chunk with conflict handling - handles chunk_id duplicates from overlap
                                result = conn.execute(text('''
                                    INSERT INTO chunks (chunk_id, doc_id, text, embedding) 
                                    VALUES (:chunk_id, :doc_id, :text, CAST(:embedding_vec AS vector)) 
                                    ON CONFLICT (chunk_id) DO NOTHING
                                    RETURNING chunk_id
                                '''), {
                                    'chunk_id': chunk_id,
                                    'doc_id': doc_id,
                                    'text': text_content,
                                    'embedding_vec': embedding_str
                                })
                                
                                if result.rowcount > 0:
                                    chunks_inserted += 1
                                else:
                                    chunks_skipped += 1
                        
                        conn.commit()
                        print(f"      Inserted: {chunks_inserted}, Skipped: {chunks_skipped}")
                        
                        # Clear batch from memory
                        del batch_df
                    
                    # Clear row group from memory
                    del df
                    
                    # Check progress after each row group
                    current_count = conn.execute(text('SELECT COUNT(*) FROM chunks')).scalar()
                    print(f"Row group {i + 1} completed. Total chunks in database: {current_count}")
                    
                except Exception as e:
                    print(f"Error processing row group {i + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"Continuing with next row group...")
                    continue
        
        # Check if we need to continue with more row groups
        if end_row_group < parquet_file.num_row_groups:
            print(f"\nSession completed. Processed row groups {start_row_group + 1}-{end_row_group}")
            print(f"Remaining row groups: {parquet_file.num_row_groups - end_row_group}")
            
            # Clean memory before restart
            clean_memory()
            
            # Verify data integrity before restart
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
                chunk_count = result[0] if result else 0
                print(f"Current chunk count before restart: {chunk_count}")
            
            if auto_mode:
                # In automated mode, just return to let the caller handle the next iteration
                print(f"Session finished. Returning control to automated process...")
                return
            else:
                # In manual mode, provide restart command and exit
                restart_script_from_row_group(end_row_group, batch_size, max_row_groups_per_session)
        else:
            print(f"\nAll row groups processed successfully!")
            print(f"Final database statistics:")
            with engine.connect() as conn:
                chunks_result = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
                docs_result = conn.execute(text("SELECT COUNT(DISTINCT doc_id) FROM chunks")).fetchone()
                print(f"Total chunks: {chunks_result[0] if chunks_result else 0}")
                print(f"Unique documents: {docs_result[0] if docs_result else 0}")
            print("Data import completed successfully!")
        
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
            count_result = conn.execute(text('SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL'))
            total_embeddings = count_result.scalar()
        
        if total_embeddings == 0:
            print("Warning: No embeddings found in the database. Skipping FAISS index creation.")
            return False

        print(f"--> Found {total_embeddings} total embeddings. Processing in batches of {batch_size}...")
        
        # Initialize variables for batch processing
        chunk_ids = []
        embeddings = []
        index = None
        d = None
        
        # Process in batches to avoid memory issues
        offset = 0
        batch_count = 0
        
        while offset < total_embeddings:
            batch_count += 1
            print(f"--> Processing batch {batch_count}: offset {offset}, limit {batch_size}")
            
            with engine.connect() as conn:
                result = conn.execute(text('''
                    SELECT chunk_id, embedding 
                    FROM chunks 
                    WHERE embedding IS NOT NULL 
                    ORDER BY chunk_id
                    LIMIT :limit OFFSET :offset
                '''), {'limit': batch_size, 'offset': offset})
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
                        emb_str = emb_str.strip('[]')
                        emb_values = [float(x.strip()) for x in emb_str.split(',')]
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
        index_path = os.path.join(output_dir, 'faiss_index.bin')
        mapping_path = os.path.join(output_dir, 'id_mapping.pkl')
        
        faiss.write_index(index, index_path)
        print(f"--> FAISS index saved to: {index_path}")
        
        with open(mapping_path, 'wb') as f:
            pickle.dump(id_mapping, f)
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
        doc_count = conn.execute(text('SELECT COUNT(*) FROM documents')).scalar()
        chunk_count = conn.execute(text('SELECT COUNT(*) FROM chunks')).scalar()
        
        print(f"Documents inserted: {doc_count}")
        print(f"Chunks inserted: {chunk_count}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the database and FAISS index from a Parquet file, or build FAISS index only from existing database data.")
    
    # The container's working directory is /app
    default_input = "/app/data/processed/documents_with_embeddings.parquet"
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
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Drop existing tables and create fresh schema (default: preserve existing data)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for processing data (default: 500 to avoid memory issues)"
    )
    parser.add_argument(
        "--start-row-group",
        type=int,
        default=0,
        help="Start processing from this row group (0-indexed, default: 0)"
    )
    parser.add_argument(
        "--max-row-groups",
        type=int,
        default=1,
        help="Maximum row groups to process per session before memory cleanup restart (default: 1)"
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Automatically determine starting row group and continue from last processed"
    )
    parser.add_argument(
        "--auto-full",
        action="store_true",
        help="Automatically complete the entire database build process with memory management"
    )
    parser.add_argument(
        "--faiss-only",
        action="store_true",
        help="Only build FAISS index from existing database data (skip data import)"
    )
    parser.add_argument(
        "--faiss-batch-size",
        type=int,
        default=10000,
        help="Batch size for FAISS index building (default: 10000)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv() # Load .env file from the project root
    args = parse_arguments()

    print(f"--- Starting Database Build Process ---")
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
            # Csak FAISS index építés
            print("🔍 FAISS index építés meglévő adatbázis adatokból...")
            success = build_faiss_index(db_engine, args.output_dir, args.faiss_batch_size)
            if success:
                print("✅ FAISS index építés sikeresen befejezése!")
            else:
                print("❌ FAISS index építés sikertelen")
                sys.exit(1)
        else:
            # Adatbázis építés + FAISS index
            if args.fresh_start:
                print("Using fresh start mode - dropping existing tables...")
                create_schema_fresh(db_engine)
            else:
                print("Using safe mode - preserving existing data and handling duplicates...")
                create_tables_if_not_exist(db_engine)
            
            if args.auto_full:
                # Automatikus teljes folyamat
                print("🤖 Automatikus teljes adatbázis építés indítása...")
                success = run_automated_full_process(db_engine, args.input_file, args.batch_size)
                if success:
                    print("✅ Automatikus folyamat sikeresen befejezése!")
                    check_data_count(db_engine)
                    build_faiss_index(db_engine, args.output_dir, args.faiss_batch_size)
                else:
                    print("❌ Automatikus folyamat megszakadt hibák miatt")
                    sys.exit(1)
            else:
                # Manuális/session-alapú mód
                # Auto-determine start row group if requested
                start_row_group = args.start_row_group
                if args.auto_continue:
                    last_processed = get_last_processed_row_group(db_engine)
                    start_row_group = max(0, last_processed + 1)
                    print(f"Auto-continue mode: starting from row group {start_row_group + 1}")
                
                safe_insert_data(db_engine, args.input_file, args.batch_size, start_row_group, args.max_row_groups)
                check_data_count(db_engine)
                build_faiss_index(db_engine, args.output_dir, args.faiss_batch_size)
        
        print("--- Database Build Process Completed Successfully ---") 