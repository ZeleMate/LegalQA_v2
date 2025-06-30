import os
import pandas as pd
import faiss
import pickle
from dotenv import load_dotenv
import numpy as np

def build_and_save_index():
    """
    Builds a FAISS index from the sample Parquet file using paths from .env
    and saves it locally for notebook usage.
    """
    print("Loading environment variables for notebook setup...")
    
    # Ensure we're in the project root and load the .env file explicitly
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')
    load_dotenv('.env')

    # Use dedicated environment variables for notebook paths to avoid conflicts
    sample_parquet_path = os.getenv("NOTEBOOK_PARQUET_PATH", "data/processed/sample_data.parquet")
    faiss_index_path = os.getenv("NOTEBOOK_FAISS_PATH", "data/processed/sample_faiss.bin")
    id_mapping_path = os.getenv("NOTEBOOK_ID_MAPPING_PATH", "data/processed/sample_mapping.pkl")

    if not os.path.exists(sample_parquet_path):
        print(f"Error: Sample data file not found at '{sample_parquet_path}'")
        print("Please run 'make setup-dev' first to create the sample data.")
        return

    print(f"Loading sample data from '{sample_parquet_path}'...")
    df = pd.read_parquet(sample_parquet_path)

    if df.empty:
        print("Error: The sample data file is empty. No index will be built.")
        return

    print("Extracting embeddings...")
    embeddings_list = df['embedding'].tolist()
    
    # Check and filter embeddings to ensure consistency
    if not embeddings_list:
        print("Error: No embeddings found in the data.")
        return
    
    # Get the expected dimension from the first embedding
    expected_dim = len(embeddings_list[0])
    print(f"Expected embedding dimension: {expected_dim}")
    
    # Filter out any embeddings with incorrect dimensions
    valid_embeddings = []
    valid_indices = []
    for i, emb in enumerate(embeddings_list):
        if emb is not None and len(emb) == expected_dim:
            valid_embeddings.append(emb)
            valid_indices.append(i)
        else:
            if emb is None:
                print(f"Warning: Skipping None embedding at index {i}")
            else:
                print(f"Warning: Skipping embedding at index {i} with dimension {len(emb)} (expected {expected_dim})")
    
    if not valid_embeddings:
        print("Error: No valid embeddings found after filtering.")
        return
    
    print(f"Using {len(valid_embeddings)} valid embeddings out of {len(embeddings_list)} total.")
    
    # Filter the dataframe to match valid embeddings
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    
    embeddings_np = np.array(valid_embeddings).astype('float32')
    
    if embeddings_np.ndim != 2:
        print(f"Error: Embeddings have an incorrect shape: {embeddings_np.shape}")
        return

    dimension = embeddings_np.shape[1]
    
    print(f"Building FAISS index with dimension {dimension}...")
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_np)
    
    print(f"FAISS index built successfully with {faiss_index.ntotal} vectors.")

    print(f"Saving FAISS index to '{faiss_index_path}'...")
    faiss.write_index(faiss_index, faiss_index_path)

    print(f"Creating and saving ID mapping to '{id_mapping_path}'...")
    id_mapping = {i: chunk_id for i, chunk_id in enumerate(df_filtered['chunk_id'])}
    with open(id_mapping_path, 'wb') as f:
        pickle.dump(id_mapping, f)

    print("\nSuccessfully built and saved local index for notebook usage.")
    print(f"  - Index: {faiss_index_path}")
    print(f"  - ID Map: {id_mapping_path}")

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')
    build_and_save_index() 