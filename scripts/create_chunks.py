import math
import os
import time
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


def chunk_documents_from_df(
    df_source: pd.DataFrame, chunk_size: int, chunk_overlap: int
) -> pd.DataFrame:
    """
    Chunks documents from a source DataFrame and assigns the parent document's
    embedding and metadata to each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks_data = []
    print(f"Starting to chunk {len(df_source)} documents...")

    if "text" not in df_source.columns:
        raise ValueError("Source DataFrame must contain a 'text' column.")

    for index, row in df_source.iterrows():
        doc_text = row["text"]

        # Create a dictionary of the parent's metadata and embedding
        # This copies all columns from the original row.
        parent_data = row.to_dict()
        # The original full text is no longer needed for the chunk entry;
        # it will be replaced by the chunk text.
        del parent_data["text"]

        # Split the text into chunks
        chunks = text_splitter.split_text(doc_text)

        # Create a new record for each chunk
        for chunk_text in chunks:
            chunk_data = parent_data.copy()
            chunk_data["chunk_id"] = str(uuid.uuid4())
            chunk_data["text"] = chunk_text
            all_chunks_data.append(chunk_data)

    print(f"Finished chunking. Created {len(all_chunks_data)} chunks.")

    # Re-order columns for clarity, putting identifiers first.
    final_df = pd.DataFrame(all_chunks_data)

    # Dynamically determine column order based on the source columns
    original_cols = df_source.columns.tolist()
    # Remove 'text' as it's replaced by the chunked text
    original_cols.remove("text")

    # Define the new standard order
    new_order = ["chunk_id", "text"] + original_cols

    # Ensure all expected columns are present
    final_cols = [col for col in new_order if col in final_df.columns]

    return final_df[final_cols]


if __name__ == "__main__":
    # --- Configuration ---
    load_dotenv()  # Load variables from .env file

    # Get the input path from the environment variable set in the .env file
    INPUT_PATH = os.getenv("PARQUET_PATH")
    if not INPUT_PATH:
        raise ValueError(
            "PARQUET_PATH environment variable not set or is empty. Please check your .env file."
        )

    # The output path is derived from the input path for consistency
    OUTPUT_DIR = os.path.dirname(INPUT_PATH)
    TEMP_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "temp_chunked_data.parquet")
    FINAL_OUTPUT_PATH = INPUT_PATH  # Overwrite the source file

    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 128
    BATCH_SIZE = 1000

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input data file not found at: {INPUT_PATH}.")

    # --- Batch Processing Execution ---
    start_time = time.time()

    source_parquet_file = pq.ParquetFile(INPUT_PATH)
    total_rows = source_parquet_file.metadata.num_rows
    num_batches = math.ceil(total_rows / BATCH_SIZE)
    print(
        f"Starting memory-efficient chunking for {total_rows} documents "
        f"in {num_batches} batches..."
    )

    writer = None

    try:
        batch_iterator = source_parquet_file.iter_batches(batch_size=BATCH_SIZE)

        # Wrap the iterator with tqdm for a progress bar
        for batch in tqdm(batch_iterator, total=num_batches, desc="Chunking documents"):
            batch_df = batch.to_pandas()

            chunked_df = chunk_documents_from_df(batch_df, CHUNK_SIZE, CHUNK_OVERLAP)

            if not chunked_df.empty:
                chunked_table = pa.Table.from_pandas(chunked_df)

                if writer is None:
                    writer = pq.ParquetWriter(TEMP_OUTPUT_PATH, chunked_table.schema)

                writer.write_table(table=chunked_table)

    finally:
        if writer:
            writer.close()
            print("\nTemporary chunked file created.")
            os.replace(TEMP_OUTPUT_PATH, FINAL_OUTPUT_PATH)
            print(f"Successfully replaced original file with chunked data at: {FINAL_OUTPUT_PATH}")
        else:
            print("\nNo data was processed or written.")

    end_time = time.time()
    print("Chunking process complete!")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
