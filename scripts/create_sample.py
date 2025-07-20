import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

def create_sample_parquet_incrementally(input_path: str, output_path: str, sample_fraction: float = 0.01):
    """
    Creates a smaller, randomly sampled Parquet file from a larger one by
    incrementally writing sampled row groups to the output file. This method
    has a very low and constant memory footprint.

    Args:
        input_path (str): Path to the source Parquet file.
        output_path (str): Path where the sampled Parquet file will be saved.
        sample_fraction (float): The fraction of rows to sample from each row group.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be between 0 and 1.")

    print(f"Incrementally creating sample from: {input_path}")
    
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema.to_arrow_schema()
    num_row_groups = parquet_file.num_row_groups
    print(f"Total number of row groups to process: {num_row_groups}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    total_original_rows = 0
    total_sampled_rows = 0
    
    # Use ParquetWriter for incremental writes
    with pq.ParquetWriter(output_path, schema) as writer:
        for i in range(num_row_groups):
            print(f"  - Processing row group {i + 1}/{num_row_groups}...")
            row_group_df = parquet_file.read_row_group(i).to_pandas()
            
            # Take a sample from the current row group
            sample_df = row_group_df.sample(frac=sample_fraction, random_state=42)
            
            total_original_rows += len(row_group_df)
            total_sampled_rows += len(sample_df)

            if not sample_df.empty:
                # Convert the sampled pandas DataFrame to an Arrow Table
                table = pa.Table.from_pandas(sample_df, schema=schema)
                # Write the Arrow Table to the Parquet file
                writer.write_table(table)
    
    print(f"\nOriginal dataframe had approximately {total_original_rows} rows.")
    print(f"Created a final sample with {total_sampled_rows} rows ({sample_fraction * 100:.1f}% of original).")
    print(f"Sample data successfully and incrementally saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a memory-efficient sample from a Parquet file.")
    
    # Defaults are set for running inside the Docker container
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="data/processed/documents_with_embeddings.parquet",
        help="Path to the source Parquet file."
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="data/processed/sample_data.parquet",
        help="Path to save the sampled Parquet file."
    )
    parser.add_argument(
        "--fraction", 
        type=float, 
        default=0.01,
        help="The fraction of rows to sample (e.g., 0.01 for 1%)."
    )
    
    args = parser.parse_args()

    create_sample_parquet_incrementally(
        input_path=args.input_file,
        output_path=args.output_file,
        sample_fraction=args.fraction
    ) 