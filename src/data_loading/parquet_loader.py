import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

parquet_path = os.getenv("PARQUET_PATH")
if not parquet_path:
    raise ValueError("Missing PARQUET_PATH variable")


def load_parquet_file(parquet_path: str) -> pd.DataFrame:
    """
    Load parquet file with required field validation.

    Args:
        parquet_path (str): Path to the file.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_parquet(parquet_path)
    required_columns = ["text", "embeddings"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        msg = "Missing the following column(s): " f"{str(missing_columns)[:60]}..."
        raise ValueError(msg)
    return df
