import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

parquet_path = os.getenv("PARQUET_PATH")
if not parquet_path:
    raise ValueError("Hiányzik a PARQUET_PATH változó")


def load_parquet_file(parquet_path: str) -> pd.DataFrame:
    """
    Parquet fájl betöltése, kötelező mezők ellenőrzésével.

    Args:
        parquet_path (str): Path to the file.

    Returns:
        pd.DataFrame: Betöltött adatkeret.

    Raises:
        ValueError: Ha a kötelező oszlopok hiányoznak.
    """
    df = pd.read_parquet(parquet_path)
    required_columns = ["text", "embeddings"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Hiányzik a következő oszlop(ok): {missing_columns}")
    return df