import json
import logging
import os
import pickle  # nosec

import faiss
from dotenv import load_dotenv

load_dotenv()


def load_faiss_index(faiss_index_path: str, id_mapping_path: str) -> tuple[faiss.Index, dict]:
    """
    Loads a FAISS index and its ID mapping from a given path.

    Args:
        faiss_index_path (str): The path to the FAISS index file.
        id_mapping_path (str): The path to the ID mapping file.
    Returns:
        tuple[faiss.Index, dict]: A tuple containing the FAISS index and its ID mapping.
    """
    if not os.path.exists(faiss_index_path):
        msg = "Index file not found."
        raise FileNotFoundError(msg)
    if not os.path.exists(id_mapping_path):
        msg = "ID mapping file not found at: {}...".format(
            id_mapping_path[:20] + "..." if len(id_mapping_path) > 20 else id_mapping_path
        )
        raise FileNotFoundError(msg)

    try:
        index = faiss.read_index(faiss_index_path)

        if id_mapping_path.endswith(".json"):
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                id_mapping = json.load(f)
        elif id_mapping_path.endswith(".pkl"):
            with open(id_mapping_path, "rb") as f:
                id_mapping = pickle.load(f)  # nosec
        else:
            msg = "Unsupported ID mapping file format: " f"{id_mapping_path[:40]}..."
            raise ValueError(msg)

        return index, id_mapping
    except Exception as e:
        logging.error("Error loading FAISS index: {}...".format(str(e)[:60]))
        raise RuntimeError("Could not load FAISS index due to: {}...".format(str(e)[:60])) from e
