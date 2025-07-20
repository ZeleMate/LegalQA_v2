import pytest
from langchain_huggingface import HuggingFaceEmbeddings

@pytest.fixture(scope="session")
def embeddings_model():
    """
    Fixture to initialize and share the HuggingFaceEmbeddings model across tests.
    The model is loaded only once per test session.
    """
    return HuggingFaceEmbeddings(
        model_name="Snowflake/snowflake-arctic-embed-m"
    ) 