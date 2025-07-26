import os

import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings

google_api_key = os.getenv("GOOGLE_API_KEY")


@pytest.fixture(scope="session")
def embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Fixture to initialize and share the GeminiEmbeddings model across tests.
    The model is loaded only once per test session.
    """
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")

    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
