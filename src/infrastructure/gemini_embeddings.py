import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr


class GeminiEmbeddings:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=SecretStr(api_key)
        )

    def embed_query(self, text: str) -> np.ndarray:
        result = self.client.embed_query(text)
        embedding_values_np = np.array(result, dtype=np.float32)
        normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
        return normed_embedding.astype(np.float32)

    async def aembed_query(self, text: str) -> np.ndarray:
        # Synchronous Gemini API, so just call the sync method in a thread
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.embed_query, text)
        return result.astype(np.float32)
