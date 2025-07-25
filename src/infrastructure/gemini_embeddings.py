import numpy as np
from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


class GeminiEmbeddings:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = GoogleGenerativeAI(
            model="models/text-embedding-004", google_api_key=api_key
        )

    def embed_query(self, text: str) -> np.ndarray:
        result = self.client.embed_content(
            contents=text,
            config=GoogleGenerativeAIEmbeddings(
                api_key=self.api_key
            ),
        )
        [embedding_obj] = result.embeddings
        embedding_values_np = np.array(embedding_obj.values)
        normed_embedding = embedding_values_np / np.linalg.norm(
            embedding_values_np
        )
        return normed_embedding

    async def aembed_query(self, text: str) -> np.ndarray:
        # Synchronous Gemini API, so just call the sync method in a thread
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
