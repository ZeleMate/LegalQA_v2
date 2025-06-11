from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import pandas as pd
import faiss
import numpy as np

class CustomRetriever(BaseRetriever, BaseModel):
    embeddings: OpenAIEmbeddings = Field(...)
    faiss_index: faiss.Index = Field(...)
    id_mapping: dict = Field(...)
    documents_df: pd.DataFrame = Field(...)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """
        Get relevant documents from the FAISS index.

        Args:
            query (str): The query to search for.
            run_manager (Optional[BaseCallbackManager]): The run manager to use for callbacks.

        Returns:
            list[Document]: The relevant documents.
        """

        query_embedding = self.embeddings.embed_query(query)
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k=10)
            
        documents = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx in self.id_mapping:
                doc_id = self.id_mapping[idx]

                match = self.documents_df[self.documents_df["id"] == doc_id]
                if not match.empty:
                    row = match.iloc[0]

                    metadata = {
                        key: value
                        for key, value in row.items()
                        if key not in ["text", "embedding"]
                    }

                    documents.append(Document(
                        page_content=row["text"],
                        metadata=metadata
                    ))

        return documents