from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import pandas as pd
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CustomRetriever(BaseRetriever, BaseModel):
    alpha: float = 0.7
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
        query_vector = np.array(query_embedding).reshape(1, -1)

        distances, indices = self.faiss_index.search(query_vector, k=20)
        distances = distances[0]
        indices = indices[0]

        documents = []

        for i, (idx, distance) in enumerate(zip(indices, distances)):
            if idx in self.id_mapping:
                doc_id = self.id_mapping[idx]
                match = self.documents_df[self.documents_df["doc_id"] == doc_id]

                if not match.empty:
                    row = match.iloc[0]

                    metadata = {
                        key: value
                        for key, value in row.items()
                        if key not in ["text", "embedding"]
                    }

                    # Relevancia FAISS distance alapján
                    relevance_score = 1.0 / (1.0 + distance)

                    # Opcionális boost
                    if metadata.get("HatarozatEve") and metadata["HatarozatEve"] >= 2020:
                        relevance_score *= 1.2
                    if metadata.get("MeghozoBirosag") == "Kúria":
                        relevance_score *= 1.2

                    # Cosine similarity
                    doc_embedding = np.array(row["embedding"]).reshape(1, -1)
                    similarity_score = cosine_similarity(query_vector, doc_embedding)[0][0]

                    # Végső súlyozott pontszám
                    final_score = self.alpha * similarity_score + (1 - self.alpha) * relevance_score

                    # Minden metrikát elmentünk
                    metadata["relevancia"] = round(relevance_score, 3)
                    metadata["similarity_score"] = round(similarity_score, 3)
                    metadata["final_score"] = round(final_score, 4)

                    documents.append(Document(
                        page_content=row["text"],
                        metadata=metadata
                    ))

        # Végső sorbarendezés
        documents.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
        return documents