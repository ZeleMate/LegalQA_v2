from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import pandas as pd
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import defaultdict

class CustomRetriever(BaseRetriever, BaseModel):
    alpha: float = 0.7
    embeddings: OpenAIEmbeddings = Field(...)
    faiss_index: faiss.Index = Field(...)
    id_mapping: dict = Field(...)
    documents_df: pd.DataFrame = Field(...)
    k: int = 20

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

        distances, indices = self.faiss_index.search(query_vector, k=self.k)
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

                    # Relevance score based on FAISS distance
                    relevance_score = 1.0 / (1.0 + distance)

                    # Optional boost
                    if metadata.get("HatarozatEve") and metadata["HatarozatEve"] >= 2020:
                        relevance_score *= 1.2
                    if metadata.get("MeghozoBirosag") == "Kúria":
                        relevance_score *= 1.2

                    # Cosine similarity
                    doc_embedding = np.array(row["embedding"]).reshape(1, -1)
                    similarity_score = cosine_similarity(query_vector, doc_embedding)[0][0]

                    # Final weighted score
                    final_score = self.alpha * similarity_score + (1 - self.alpha) * relevance_score

                    # Store all metrics
                    metadata["relevancia"] = round(relevance_score, 3)
                    metadata["similarity_score"] = round(similarity_score, 3)
                    metadata["final_score"] = round(final_score, 4)

                    documents.append(Document(
                        page_content=row["text"],
                        metadata=metadata
                    ))
        if len(documents) >= 6:
            # Check if all documents have an embedding
            documents_with_embedding = [
                doc for doc in documents
                if "embedding" in doc.metadata
            ]
            
            if len(documents_with_embedding) >= 6:
                n_clusters = min(5, max(2, len(documents_with_embedding) // 4))

                embedding_matrix = np.vstack([
                    np.array(doc.metadata["embedding"]) for doc in documents_with_embedding
                ])

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embedding_matrix)
        
                for doc, label in zip(documents_with_embedding, labels):
                    doc.metadata["cluster"] = int(label)

                cluster_scores = defaultdict(list)
                for doc in documents_with_embedding:
                    cluster_scores[doc.metadata["cluster"]].append(doc.metadata["final_score"])
                avg_scores = {
                    c: np.mean(s) for c, s in cluster_scores.items()
                }
                best_cluster = max(avg_scores.items(), key=lambda x: x[1])[0]

                documents = [doc for doc in documents_with_embedding if doc.metadata["cluster"] == best_cluster]
            else:
                for doc in documents:
                    doc.metadata["cluster"] = 0
        else:
            for doc in documents:
                doc.metadata["cluster"] = 0

        # Final sort
        documents.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
        return documents