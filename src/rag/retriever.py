from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import pandas as pd
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

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
                        if key not in ["text"]
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

        # Final sort
        documents.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
        return documents

class RerankingRetriever(BaseRetriever, BaseModel):
    retriever: CustomRetriever = Field(...)
    llm: ChatOpenAI = Field(...)
    reranker_prompt: PromptTemplate = Field(...)
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """
        Retrieves documents from the base retriever and reranks them using an LLM.

        Args:
            query (str): The user's query.

        Returns:
            list[Document]: A list of reranked and relevant documents.
        """
        # 1. Get initial documents from the base retriever
        initial_docs = self.retriever.get_relevant_documents(query)

        # 2. Format documents for the reranker prompt
        doc_texts = ""
        for i, doc in enumerate(initial_docs):
            doc_texts += f"### Document {i+1} (ID: {doc.metadata.get('doc_id', 'N/A')})\n"
            doc_texts += doc.page_content
            doc_texts += "\n---\n"

        # 3. Invoke the reranker LLM
        parser = JsonOutputParser()
        chain = self.reranker_prompt | self.llm | parser
        
        reranked_results = chain.invoke({
            "query": query,
            "documents": doc_texts,
            "k": self.k
        })

        # 4. Process the reranked results
        final_docs = []
        seen_doc_ids = set()

        if "ranked_documents" in reranked_results:
            for res in reranked_results["ranked_documents"]:
                doc_id = res.get("doc_id")
                relevance_score = res.get("relevance_score")
                reason = res.get("reason")
                
                # Find the original document by its ID
                original_doc = next((doc for doc in initial_docs if doc.metadata.get("doc_id") == doc_id), None)
                
                if original_doc and doc_id not in seen_doc_ids:
                    original_doc.metadata['reranker_score'] = relevance_score
                    original_doc.metadata['reranker_reason'] = reason
                    final_docs.append(original_doc)
                    seen_doc_ids.add(doc_id)

        # Add remaining initial docs if we didn't get enough from the reranker
        for doc in initial_docs:
            if len(final_docs) >= self.k:
                break
            if doc.metadata.get("doc_id") not in seen_doc_ids:
                final_docs.append(doc)

        return final_docs[:self.k]