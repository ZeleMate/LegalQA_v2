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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

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
    embeddings: OpenAIEmbeddings = Field(...)
    k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50

    def _get_top_snippets_from_text(self, text: str, query_vector: np.ndarray, k: int) -> list[str]:
        """Extracts the top k most relevant snippets from a given text."""
        if not text:
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            return [text]

        chunk_embeddings = self.embeddings.embed_documents(chunks)
        similarities = cosine_similarity(query_vector, chunk_embeddings)[0]
        
        # Get indices of top k chunks, handling cases where k > len(chunks)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [chunks[i] for i in top_k_indices]

    def _get_prioritized_snippets(self, doc: Document, query_vector: np.ndarray, k: int = 3) -> str:
        """
        Selects top snippets by giving a score boost to those containing 'indokolás'.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(doc.page_content)
        
        if not chunks:
            return doc.page_content

        chunk_embeddings = self.embeddings.embed_documents(chunks)
        similarities = cosine_similarity(query_vector, chunk_embeddings)[0]

        def keyword_score_boost(chunk: str, score: float) -> float:
            # Normalize chunk by removing spaces and making it lowercase
            normalized_chunk = chunk.replace(" ", "").lower()
            # Check for 'indokolás' in its various forms
            if "indokolás" in normalized_chunk:
                return score * 1.5
            return score

        # Apply boost to similarities
        boosted_similarities = np.array([
            keyword_score_boost(chunk, score) for chunk, score in zip(chunks, similarities)
        ])
        
        # Get indices of top k chunks based on the boosted scores
        top_k_indices = np.argsort(boosted_similarities)[-k:][::-1]
        
        top_snippets = [chunks[i] for i in top_k_indices]
        
        # Combine snippets into a single context string
        return "\n\n---\n\n".join(top_snippets)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """
        Retrieves documents, creates a sophisticated context for each, and then reranks them.
        """
        initial_docs = self.retriever.invoke(query)
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array(query_embedding).reshape(1, -1)

        # Create context and format for reranker
        doc_texts = ""
        for i, doc in enumerate(initial_docs):
            context = self._get_prioritized_snippets(doc, query_vector, k=3)
            doc.metadata['best_snippet_for_reranker'] = context
            doc_texts += f"### Document {i+1} (ID: {doc.metadata.get('doc_id')})\\n"
            doc_texts += context
            doc_texts += "\\n---\\n"
            
        # Invoke the reranker LLM
        parser = JsonOutputParser()
        chain = self.reranker_prompt | self.llm | parser
        
        try:
            reranked_results = chain.invoke({
                "query": query,
                "documents": doc_texts,
                "k": self.k
            })
        except Exception as e:
            print(f"Error during reranker invocation: {e}")
            return initial_docs[:self.k]

        # Process and return the final list of documents
        final_docs = []
        seen_doc_ids = set()
        if "ranked_documents" in reranked_results:
            for res in reranked_results["ranked_documents"]:
                doc_id = res.get("doc_id")
                original_doc = next((doc for doc in initial_docs if doc.metadata.get("doc_id") == doc_id), None)
                
                if original_doc and doc_id not in seen_doc_ids:
                    original_doc.metadata['reranker_score'] = res.get("relevance_score")
                    original_doc.metadata['reranker_reason'] = res.get("reason")
                    final_docs.append(original_doc)
                    seen_doc_ids.add(doc_id)

        # Fill with initial documents if reranker returns fewer than k
        for doc in initial_docs:
            if len(final_docs) >= self.k:
                break
            if doc.metadata.get("doc_id") not in seen_doc_ids:
                final_docs.append(doc)

        return final_docs[:self.k]