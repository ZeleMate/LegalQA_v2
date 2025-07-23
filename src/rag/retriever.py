"""
Retriever implementation with caching and performance improvements.
"""

import asyncio
import logging
from typing import List, Optional, Any
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ConfigDict
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from src.infrastructure import get_cache_manager, get_db_manager, cache_embedding_query

logger = logging.getLogger(__name__)

class CustomRetriever(BaseRetriever, BaseModel):
    """Retriever with caching and async database operations."""
    
    alpha: float = 0.7
    embeddings: GoogleGenerativeAIEmbeddings = Field(...)
    faiss_index: faiss.Index = Field(...)
    id_mapping: dict = Field(...)
    k: int = 20

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _get_docs_from_db_async(self, chunk_ids: List[str]) -> dict:
        """
        Async method to fetch document chunks with caching.
        """
        if not chunk_ids:
            return {}

        # Use the database manager
        db_manager = get_db_manager()
        docs_data = await db_manager.fetch_chunks_by_ids(chunk_ids)
        
        logger.debug(f"Fetched {len(docs_data)} chunks from database")
        return docs_data

    async def _get_cached_embedding(self, query: str) -> np.ndarray:
        """Get embedding with caching support."""
        return await cache_embedding_query(query, self.embeddings)

    async def _aget_relevant_documents_async(self, query: str) -> List[Document]:
        """
        Async version of document retrieval with caching.
        """
        logger.debug(f"Starting document retrieval for query: {query[:50]}...")
        
        try:
            # Get cached embedding
            query_embedding = await self._get_cached_embedding(query)
            query_vector = np.array(query_embedding).reshape(1, -1)

            # Search FAISS index
            logger.debug("Searching FAISS index...")
            distances, indices = self.faiss_index.search(query_vector, k=self.k)
            
            # Map FAISS indices to chunk_ids
            retrieved_chunk_ids = []
            for idx in indices[0]:
                if idx in self.id_mapping:
                    retrieved_chunk_ids.append(self.id_mapping[idx])
            
            logger.debug(f"Found {len(retrieved_chunk_ids)} chunks from FAISS")
            
            # Fetch documents from database (async)
            try:
                docs_from_db = await self._get_docs_from_db_async(list(set(retrieved_chunk_ids)))
            except Exception as e:
                logger.warning(f"Failed to fetch documents from DB: {e}")
                docs_from_db = {}
            
            if not docs_from_db:
                logger.warning("No documents found in DB for retrieved IDs")
                return []

            # Process documents with scoring
            documents = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx in self.id_mapping:
                    chunk_id = self.id_mapping[idx]
                    doc_data = docs_from_db.get(chunk_id)

                    if doc_data:
                        metadata = doc_data.copy()
                        text = metadata.pop("text", "")
                        
                        # Parse embedding more efficiently
                        embedding_hex = doc_data.get("embedding")
                        if not embedding_hex:
                            continue

                        try:
                            # Convert hex string back to numpy array
                            doc_embedding_bytes = bytes.fromhex(embedding_hex)
                            doc_embedding_array = np.frombuffer(doc_embedding_bytes, dtype=np.float32)
                            doc_embedding = doc_embedding_array.reshape(1, -1)
                        except Exception as e:
                            logger.warning(f"Failed to parse embedding: {e}")
                            continue

                        # Calculate scores
                        relevance_score = 1.0 / (1.0 + distance)
                        similarity_score = cosine_similarity(query_vector, doc_embedding)[0][0]
                        final_score = self.alpha * similarity_score + (1 - self.alpha) * relevance_score

                        metadata.update({
                            "relevancia": round(relevance_score, 3),
                            "similarity_score": round(similarity_score, 3),
                            "final_score": round(final_score, 4)
                        })

                        documents.append(Document(
                            page_content=text,
                            metadata=metadata
                        ))

            # Sort by final score
            documents.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
            logger.debug(f"Retrieval completed, returning {len(documents)} documents")
            return documents
        
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            raise

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> list:
        """
        Dummy method required by BaseRetriever abstract interface. Do not use!
        """
        raise NotImplementedError("Use only the async _get_relevant_documents_async method!")

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> list:
        """
        Async interface for LCEL pipeline compatibility.
        """
        return await self._aget_relevant_documents_async(query)

class RerankingRetriever(BaseRetriever, BaseModel):
    """Reranking retriever with caching and batch processing."""
    
    retriever: CustomRetriever = Field(...)
    llm: ChatGoogleGenerativeAI = Field(...)
    reranker_prompt: PromptTemplate = Field(...)
    embeddings: GoogleGenerativeAIEmbeddings = Field(...)
    k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    reranking_enabled: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _get_prioritized_snippets_cached(self, 
                                             doc: Document, 
                                             query_vector: np.ndarray, 
                                             k: int = 3) -> str:
        """
        Cached version of snippet extraction with keyword boosting.
        """
        cache_manager = get_cache_manager()
        
        # Create cache key based on document content and query
        doc_hash = cache_manager._generate_key("doc", doc.page_content)
        query_hash = cache_manager._generate_key("query", query_vector.tobytes())
        cache_key = f"snippets:{doc_hash}:{query_hash}:{k}"
        
        # Try to get from cache
        cached_snippets = await cache_manager.get(cache_key)
        if cached_snippets:
            logger.debug("Cache hit for snippet extraction")
            return cached_snippets
        
        # Compute snippets if not cached
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(doc.page_content)
        
        if not chunks:
            result = doc.page_content
        else:
            # Get embeddings for chunks (with caching)
            chunk_embeddings = []
            for chunk in chunks:
                embedding = await cache_embedding_query(chunk, self.embeddings)
                chunk_embeddings.append(embedding)
            
            chunk_embeddings_array = np.array(chunk_embeddings)
            similarities = cosine_similarity(query_vector, chunk_embeddings_array)[0]

            def keyword_score_boost(chunk: str, score: float) -> float:
                normalized_chunk = chunk.replace(" ", "").lower()
                if "indokolás" in normalized_chunk:
                    return score * 1.5
                return score

            # Apply boost to similarities
            boosted_similarities = np.array([
                keyword_score_boost(chunk, score) 
                for chunk, score in zip(chunks, similarities)
            ])
            
            # Get top k chunks
            top_k_indices = np.argsort(boosted_similarities)[-k:][::-1]
            top_snippets = [chunks[i] for i in top_k_indices]
            result = "\\n\\n---\\n\\n".join(top_snippets)
        
        # Cache the result
        await cache_manager.set(cache_key, result, ttl=1800)
        return result

    async def _batch_process_documents(self, 
                                     initial_docs: List[Document], 
                                     query: str) -> List[Document]:
        """
        Process documents in batches for better performance.
        """
        query_embedding = await cache_embedding_query(query, self.embeddings)
        query_vector = np.array(query_embedding).reshape(1, -1)

        # Process documents in parallel
        tasks = []
        for doc in initial_docs:
            task = self._get_prioritized_snippets_cached(doc, query_vector, k=3)
            tasks.append(task)
        
        # Wait for all snippet extractions to complete
        snippets = await asyncio.gather(*tasks)
        
        # Update documents with best snippets
        for doc, snippet in zip(initial_docs, snippets):
            doc.metadata['best_snippet_for_reranker'] = snippet
        
        return initial_docs

    async def _aget_relevant_documents_async(self, query: str) -> List[Document]:
        """
        Async version of reranking retrieval.
        """
        logger.debug(f"Starting reranking for query: {query[:50]}...")
        
        # Get initial documents
        initial_docs = await self.retriever._aget_relevant_documents_async(query)
        
        if not self.reranking_enabled:
            return initial_docs[:self.k]

        if not initial_docs:
            return []
        
        # Process documents in batch
        processed_docs = await self._batch_process_documents(initial_docs, query)
        
        # Create context for reranker
        doc_texts = ""
        for i, doc in enumerate(processed_docs):
            context = doc.metadata.get('best_snippet_for_reranker', doc.page_content[:500])
            doc_texts += f"### Document {i+1} (ID: {doc.metadata.get('chunk_id')})\\n"
            doc_texts += context
            doc_texts += "\\n---\\n"
        
        # Cache reranker results
        cache_manager = get_cache_manager()
        reranker_key = cache_manager._generate_key("rerank", f"{query}:{doc_texts[:200]}")
        
        cached_result = await cache_manager.get(reranker_key)
        if cached_result:
            logger.debug("Cache hit for reranker results")
            reranked_results = cached_result
        else:
            # Invoke reranker
            parser = JsonOutputParser()
            chain = self.reranker_prompt | self.llm | parser
            
            try:
                reranked_results = await chain.ainvoke({
                    "query": query,
                    "documents": doc_texts,
                    "k": self.k
                })
                # Cache the result
                await cache_manager.set(reranker_key, reranked_results, ttl=600)
            except Exception as e:
                logger.error(f"Error during reranker invocation: {e}")
                return initial_docs[:self.k]

        # Process final results
        final_docs = []
        seen_chunk_ids = set()
        
        if "ranked_documents" in reranked_results:
            for res in reranked_results["ranked_documents"]:
                chunk_id = res.get("chunk_id")
                original_doc = next(
                    (doc for doc in initial_docs if doc.metadata.get("chunk_id") == chunk_id), 
                    None
                )
                
                if original_doc and chunk_id not in seen_chunk_ids:
                    original_doc.metadata.update({
                        'reranker_score': res.get("relevance_score"),
                        'reranker_reason': res.get("reason")
                    })
                    final_docs.append(original_doc)
                    seen_chunk_ids.add(chunk_id)

        # Fill with initial documents if needed
        for doc in initial_docs:
            if len(final_docs) >= self.k:
                break
            if doc.metadata.get("chunk_id") not in seen_chunk_ids:
                final_docs.append(doc)

        logger.debug(f"Reranking completed, returning {len(final_docs)} documents")
        return final_docs[:self.k]

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> list:
        """
        Dummy method required by BaseRetriever abstract interface. Do not use!
        """
        raise NotImplementedError("Use only the async _get_relevant_documents_async method!")

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> list:
        """
        Async interface for LCEL pipeline compatibility.
        """
        return await self._aget_relevant_documents_async(query)

def initialize_retriever(embeddings, faiss_index, id_mapping, llm, reranker_prompt, k_retriever=25, k_reranker=5):
    """
    Initialize a new retriever with the given parameters.
    """
    # ... existing code ...