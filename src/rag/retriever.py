from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
import os
from dotenv import load_dotenv
import sys
import pandas as pd

class CustomRetriever(BaseRetriever, BaseModel):
    alpha: float = 0.7
    embeddings: OpenAIEmbeddings = Field(...)
    faiss_index: faiss.Index = Field(...)
    id_mapping: dict = Field(...)
    k: int = 20

    class Config:
        arbitrary_types_allowed = True

    def _get_db_connection(self):
        """Establishes a connection to the PostgreSQL database."""
        load_dotenv() # Loads variables from .env file
        try:
            return psycopg2.connect(
                dbname=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT")
            )
        except psycopg2.OperationalError as e:
            print(f"Error connecting to DB in retriever: {e}")
            return None

    def _get_docs_from_db(self, chunk_ids: list[str]) -> dict:
        """
        Fetches document chunk data for a list of chunk_ids.
        Switches between PostgreSQL and a local Parquet file based on the
        DATA_SOURCE environment variable.
        """
        data_source = os.getenv("DATA_SOURCE", "postgres")

        if not chunk_ids:
            return {}

        # --- Local Parquet Fallback for Notebook Testing ---
        if data_source == "local_parquet":
            print("--> [Retriever] Fetching data from LOCAL PARQUET file...")
            parquet_path = os.getenv("PARQUET_PATH")
            if not parquet_path or not os.path.exists(parquet_path):
                print(f"Error: PARQUET_PATH environment variable not set or file not found at '{parquet_path}'.", file=sys.stderr)
                return {}
            
            try:
                df = pd.read_parquet(parquet_path)
                # Ensure the index is set to chunk_id for efficient lookup
                if 'chunk_id' in df.columns:
                    df.set_index('chunk_id', inplace=True)
                
                # Fetch rows corresponding to the chunk_ids
                relevant_rows_df = df.loc[df.index.isin(chunk_ids)]
                
                # Convert DataFrame to the dictionary format expected by the calling function
                docs_data = relevant_rows_df.to_dict(orient='index')
                print(f"--> [Retriever] Found {len(docs_data)} documents in Parquet file.")
                return docs_data
            except Exception as e:
                print(f"!!!!!!!!!! [Retriever] Failed to read or process Parquet file: {e}", file=sys.stderr)
                return {}

        # --- Default: PostgreSQL Database ---
        print("--> [Retriever] _get_docs_from_db: Attempting to fetch docs from DB.")
        docs_data = {}
        conn = self._get_db_connection()
        if not conn:
            print("--> [Retriever] _get_docs_from_db: DB connection failed.")
            return docs_data
            
        try:
            with conn.cursor() as cursor:
                # Query by chunk_id, as FAISS maps to chunks, not documents.
                query = "SELECT * FROM chunks WHERE chunk_id = ANY(%s)"
                print(f"--> [Retriever] _get_docs_from_db: Executing query for {len(chunk_ids)} chunk_ids.")
                cursor.execute(query, (chunk_ids,))
                
                rows = cursor.fetchall()
                print(f"--> [Retriever] _get_docs_from_db: Fetched {len(rows)} rows from DB.")
                
                colnames = [desc[0] for desc in cursor.description]
                
                for row in rows:
                    row_dict = dict(zip(colnames, row))
                    # Key the dictionary by chunk_id for correct lookup later.
                    docs_data[row_dict['chunk_id']] = row_dict
        except Exception as e:
            print(f"!!!!!!!!!! [Retriever] _get_docs_from_db ERROR: {e}", file=sys.stderr)
            # Re-raise the exception to see it in the server response
            raise e
        finally:
            conn.close()
            
        print("--> [Retriever] _get_docs_from_db: Successfully fetched and processed docs.")
        return docs_data

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """
        Get relevant documents from the FAISS index.

        Args:
            query (str): The query to search for.
            run_manager (Optional[BaseCallbackManager]): The run manager to use for callbacks.

        Returns:
            list[Document]: The relevant documents.
        """
        print("\n--> [Retriever] Starting document retrieval process...")
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array(query_embedding).reshape(1, -1)

            print("--> [Retriever] Searching FAISS index...")
            distances, indices = self.faiss_index.search(query_vector, k=self.k)
            
            # The id_mapping maps FAISS indices to chunk_ids.
            retrieved_chunk_ids = []
            for idx in indices[0]:
                if idx in self.id_mapping:
                    retrieved_chunk_ids.append(self.id_mapping[idx])
            
            print(f"--> [Retriever] Found {len(retrieved_chunk_ids)} potential chunks from FAISS. Fetching from DB.")
            
            docs_from_db = self._get_docs_from_db(list(set(retrieved_chunk_ids)))
            
            if not docs_from_db:
                print("--> [Retriever] No documents found in DB for the retrieved IDs.")
                return []

            documents = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx in self.id_mapping:
                    # The ID from mapping is a chunk_id.
                    chunk_id = self.id_mapping[idx]
                    doc_data = docs_from_db.get(chunk_id)

                    if doc_data:
                        metadata = doc_data.copy()
                        text = metadata.pop("text", "") # Use .pop with a default value
                        
                        embedding_str = doc_data.get("embedding")
                        if not embedding_str:
                            continue

                        # pgvector returns the vector as a string representation of a list (e.g., "[1,2,3]").
                        # We need to parse this string back into a numpy array.
                        try:
                            # np.fromstring is a robust way to handle this format.
                            # We remove the brackets from the string before parsing.
                            doc_embedding_array = np.fromstring(embedding_str.strip('[]'), sep=',')
                        except Exception:
                            # If parsing fails for any reason, skip this document.
                            continue

                        doc_embedding = doc_embedding_array.reshape(1, -1)
                        
                        relevance_score = 1.0 / (1.0 + distance)
                        similarity_score = cosine_similarity(query_vector, doc_embedding)[0][0]
                        final_score = self.alpha * similarity_score + (1 - self.alpha) * relevance_score

                        metadata["relevancia"] = round(relevance_score, 3)
                        metadata["similarity_score"] = round(similarity_score, 3)
                        metadata["final_score"] = round(final_score, 4)

                        documents.append(Document(
                            page_content=text,
                            metadata=metadata
                        ))

            documents.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
            print("--> [Retriever] Document retrieval process finished successfully.")
            return documents
        
        except Exception as e:
            print(f"!!!!!!!!!! [Retriever] ERROR during document retrieval: {e}", file=sys.stderr)
            raise e

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
        return "\\n\\n---\\n\\n".join(top_snippets)

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
            doc_texts += f"### Document {i+1} (ID: {doc.metadata.get('chunk_id')})\\n"
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
        seen_chunk_ids = set()
        if "ranked_documents" in reranked_results:
            for res in reranked_results["ranked_documents"]:
                chunk_id = res.get("chunk_id")
                original_doc = next((doc for doc in initial_docs if doc.metadata.get("chunk_id") == chunk_id), None)
                
                if original_doc and chunk_id not in seen_chunk_ids:
                    original_doc.metadata['reranker_score'] = res.get("relevance_score")
                    original_doc.metadata['reranker_reason'] = res.get("reason")
                    final_docs.append(original_doc)
                    seen_chunk_ids.add(chunk_id)

        # Fill with initial documents if reranker returns fewer than k
        for doc in initial_docs:
            if len(final_docs) >= self.k:
                break
            if doc.metadata.get("chunk_id") not in seen_chunk_ids:
                final_docs.append(doc)

        return final_docs[:self.k]