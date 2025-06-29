from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager
import traceback
import sys
import logging
import pyarrow.parquet as pq
from pathlib import Path
from langchain_core.prompts import PromptTemplate

from src.data.faiss_loader import load_faiss_index
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.rag.retriever import CustomRetriever, RerankingRetriever
from src.chain.qa_chain import build_qa_chain

# Setup basic logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# This context is no longer needed for the model, but can be kept for other purposes if necessary
# context = {}

def load_data_for_retriever(parquet_path: str):
    """
    Loads data from a Parquet file into memory-efficient dictionaries for the retriever.
    This avoids loading the entire DataFrame into memory.
    """
    print("--> Loading data from Parquet file efficiently...")
    text_db = {}
    metadata_db = {}
    
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Define which columns we absolutely need. This helps reduce memory usage.
    # We assume all other columns are metadata.
    required_cols = {'doc_id', 'text', 'embedding'}
    all_cols = parquet_file.schema.names
    metadata_cols = [col for col in all_cols if col not in required_cols]
    
    # Iterate over row groups to keep memory usage low
    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(i, columns=all_cols)
        df_chunk = row_group.to_pandas()
        
        for _, row in df_chunk.iterrows():
            doc_id = row['doc_id']
            text_db[doc_id] = row['text']
            
            # Collect all metadata, including the embedding
            metadata = {col: row[col] for col in metadata_cols}
            metadata['embedding'] = row['embedding']
            metadata_db[doc_id] = metadata
            
    print(f"--> Data loaded. Total documents in text_db: {len(text_db)}")
    return text_db, metadata_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    logger.info("--> Loading FAISS index and mappings...")
    load_dotenv()

    # Load environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
    ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH")
    RERANKER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reranker_prompt.txt"

    # Load components
    logger.info("Loading embeddings model...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    logger.info("Loading FAISS index and ID mapping...")
    faiss_index, id_mapping = load_faiss_index(FAISS_INDEX_PATH, ID_MAPPING_PATH)
    
    logger.info("Loading reranker prompt...")
    try:
        reranker_template = RERANKER_PROMPT_PATH.read_text(encoding="utf-8")
        reranker_prompt = PromptTemplate.from_template(reranker_template)
    except FileNotFoundError:
        logger.error(f"Reranker prompt not found at: {RERANKER_PROMPT_PATH}")
        raise
    
    # Instantiate the base retriever
    base_retriever = CustomRetriever(
        embeddings=embeddings,
        faiss_index=faiss_index,
        id_mapping=id_mapping
    )
    
    # Instantiate the reranking retriever
    reranker_llm = ChatOpenAI(model_name="o4-mini-2025-04-16", temperature=0, openai_api_key=OPENAI_API_KEY)
    
    reranking_retriever = RerankingRetriever(
        retriever=base_retriever,
        llm=reranker_llm,
        reranker_prompt=reranker_prompt,
        embeddings=embeddings
    )
    
    # Build the final QA chain with the reranking retriever
    qa_chain = build_qa_chain(reranking_retriever)
    
    # Store the loaded objects in the application state
    app.state.qa_chain = qa_chain
    logger.info("--> Lightweight components loaded successfully. Application is ready.")
    
    yield
    
    # This code runs on shutdown
    logger.info("--> Cleaning up resources...")
    app.state.qa_chain = None


app = FastAPI(
    title="LegalQA API",
    description="An API for answering legal questions.",
    version="1.0.0",
    lifespan=lifespan  # Register the lifespan event handler
)


class QuestionRequest(BaseModel):
    question: str
    # user_id: str | None = None # Example of an optional field

@app.get("/health", status_code=200, tags=["Status"])
async def health_check():
    """Endpoint to check if the API is running."""
    return {"status": "ok"}

@app.post("/ask", tags=["Q&A"])
async def ask(req: QuestionRequest, request: Request):
    """
    Receives a question, invokes the QA chain, and returns the answer and sources.
    """
    qa_chain = request.app.state.qa_chain
    if not qa_chain:
        logger.error("QA chain is not available. Service might be starting up or has failed.")
        raise HTTPException(status_code=503, detail="Service not available. Please try again later.")
    
    try:
        logger.info(f"Invoking QA chain for question: \"{req.question[:50]}...\"")
        result = qa_chain.invoke({
            "question": req.question,
            "chat_history": [] # Passing an empty history for now
        })
        logger.info("QA chain invocation successful.")
        return {
            "answer": result["answer"],
            "sources": [doc.metadata["doc_id"] for doc in result["source_documents"]]
        }
    except Exception as e:
        # Log the full error and traceback for server-side debugging
        logger.error(f"An error occurred in /ask endpoint: {e}", exc_info=True)
        # Return a generic error message to the client to avoid exposing internal details
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the LegalQA API!",
        "endpoints": {
            "/ask": "POST - Ask a question to the system"
        }
    }