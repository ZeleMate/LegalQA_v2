from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
from langchain_core.prompts import PromptTemplate

from src.data.faiss_loader import load_faiss_index
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.rag.retriever import CustomRetriever, RerankingRetriever
from src.chain.qa_chain import build_qa_chain

# Setup basic logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# The load_data_for_retriever function is no longer needed as the app
# relies on the retriever to fetch data from the database.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    logger.info("--> Loading application components...")
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
    reranker_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    reranking_retriever = RerankingRetriever(
        retriever=base_retriever,
        llm=reranker_llm,
        reranker_prompt=reranker_prompt,
        embeddings=embeddings
    )
    
    # Build the final QA chain with the reranking retriever
    final_qa_chain = build_qa_chain(reranking_retriever)
    
    # Store the loaded objects in the application state
    app.state.qa_chain = final_qa_chain
    logger.info("--> Application components loaded successfully. Ready to serve.")
    
    yield
    
    # This code runs on shutdown
    logger.info("--> Cleaning up resources...")
    app.state.qa_chain = None


app = FastAPI(
    # ... (app definition is correct) ...
)


class QuestionRequest(BaseModel):
    question: str

@app.get("/health", status_code=200, tags=["Status"])
async def health_check():
    """Endpoint to check if the API is running."""
    return {"status": "ok"}

@app.post("/ask", tags=["Q&A"])
async def ask(req: QuestionRequest, request: Request):
    """
    Receives a question, invokes the QA chain, and returns the answer.
    """
    qa_chain = request.app.state.qa_chain
    if not qa_chain:
        raise HTTPException(status_code=503, detail="Service not available.")
    
    try:
        logger.info(f"Invoking QA chain for question: \"{req.question[:50]}...\"")
        # The chain expects a simple string input now
        answer = qa_chain.invoke(req.question)
        logger.info("QA chain invocation successful.")
        
        # The current chain only returns a string.
        # Returning sources would require chain modification.
        return {
            "answer": answer,
            "sources": [] 
        }
    except Exception as e:
        logger.error(f"An error occurred in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the LegalQA API!",
        "endpoints": {
            "/ask": "POST - Ask a question to the system"
        }
    }