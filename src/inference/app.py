"""
FastAPI application with performance improvements.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Performance monitoring
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, SecretStr

from src.chain.qa_chain import build_qa_chain

# Import components
from src.data.faiss_loader import load_faiss_index
from src.infrastructure import ensure_database_setup, get_cache_manager, get_db_manager
from src.infrastructure.gemini_embeddings import GeminiEmbeddings
from src.rag.retriever import CustomRetriever, RerankingRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "legalqa_requests_total",
    "Total requests processed",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "legalqa_request_duration_seconds", "Request processing time in seconds"
)
CACHE_HITS = Counter("legalqa_cache_hits_total", "Total cache hits", ["cache_type"])
DATABASE_QUERIES = Counter("legalqa_database_queries_total", "Total database queries")
EMBEDDING_REQUESTS = Counter("legalqa_embedding_requests_total", "Total embedding requests")

# Global application state
app_state: Dict[str, Optional[Any]] = {
    "qa_chain": None,
    "cache_manager": None,
    "db_manager": None,
    "startup_time": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with async initialization."""

    @REQUEST_LATENCY.time()
    async def startup_event():
        """Initializes all application components asynchronously."""
        logger.info("🚀 Starting LegalQA application...")
        load_dotenv()

        # Initialize managers
        get_db_manager()
        get_cache_manager()

        # Load components in parallel
        logger.info("📊 Loading FAISS components...")
        faiss_index, id_mapping = await load_faiss_components()

        logger.info("🤖 Initializing AI models...")
        embeddings, reranker_llm, reranker_prompt, google_api_key = await initialize_models()

        # Initialize cache and database managers
        logger.info("🗄️ Initializing cache and database managers...")
        cache_manager = await initialize_cache()
        db_manager = await initialize_database()

        # Build retrievers
        logger.info("⚡ Building retrieval pipeline...")
        base_retriever = CustomRetriever(
            embeddings=embeddings,
            faiss_index=faiss_index,
            id_mapping=id_mapping,
        )

        reranking_retriever = RerankingRetriever(
            retriever=base_retriever,
            llm=reranker_llm,
            reranker_prompt=reranker_prompt,
            embeddings=embeddings,
        )

        # Build the final QA chain
        final_qa_chain = build_qa_chain(reranking_retriever, google_api_key)

        # Store components in app state
        startup_time = time.time() - startup_start_time
        app_state.update(
            {
                "qa_chain": final_qa_chain,
                "cache_manager": cache_manager,
                "db_manager": db_manager,
                "startup_time": startup_time,
            }
        )

        # Ensure database is set up
        await ensure_database_setup()

        startup_time = app_state.get("startup_time", 0.0) or 0.0
        logger.info("✅ Application startup complete in {:.2f} seconds.".format(startup_time))

    startup_start_time = time.time()
    await startup_event()

    yield

    # Shutdown
    logger.info("🌙 Shutting down application...")
    db_manager = get_db_manager()
    await db_manager.close()

    cache_manager = get_cache_manager()
    await cache_manager.close()
    logger.info("✅ Application shutdown complete.")


async def initialize_cache():
    """Initialize cache manager."""
    logger.info("Initializing cache manager...")
    cache_manager = get_cache_manager()
    return cache_manager


async def initialize_database():
    """Initialize database manager."""
    logger.info("Initializing database manager...")
    db_manager = get_db_manager()
    await db_manager.initialize()
    return db_manager


async def initialize_models():
    """Initialize AI models."""
    logger.info("Initializing AI models...")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    embeddings = GeminiEmbeddings(api_key=google_api_key)

    # Reranker LLM és prompt marad
    google_api_key = os.getenv("GOOGLE_API_KEY")
    reranker_llm = await asyncio.to_thread(
        ChatGoogleGenerativeAI,
        model="gemini-2.5-pro",
        temperature=0,
        api_key=SecretStr(google_api_key) if google_api_key else None,
    )
    reranker_prompt_path = Path(__file__).parent.parent / "prompts" / "reranker_prompt.txt"
    try:
        reranker_template = reranker_prompt_path.read_text(encoding="utf-8")
        reranker_prompt = PromptTemplate.from_template(reranker_template)
    except FileNotFoundError:
        logger.error(f"Reranker prompt not found at: {reranker_prompt_path}")
        raise
    return embeddings, reranker_llm, reranker_prompt, google_api_key


async def load_faiss_components():
    """Load FAISS index components."""
    faiss_index_path = os.getenv("FAISS_INDEX_PATH")
    id_mapping_path = os.getenv("ID_MAPPING_PATH")

    if not faiss_index_path or not id_mapping_path:
        raise ValueError("FAISS_INDEX_PATH and ID_MAPPING_PATH environment variables are required")

    # Load FAISS components in thread pool to avoid blocking
    faiss_index, id_mapping = await asyncio.to_thread(
        load_faiss_index, faiss_index_path, id_mapping_path
    )

    return faiss_index, id_mapping


# Create FastAPI app with settings
app = FastAPI(
    title="LegalQA API",
    description="A high-performance API for answering legal questions.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Add performance monitoring to all requests."""
    start_time = time.time()
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        status = response.status_code

        # Record metrics
        process_time = time.time() - start_time
        REQUEST_LATENCY.observe(process_time)
        REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()

        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Startup-Time"] = str(app_state.get("startup_time", 0))

        return response

    except Exception as e:
        process_time = time.time() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=path, status=500).inc()
        logger.error(f"Request failed: {e}")
        raise


class QuestionRequest(BaseModel):
    """Request model for questions."""

    question: str
    use_cache: bool = True
    max_documents: int = Field(5, ge=1, le=20)


class QuestionResponse(BaseModel):
    """Response model for answers."""

    answer: str
    sources: list
    processing_time: float
    cache_hit: bool
    metadata: dict


@app.get("/health", status_code=200, tags=["Status"])
async def health_check():
    """Enhanced health check with system status."""
    qa_chain = app_state.get("qa_chain")
    cache_manager = app_state.get("cache_manager")
    db_manager = app_state.get("db_manager")

    # Check database connectivity
    db_status = "ok"
    try:
        if db_manager:
            stats = await db_manager.get_database_stats()
            if not stats:
                db_status = "warning"
    except Exception:
        db_status = "error"

    return {
        "status": "ok" if qa_chain else "initializing",
        "components": {
            "qa_chain": "ready" if qa_chain else "not_ready",
            "cache": "ready" if cache_manager else "not_ready",
            "database": db_status,
        },
        "startup_time": app_state.get("startup_time"),
        "uptime": time.time() - (app_state.get("startup_time") or time.time()),
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """Get application performance statistics."""
    db_manager = app_state.get("db_manager")

    stats = {
        "startup_time": app_state.get("startup_time"),
        "uptime": time.time() - (app_state.get("startup_time") or time.time()),
    }

    # Add database stats if available
    if db_manager:
        try:
            db_stats = await db_manager.get_database_stats()
            stats["database"] = db_stats
        except Exception as e:
            logger.warning(f"Failed to get database stats: {e}")

    return stats


@app.post("/ask", response_model=QuestionResponse, tags=["Q&A"])
async def ask_question(req: QuestionRequest, request: Request):
    """
    High-performance question answering endpoint with caching and monitoring.
    """
    start_time = time.time()
    cache_hit = False

    qa_chain = app_state.get("qa_chain")
    cache_manager = app_state.get("cache_manager")

    if not qa_chain:
        raise HTTPException(
            status_code=503,
            detail="Service not available - still initializing",
        )

    try:
        logger.info(f'Processing question: "{req.question[:50]}..."')

        # Check cache if enabled
        answer = None
        if req.use_cache and cache_manager:
            cache_key = cache_manager._generate_key("qa", req.question)
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                answer = cached_result
                cache_hit = True
                CACHE_HITS.labels(cache_type="qa_result").inc()
                logger.info("Cache hit for question")

        # Generate answer if not cached
        if not answer:
            EMBEDDING_REQUESTS.inc()

            # Use async version if available
            if hasattr(qa_chain, "ainvoke"):
                answer = await qa_chain.ainvoke(req.question)
            else:
                # Fallback to sync version in thread pool
                answer = await asyncio.to_thread(qa_chain.invoke, req.question)

            # Cache the result
            if req.use_cache and cache_manager:
                await cache_manager.set(cache_key, answer, ttl=1800)

        processing_time = time.time() - start_time

        logger.info("Question processed successfully in {:.3f}s".format(processing_time))

        return QuestionResponse(
            answer=str(answer) if answer is not None else "No answer generated",
            sources=[],  # Could be enhanced to include actual sources
            processing_time=processing_time,
            cache_hit=cache_hit,
            metadata={
                "question_length": len(req.question),
                "startup_time": app_state.get("startup_time"),
                "max_documents": req.max_documents,
            },
        )

    except Exception as e:
        logger.error("Error processing question: {}".format(str(e)[:60]))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/clear-cache", tags=["Management"])
async def clear_cache():
    """Clear all caches."""
    cache_manager = app_state.get("cache_manager")
    if cache_manager:
        await cache_manager.clear_all()
        return {"status": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=503, detail="Cache manager not available")


@app.get("/", tags=["General"])
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the LegalQA API!",
        "version": app.version,
        "docs_url": "/docs",
        "features": [
            "Multi-level caching",
            "Async database operations",
            "Performance monitoring",
            "Optimized FAISS search",
            "Batch processing",
        ],
        "endpoints": {
            "/ask": "POST - Ask a question to the system",
            "/health": "GET - Health check with component status",
            "/metrics": "GET - Prometheus metrics",
            "/stats": "GET - Performance statistics",
            "/clear-cache": "POST - Clear all caches",
        },
        "startup_time": app_state.get("startup_time"),
        "status": "ready" if app_state.get("qa_chain") else "initializing",
    }
