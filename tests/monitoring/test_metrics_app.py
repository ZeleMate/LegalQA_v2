#!/usr/bin/env python3
"""
Simple test application for metrics testing.
"""

import random
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# Prometheus metrics
REQUEST_COUNT = Counter(
    "legalqa_requests_total",
    "Total requests processed",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "legalqa_request_duration_seconds", "Request processing time in seconds"
)

# RAG-specific metrics
RAG_RETRIEVAL_TIME = Histogram("legalqa_rag_retrieval_seconds", "RAG retrieval time")
RAG_RERANK_TIME = Histogram("legalqa_rag_rerank_seconds", "RAG reranking time")
RAG_LLM_TIME = Histogram("legalqa_rag_llm_seconds", "RAG LLM generation time")
RAG_DOCUMENTS_RETRIEVED = Histogram("legalqa_documents_retrieved", "Number of documents retrieved")
RAG_RELEVANCE_SCORE = Histogram("legalqa_relevance_score", "Document relevance scores")
RAG_CACHE_HIT_RATE = Gauge("legalqa_cache_hit_rate", "Cache hit rate percentage")

# SLI/SLO metrics
LATENCY_P95 = Gauge("legalqa_latency_p95_seconds", "95th percentile latency")
LATENCY_P99 = Gauge("legalqa_latency_p99_seconds", "99th percentile latency")
ERROR_RATE = Gauge("legalqa_error_rate", "Error rate percentage")
QPS = Gauge("legalqa_queries_per_second", "Queries per second")
COST_PER_QUERY = Gauge("legalqa_cost_per_query_usd", "Cost per query in USD")

# Canary/Rollback metrics
CANARY_SUCCESS_RATE = Gauge("legalqa_canary_success_rate", "Canary deployment success rate")
CANARY_LATENCY_DIFF = Gauge("legalqa_canary_latency_diff", "Latency difference in canary")
ROLLBACK_TRIGGERED = Counter("legalqa_rollback_triggered", "Number of rollbacks triggered")

app = FastAPI(title="LegalQA Metrics Test", version="1.0.0")


# Performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next: Any) -> Response:
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
        return response

    except Exception:
        process_time = time.time() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=path, status=500).inc()
        raise


class QuestionRequest(BaseModel):
    question: str
    use_cache: bool = True


class QuestionResponse(BaseModel):
    answer: str
    processing_time: float
    cache_hit: bool
    metadata: dict


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "metrics": "ready",
            "cache": "ready",
            "database": "ready",
        },
        "uptime": time.time(),
    }


@app.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    content = generate_latest()
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", tags=["Monitoring"])
async def get_stats() -> dict[str, Any]:
    """Get application performance statistics."""
    return {
        "uptime": time.time(),
        "metrics": {
            "cache_hit_rate": RAG_CACHE_HIT_RATE._value.get(),
            "error_rate": ERROR_RATE._value.get(),
            "latency_p95": LATENCY_P95._value.get(),
            "qps": QPS._value.get(),
            "cost_per_query": COST_PER_QUERY._value.get(),
        },
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest) -> QuestionResponse:
    """Mock QA endpoint with metrics."""
    start_time = time.time()
    cache_hit = random.choice([True, False]) if req.use_cache else False

    # Simulate RAG pipeline with metrics
    if not cache_hit:
        # Simulate retrieval
        retrieval_time = random.uniform(0.1, 0.5)
        RAG_RETRIEVAL_TIME.observe(retrieval_time)
        time.sleep(retrieval_time)

        # Simulate reranking
        rerank_time = random.uniform(0.2, 0.8)
        RAG_RERANK_TIME.observe(rerank_time)
        time.sleep(rerank_time)

        # Simulate LLM generation
        llm_time = random.uniform(0.5, 2.0)
        RAG_LLM_TIME.observe(llm_time)
        time.sleep(llm_time)

        # Simulate documents retrieved
        docs_count = random.randint(3, 8)
        RAG_DOCUMENTS_RETRIEVED.observe(docs_count)

        # Simulate relevance scores
        for _ in range(docs_count):
            relevance = random.uniform(0.3, 0.9)
            RAG_RELEVANCE_SCORE.observe(relevance)

    # Update cache hit rate
    if cache_hit:
        RAG_CACHE_HIT_RATE.set(85.5)  # Mock 85.5% cache hit rate
    else:
        RAG_CACHE_HIT_RATE.set(75.2)  # Mock 75.2% cache hit rate

    # Update other metrics
    LATENCY_P95.set(random.uniform(1.5, 2.5))
    LATENCY_P99.set(random.uniform(3.0, 5.0))
    ERROR_RATE.set(random.uniform(0.1, 0.5))
    QPS.set(random.uniform(10, 50))
    COST_PER_QUERY.set(random.uniform(0.01, 0.05))

    processing_time = time.time() - start_time

    # Generate mock Hungarian legal answer based on question
    mock_answer = generate_mock_hungarian_answer(req.question)

    return QuestionResponse(
        answer=mock_answer,
        processing_time=processing_time,
        cache_hit=cache_hit,
        metadata={
            "question_length": len(req.question),
            "mock_response": True,
            "language": "hungarian",
        },
    )


def generate_mock_hungarian_answer(question: str) -> str:
    """Generate mock Hungarian court decision answers based on question keywords."""
    question_lower = question.lower()

    if "emberölés" in question_lower or "büntetés" in question_lower:
        return (
            "A Fővárosi Törvényszék a vádlottat emberölés bűntette miatt "
            "12 év szabadságvesztésre ítélte. A bíróság súlyosító körülményként "
            "értékelte a cselekmény kegyetlenségét, enyhítőként pedig a vádlott "
            "beismerő vallomását."
        )
    elif "vagyon elleni" in question_lower or "lopás" in question_lower:
        return (
            "A Pest Megyei Törvényszék a vádlottat folytatólagosan elkövetett "
            "lopás miatt 3 év börtönbüntetésre és 300.000 Ft pénzbüntetésre "
            "ítélte. A kár összege 850.000 Ft volt."
        )
    elif "csalás" in question_lower and "kúria" in question_lower:
        return (
            "A Kúria helybenhagyta a másodfokon 5 év szabadságvesztésre "
            "módosított ítéletet. A csalással okozott kár meghaladta a "
            "10 millió forintot, a vádlott több sértettet károsított meg."
        )
    elif "kártérítés" in question_lower or "közlekedési" in question_lower:
        return (
            "A bíróság 2.5 millió forint kártérítést ítélt meg a sértettnek. "
            "A gépjárművezetőt ittas vezetés és súlyos testi sértés okozása "
            "miatt 2 év felfüggesztett börtönbüntetésre ítélte."
        )
    elif "munkaügyi" in question_lower or "munkajogi" in question_lower:
        return (
            "A Fővárosi Munkaügyi Bíróság jogtalannak minősítette a felmondást. "
            "A munkáltatót kötelezte 800.000 Ft kártérítés megfizetésére és "
            "a munkavállaló visszafoglalására."
        )
    elif "családjogi" in question_lower or "válás" in question_lower:
        return (
            "A bíróság kimondta a házasság felbontását, a kiskorú gyermek "
            "elhelyezését az anya mellett rendelte el. Az apa részére kontakt "
            "jogot biztosított minden második hétvégén."
        )
    elif "pénzbüntetés" in question_lower or "szabálysértés" in question_lower:
        return (
            "A szabálysértési hatóság 80.000 Ft pénzbírságot szabott ki "
            "közúti közlekedés szabályainak megsértése miatt. A jogsértő "
            "sebességtúllépést követett el lakott területen."
        )
    elif "bizonyíték" in question_lower or "büntetőügy" in question_lower:
        return (
            "A bíróság a tanúvallomásokat ellentmondásosnak minősítette, "
            "a tárgyi bizonyítékok alapján azonban megállapította a vádlott "
            "bűnösségét. A DNS-vizsgálat eredménye döntő jelentőségű volt."
        )
    elif "felmentő" in question_lower or "indokolás" in question_lower:
        return (
            "A bíróság felmentette a vádlottat, mivel a vád terhére felhozott "
            "bizonyítékok nem voltak elegendőek a bűnösség megállapításához. "
            "A kétség a vádlott javára szolgált."
        )
    elif "rablás" in question_lower or "börtön" in question_lower:
        return (
            "A Fővárosi Törvényszék a vádlottat rablás bűntette miatt "
            "8 év szabadságvesztésre ítélte. A bűncselekmény során a sértett "
            "súlyos sérüléseket szenvedett."
        )
    elif "örökség" in question_lower or "hagyaték" in question_lower:
        return (
            "A bíróság az öröklési vitában az okiratos örökösöket "
            "részesítette előnyben. A hagyaték értéke 15 millió forint volt, "
            "amelyet egyenlő arányban osztottak fel."
        )
    elif "gazdasági" in question_lower or "költségvetési" in question_lower:
        return (
            "A Fővárosi Törvényszék a vádlottat költségvetési csalás miatt "
            "4 év szabadságvesztésre és vagyonelkobzásra ítélte. Az okozott "
            "kár meghaladta az 50 millió forintot."
        )
    elif "tárgyi kár" in question_lower or "rongálás" in question_lower:
        return (
            "A bíróság 450.000 Ft tárgyi kárt állapított meg a rongálás "
            "következtében. A vádlottat 1 év felfüggesztett börtönbüntetésre "
            "és a kár megtérítésére ítélte."
        )
    elif "fájdalomdíj" in question_lower or "sértett" in question_lower:
        return (
            "A bíróság 1.2 millió forint fájdalomdíjat ítélt meg a sértettnek "
            "a súlyos testi sértés miatt. A gyógyulási idő 6 hónapot vett "
            "igénybe."
        )
    elif "feltételes" in question_lower or "felfüggesztett" in question_lower:
        return (
            "A bíróság 2 év börtönbüntetést szabott ki, amelynek végrehajtását "
            "3 év próbaidőre felfüggesztette. A vádlottnak közérdekű munkát "
            "kell végeznie."
        )
    elif "gyermek elhelyezése" in question_lower:
        return (
            "A bíróság a gyermek elhelyezését az apa mellett rendelte el, "
            "mivel az anya életvitele nem biztosította a gyermek megfelelő "
            "fejlődését. A láthatási jog kétheti rendszerességgel került "
            "megállapításra."
        )
    elif "kereset elutasítás" in question_lower:
        return (
            "A bíróság elutasította a keresetet, mivel a felperes nem tudta "
            "bizonyítani a szerződésszegést. A pervesztesség költségeit a "
            "felperesre hárította."
        )
    elif "szerződésszegés" in question_lower:
        return (
            "A bíróság 2.8 millió forint kártérítést ítélt meg "
            "szerződésszegés miatt. A kötelezett nem teljesítette határidőben "
            "a vállalt szolgáltatást."
        )
    elif "tanúvallomás" in question_lower:
        return (
            "A bíróság a tanúvallomásokat hiteles bizonyítéknak értékelte. "
            "Három tanú egybehangzó vallomása alapján állapította meg a "
            "tényállást."
        )
    elif "közigazgatási" in question_lower:
        return (
            "A Fővárosi Közigazgatási és Munkaügyi Bíróság megsemmisítette "
            "a hatósági határozatot jogszabálysértés miatt. Az ügyet új "
            "eljárásra utalta vissza."
        )
    else:
        return (
            f"A bírósági határozat szerint: {question}. Ez egy mock válasz, "
            "amely szimulálja a valós magyar bírósági határozatok tartalmát "
            "és stílusát."
        )


@app.post("/clear-cache")
async def clear_cache() -> dict[str, str]:
    """Clear cache endpoint."""
    RAG_CACHE_HIT_RATE.set(0.0)
    return {"status": "Cache cleared successfully"}


@app.get("/")
async def read_root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "LegalQA Metrics Test API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics",
            "/stats": "GET - Performance statistics",
            "/ask": "POST - Mock QA endpoint",
            "/clear-cache": "POST - Clear cache",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
