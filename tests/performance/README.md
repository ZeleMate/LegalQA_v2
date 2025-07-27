# Performance Testing

This directory contains performance and load testing scripts for the LegalQA system.

## Test Files

### `smoke.js`
Basic smoke test for system health and functionality.
- **Duration**: 15 seconds
- **Virtual Users**: 2
- **Thresholds**: 
  - P95 response time < 2s
  - Error rate < 1%

### `rag_performance.js`
Comprehensive RAG performance test with realistic load patterns.
- **Duration**: 11 minutes (ramp up, steady load, stress test, ramp down)
- **Virtual Users**: 10-20 (variable)
- **Thresholds**:
  - P95 response time < 2s
  - P99 response time < 5s
  - Error rate < 1%
  - RAG retrieval time < 1s
  - LLM generation time < 3s
  - Cache hit rate > 80%

## Running Tests

### Prerequisites
- k6 installed: `brew install k6`
- Application running on localhost:8000

### Smoke Test
```bash
k6 run tests/performance/smoke.js
```

### RAG Performance Test
```bash
k6 run tests/performance/rag_performance.js
```

### With Custom Base URL
```bash
BASE_URL=http://your-app-url:8000 k6 run tests/performance/smoke.js
```

## Metrics Collected

### Custom k6 Metrics
- `rag_retrieval_time`: Document retrieval performance
- `rag_rerank_time`: Reranking performance  
- `rag_llm_time`: LLM generation performance
- `rag_cache_hit_rate`: Cache effectiveness
- `rag_relevance_score`: Response relevance

### Prometheus Metrics
- `legalqa_requests_total`: Request counts by endpoint
- `legalqa_request_duration_seconds`: Response time distribution
- `legalqa_rag_retrieval_seconds`: RAG retrieval timing
- `legalqa_cache_hit_rate`: Cache hit rate percentage
- `legalqa_latency_p95_seconds`: 95th percentile latency

## Industry Standards

The tests are designed to meet industry standards for RAG systems:
- **Latency**: P95 < 2s, P99 < 5s
- **Reliability**: Error rate < 1%
- **Cache Performance**: Hit rate > 80%
- **Component Performance**: Retrieval < 1s, LLM < 3s

## Test Scenarios

### Smoke Test Scenarios
1. Health check endpoint
2. Metrics endpoint (Prometheus format)
3. Statistics endpoint
4. QA endpoint with cache miss
5. Cache clear endpoint

### Performance Test Scenarios
1. Realistic legal questions
2. Variable cache usage (70% cache hits)
3. Component timing simulation
4. Health and metrics monitoring
5. Load progression (ramp up → steady → stress → ramp down) 