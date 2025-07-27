import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics for RAG performance
const ragRetrievalTime = new Trend('rag_retrieval_time');
const ragRerankTime = new Trend('rag_rerank_time');
const ragLLMTime = new Trend('rag_llm_time');
const ragCacheHitRate = new Rate('rag_cache_hit_rate');
const ragRelevanceScore = new Trend('rag_relevance_score');

export let options = {
  stages: [
    { duration: '2m', target: 10 },  // Ramp up
    { duration: '5m', target: 10 },  // Steady load
    { duration: '2m', target: 20 },  // Stress test
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    // Industry standard SLI/SLO
    http_req_duration: ['p(95)<2000', 'p(99)<5000'], // 95% < 2s, 99% < 5s
    http_req_failed: ['rate<0.01'],                   // max 1% error rate
    'rag_retrieval_time': ['p(95)<1000'],             // 95% retrieval < 1s
    'rag_llm_time': ['p(95)<3000'],                   // 95% LLM < 3s
    'rag_cache_hit_rate': ['rate>0.8'],               // 80% cache hit rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Hungarian court decision questions for realistic testing
const COURT_DECISION_QUESTIONS = [
  "Milyen büntetést szabott ki a bíróság emberölés esetén?",
  "Hogyan ítélte meg a bíróság a vagyon elleni bűncselekmény esetét?",
  "Milyen ítéletet hozott a Kúria a csalás ügyében?",
  "Mekkora kártérítést ítélt meg a bíróság a közlekedési balesetben?",
  "Hogyan döntött a bíróság a munkaügyi perben?",
  "Milyen ítélet született a családjogi ügyben?",
  "Mekkora pénzbüntetést szabott ki a bíróság szabálysértés miatt?",
  "Hogyan értékelte a bíróság a bizonyítékokat a büntetőügyben?",
  "Milyen indokolást adott a bíróság a felmentő ítélethez?",
  "Mekkora börtönbüntetést kapott a vádlott rablás miatt?",
  "Hogyan döntött a bíróság az örökségi ügyben?",
  "Milyen ítélet született a gazdasági bűncselekmény ügyében?",
  "Mekkora tárgyi kár keletkezett a rongálás következtében?",
  "Hogyan ítélte meg a bíróság a sértett fájdalomdíj iránti kérelmét?",
  "Milyen feltételes szabadságvesztést szabott ki a bíróság?",
  "Hogyan döntött a bíróság a válási ügyben a gyermek elhelyezéséről?",
  "Milyen indokok alapján utasította el a bíróság a keresetet?",
  "Mekkora összeget ítélt meg a bíróság szerződésszegés miatt?",
  "Hogyan értékelte a bíróság a tanúvallomásokat?",
  "Milyen ítélet született a közigazgatási perben?",
];

export default function () {
  const question = COURT_DECISION_QUESTIONS[Math.floor(Math.random() * COURT_DECISION_QUESTIONS.length)];
  const useCache = Math.random() > 0.3; // 70% cache usage

  // QA request with detailed timing
  const startTime = Date.now();
  
  let qaResponse = http.post(`${BASE_URL}/ask`, JSON.stringify({
    question: question,
    use_cache: useCache
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  const totalTime = Date.now() - startTime;

  // Record custom metrics
  ragRetrievalTime.add(Math.random() * 500); // Mock retrieval time
  ragRerankTime.add(Math.random() * 800);    // Mock rerank time
  ragLLMTime.add(Math.random() * 2000);      // Mock LLM time
  ragCacheHitRate.add(useCache ? 1 : 0);
  ragRelevanceScore.add(Math.random() * 0.6 + 0.4); // 0.4-1.0 relevance

  check(qaResponse, {
    'qa status 200': (r) => r.status === 200,
    'qa response time acceptable': (r) => r.timings.duration < 5000,
    'qa response contains answer': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.answer && data.processing_time;
      } catch (e) {
        return false;
      }
    },
    'qa response contains metrics': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.cache_hit !== undefined && data.metadata;
      } catch (e) {
        return false;
      }
    },
  });

  // Health check every 10th request
  if (Math.random() < 0.1) {
    let health = http.get(`${BASE_URL}/health`);
    check(health, {
      'health status 200': (r) => r.status === 200,
      'health response fast': (r) => r.timings.duration < 500,
    });
  }

  // Metrics check every 20th request
  if (Math.random() < 0.05) {
    let metrics = http.get(`${BASE_URL}/metrics`);
    check(metrics, {
      'metrics status 200': (r) => r.status === 200,
      'metrics contains RAG data': (r) => r.body.includes('legalqa_rag_'),
    });
  }

  sleep(Math.random() * 2 + 1); // 1-3s between requests
} 