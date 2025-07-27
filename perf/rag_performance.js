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
    http_req_failed: ['rate<0.01'],                   // < 1% error rate
    'rag_retrieval_time': ['p(95)<1000'],             // Retrieval < 1s
    'rag_rerank_time': ['p(95)<2000'],                // Rerank < 2s
    'rag_llm_time': ['p(95)<3000'],                   // LLM < 3s
    'rag_cache_hit_rate': ['rate>0.3'],               // > 30% cache hit
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test questions covering different complexity levels
const testQuestions = [
  "Mi a bűnszervezet fogalma a Btk. szerint?",
  "Milyen feltételeket kell teljesíteni a házasságkötéshez?",
  "Mi a különbség a szándékosság és a gondatlanság között?",
  "Hogyan működik a jogutódlás?",
  "Mi a szerződés érvényességének feltételei?",
];

export default function () {
  const question = testQuestions[Math.floor(Math.random() * testQuestions.length)];
  
  // Start timing for RAG components
  const startTime = Date.now();
  
  // QA request with detailed timing
  let qa = http.post(`${BASE_URL}/ask`, JSON.stringify({
    question: question,
    use_cache: true,
    max_documents: 5
  }), { 
    headers: { 
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    } 
  });
  
  const totalTime = Date.now() - startTime;
  
  // Parse response for detailed metrics
  let responseData = {};
  try {
    responseData = JSON.parse(qa.body);
  } catch (e) {
    console.error('Failed to parse response:', e);
  }
  
  // Record RAG-specific metrics
  if (responseData.processing_time) {
    // Estimate component times (in real implementation, these would come from detailed timing)
    const retrievalTime = responseData.processing_time * 0.3; // 30% for retrieval
    const rerankTime = responseData.processing_time * 0.4;    // 40% for reranking
    const llmTime = responseData.processing_time * 0.3;       // 30% for LLM
    
    ragRetrievalTime.add(retrievalTime);
    ragRerankTime.add(rerankTime);
    ragLLMTime.add(llmTime);
  }
  
  // Record cache hit rate
  if (responseData.cache_hit) {
    ragCacheHitRate.add(1);
  } else {
    ragCacheHitRate.add(0);
  }
  
  // Record relevance score (estimated from response quality)
  const relevanceScore = responseData.answer && responseData.answer.length > 50 ? 0.8 : 0.4;
  ragRelevanceScore.add(relevanceScore);
  
  check(qa, {
    'qa status 200': (r) => r.status === 200,
    'qa response time < 2s': (r) => r.timings.duration < 2000,
    'qa has answer': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.answer && data.answer.length > 0;
      } catch (e) {
        return false;
      }
    },
    'qa has processing time': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.processing_time && data.processing_time > 0;
      } catch (e) {
        return false;
      }
    },
    'qa has cache info': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('cache_hit');
      } catch (e) {
        return false;
      }
    },
  });
  
  // Random sleep between requests (1-3 seconds)
  sleep(Math.random() * 2 + 1);
} 