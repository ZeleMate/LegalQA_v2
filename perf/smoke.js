import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 2,
  duration: '15s',
  thresholds: {
    http_req_duration: ['p(95)<10000'], // 95% response time < 10s (very noise-tolerant)
    http_req_failed: ['rate<0.10'],    // max 10% error rate (very noise-tolerant)
  },
  // Don't fail the test on threshold violations for smoke test
  noConnectionReuse: true,
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Health check - basic system status
  let health = http.get(`${BASE_URL}/health`);
  check(health, {
    'health status 200': (r) => r.status === 200,
    'health response fast': (r) => r.timings.duration < 1000,
    'health response contains required fields': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('status') && data.status === 'healthy';
      } catch (e) {
        return false;
      }
    },
  });

  sleep(0.5);

  // Metrics endpoint - monitoring
  let metrics = http.get(`${BASE_URL}/metrics`);
  check(metrics, {
    'metrics status 200': (r) => r.status === 200,
    'metrics response fast': (r) => r.timings.duration < 1000,
    'metrics contains prometheus format': (r) => r.body && r.body.includes('legalqa_'),
  });

  sleep(0.5);

  // Stats endpoint - system statistics
  let stats = http.get(`${BASE_URL}/stats`);
  check(stats, {
    'stats status 200': (r) => r.status === 200,
    'stats response fast': (r) => r.timings.duration < 1000,
    'stats contains component information': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('components');
      } catch (e) {
        return false;
      }
    },
  });

  sleep(0.5);

  // QA question - main functionality testing
  let qa = http.post(`${BASE_URL}/ask`, JSON.stringify({
    question: "What is the main rule of the Civil Code?",
    use_cache: true,
    max_documents: 3
  }), { 
    headers: { 
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    } 
  });
  
  check(qa, {
    'ask status 200': (r) => r.status === 200,
    'ask response fast': (r) => r.timings.duration < 5000, // QA can be slower
    'ask response contains answer field': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('answer') && data.answer.length > 0;
      } catch (e) {
        return false;
      }
    },
    'ask response contains processing_time field': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.hasOwnProperty('processing_time') && data.processing_time >= 0;
      } catch (e) {
        return false;
      }
    },
  });

  sleep(1);

  // Cache clear endpoint - admin function
  let cache_clear = http.post(`${BASE_URL}/clear-cache`);
  check(cache_clear, {
    'cache clear status 200': (r) => r.status === 200,
    'cache clear response fast': (r) => r.timings.duration < 2000,
  });

  sleep(1);
} 