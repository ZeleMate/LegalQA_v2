import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 2,
  duration: '15s',
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% response time < 2s (industry standard)
    http_req_failed: ['rate<0.01'],    // max 1% error rate (industry standard)
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
    'health response fast': (r) => r.timings.duration < 500, // 500ms
    'health response contains required fields': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.status === 'healthy' && data.components;
      } catch (e) {
        return false;
      }
    },
  });

  // Metrics endpoint - Prometheus format
  let metrics = http.get(`${BASE_URL}/metrics`);
  check(metrics, {
    'metrics status 200': (r) => r.status === 200,
    'metrics response fast': (r) => r.timings.duration < 1000, // 1s
    'metrics contains prometheus format': (r) => r.body.includes('# HELP') && r.body.includes('# TYPE'),
  });

  // Stats endpoint - performance statistics
  let stats = http.get(`${BASE_URL}/stats`);
  check(stats, {
    'stats status 200': (r) => r.status === 200,
    'stats response fast': (r) => r.timings.duration < 1000, // 1s
    'stats contains component information': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.metrics && typeof data.uptime === 'number';
      } catch (e) {
        return false;
      }
    },
  });

  // QA endpoint - Hungarian court decision question
  let qaResponse = http.post(`${BASE_URL}/ask`, JSON.stringify({
    question: "Milyen büntetést szabott ki a bíróság emberölés esetén?",
    use_cache: false
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(qaResponse, {
    'ask status 200': (r) => r.status === 200,
    'ask response fast': (r) => r.timings.duration < 5000, // 5s for RAG pipeline
    'ask response contains answer field': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.answer && data.processing_time;
      } catch (e) {
        return false;
      }
    },
    'ask response contains processing_time field': (r) => {
      try {
        const data = JSON.parse(r.body);
        return typeof data.processing_time === 'number';
      } catch (e) {
        return false;
      }
    },
  });

  // Cache clear endpoint
  let cacheClear = http.post(`${BASE_URL}/clear-cache`);
  check(cacheClear, {
    'cache clear status 200': (r) => r.status === 200,
    'cache clear response fast': (r) => r.timings.duration < 500, // 500ms
  });

  sleep(1);
} 