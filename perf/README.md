# Performance Testing

This directory contains k6 scripts for system performance testing.

## Smoke Performance Test

The `smoke.js` script is a quick, basic performance test that:

- **Simulates 2 virtual users**
- **Runs for 15 seconds**
- **Tests all main API endpoints**:
  - `/health` - system status
  - `/metrics` - monitoring
  - `/stats` - statistics
  - `/ask` - main QA functionality
  - `/clear-cache` - admin function

## Thresholds

- **95% response time < 5 seconds**
- **Error rate < 5%**
- **Minimum 1 request/second**

## How to run locally

```bash
# Default localhost:8000
k6 run perf/smoke.js

# With custom URL
BASE_URL=http://your-app-url:8000 k6 run perf/smoke.js

# With more VUs (stress test)
k6 run --vus 10 --duration 60s perf/smoke.js
```

## CI/CD Integration

GitHub Actions automatically runs this test on pull requests and main/develop branches as part of the integrated CI workflow.

### Environment Variables

- `APP_URL_STAGING`: The staging environment URL (GitHub Secrets)
- `BASE_URL`: The application URL to test

## Results

Test results are available in the following format:

- **k6-summary.json**: Summary metrics
- **k6-results.json**: Detailed results

## Troubleshooting

If the test fails:

1. Check if the application is running at the specified URL
2. Look at the thresholds - they might be too strict
3. Check network connectivity
4. Look at k6 logs for detailed error messages 