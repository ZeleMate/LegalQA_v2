# LegalQA Performance Optimization Testing Checklist

## 🎯 Overview

This checklist ensures that all performance optimizations work correctly and don't break existing functionality. **Run all tests before deployment or merging changes.**

## 📋 Pre-Deployment Testing Checklist

### ✅ **Quick Validation** (2-3 minutes)
```bash
make -f Makefile.optimized validate
```

**Expected Result:** All validation checks should pass
- [ ] Import validation ✅
- [ ] Cache functionality ✅  
- [ ] Database manager ✅
- [ ] File structure ✅
- [ ] Performance configs ✅
- [ ] Backward compatibility ✅
- [ ] Quick performance check ✅

---

### ✅ **Comprehensive Testing** (10-15 minutes)
```bash
make -f Makefile.optimized test
```

**Expected Results:**
- [ ] **Import Tests:** All modules importable (8+ tests)
- [ ] **Functionality Tests:** Core features working (6+ tests)
- [ ] **Performance Tests:** Optimizations effective (8+ tests)
- [ ] **Integration Tests:** Components work together (6+ tests)
- [ ] **Configuration Tests:** Files and configs correct (5+ tests)
- [ ] **Docker Tests:** Container setup optimized (2+ tests)

**Success Criteria:** 90%+ tests passing, detailed report generated

---

### ✅ **Component-Specific Testing**

#### **Cache System Testing**
```bash
make -f Makefile.optimized test-functionality
```

**Verify:**
- [ ] Memory cache operations (set/get/expire)
- [ ] Cache key generation consistency
- [ ] Cache manager initialization
- [ ] Error handling for invalid operations

#### **Database Optimization Testing**
```bash
# Check database optimizations work
make -f Makefile.optimized test-integration
```

**Verify:**
- [ ] Connection pooling simulation
- [ ] Batch query performance
- [ ] Index usage improvements
- [ ] Async database operations

#### **Performance Validation**
```bash
make -f Makefile.optimized test-performance
```

**Verify:**
- [ ] Response time benchmarks (< 2.0s threshold)
- [ ] Memory usage monitoring (< 2GB threshold)  
- [ ] Cache hit rate simulation (> 30%)
- [ ] Concurrent request handling
- [ ] Async vs sync performance benefits

---

### ✅ **Environment & Configuration Testing**

#### **Docker Setup Validation**
```bash
# Test optimized Docker build
make -f Makefile.optimized prod-build

# Verify multi-stage build
docker images | grep legalqa
```

**Verify:**
- [ ] Multi-stage Dockerfile builds successfully
- [ ] Production image size < 1GB
- [ ] All required files copied correctly
- [ ] Health checks configured

#### **Dependencies Validation**
```bash
# Check performance dependencies
grep -A 10 "Performance optimizations" pyproject.toml
```

**Verify performance dependencies present:**
- [ ] `asyncpg>=0.28.0` - Async PostgreSQL driver
- [ ] `aioredis>=2.0.0` - Redis caching
- [ ] `redis[hiredis]>=4.0.0` - High-performance Redis
- [ ] `sqlalchemy[asyncio]>=2.0.0` - Async SQLAlchemy
- [ ] `prometheus-client>=0.17.0` - Monitoring
- [ ] `psutil>=5.9.0` - Performance monitoring

---

### ✅ **Integration & Compatibility Testing**

#### **API Compatibility**
Test that API interface remains backward compatible:

```bash
# Test old request format still works
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test question"}'

# Test new request format works
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test question", "use_cache": true, "max_documents": 5}'
```

**Verify:**
- [ ] Old request format returns valid response
- [ ] New request format includes performance metadata
- [ ] Response structure includes new fields (processing_time, cache_hit)
- [ ] HTTP status codes remain consistent

#### **End-to-End Workflow**
```bash
# Start optimized environment
make -f Makefile.optimized dev-up

# Test complete workflow
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Mi a bűnszervezet fogalma a Btk. szerint?"}'
```

**Verify:**
- [ ] Application starts successfully (< 10s)
- [ ] Health check responds ✅
- [ ] Question processing works
- [ ] Performance headers present (X-Process-Time)
- [ ] Caching system active

---

### ✅ **Performance Validation**

#### **Memory Usage Testing**
```bash
# Monitor memory during operation
docker stats --no-stream

# Check for memory leaks
make -f Makefile.optimized benchmark
```

**Verify:**
- [ ] Memory usage < 1GB per container
- [ ] No significant memory growth over time
- [ ] Cache memory usage reasonable

#### **Response Time Testing**
```bash
# Test response times
make -f Makefile.optimized benchmark

# Load testing (if Apache Bench available)
ab -n 100 -c 5 -T 'application/json' \
   -p question.json \
   http://localhost:8000/ask
```

**Performance Targets:**
- [ ] Average response time < 1.5s
- [ ] 95th percentile < 2.0s
- [ ] Throughput > 10 requests/second
- [ ] Zero error rate under normal load

#### **Cache Performance**
```bash
# Test cache hit rates
curl http://localhost:8000/metrics | grep cache_hits

# Clear cache and test
curl -X POST http://localhost:8000/clear-cache
```

**Verify:**
- [ ] Cache hit rate > 30% after warm-up
- [ ] Cache clearing works
- [ ] No cache-related errors

---

## 🚨 Failure Investigation

### **If Tests Fail:**

1. **Check Error Details**
   ```bash
   cat test_report.json | grep -A 5 "FAIL"
   ```

2. **Verify File Structure**
   ```bash
   ls -la src/infrastructure/
   ls -la config/
   ```

3. **Check Dependencies**
   ```bash
   pip list | grep -E "(asyncpg|aioredis|prometheus)"
   ```

4. **Validate Import Issues**
   ```bash
   python3 -c "from src.infrastructure.cache_manager import CacheManager; print('OK')"
   ```

### **Common Issues & Fixes:**

| Issue | Symptoms | Fix |
|-------|----------|-----|
| Import errors | `ImportError: No module named...` | Install missing dependencies: `pip install -e ".[dev]"` |
| Cache errors | Cache operations fail | Check Redis configuration in `config/redis.conf` |
| Docker issues | Build failures | Verify Dockerfile.optimized syntax |
| Performance degradation | Tests timeout | Check resource allocation in docker-compose |

---

## 📊 Success Criteria

### **Minimum Requirements:**
- [ ] **90%+ tests passing** in comprehensive test suite
- [ ] **All validation checks pass** in quick validation
- [ ] **No import errors** for optimized components
- [ ] **Backward compatibility maintained** for existing API
- [ ] **Performance improvements verified** (response time, memory)

### **Optimal Results:**
- [ ] **95%+ tests passing** with detailed performance metrics
- [ ] **Cache hit rate > 50%** in realistic scenarios
- [ ] **Memory usage < 500MB** per container
- [ ] **Response time < 1.0s** average
- [ ] **Zero critical errors** in error handling tests

---

## 🎯 Final Deployment Checklist

Before deploying to production:

- [ ] ✅ All tests pass (`make test-ci`)
- [ ] ✅ Performance benchmarks meet targets
- [ ] ✅ Memory usage optimized
- [ ] ✅ Cache system operational  
- [ ] ✅ Database optimizations applied
- [ ] ✅ Monitoring configured
- [ ] ✅ Backward compatibility verified
- [ ] ✅ Documentation updated
- [ ] ✅ Environment variables configured
- [ ] ✅ Docker images built and tagged

---

## 📞 Support

If issues arise during testing:

1. **Check logs:** `make logs-follow`
2. **Review test report:** `cat test_report.json`
3. **Validate configuration:** `make validate`
4. **Check health status:** `make health`

**For urgent issues:** Run `make clean` and restart with `make dev-setup && make dev-up`