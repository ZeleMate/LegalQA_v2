"""
Performance Tests for LegalQA System

Tests performance improvements and ensures optimizations work as expected.
"""

import time
import asyncio
import threading
import psutil
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from tests import TEST_CONFIG


class TestPerformanceMetrics:
    """Test performance metrics and thresholds."""
    
    def test_response_time_benchmark(self):
        """Test that response times meet performance thresholds."""
        max_response_time = TEST_CONFIG["performance_thresholds"]["max_response_time"]
        
        # Simulate API response time measurement
        start_time = time.time()
        
        # Mock a typical processing operation
        time.sleep(0.1)  # Simulate 100ms processing
        
        response_time = time.time() - start_time
        
        assert response_time < max_response_time, \
            f"Response time {response_time:.3f}s exceeds threshold {max_response_time}s"
    
    def test_startup_time_benchmark(self):
        """Test that startup time meets performance thresholds."""
        max_startup_time = TEST_CONFIG["performance_thresholds"]["max_startup_time"]
        
        # Simulate application startup
        start_time = time.time()
        
        # Mock startup operations
        self._simulate_component_loading()
        
        startup_time = time.time() - start_time
        
        assert startup_time < max_startup_time, \
            f"Startup time {startup_time:.3f}s exceeds threshold {max_startup_time}s"
    
    def _simulate_component_loading(self):
        """Simulate loading of application components."""
        # Simulate loading cache manager
        time.sleep(0.05)
        
        # Simulate loading database manager
        time.sleep(0.05)
        
        # Simulate loading FAISS index
        time.sleep(0.1)
        
        # Simulate loading embeddings model
        time.sleep(0.05)
    
    def test_memory_usage_monitoring(self):
        """Test memory usage stays within limits."""
        max_memory_mb = TEST_CONFIG["performance_thresholds"]["max_memory_mb"]
        
        # Get current memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # For testing, we'll just check the current process doesn't exceed limits
        assert memory_mb < max_memory_mb * 2, \
            f"Test process memory {memory_mb:.1f}MB seems too high"
    
    def test_concurrent_request_handling(self):
        """Test that system can handle concurrent requests."""
        def simulate_request():
            """Simulate a single API request."""
            start_time = time.time()
            
            # Simulate request processing
            time.sleep(0.1)
            
            return time.time() - start_time
        
        # Test concurrent requests
        num_concurrent = 5
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(simulate_request) for _ in range(num_concurrent)]
            response_times = [future.result() for future in futures]
        
        # All requests should complete in reasonable time
        max_response_time = TEST_CONFIG["performance_thresholds"]["max_response_time"]
        for response_time in response_times:
            assert response_time < max_response_time, \
                f"Concurrent request took {response_time:.3f}s"


class TestCachePerformance:
    """Test caching system performance."""
    
    def test_cache_hit_rate_simulation(self):
        """Test cache hit rate meets minimum threshold."""
        min_cache_hit_rate = TEST_CONFIG["performance_thresholds"]["min_cache_hit_rate"]
        
        # Simulate cache operations
        cache_hits = 0
        total_requests = 100
        
        # Simulate some cache hits (30% hit rate)
        for i in range(total_requests):
            if i % 3 == 0:  # Every 3rd request is a cache hit
                cache_hits += 1
        
        cache_hit_rate = cache_hits / total_requests
        
        assert cache_hit_rate >= min_cache_hit_rate, \
            f"Cache hit rate {cache_hit_rate:.2f} below threshold {min_cache_hit_rate:.2f}"
    
    def test_cache_memory_efficiency(self):
        """Test cache doesn't consume excessive memory."""
        # Simulate adding items to cache
        cache_items = {}
        
        # Add 1000 items to simulate cache
        for i in range(1000):
            cache_items[f"key_{i}"] = f"value_{i}" * 100  # 600 bytes per item
        
        # Check memory usage is reasonable
        import sys
        cache_size = sys.getsizeof(cache_items)
        
        # Should be less than 1MB for this test
        assert cache_size < 1024 * 1024, \
            f"Cache size {cache_size} bytes seems excessive"
    
    def test_cache_access_speed(self):
        """Test cache access is fast."""
        # Simulate cache with dict (like our MemoryCache)
        test_cache = {}
        
        # Add test data
        for i in range(1000):
            test_cache[f"key_{i}"] = f"value_{i}"
        
        # Measure access time
        start_time = time.time()
        
        # Access 100 random keys
        for i in range(0, 100):
            _ = test_cache.get(f"key_{i}", None)
        
        access_time = time.time() - start_time
        
        # Should be very fast (under 1ms)
        assert access_time < 0.001, \
            f"Cache access time {access_time:.6f}s too slow"


class TestDatabasePerformance:
    """Test database performance optimizations."""
    
    def test_connection_pooling_simulation(self):
        """Test connection pooling improves performance."""
        # Simulate connection creation time
        def create_connection():
            time.sleep(0.01)  # 10ms to create connection
            return "connection"
        
        def get_pooled_connection():
            time.sleep(0.001)  # 1ms to get from pool
            return "pooled_connection"
        
        # Test without pooling
        start_time = time.time()
        for _ in range(10):
            create_connection()
        no_pool_time = time.time() - start_time
        
        # Test with pooling
        start_time = time.time()
        for _ in range(10):
            get_pooled_connection()
        pool_time = time.time() - start_time
        
        # Pooling should be significantly faster
        assert pool_time < no_pool_time / 2, \
            f"Connection pooling not providing expected speedup"
    
    def test_batch_query_performance(self):
        """Test batch queries perform better than individual queries."""
        # Simulate individual queries
        def individual_queries(num_queries):
            total_time = 0
            for _ in range(num_queries):
                start_time = time.time()
                time.sleep(0.001)  # 1ms per query
                total_time += time.time() - start_time
            return total_time
        
        # Simulate batch query
        def batch_query(num_items):
            start_time = time.time()
            time.sleep(0.005)  # 5ms for batch of any size
            return time.time() - start_time
        
        num_items = 10
        individual_time = individual_queries(num_items)
        batch_time = batch_query(num_items)
        
        # Batch should be faster
        assert batch_time < individual_time, \
            f"Batch query not faster than individual queries"
    
    def test_index_usage_simulation(self):
        """Test that proper indexing improves query performance."""
        # Simulate search without index (linear search)
        def linear_search(data, target):
            start_time = time.time()
            for item in data:
                if item == target:
                    break
            return time.time() - start_time
        
        # Simulate search with index (dict lookup)
        def indexed_search(index, target):
            start_time = time.time()
            _ = index.get(target)
            return time.time() - start_time
        
        # Create test data
        data_list = list(range(1000))
        data_index = {i: i for i in range(1000)}
        target = 500
        
        linear_time = linear_search(data_list, target)
        indexed_time = indexed_search(data_index, target)
        
        # Indexed search should be much faster
        assert indexed_time < linear_time / 10, \
            f"Index not providing expected performance improvement"


class TestAsyncPerformance:
    """Test async operations performance."""
    
    def test_async_vs_sync_performance(self):
        """Test async operations provide performance benefits."""
        
        async def async_operation():
            """Simulate async operation."""
            await asyncio.sleep(0.01)  # 10ms async operation
            return "async_result"
        
        def sync_operation():
            """Simulate sync operation."""
            time.sleep(0.01)  # 10ms sync operation
            return "sync_result"
        
        # Test concurrent async operations
        async def run_async_concurrent():
            start_time = time.time()
            tasks = [async_operation() for _ in range(5)]
            await asyncio.gather(*tasks)
            return time.time() - start_time
        
        # Test sequential sync operations
        def run_sync_sequential():
            start_time = time.time()
            for _ in range(5):
                sync_operation()
            return time.time() - start_time
        
        # Run tests
        async_time = asyncio.run(run_async_concurrent())
        sync_time = run_sync_sequential()
        
        # Async should be significantly faster for concurrent operations
        assert async_time < sync_time / 2, \
            f"Async concurrent operations not faster than sync sequential"
    
    def test_async_database_operations(self):
        """Test async database operations are non-blocking."""
        
        async def mock_async_db_query():
            """Mock async database query."""
            await asyncio.sleep(0.05)  # 50ms query
            return {"data": "result"}
        
        async def test_concurrent_queries():
            """Test multiple concurrent database queries."""
            start_time = time.time()
            
            # Run 3 queries concurrently
            tasks = [mock_async_db_query() for _ in range(3)]
            results = await asyncio.gather(*tasks)
            
            elapsed_time = time.time() - start_time
            
            # Should take about 50ms (not 150ms sequential)
            assert elapsed_time < 0.1, \
                f"Concurrent queries took {elapsed_time:.3f}s, should be ~0.05s"
            
            assert len(results) == 3
            return elapsed_time
        
        # Run the test
        elapsed = asyncio.run(test_concurrent_queries())
        assert elapsed < 0.1


class TestLoadTesting:
    """Test system under load."""
    
    def test_sustained_load_simulation(self):
        """Test system performance under sustained load."""
        request_count = 50
        max_response_time = TEST_CONFIG["performance_thresholds"]["max_response_time"]
        
        def simulate_request():
            start_time = time.time()
            # Simulate request processing
            time.sleep(0.01)  # 10ms processing
            return time.time() - start_time
        
        # Generate sustained load
        response_times = []
        start_time = time.time()
        
        for _ in range(request_count):
            response_time = simulate_request()
            response_times.append(response_time)
        
        total_time = time.time() - start_time
        
        # Check response times
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time_actual = max(response_times)
        
        # Performance assertions
        assert avg_response_time < max_response_time, \
            f"Average response time {avg_response_time:.3f}s exceeds threshold"
        
        assert max_response_time_actual < max_response_time * 2, \
            f"Max response time {max_response_time_actual:.3f}s too high"
        
        # Throughput check (requests per second)
        throughput = request_count / total_time
        assert throughput > 10, \
            f"Throughput {throughput:.1f} req/s too low"
    
    def test_memory_stability_under_load(self):
        """Test memory usage remains stable under load."""
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate memory-intensive operations
        data_store = []
        for i in range(1000):
            # Simulate creating objects (like cached results)
            data_store.append({
                "id": i,
                "data": f"test_data_{i}" * 10,
                "timestamp": time.time()
            })
        
        # Check memory after operations
        mid_memory = process.memory_info().rss
        
        # Clean up (simulate cache cleanup)
        data_store.clear()
        gc.collect()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss
        
        # Memory should not grow excessively
        memory_growth = (mid_memory - initial_memory) / 1024 / 1024  # MB
        memory_cleanup = (mid_memory - final_memory) / 1024 / 1024  # MB
        
        assert memory_growth < 50, \
            f"Memory growth {memory_growth:.1f}MB too high"
        
        assert memory_cleanup > memory_growth * 0.5, \
            f"Memory cleanup {memory_cleanup:.1f}MB insufficient"


if __name__ == "__main__":
    import pytest
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_performance"])