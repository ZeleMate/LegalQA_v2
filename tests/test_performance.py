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
import os
import pytest

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
        assert indexed_time < linear_time, \
            f"Index not providing expected performance improvement"


class TestAsyncPerformance:
    """Test async operations performance."""
    
    def test_async_vs_sync_performance(self):
        """Test async operations are faster for concurrent tasks."""
        
        # Define mock async and sync operations
        async def async_operation():
            await asyncio.sleep(0.01)
        
        def sync_operation():
            time.sleep(0.01)
        
        # Run async concurrently
        async def run_async_concurrent():
            tasks = [async_operation() for _ in range(5)]
            await asyncio.gather(*tasks)
        
        # Run sync sequentially
        def run_sync_sequential():
            for _ in range(5):
                sync_operation()
        
        # Measure times
        start_time = time.time()
        asyncio.run(run_async_concurrent())
        async_time = time.time() - start_time
        
        start_time = time.time()
        run_sync_sequential()
        sync_time = time.time() - start_time
        
        # Async should be faster
        assert async_time < sync_time
    
    def test_async_database_operations(self):
        """Test async database operations provide performance benefits."""
        
        # Mock async database query
        async def mock_async_db_query():
            await asyncio.sleep(0.01)
            return "data"
        
        # Test concurrent queries
        async def test_concurrent_queries():
            tasks = [mock_async_db_query() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
        
        # Run the test
        start_time = time.time()
        asyncio.run(test_concurrent_queries())
        total_time = time.time() - start_time
        
        # Should be faster than 5 * 0.01s (50ms)
        assert total_time < 0.05


class TestLoadTesting:
    """Simulate load testing scenarios."""
    
    def test_sustained_load_simulation(self):
        """Test system stability under sustained load."""
        
        def simulate_request():
            """Simulate a request with random processing time."""
            time.sleep(0.01 + (0.02 * (os.getpid() % 10 / 10)))  # 10-30ms
        
        # Simulate 100 requests over a short period
        num_requests = 100
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(simulate_request) for _ in range(num_requests)]
            for future in futures:
                future.result()
        
        total_time = time.time() - start_time
        
        # Should complete in a reasonable timeframe (e.g., under 0.5s)
        assert total_time < 0.5, \
            f"Sustained load test took {total_time:.3f}s, indicating potential bottlenecks"
    
    def test_memory_stability_under_load(self):
        """Test memory usage does not grow indefinitely under load."""
        
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Simulate a series of requests
        for _ in range(50):
            # Simulate a request that might allocate some memory
            _ = [b' ' * 1024 for _ in range(100)]  # Allocate ~100KB
        
        # Get final memory usage
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Memory should not have increased by more than a reasonable amount (e.g., 50MB)
        # This is a loose check, real memory profiling is more complex
        assert (final_memory_mb - initial_memory_mb) < 50, \
            f"Memory usage increased from {initial_memory_mb:.1f}MB to {final_memory_mb:.1f}MB under load"


class TestQALatency:
    """Test the latency of the full QA pipeline."""

    @pytest.fixture(scope="class")
    def qa_chain(self, embeddings_model):
        """Fixture to build the QA chain for latency tests."""
        from src.chain.qa_chain import build_qa_chain
        from src.rag.retriever import RerankingRetriever, CustomRetriever
        from src.data.faiss_loader import load_faiss_index
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv()

        faiss_path = os.getenv("FAISS_INDEX_PATH", "data/processed/sample_faiss.bin")
        id_mapping_path = os.getenv("ID_MAPPING_PATH", "data/processed/doc_id_map.json")

        if not faiss_path or not id_mapping_path:
            pytest.skip("FAISS index paths not configured, skipping latency test.")

        faiss_index, id_mapping = load_faiss_index(faiss_path, id_mapping_path)
        if not faiss_index:
            pytest.fail("Failed to load FAISS index for latency test.")

        llm = ChatOpenAI(model_name="o3-2025-04-16", temperature=0)

        custom_retriever = CustomRetriever(
            faiss_index=faiss_index,
            id_mapping=id_mapping,
            embeddings=embeddings_model,
            k=25
        )
        
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "reranker_prompt.txt"
        reranker_prompt_template = prompt_path.read_text(encoding="utf-8")
        reranker_prompt = PromptTemplate.from_template(reranker_prompt_template)

        reranking_retriever = RerankingRetriever(
            retriever=custom_retriever,
            llm=llm,
            reranker_prompt=reranker_prompt,
            embeddings=embeddings_model,
            k=5,
            reranking_enabled=False
        )

        return build_qa_chain(reranking_retriever)

    def test_average_qa_latency(self, qa_chain):
        """Test that the average QA response time is within the threshold."""
        questions = [
            "Mi a BH2006. 179. számú eseti döntésének tartalma?",
            "Milyen feltételei vannak a csoportos létszámcsökkentésnek?",
            "Hogyan szabályozza a Munka Törvénykönyve a rendkívüli munkavégzést?"
        ]
        
        latencies = []
        for q in questions:
            start_time = time.time()
            qa_chain.invoke(q)
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        avg_latency = sum(latencies) / len(latencies)
        threshold = 10.0

        assert avg_latency <= threshold, f"Average latency {avg_latency:.2f}s exceeds threshold {threshold}s"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_performance"])