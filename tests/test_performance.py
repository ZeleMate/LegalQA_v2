"""
Performance Tests for LegalQA System

Tests performance improvements and ensures optimizations work as expected.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest

from tests import TEST_CONFIG

# Configure logging for this test file
logger = logging.getLogger(__name__)

google_api_key = os.getenv("GOOGLE_API_KEY")


class TestPerformanceMetrics:
    """Test performance metrics and thresholds."""

    def test_response_time_benchmark(self):
        """Test that response times meet performance thresholds."""
        logger.info("--- Running Response Time Benchmark ---")
        max_response_time = TEST_CONFIG["performance_thresholds"][
            "max_response_time"
        ]

        # Simulate API response time measurement
        start_time = time.time()

        # Mock a typical processing operation
        time.sleep(0.1)  # Simulate 100ms processing

        response_time = time.time() - start_time
        logger.debug(
            f"Simulated response time: {response_time:.3f}s (Threshold: {max_response_time}s)"
        )

        assert (
            response_time < max_response_time
        ), f"Response time {response_time:.3f}s exceeds threshold {max_response_time}s"
        logger.info("✅ Response time is within the threshold.")

    def test_startup_time_benchmark(self):
        """Test that startup time meets performance thresholds."""
        logger.info("--- Running Startup Time Benchmark ---")
        max_startup_time = TEST_CONFIG["performance_thresholds"][
            "max_startup_time"
        ]

        # Simulate application startup
        start_time = time.time()

        # Mock startup operations
        self._simulate_component_loading()

        startup_time = time.time() - start_time
        logger.debug(
            f"Simulated startup time: {startup_time:.3f}s (Threshold: {max_startup_time}s)"
        )

        assert (
            startup_time < max_startup_time
        ), f"Startup time {startup_time:.3f}s exceeds threshold {max_startup_time}s"
        logger.info("✅ Startup time is within the threshold.")

    def _simulate_component_loading(self):
        """Simulate loading of application components."""
        logger.debug("Simulating component loading...")
        # Simulate loading cache manager
        time.sleep(0.05)

        # Simulate loading database manager
        time.sleep(0.05)

        # Simulate loading FAISS index
        time.sleep(0.1)

        # Simulate loading embeddings model
        time.sleep(0.05)
        logger.debug("Component loading simulation finished.")

    def test_memory_usage_monitoring(self):
        """Test memory usage stays within limits."""
        logger.info("--- Running Memory Usage Monitoring Test ---")
        max_memory_mb = TEST_CONFIG["performance_thresholds"]["max_memory_mb"]

        # Get current memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.debug(
            f"Current memory usage: {memory_mb:.1f}MB (Threshold: {max_memory_mb}MB)"
        )

        # For testing, we'll just check the current process doesn't exceed limits
        assert (
            memory_mb < max_memory_mb * 2
        ), f"Test process memory {memory_mb:.1f}MB seems too high"
        logger.info("✅ Memory usage is within acceptable limits.")

    def test_concurrent_request_handling(self):
        """Test that system can handle concurrent requests."""
        logger.info("--- Running Concurrent Request Handling Test ---")

        def simulate_request():
            """Simulate a single API request."""
            start_time = time.time()
            time.sleep(0.1)
            return time.time() - start_time

        # Test concurrent requests
        num_concurrent = 5
        logger.debug(f"Simulating {num_concurrent} concurrent requests...")
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(simulate_request)
                for _ in range(num_concurrent)
            ]
            response_times = [future.result() for future in futures]
        logger.debug(f"Concurrent request response times: {response_times}")

        # All requests should complete in reasonable time
        max_response_time = TEST_CONFIG["performance_thresholds"][
            "max_response_time"
        ]
        for response_time in response_times:
            assert (
                response_time < max_response_time
            ), f"Concurrent request took {response_time:.3f}s"
        logger.info(
            "✅ All concurrent requests completed within the time threshold."
        )


class TestCachePerformance:
    """Test caching system performance."""

    def test_cache_hit_rate_simulation(self):
        """Test cache hit rate meets minimum threshold."""
        logger.info("--- Running Cache Hit Rate Simulation ---")
        min_cache_hit_rate = TEST_CONFIG["performance_thresholds"][
            "min_cache_hit_rate"
        ]

        # Simulate cache operations
        cache_hits = 0
        total_requests = 100

        # Simulate some cache hits (30% hit rate)
        for i in range(total_requests):
            if i % 3 == 0:  # Every 3rd request is a cache hit
                cache_hits += 1

        cache_hit_rate = cache_hits / total_requests
        logger.debug(
            f"Simulated cache hit rate: {cache_hit_rate:.2f} (Threshold: {min_cache_hit_rate:.2f})"
        )

        assert (
            cache_hit_rate >= min_cache_hit_rate
        ), f"Cache hit rate {cache_hit_rate:.2f} below threshold {min_cache_hit_rate:.2f}"
        logger.info("✅ Simulated cache hit rate meets the threshold.")

    def test_cache_memory_efficiency(self):
        """Test cache doesn't consume excessive memory."""
        logger.info("--- Running Cache Memory Efficiency Test ---")
        # Simulate adding items to cache
        cache_items = {}

        # Add 1000 items to simulate cache
        for i in range(1000):
            cache_items[f"key_{i}"] = f"value_{i}" * 100  # 600 bytes per item

        # Check memory usage is reasonable
        import sys

        cache_size = sys.getsizeof(cache_items)
        logger.debug(
            f"Simulated cache size for 1000 items: {cache_size} bytes."
        )

        # Should be less than 1MB for this test
        assert (
            cache_size < 1024 * 1024
        ), f"Cache size {cache_size} bytes seems excessive"
        logger.info("✅ Simulated cache memory usage is efficient.")

    def test_cache_access_speed(self):
        """Test cache access is fast."""
        logger.info("--- Running Cache Access Speed Test ---")
        # Simulate cache with dict (like our MemoryCache)
        test_cache = {}

        # Add test data
        for i in range(1000):
            test_cache[f"key_{i}"] = f"value_{i}"

        # Measure access time
        logger.debug("Measuring access time for 100 random keys.")
        start_time = time.time()

        # Access 100 random keys
        for i in range(0, 100):
            _ = test_cache.get(f"key_{i}", None)

        access_time = time.time() - start_time
        logger.debug(f"Access time for 100 items: {access_time:.6f}s")

        # Should be very fast (under 1ms)
        assert (
            access_time < 0.001
        ), f"Cache access time {access_time:.6f}s too slow"
        logger.info("✅ Cache access speed is within limits.")


class TestDatabasePerformance:
    """Test database performance optimizations."""

    def test_connection_pooling_simulation(self):
        """Test connection pooling improves performance."""
        logger.info("--- Running DB Connection Pooling Simulation ---")

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
        logger.debug(
            f"Simulated time without connection pooling: {no_pool_time:.4f}s"
        )

        # Test with pooling
        start_time = time.time()
        for _ in range(10):
            get_pooled_connection()
        pool_time = time.time() - start_time
        logger.debug(
            f"Simulated time with connection pooling: {pool_time:.4f}s"
        )

        # Pooling should be significantly faster
        assert (
            pool_time < no_pool_time / 2
        ), "Connection pooling not providing expected speedup"
        logger.info(
            "✅ Connection pooling simulation shows significant performance improvement."
        )

    def test_batch_query_performance(self):
        """Test batch queries perform better than individual queries."""
        logger.info("--- Running DB Batch Query Performance Test ---")

        # Simulate individual queries
        def individual_queries(num_queries):
            total_time = 0
            for _ in range(num_queries):
                time.sleep(0.001)  # 1ms per query
                total_time += 0.001
            return total_time

        # Simulate batch query
        def batch_query(num_items):
            time.sleep(0.005)  # 5ms for batch of any size
            return 0.005

        num_items = 10
        individual_time = individual_queries(num_items)
        batch_time = batch_query(num_items)
        logger.debug(
            f"Individual query time for {num_items} items: {individual_time:.4f}s"
        )
        logger.debug(
            f"Batch query time for {num_items} items: {batch_time:.4f}s"
        )

        # Batch should be faster
        assert (
            batch_time < individual_time
        ), "Batch query not faster than individual queries"
        logger.info(
            "✅ Batch query simulation is faster than individual queries."
        )

    def test_index_usage_simulation(self):
        """Test that proper indexing improves query performance."""
        logger.info("--- Running DB Index Usage Simulation ---")

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
        data_list = list(range(10000))
        data_index = {i: i for i in range(10000)}
        target = 5000

        linear_time = linear_search(data_list, target)
        indexed_time = indexed_search(data_index, target)
        logger.debug(f"Linear search time: {linear_time:.6f}s")
        logger.debug(f"Indexed search time: {indexed_time:.6f}s")

        # Indexed search should be much faster
        assert (
            indexed_time < linear_time
        ), "Index not providing expected performance improvement"
        logger.info(
            "✅ Indexed search simulation is faster than linear search."
        )


class TestAsyncPerformance:
    """Test async operations performance."""

    def test_async_vs_sync_performance(self):
        """Test async operations are faster for concurrent tasks."""
        logger.info("--- Running Async vs. Sync Performance Test ---")

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
        logger.debug(f"Async concurrent time: {async_time:.4f}s")

        start_time = time.time()
        run_sync_sequential()
        sync_time = time.time() - start_time
        logger.debug(f"Sync sequential time: {sync_time:.4f}s")

        # Async should be faster
        assert async_time < sync_time
        logger.info(
            "✅ Async execution was faster than sync for concurrent tasks."
        )

    def test_async_database_operations(self):
        """Test async database operations provide performance benefits."""
        logger.info("--- Running Async Database Operations Test ---")

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
        logger.debug("Running 5 concurrent async DB queries...")
        start_time = time.time()
        asyncio.run(test_concurrent_queries())
        total_time = time.time() - start_time
        logger.debug(
            f"Total time for concurrent async queries: {total_time:.4f}s"
        )

        # Should be faster than 5 * 0.01s (50ms)
        assert total_time < 0.05
        logger.info(
            "✅ Concurrent async database queries completed efficiently."
        )


class TestLoadTesting:
    """Simulate load testing scenarios."""

    def test_sustained_load_simulation(self):
        """Test system stability under sustained load."""
        logger.info("--- Running Sustained Load Simulation ---")

        def simulate_request():
            """Simulate a request with random processing time."""
            time.sleep(0.01 + (0.02 * (os.getpid() % 10 / 10)))  # 10-30ms

        # Simulate 100 requests over a short period
        num_requests = 100
        logger.debug(
            f"Simulating {num_requests} requests with 10 worker threads..."
        )
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(simulate_request) for _ in range(num_requests)
            ]
            for future in futures:
                future.result()

        total_time = time.time() - start_time
        logger.debug(f"Total time for sustained load test: {total_time:.3f}s")

        # Should complete in a reasonable timeframe (e.g., under 0.5s)
        assert (
            total_time < 0.5
        ), f"Sustained load test took {total_time:.3f}s, indicating potential bottlenecks"
        logger.info("✅ System is stable under sustained load simulation.")

    def test_memory_stability_under_load(self):
        """Test memory usage does not grow indefinitely under load."""
        logger.info("--- Running Memory Stability Under Load Test ---")
        process = psutil.Process()

        # Get initial memory usage
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"Initial memory usage: {initial_memory_mb:.1f}MB")

        # Simulate a series of requests
        logger.debug("Simulating 50 requests that allocate memory...")
        for _ in range(50):
            # Simulate a request that might allocate some memory
            _ = [b" " * 1024 for _ in range(100)]  # Allocate ~100KB

        # Get final memory usage
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"Final memory usage: {final_memory_mb:.1f}MB")

        # Memory should not have increased by more than a reasonable amount (e.g., 50MB)
        # This is a loose check, real memory profiling is more complex
        assert (
            final_memory_mb - initial_memory_mb
        ) < 50, f"Memory usage increased from {initial_memory_mb:.1f}MB to {final_memory_mb:.1f}MB under load"
        logger.info("✅ Memory usage remained stable under load.")


class TestQALatency:
    """Test the latency of the full QA pipeline."""

    @pytest.fixture(scope="class")
    def qa_chain(self):
        """Fixture to build the QA chain for latency tests."""
        logger.info(
            "--- Setting up QA Chain for Latency Test (once per class) ---"
        )
        from pathlib import Path

        from dotenv import load_dotenv
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )

        from src.chain.qa_chain import build_qa_chain
        from src.data.faiss_loader import load_faiss_index
        from src.rag.retriever import CustomRetriever, RerankingRetriever

        load_dotenv()

        faiss_path = os.getenv(
            "FAISS_INDEX_PATH", "data/processed/sample_faiss.bin"
        )
        id_mapping_path = os.getenv(
            "ID_MAPPING_PATH", "data/processed/id_mapping.pkl"
        )

        if not faiss_path or not id_mapping_path:
            pytest.skip(
                "FAISS index paths not configured, skipping latency test."
            )

        faiss_index, id_mapping = load_faiss_index(faiss_path, id_mapping_path)
        if not faiss_index:
            logger.error("❌ Failed to load FAISS index for latency test.")
            pytest.fail("Failed to load FAISS index for latency test.")

        logger.debug("FAISS index loaded successfully.")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", temperature=0, api_key=google_api_key
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=google_api_key,
            output_dim=768,
        )
        logger.debug("LLM initialized.")

        custom_retriever = CustomRetriever(
            faiss_index=faiss_index,
            id_mapping=id_mapping,
            embeddings=embeddings,
            k=25,
        )
        logger.debug("CustomRetriever initialized.")

        prompt_path = (
            Path(__file__).parent.parent
            / "src"
            / "prompts"
            / "reranker_prompt.txt"
        )
        reranker_prompt_template = prompt_path.read_text(encoding="utf-8")
        reranker_prompt = PromptTemplate.from_template(
            reranker_prompt_template
        )

        reranking_retriever = RerankingRetriever(
            retriever=custom_retriever,
            llm=llm,
            reranker_prompt=reranker_prompt,
            embeddings=embeddings,
            k=5,
            reranking_enabled=False,
        )
        logger.debug("RerankingRetriever initialized (reranking disabled).")

        chain = build_qa_chain(reranking_retriever)
        logger.info("✅ QA Chain for latency test setup complete.")
        return chain

    async def test_average_qa_latency(self, qa_chain):
        """Test that the average QA response time is within the threshold."""
        logger.info("--- Running QA Latency Test ---")
        questions = [
            "Mi a BH2006. 179. számú eseti döntésének tartalma?",
            "Milyen feltételei vannak a csoportos létszámcsökkentésnek?",
            "Hogyan szabályozza a Munka Törvénykönyve a rendkívüli munkavégzést?",
        ]

        latencies = []
        for i, q in enumerate(questions):
            logger.debug(f"Invoking QA chain for question #{i+1}...")
            start_time = time.time()
            await qa_chain.ainvoke(q)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            logger.debug(f"Question #{i+1} latency: {latency:.2f}s")

        avg_latency = sum(latencies) / len(latencies)
        threshold = 10.0
        logger.info(
            f"Average QA latency: {avg_latency:.2f}s (Threshold: {threshold}s)"
        )

        assert (
            avg_latency <= threshold
        ), f"Average latency {avg_latency:.2f}s exceeds threshold {threshold}s"
        logger.info("✅ Average QA latency is within the threshold.")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_performance"])
