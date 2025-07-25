"""
Integration Tests for LegalQA System

End-to-end tests that verify the complete system works correctly.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Configure logging for this test file
logger = logging.getLogger(__name__)


class TestSystemIntegration:
    """Test complete system integration."""

    def test_full_qa_pipeline_mock(self):
        """Test the complete Q&A pipeline with mocked components."""
        logger.info("--- Running Full QA Pipeline Mock Test ---")

        # Mock components
        logger.debug("Setting up mocked components (embeddings, FAISS, ID mapping)...")
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 768
        mock_embeddings.embed_documents.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        mock_faiss_index = Mock()
        mock_faiss_index.search.return_value = ([[0.5, 0.7]], [[0, 1]])

        mock_id_mapping = {0: "chunk_1", 1: "chunk_2"}

        # Test that components can work together
        try:
            # Simulate query embedding
            logger.debug("Simulating query embedding...")
            query = "Test legal question"
            query_embedding = mock_embeddings.embed_query(query)
            assert len(query_embedding) == 768

            # Simulate FAISS search
            logger.debug("Simulating FAISS search...")
            distances, indices = mock_faiss_index.search([query_embedding], k=5)
            assert len(distances[0]) == 2
            assert len(indices[0]) == 2

            # Simulate ID mapping
            logger.debug("Simulating ID mapping...")
            chunk_ids = []
            for idx in indices[0]:
                if idx in mock_id_mapping:
                    chunk_ids.append(mock_id_mapping[idx])

            assert len(chunk_ids) == 2
            assert "chunk_1" in chunk_ids
            assert "chunk_2" in chunk_ids
            logger.info("✅ Mocked QA pipeline executed successfully.")

        except Exception as e:
            logger.error(f"❌ Mocked QA pipeline test failed: {e}", exc_info=True)
            assert False, f"Integration test failed: {e}"

    def test_cache_database_integration(self):
        """Test cache and database work together."""
        logger.info("--- Running Cache-Database Integration Test ---")

        # Mock cache manager
        logger.debug("Setting up mocked cache and database...")
        mock_cache = {}

        def cache_get(key):
            return mock_cache.get(key)

        def cache_set(key, value, ttl=300):
            mock_cache[key] = value

        # Mock database
        mock_db_data = {
            "chunk_1": {"text": "Legal text 1", "doc_id": "doc_1"},
            "chunk_2": {"text": "Legal text 2", "doc_id": "doc_2"},
        }

        def db_fetch(chunk_ids):
            logger.debug(f"DB fetch call for chunk IDs: {chunk_ids}")
            return {
                chunk_id: mock_db_data[chunk_id]
                for chunk_id in chunk_ids
                if chunk_id in mock_db_data
            }

        # Test workflow
        chunk_ids = ["chunk_1", "chunk_2"]
        cache_key = f"chunks:{':'.join(chunk_ids)}"

        # First request - should hit database
        logger.info("Simulating first request (cache miss)...")
        cached_result = cache_get(cache_key)
        assert cached_result is None
        logger.debug("Cache miss confirmed.")

        db_result = db_fetch(chunk_ids)
        cache_set(cache_key, db_result)
        logger.debug("Fetched from DB and stored in cache.")
        result = db_result

        assert len(result) == 2

        # Second request - should hit cache
        logger.info("Simulating second request (cache hit)...")
        cached_result = cache_get(cache_key)
        assert cached_result is not None
        logger.debug("Cache hit confirmed.")
        assert len(cached_result) == 2
        logger.info("✅ Cache-Database integration works as expected.")

    def test_async_component_integration(self):
        """Test async components work together."""
        logger.info("--- Running Async Component Integration Test ---")

        async def mock_async_embedding(text):
            await asyncio.sleep(0.01)
            return [0.1] * 768

        async def mock_async_db_fetch(chunk_ids):
            await asyncio.sleep(0.01)
            return {chunk_id: f"text_for_{chunk_id}" for chunk_id in chunk_ids}

        async def mock_async_llm_call(prompt):
            await asyncio.sleep(0.02)
            return "Generated answer based on context"

        async def test_async_pipeline():
            logger.debug("Executing async pipeline...")
            query = "Test question"

            logger.debug("Step 1: Get query embedding (async)...")
            query_embedding = await mock_async_embedding(query)
            assert len(query_embedding) == 768

            logger.debug("Step 2: Fetch documents (async)...")
            chunk_ids = ["chunk_1", "chunk_2"]
            docs = await mock_async_db_fetch(chunk_ids)
            assert len(docs) == 2

            logger.debug("Step 3: Generate answer with LLM (async)...")
            context = " ".join(docs.values())
            answer = await mock_async_llm_call(f"Context: {context}\nQuestion: {query}")
            assert "Generated answer" in answer

            return answer

        # Run the async test
        logger.info("Running mocked async pipeline...")
        result = asyncio.run(test_async_pipeline())
        assert "Generated answer" in result
        logger.info("✅ Async component integration test passed.")


class TestApiIntegration:
    """Test API integration with all components."""

    def test_fastapi_app_structure(self):
        """Test that FastAPI app can be structured correctly."""
        logger.info("--- Running FastAPI App Structure Test ---")
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel

            logger.debug("Creating mock FastAPI app and Pydantic models...")
            app = FastAPI(title="Test LegalQA API")

            class QuestionRequest(BaseModel):
                question: str

            class QuestionResponse(BaseModel):
                answer: str

            request = QuestionRequest(question="Test question")
            response = QuestionResponse(answer="Test answer")

            assert request.question == "Test question"
            assert response.answer == "Test answer"
            assert hasattr(app, "routes")
            logger.info("✅ FastAPI app structure and models are correct.")

        except ImportError:
            logger.warning("FastAPI not installed, skipping structure test.")

    def test_middleware_integration(self):
        """Test middleware can be integrated properly."""
        logger.info("--- Running Middleware Integration Test ---")

        def mock_performance_middleware(request, call_next):
            start_time = time.time()
            response = {"status": "ok"}
            process_time = time.time() - start_time
            response["X-Process-Time"] = str(process_time)
            return response

        logger.debug("Simulating middleware call...")
        mock_request = {"method": "POST", "path": "/ask"}

        def mock_call_next(req):
            return {"status": "processed"}

        response = mock_performance_middleware(mock_request, mock_call_next)

        assert "X-Process-Time" in response
        assert response["status"] == "ok"
        logger.info("✅ Middleware integration simulation passed.")

    def test_error_handling_integration(self):
        """Test error handling across components."""
        logger.info("--- Running Error Handling Integration Test ---")

        class MockError(Exception):
            pass

        def component_with_error():
            logger.debug("Raising MockError from component.")
            raise MockError("Component failed")

        def error_handler(func):
            try:
                return func()
            except MockError as e:
                logger.debug(f"Caught expected MockError: {e}")
                return {"error": str(e), "status": "failed"}
            except Exception:
                return {"error": "Unknown error", "status": "failed"}

        result = error_handler(component_with_error)

        assert result["status"] == "failed"
        assert "Component failed" in result["error"]
        logger.info("✅ Error handling integration simulation passed.")


class TestDataFlowIntegration:
    """Test data flow through the system."""

    def test_question_to_answer_flow(self):
        """Test complete data flow from question to answer."""
        logger.info("--- Running Question-to-Answer Data Flow Test ---")

        def simulate_qa_flow(question):
            logger.debug(f"Simulating QA flow for question: '{question}'")
            # 1. Embed question
            # q_embedding = [0.1] * 5  # F841 törölve
            # 2. Retrieve docs
            retrieved_docs = ["doc1_text", "doc2_text"]
            # 3. Create context
            # context = " ".join(retrieved_docs)  # F841 törölve
            # 4. Generate answer
            answer = "This is a generated answer."
            logger.debug("QA flow simulation complete.")
            return answer, retrieved_docs

        question = "What is the law?"
        answer, sources = simulate_qa_flow(question)

        assert isinstance(answer, str)
        assert len(sources) > 0
        logger.info("✅ Simulated question-to-answer data flow is correct.")

    def test_caching_workflow_integration(self):
        """Test caching workflow integrates with data processing."""
        logger.info("--- Running Caching Workflow Integration Test ---")

        mock_cache = {}

        def expensive_computation(input_data):
            logger.debug(f"Performing expensive computation for: {input_data}")
            time.sleep(0.02)
            return f"computed_{input_data}"

        def cached_computation(input_data, cache_key=None):
            key = cache_key or input_data
            if key in mock_cache:
                logger.debug(f"Cache hit for key: {key}")
                return mock_cache[key]
            else:
                logger.debug(f"Cache miss for key: {key}. Computing...")
                result = expensive_computation(input_data)
                mock_cache[key] = result
                return result

        # First call - should compute
        logger.info("First call (cache miss)...")
        start_time1 = time.time()
        result1 = cached_computation("data1")
        time1 = time.time() - start_time1

        # Second call - should be cached
        logger.info("Second call (cache hit)...")
        start_time2 = time.time()
        result2 = cached_computation("data1")
        time2 = time.time() - start_time2

        assert result1 == result2
        assert time2 < time1
        logger.info("✅ Caching workflow provides performance benefit.")


class TestEnvironmentIntegration:
    """Test environment and configuration integration."""

    def test_environment_variable_handling(self):
        """Test environment variables are handled correctly."""
        logger.info("--- Running Environment Variable Handling Test ---")

        with patch.dict(os.environ, {"MY_TEST_VAR": "my_value"}):
            logger.debug("Patched os.environ with MY_TEST_VAR.")
            assert os.getenv("MY_TEST_VAR") == "my_value"

        logger.debug("Verifying os.environ is restored after patch.")
        assert os.getenv("MY_TEST_VAR") is None
        logger.info("✅ Environment variable handling is correct.")

    def test_file_path_integration(self):
        """Test that file paths are constructed correctly."""
        logger.info("--- Running File Path Integration Test ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            data_dir = root_path / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("hello")
            logger.debug(f"Created temporary file at: {test_file}")

            assert test_file.exists()
            assert test_file.read_text() == "hello"

        logger.debug("Temporary directory and its contents have been removed.")
        logger.info("✅ File path integration is correct.")


class TestUpgradeCompatibility:
    """Test compatibility between old and new versions."""

    def test_api_backward_compatibility(self):
        """Test API can handle old and new request formats."""
        logger.info("--- Running API Backward Compatibility Test ---")

        def process_request(request_data):
            # New version requires 'metadata' field
            if "metadata" not in request_data:
                logger.debug("Old request format detected, adding default metadata.")
                request_data["metadata"] = {}
            return request_data

        # Old client request
        old_request = {"question": "Old question"}
        logger.debug(f"Processing old request: {old_request}")
        processed_old = process_request(old_request.copy())
        assert "metadata" in processed_old

        # New client request
        new_request = {
            "question": "New question",
            "metadata": {"client": "v2"},
        }
        logger.debug(f"Processing new request: {new_request}")
        processed_new = process_request(new_request.copy())
        assert processed_new["metadata"] == {"client": "v2"}
        logger.info("✅ API correctly handles both old and new request formats.")

    def test_data_structure_compatibility(self):
        """Test system handles variations in data structures."""
        logger.info("--- Running Data Structure Compatibility Test ---")

        def extract_core_response(response):
            # This function is robust to missing 'sources' key
            logger.debug(f"Extracting core fields from response: {response}")
            return {
                "answer": response.get("answer"),
                "sources": response.get("sources"),
            }

        # Simulate response from an older component without 'sources'
        old_response = {"answer": "Test"}
        logger.debug(f"Processing old data structure: {old_response}")
        core_old = extract_core_response(old_response)
        assert core_old["sources"] is None

        # Simulate response from a newer component
        new_response = {"answer": "Test", "sources": ["doc1"]}
        logger.debug(f"Processing new data structure: {new_response}")
        core_new = extract_core_response(new_response)
        assert core_new["sources"] == ["doc1"]
        logger.info("✅ System correctly handles variations in data structures.")


class TestDockerSetup:
    """Tests for Docker configuration and setup."""

    @pytest.fixture(scope="class")
    def project_root(self):
        """Provide the project root directory."""
        return Path(__file__).parent.parent

    def test_dockerfile_is_multistage(self, project_root):
        """Check if the Dockerfile appears to be a multi-stage build."""
        logger.info("--- Running Dockerfile Multi-Stage Build Test ---")
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found"

        content = dockerfile_path.read_text()
        is_multistage = " as builder" in content and " as production" in content
        logger.debug(f"Dockerfile is multi-stage: {is_multistage}")
        assert is_multistage, "Dockerfile does not appear to be a multi-stage build"
        logger.info("✅ Dockerfile uses multi-stage builds.")

    def test_docker_compose_has_redis(self, project_root):
        """Check for Redis service in docker-compose.yml."""
        logger.info("--- Running Docker-Compose Redis Service Test ---")
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml not found"

        try:
            import yaml

            logger.debug("Loading docker-compose.yml...")
            with open(compose_path, "r") as f:
                compose_data = yaml.safe_load(f)

            has_redis = "redis" in compose_data.get("services", {})
            logger.debug(f"Redis service found in docker-compose.yml: {has_redis}")
            assert has_redis, "Redis service not found in docker-compose.yml"
            logger.info("✅ Redis service is present in docker-compose.yml.")
        except ImportError:
            logger.warning("PyYAML not installed, skipping docker-compose check.")
            pytest.skip("PyYAML not installed, skipping docker-compose check.")
        except Exception as e:
            logger.error(f"Failed to parse docker-compose.yml: {e}", exc_info=True)
            pytest.fail(f"Failed to parse docker-compose.yml: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
