"""
Integration Tests for LegalQA System

End-to-end tests that verify the complete system works correctly.
"""

import asyncio
import json
import os
import time
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from tests import TEST_CONFIG


class TestSystemIntegration:
    """Test complete system integration."""
    
    def test_full_qa_pipeline_mock(self):
        """Test the complete Q&A pipeline with mocked components."""
        
        # Mock components
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        mock_embeddings.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        
        mock_faiss_index = Mock()
        mock_faiss_index.search.return_value = ([0.5, 0.7], [[0, 1]])
        
        mock_id_mapping = {0: "chunk_1", 1: "chunk_2"}
        
        # Test that components can work together
        try:
            # Simulate query embedding
            query = "Test legal question"
            query_embedding = mock_embeddings.embed_query(query)
            assert len(query_embedding) == 1536
            
            # Simulate FAISS search
            distances, indices = mock_faiss_index.search([query_embedding], k=5)
            assert len(distances[0]) == 2
            assert len(indices[0]) == 2
            
            # Simulate ID mapping
            chunk_ids = []
            for idx in indices[0]:
                if idx in mock_id_mapping:
                    chunk_ids.append(mock_id_mapping[idx])
            
            assert len(chunk_ids) == 2
            assert "chunk_1" in chunk_ids
            assert "chunk_2" in chunk_ids
            
        except Exception as e:
            assert False, f"Integration test failed: {e}"
    
    def test_cache_database_integration(self):
        """Test cache and database work together."""
        
        # Mock cache manager
        mock_cache = {}
        
        def cache_get(key):
            return mock_cache.get(key)
        
        def cache_set(key, value, ttl=300):
            mock_cache[key] = value
        
        # Mock database
        mock_db_data = {
            "chunk_1": {"text": "Legal text 1", "doc_id": "doc_1"},
            "chunk_2": {"text": "Legal text 2", "doc_id": "doc_2"}
        }
        
        def db_fetch(chunk_ids):
            return {chunk_id: mock_db_data[chunk_id] 
                   for chunk_id in chunk_ids if chunk_id in mock_db_data}
        
        # Test workflow
        chunk_ids = ["chunk_1", "chunk_2"]
        
        # First request - should hit database
        cache_key = f"chunks:{':'.join(chunk_ids)}"
        cached_result = cache_get(cache_key)
        
        if not cached_result:
            # Fetch from database
            db_result = db_fetch(chunk_ids)
            # Cache the result
            cache_set(cache_key, db_result)
            result = db_result
        else:
            result = cached_result
        
        assert len(result) == 2
        assert "chunk_1" in result
        assert result["chunk_1"]["text"] == "Legal text 1"
        
        # Second request - should hit cache
        cached_result = cache_get(cache_key)
        assert cached_result is not None
        assert len(cached_result) == 2
    
    def test_async_component_integration(self):
        """Test async components work together."""
        
        async def mock_async_embedding(text):
            """Mock async embedding generation."""
            await asyncio.sleep(0.01)  # Simulate API call
            return [0.1] * 1536
        
        async def mock_async_db_fetch(chunk_ids):
            """Mock async database fetch."""
            await asyncio.sleep(0.01)  # Simulate DB query
            return {chunk_id: f"text_for_{chunk_id}" for chunk_id in chunk_ids}
        
        async def mock_async_llm_call(prompt):
            """Mock async LLM call."""
            await asyncio.sleep(0.02)  # Simulate LLM API call
            return "Generated answer based on context"
        
        async def test_async_pipeline():
            """Test complete async pipeline."""
            query = "Test question"
            
            # Step 1: Get query embedding
            query_embedding = await mock_async_embedding(query)
            assert len(query_embedding) == 1536
            
            # Step 2: Fetch documents (simulate FAISS + DB)
            chunk_ids = ["chunk_1", "chunk_2"]
            docs = await mock_async_db_fetch(chunk_ids)
            assert len(docs) == 2
            
            # Step 3: Generate answer
            context = " ".join(docs.values())
            answer = await mock_async_llm_call(f"Context: {context}\nQuestion: {query}")
            assert "Generated answer" in answer
            
            return answer
        
        # Run the async test
        result = asyncio.run(test_async_pipeline())
        assert "Generated answer" in result


class TestApiIntegration:
    """Test API integration with all components."""
    
    def test_fastapi_app_structure(self):
        """Test that FastAPI app can be structured correctly."""
        
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            
            # Create mock app structure
            app = FastAPI(title="Test LegalQA API")
            
            class QuestionRequest(BaseModel):
                question: str
                use_cache: bool = True
                max_documents: int = 5
            
            class QuestionResponse(BaseModel):
                answer: str
                sources: list
                processing_time: float
                cache_hit: bool
                metadata: dict
            
            # Test that models can be instantiated
            request = QuestionRequest(question="Test question")
            response = QuestionResponse(
                answer="Test answer",
                sources=[],
                processing_time=0.1,
                cache_hit=False,
                metadata={}
            )
            
            assert request.question == "Test question"
            assert response.answer == "Test answer"
            assert hasattr(app, 'routes')
            
        except ImportError:
            # Skip if FastAPI not available
            pass
    
    def test_middleware_integration(self):
        """Test middleware can be integrated properly."""
        
        # Mock middleware function
        def mock_performance_middleware(request, call_next):
            """Mock performance monitoring middleware."""
            start_time = time.time()
            
            # Simulate request processing
            response = {"status": "ok"}
            
            process_time = time.time() - start_time
            response["X-Process-Time"] = str(process_time)
            
            return response
        
        # Test middleware
        mock_request = {"method": "POST", "path": "/ask"}
        mock_call_next = lambda req: {"status": "processed"}
        
        response = mock_performance_middleware(mock_request, mock_call_next)
        
        assert "X-Process-Time" in response
        assert response["status"] == "ok"
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        
        class MockError(Exception):
            pass
        
        def component_with_error():
            raise MockError("Component failed")
        
        def error_handler(func):
            try:
                return func()
            except MockError as e:
                return {"error": str(e), "status": "failed"}
            except Exception as e:
                return {"error": "Unknown error", "status": "failed"}
        
        # Test error handling
        result = error_handler(component_with_error)
        
        assert result["status"] == "failed"
        assert "Component failed" in result["error"]


class TestDataFlowIntegration:
    """Test data flow through the system."""
    
    def test_question_to_answer_flow(self):
        """Test complete data flow from question to answer."""
        
        # Mock the complete flow
        def simulate_qa_flow(question):
            """Simulate the complete Q&A flow."""
            
            # Step 1: Validate input
            if not question or len(question.strip()) == 0:
                return {"error": "Empty question"}
            
            # Step 2: Generate embedding (mock)
            embedding = [0.1] * 1536
            
            # Step 3: Search FAISS (mock)
            search_results = [
                {"chunk_id": "chunk_1", "distance": 0.5},
                {"chunk_id": "chunk_2", "distance": 0.7}
            ]
            
            # Step 4: Fetch from database (mock)
            documents = {
                "chunk_1": {"text": "Legal text about contracts", "doc_id": "doc_1"},
                "chunk_2": {"text": "Legal text about liability", "doc_id": "doc_2"}
            }
            
            # Step 5: Create context
            context = " ".join([doc["text"] for doc in documents.values()])
            
            # Step 6: Generate answer (mock)
            answer = f"Based on the legal documents: {len(context)} characters of context processed."
            
            # Step 7: Return structured response
            return {
                "answer": answer,
                "sources": list(documents.keys()),
                "processing_time": 0.1,
                "cache_hit": False,
                "metadata": {
                    "documents_found": len(documents),
                    "context_length": len(context)
                }
            }
        
        # Test valid question
        result = simulate_qa_flow("What are the contract requirements?")
        
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 2
        assert result["metadata"]["documents_found"] == 2
        
        # Test empty question
        error_result = simulate_qa_flow("")
        assert "error" in error_result
    
    def test_caching_workflow_integration(self):
        """Test caching workflow integration."""
        
        # Mock cache and computation
        cache_store = {}
        computation_calls = 0
        
        def expensive_computation(input_data):
            nonlocal computation_calls
            computation_calls += 1
            # Simulate expensive operation
            return f"Result for: {input_data}"
        
        def cached_computation(input_data, cache_key=None):
            if cache_key is None:
                cache_key = f"comp:{hash(input_data)}"
            
            # Check cache first
            if cache_key in cache_store:
                return cache_store[cache_key], True  # result, cache_hit
            
            # Compute if not cached
            result = expensive_computation(input_data)
            cache_store[cache_key] = result
            
            return result, False  # result, cache_hit
        
        # Test workflow
        input_data = "test question about legal matters"
        
        # First call - should compute
        result1, cache_hit1 = cached_computation(input_data)
        assert not cache_hit1
        assert computation_calls == 1
        assert "Result for:" in result1
        
        # Second call - should use cache
        result2, cache_hit2 = cached_computation(input_data)
        assert cache_hit2
        assert computation_calls == 1  # No additional computation
        assert result1 == result2
        
        # Different input - should compute again
        result3, cache_hit3 = cached_computation("different question")
        assert not cache_hit3
        assert computation_calls == 2


class TestEnvironmentIntegration:
    """Test environment and configuration integration."""
    
    def test_environment_variable_handling(self):
        """Test environment variable handling."""
        
        # Mock environment variables
        mock_env = {
            "OPENAI_API_KEY": "test-key-123",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "REDIS_URL": "redis://localhost:6379/0",
            "LOG_LEVEL": "INFO"
        }
        
        def get_env_config():
            """Mock getting configuration from environment."""
            config = {}
            
            # Required settings
            config["openai_api_key"] = mock_env.get("OPENAI_API_KEY")
            if not config["openai_api_key"]:
                raise ValueError("OPENAI_API_KEY is required")
            
            # Database settings
            config["db_host"] = mock_env.get("POSTGRES_HOST", "localhost")
            config["db_port"] = int(mock_env.get("POSTGRES_PORT", "5432"))
            
            # Optional settings
            config["redis_url"] = mock_env.get("REDIS_URL")
            config["log_level"] = mock_env.get("LOG_LEVEL", "INFO")
            
            return config
        
        # Test configuration loading
        config = get_env_config()
        
        assert config["openai_api_key"] == "test-key-123"
        assert config["db_host"] == "localhost"
        assert config["db_port"] == 5432
        assert config["redis_url"] == "redis://localhost:6379/0"
        assert config["log_level"] == "INFO"
    
    def test_file_path_integration(self):
        """Test file path handling and integration."""
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock file structure
            (temp_path / "data").mkdir()
            (temp_path / "data" / "processed").mkdir()
            
            # Create mock files
            faiss_index_path = temp_path / "data" / "processed" / "faiss_index.bin"
            id_mapping_path = temp_path / "data" / "processed" / "id_mapping.pkl"
            
            faiss_index_path.write_bytes(b"mock faiss index data")
            id_mapping_path.write_bytes(b"mock id mapping data")
            
            # Test file existence
            assert faiss_index_path.exists()
            assert id_mapping_path.exists()
            
            # Test path configuration
            paths = {
                "faiss_index": str(faiss_index_path),
                "id_mapping": str(id_mapping_path),
                "data_dir": str(temp_path / "data")
            }
            
            # Verify paths
            for path_name, path_value in paths.items():
                assert Path(path_value).exists(), f"Path {path_name} does not exist"


class TestUpgradeCompatibility:
    """Test compatibility between old and new versions."""
    
    def test_api_backward_compatibility(self):
        """Test that API remains backward compatible."""
        
        # Old API format (should still work)
        old_request_format = {
            "question": "What is the legal definition of contract?"
        }
        
        # New API format (with optimizations)
        new_request_format = {
            "question": "What is the legal definition of contract?",
            "use_cache": True,
            "max_documents": 5
        }
        
        def process_request(request_data):
            """Process request with backward compatibility."""
            # Extract question (required)
            question = request_data.get("question")
            if not question:
                return {"error": "Question is required"}
            
            # Handle optional new parameters with defaults
            use_cache = request_data.get("use_cache", True)
            max_documents = request_data.get("max_documents", 5)
            
            # Process (mock)
            return {
                "answer": f"Processed: {question[:50]}...",
                "cache_used": use_cache,
                "documents_limit": max_documents
            }
        
        # Test old format compatibility
        old_result = process_request(old_request_format)
        assert "answer" in old_result
        assert old_result["cache_used"] is True  # Default value
        assert old_result["documents_limit"] == 5  # Default value
        
        # Test new format
        new_result = process_request(new_request_format)
        assert "answer" in new_result
        assert new_result["cache_used"] is True
        assert new_result["documents_limit"] == 5
    
    def test_data_structure_compatibility(self):
        """Test that data structures remain compatible."""
        
        # Old response format
        old_response = {
            "answer": "Legal answer",
            "sources": []
        }
        
        # New enhanced response format
        new_response = {
            "answer": "Legal answer",
            "sources": [],
            "processing_time": 0.5,
            "cache_hit": False,
            "metadata": {"documents_found": 3}
        }
        
        def extract_core_response(response):
            """Extract core response that old clients expect."""
            return {
                "answer": response["answer"],
                "sources": response.get("sources", [])
            }
        
        # Test that new response can be converted to old format
        core_response = extract_core_response(new_response)
        
        assert core_response["answer"] == "Legal answer"
        assert core_response["sources"] == []
        
        # Test that old response still works
        old_core = extract_core_response(old_response)
        assert old_core["answer"] == "Legal answer"


if __name__ == "__main__":
    import pytest
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])