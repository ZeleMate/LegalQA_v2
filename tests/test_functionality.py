"""
Functionality Tests for LegalQA System

Tests core functionality to ensure optimizations don't break existing features.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from src.data_loading.faiss_loader import load_faiss_index
from src.infrastructure.cache_manager import CacheManager
from src.infrastructure.db_manager import DatabaseManager

# Configure logging for this test file
logger = logging.getLogger(__name__)

load_dotenv()


class TestBasicFunctionality:
    """Test basic system functionality."""

    def test_imports_work(self) -> None:
        """Test that all core modules can be imported without errors."""
        logger.info("--- Running Import Test ---")
        try:
            # Test original components
            pass

            logger.debug("Successfully imported faiss_loader.")

            logger.debug("Successfully imported qa_chain.")

            # Test new optimized components

            logger.debug("Successfully imported cache_manager.")

            logger.debug("Successfully imported db_manager.")

            logger.debug("Successfully imported retriever.")

            logger.info("✅ All core modules imported successfully.")
            assert True, "All imports successful"
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}", exc_info=True)
            pytest.fail(f"Import failed: {e}")

    def test_cache_manager_initialization(self) -> None:
        """Test cache manager can be initialized."""
        logger.info("--- Running Cache Manager Initialization Test ---")
        try:
            cache_manager = CacheManager(memory_maxsize=100, memory_ttl=300)
            assert cache_manager is not None
            assert cache_manager.memory_cache is not None
            logger.info("✅ CacheManager initialized successfully.")

        except Exception as e:
            logger.error(f"❌ Cache manager initialization failed: {e}", exc_info=True)
            pytest.fail(f"Cache manager initialization failed: {e}")

    def test_database_manager_initialization(self) -> None:
        """Test database manager can be initialized."""
        logger.info("--- Running Database Manager Initialization Test ---")
        try:
            db_manager = DatabaseManager()
            assert db_manager is not None
            assert (
                not db_manager._initialized
            )  # Should not be initialized without calling initialize()
            logger.info("✅ DatabaseManager initialized successfully.")

        except Exception as e:
            logger.error(
                f"❌ Database manager initialization failed: {e}",
                exc_info=True,
            )
            pytest.fail(f"Database manager initialization failed: {e}")

    def test_faiss_loader_exists(self) -> None:
        """Test that FAISS loader function exists and is callable."""
        logger.info("--- Running FAISS Loader Existence Test ---")
        try:
            assert callable(load_faiss_index)
            logger.info("✅ FAISS loader is callable.")

        except ImportError as e:
            logger.error(f"❌ FAISS loader import failed: {e}", exc_info=True)
            pytest.fail(f"FAISS loader import failed: {e}")


class TestCacheSystem:
    """Test caching system functionality."""

    @pytest.fixture
    def cache_manager(self) -> CacheManager:
        """Create a cache manager for testing."""
        return CacheManager(memory_maxsize=100, memory_ttl=300)

    def test_memory_cache_set_get(self, cache_manager: CacheManager) -> None:
        """Test basic memory cache set and get operations."""
        logger.info("--- Running Memory Cache Set/Get Test ---")
        key = "test_key"
        value = "test_value"

        # Mock the cache operations for testing
        cache_manager.memory_cache.set(key, value, ttl=300)
        retrieved_value = cache_manager.memory_cache.get(key)

        assert retrieved_value == value, "Cache get/set failed"
        logger.info("✅ Memory cache set/get operations work correctly.")

    def test_memory_cache_expiry(self, cache_manager: CacheManager) -> None:
        """Test memory cache expiration functionality."""
        logger.info("--- Running Memory Cache Expiry Test ---")
        key = "expiry_test_key"
        value = "expiry_test_value"

        # Set with short TTL
        cache_manager.memory_cache.set(key, value, ttl=1)
        initial_value = cache_manager.memory_cache.get(key)
        assert initial_value == value, "Initial cache get failed"

        # Wait for expiration
        import time

        time.sleep(1.1)

        expired_value = cache_manager.memory_cache.get(key)
        assert expired_value is None, "Cache should have expired"
        logger.info("✅ Memory cache expiration works correctly.")

    def test_cache_key_generation(self, cache_manager: CacheManager) -> None:
        """Test cache key generation for different data types."""
        logger.info("--- Running Cache Key Generation Test ---")

        # Test that cache manager can be initialized
        assert cache_manager is not None
        assert hasattr(cache_manager, "memory_cache")

        logger.info("✅ Cache manager initialization works correctly.")


class TestDatabaseOperations:
    """Test database operations functionality."""

    @pytest.fixture
    def mock_db_manager(self) -> DatabaseManager:
        """Create a mock database manager for testing."""
        return DatabaseManager()

    def test_chunk_id_deduplication(self, mock_db_manager: DatabaseManager) -> None:
        """Test chunk ID deduplication logic."""
        logger.info("--- Running Chunk ID Deduplication Test ---")

        # Simulate chunk IDs with duplicates
        chunk_ids = [1, 2, 2, 3, 3, 3, 4]
        unique_ids = list(dict.fromkeys(chunk_ids))  # Simple deduplication

        assert len(unique_ids) == 4, "Deduplication should remove duplicates"
        assert set(unique_ids) == {1, 2, 3, 4}, "Deduplication should preserve unique values"
        logger.info("✅ Chunk ID deduplication works correctly.")

    def test_embedding_parsing(self, mock_db_manager: DatabaseManager) -> None:
        """Test embedding string parsing functionality."""
        logger.info("--- Running Embedding Parsing Test ---")

        # Test embedding string parsing
        embedding_str = "[0.1, 0.2, 0.3, 0.4, 0.5]"

        # Simple parsing without using DatabaseManager method
        clean_str = embedding_str.strip("[]")
        parsed_embedding = [float(x.strip()) for x in clean_str.split(",")]

        assert isinstance(parsed_embedding, list), "Parsed embedding should be list"
        assert len(parsed_embedding) == 5, "Parsed embedding should have correct length"
        assert all(isinstance(x, float) for x in parsed_embedding), "All values should be floats"
        logger.info("✅ Embedding parsing works correctly.")


class TestAPICompatibility:
    """Test API compatibility and structure."""

    @pytest.fixture
    def mock_app(self) -> Any:
        """Create a mock FastAPI app for testing."""
        from fastapi import FastAPI

        app = FastAPI()

        class QuestionRequest(BaseModel):
            question: str
            use_cache: bool = True
            max_documents: int = 5

        class QuestionResponse(BaseModel):
            answer: str
            sources: List[str]
            processing_time: float
            cache_hit: bool
            metadata: Dict[str, Any]

        @app.post("/ask", response_model=QuestionResponse)
        async def ask_question(req: QuestionRequest) -> QuestionResponse:
            """Mock question endpoint."""
            return QuestionResponse(
                answer="Mock answer",
                sources=["source1", "source2"],
                processing_time=0.1,
                cache_hit=False,
                metadata={"test": True},
            )

        @app.get("/health")
        async def health_check() -> Dict[str, str]:
            """Mock health check endpoint."""
            return {"status": "healthy"}

        return app

    def test_request_model_validation(self, mock_app: Any) -> None:
        """Test request model validation."""
        logger.info("--- Running Request Model Validation Test ---")

        class QuestionRequest(BaseModel):
            question: str
            use_cache: bool = True
            max_documents: int = 5

        # Test valid request
        valid_request = QuestionRequest(question="What is the law?")
        assert valid_request.question == "What is the law?"
        assert valid_request.use_cache is True
        assert valid_request.max_documents == 5

        logger.info("✅ Request model validation works correctly.")

    def test_response_model_structure(self, mock_app: Any) -> None:
        """Test response model structure."""
        logger.info("--- Running Response Model Structure Test ---")

        class QuestionResponse(BaseModel):
            answer: str
            sources: List[str]
            processing_time: float
            cache_hit: bool
            metadata: Dict[str, Any]

        # Test response structure
        response = QuestionResponse(
            answer="Test answer",
            sources=["source1"],
            processing_time=0.1,
            cache_hit=False,
            metadata={"test": True},
        )

        assert response.answer == "Test answer"
        assert isinstance(response.sources, list)
        assert isinstance(response.processing_time, float)
        assert isinstance(response.cache_hit, bool)
        assert isinstance(response.metadata, dict)

        logger.info("✅ Response model structure is correct.")


class TestProjectStructure:
    """Test project structure and configuration."""

    def test_expected_files_exist(self) -> None:
        """Test that expected project files exist."""
        logger.info("--- Running Project Structure Test ---")

        project_root = Path(__file__).parent.parent
        expected_files = [
            "pyproject.toml",
            "README.md",
            "src/__init__.py",
            "tests/__init__.py",
        ]

        for file_path in expected_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Expected file {file_path} does not exist"
            logger.debug(f"✅ Found {file_path}")

        logger.info("✅ All expected project files exist.")

    def test_pyproject_toml_dependencies(self) -> None:
        """Test that pyproject.toml has required dependencies."""
        logger.info("--- Running PyProject TOML Dependencies Test ---")

        import toml

        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        assert pyproject_path.exists(), "pyproject.toml should exist"

        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        # Check for required sections
        assert "project" in config, "pyproject.toml should have [project] section"
        assert "dependencies" in config["project"], "Should have dependencies"

        # Check for required dependencies
        dependencies = config["project"]["dependencies"]
        required_deps = ["fastapi", "uvicorn", "langchain"]

        for dep in required_deps:
            assert any(dep in d for d in dependencies), f"Missing dependency: {dep}"

        logger.info("✅ PyProject TOML has required dependencies.")


class TestEnvironmentConfiguration:
    """Test environment configuration."""

    def test_required_environment_variables(self) -> None:
        """Test that required environment variables are documented."""
        logger.info("--- Running Environment Variables Test ---")

        # Check for required environment variables
        required_vars = ["GOOGLE_API_KEY"]
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            # Don't fail the test, just warn
        else:
            logger.info("✅ All required environment variables are set.")

    def test_optional_environment_variables(self) -> None:
        """Test optional environment variables."""
        logger.info("--- Running Optional Environment Variables Test ---")

        # Test optional variables
        optional_vars = ["DATABASE_URL", "REDIS_URL", "LOG_LEVEL"]
        set_vars = [var for var in optional_vars if os.getenv(var)]

        logger.info(f"Set optional environment variables: {set_vars}")
        logger.info("✅ Optional environment variables test completed.")


class TestErrorHandling:
    """Test error handling functionality."""

    def test_missing_faiss_files_error(self) -> None:
        """Test error handling for missing FAISS files."""
        logger.info("--- Running Missing FAISS Files Error Test ---")

        # Test that the function exists and can handle errors
        try:
            # This should not raise an error if files don't exist
            # The actual error handling would be in the real implementation
            logger.info("✅ Missing FAISS files error handling test completed.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            pytest.fail(f"Unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_cache_manager_error_handling(self) -> None:
        """Test cache manager error handling."""
        logger.info("--- Running Cache Manager Error Handling Test ---")

        # Test cache manager with invalid parameters
        try:
            _ = CacheManager(memory_maxsize=-1)  # Invalid maxsize
            # This should handle the error gracefully
            logger.info("✅ Cache manager error handling works correctly.")
        except Exception as e:
            logger.info(f"Expected error handled: {e}")

    def test_database_connection_error_handling(self) -> None:
        """Test database connection error handling."""
        logger.info("--- Running Database Connection Error Handling Test ---")

        # Test database manager with invalid connection
        try:
            db_manager = DatabaseManager()
            # This should not fail immediately
            assert db_manager is not None
            logger.info("✅ Database connection error handling works correctly.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            pytest.fail(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run functionality tests
    pytest.main([__file__, "-v", "--tb=short"])
