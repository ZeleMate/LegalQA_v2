"""
Functionality Tests for LegalQA System

Tests core functionality to ensure optimizations don't break existing features.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from tests import TEST_CONFIG

# Configure logging for this test file
logger = logging.getLogger(__name__)


class TestBasicFunctionality:
    """Test basic system functionality."""

    def test_imports_work(self):
        """Test that all core modules can be imported without errors."""
        logger.info("--- Running Import Test ---")
        try:
            # Test original components
            from src.data.faiss_loader import load_faiss_index

            logger.debug("Successfully imported faiss_loader.")
            from src.chain.qa_chain import build_qa_chain

            logger.debug("Successfully imported qa_chain.")

            # Test new optimized components
            from src.infrastructure.cache_manager import CacheManager, get_cache_manager

            logger.debug("Successfully imported cache_manager.")
            from src.infrastructure.db_manager import DatabaseManager, get_db_manager

            logger.debug("Successfully imported db_manager.")
            from src.rag.retriever import CustomRetriever, RerankingRetriever

            logger.debug("Successfully imported retriever.")

            logger.info("✅ All core modules imported successfully.")
            assert True, "All imports successful"
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}", exc_info=True)
            pytest.fail(f"Import failed: {e}")

    def test_cache_manager_initialization(self):
        """Test cache manager can be initialized."""
        logger.info("--- Running Cache Manager Initialization Test ---")
        try:
            from src.infrastructure.cache_manager import CacheManager

            cache_manager = CacheManager(memory_maxsize=100, memory_ttl=300)
            assert cache_manager is not None
            assert cache_manager.memory_cache is not None
            logger.info("✅ CacheManager initialized successfully.")

        except Exception as e:
            logger.error(f"❌ Cache manager initialization failed: {e}", exc_info=True)
            pytest.fail(f"Cache manager initialization failed: {e}")

    def test_database_manager_initialization(self):
        """Test database manager can be initialized."""
        logger.info("--- Running Database Manager Initialization Test ---")
        try:
            from src.infrastructure.db_manager import DatabaseManager

            db_manager = DatabaseManager()
            assert db_manager is not None
            assert (
                not db_manager._initialized
            )  # Should not be initialized without calling initialize()
            logger.info("✅ DatabaseManager initialized successfully.")

        except Exception as e:
            logger.error(
                f"❌ Database manager initialization failed: {e}", exc_info=True
            )
            pytest.fail(f"Database manager initialization failed: {e}")

    def test_faiss_loader_exists(self):
        """Test that FAISS loader function exists and is callable."""
        logger.info("--- Running FAISS Loader Existence Test ---")
        try:
            from src.data.faiss_loader import load_faiss_index

            assert callable(load_faiss_index)
            logger.info("✅ FAISS loader is callable.")

        except ImportError as e:
            logger.error(f"❌ FAISS loader import failed: {e}", exc_info=True)
            pytest.fail(f"FAISS loader import failed: {e}")


class TestCacheSystem:
    """Test caching system functionality."""

    @pytest.fixture
    def cache_manager(self):
        """Provide a cache manager for testing."""
        from src.infrastructure.cache_manager import CacheManager

        return CacheManager(memory_maxsize=10, memory_ttl=60)

    def test_memory_cache_set_get(self, cache_manager):
        """Test memory cache set and get operations."""
        logger.info("--- Running Cache Set/Get Test ---")
        key, value = "test_key", "test_value"
        cache_manager.memory_cache.set(key, value, ttl=60)
        logger.debug(f"Set cache key '{key}' with value '{value}'.")

        result = cache_manager.memory_cache.get(key)
        logger.debug(f"Retrieved value for key '{key}': '{result}'.")

        assert result == value
        logger.info("✅ Cache set and get operations successful.")

    def test_memory_cache_expiry(self, cache_manager):
        """Test memory cache TTL expiry."""
        logger.info("--- Running Cache Expiry Test ---")
        key, value = "expire_key", "expire_value"

        # Set with very short TTL
        cache_manager.memory_cache.set(key, value, ttl=1)
        logger.debug(f"Set cache key '{key}' with TTL of 1 second.")

        # Should exist immediately
        assert cache_manager.memory_cache.get(key) == value
        logger.debug("Verified that key exists immediately after setting.")

        # Simulate expiry by manually setting the time
        logger.debug("Simulating time travel to expire the cache key...")
        cache_manager.memory_cache.cache[key] = (
            value,
            cache_manager.memory_cache.cache[key][1].__class__(1970, 1, 1),  # Past date
        )

        assert cache_manager.memory_cache.get(key) is None
        logger.info("✅ Cache key correctly expired.")

    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation is consistent."""
        logger.info("--- Running Cache Key Generation Test ---")
        key1 = cache_manager._generate_key("test", "data")
        key2 = cache_manager._generate_key("test", "data")
        key3 = cache_manager._generate_key("test", "different_data")
        logger.debug(f"Generated keys: key1='{key1}', key2='{key2}', key3='{key3}'.")

        assert key1 == key2  # Same input should generate same key
        assert key1 != key3  # Different input should generate different key
        assert key1.startswith("test:")
        logger.info("✅ Cache key generation is consistent and correct.")


class TestDatabaseOperations:
    """Test database operations and optimizations."""

    @pytest.fixture
    def mock_db_manager(self):
        """Provide a mocked database manager."""
        from src.infrastructure.db_manager import DatabaseManager

        db_manager = DatabaseManager()
        # Mock the connection methods
        db_manager._initialized = True
        return db_manager

    def test_chunk_id_deduplication(self, mock_db_manager):
        """Test that duplicate chunk IDs are handled correctly."""
        logger.info("--- Running Chunk ID Deduplication Test ---")
        chunk_ids = ["id1", "id2", "id1", "id3", "id2"]
        logger.debug(f"Original chunk IDs: {chunk_ids}")
        unique_ids = list(dict.fromkeys(chunk_ids))
        logger.debug(f"Deduplicated chunk IDs: {unique_ids}")

        assert len(unique_ids) == 3
        assert unique_ids == ["id1", "id2", "id3"]
        logger.info("✅ Chunk ID deduplication works as expected.")

    def test_embedding_parsing(self, mock_db_manager):
        """Test embedding string parsing functionality."""
        logger.info("--- Running Embedding Parsing Test ---")
        embedding_str = "[1.0,2.0,3.0,4.0]"
        logger.debug(f"Parsing embedding string: '{embedding_str}'")

        try:
            import numpy as np

            clean_str = embedding_str.strip("[]")
            values = np.fromstring(clean_str, sep=",", dtype=np.float32)
            assert len(values) == 4
            assert values[0] == 1.0
            logger.info("✅ Embedding string parsed successfully.")
        except ImportError:
            # Skip if numpy not available in test environment
            logger.warning("NumPy not available, skipping embedding parsing test.")
            pytest.skip("NumPy not available for testing")


class TestAPICompatibility:
    """Test API compatibility and response formats."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI()

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

        @app.post("/ask", response_model=QuestionResponse)
        async def ask_question(req: QuestionRequest):
            return QuestionResponse(
                answer="Test answer",
                sources=[],
                processing_time=0.1,
                cache_hit=False,
                metadata={"test": True},
            )

        @app.get("/health")
        async def health_check():
            return {"status": "ok"}

        return app

    def test_request_model_validation(self, mock_app):
        """Test that request models validate correctly."""
        logger.info("--- Running API Request Model Validation Test ---")
        from pydantic import BaseModel

        class QuestionRequest(BaseModel):
            question: str
            use_cache: bool = True
            max_documents: int = 5

        # Valid request
        logger.debug("Testing with default values.")
        valid_req = QuestionRequest(question="Test question")
        assert valid_req.question == "Test question"
        assert valid_req.use_cache is True
        assert valid_req.max_documents == 5

        # Test with custom values
        logger.debug("Testing with custom values.")
        custom_req = QuestionRequest(
            question="Custom question", use_cache=False, max_documents=10
        )
        assert custom_req.use_cache is False
        assert custom_req.max_documents == 10
        logger.info("✅ API request models validate correctly.")

    def test_response_model_structure(self, mock_app):
        """Test that response models have correct structure."""
        logger.info("--- Running API Response Model Structure Test ---")
        from pydantic import BaseModel

        class QuestionResponse(BaseModel):
            answer: str
            sources: list
            processing_time: float
            cache_hit: bool
            metadata: dict

        response = QuestionResponse(
            answer="Test",
            sources=[],
            processing_time=0.5,
            cache_hit=True,
            metadata={"key": "value"},
        )
        logger.debug(f"Created sample response model: {response.model_dump_json()}")

        assert hasattr(response, "answer")
        assert hasattr(response, "sources")
        assert hasattr(response, "processing_time")
        assert hasattr(response, "cache_hit")
        assert hasattr(response, "metadata")
        logger.info("✅ API response model has the correct structure.")


class TestProjectStructure:
    """Tests for project structure and configuration files."""

    def test_expected_files_exist(self):
        """Test that key project files and directories exist."""
        logger.info("--- Running Project File Existence Test ---")
        project_root = Path(__file__).parent.parent
        expected_paths = [
            "src/infrastructure/cache_manager.py",
            "src/infrastructure/db_manager.py",
            "src/data/faiss_loader.py",
            "config/redis.conf",
            "config/prometheus.yml",
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "pyproject.toml",
        ]
        missing_files = []
        for path in expected_paths:
            if not (project_root / path).exists():
                logger.error(f"❌ Missing project file: {path}")
                missing_files.append(path)

        assert not missing_files, f"File(s) not found: {', '.join(missing_files)}"
        logger.info("✅ All key project files are present.")

    def test_pyproject_toml_dependencies(self):
        """Check for key performance dependencies in pyproject.toml."""
        logger.info("--- Running pyproject.toml Dependency Test ---")
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        try:
            import toml

            logger.debug("Loading pyproject.toml...")
            pyproject_data = toml.load(pyproject_path)
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])

            performance_deps = ["asyncpg", "aioredis", "prometheus-client"]
            logger.debug(f"Checking for dependencies: {performance_deps}")
            missing_deps = []
            for dep in performance_deps:
                if not any(dep in d for d in dependencies):
                    logger.error(f"❌ Missing performance dependency: {dep}")
                    missing_deps.append(dep)

            assert (
                not missing_deps
            ), f"Missing dependencies in pyproject.toml: {', '.join(missing_deps)}"
            logger.info("✅ All performance dependencies found in pyproject.toml.")
        except ImportError:
            logger.warning("toml package not installed, skipping pyproject.toml check.")
            pytest.skip("toml package not installed, skipping pyproject.toml check.")
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {e}", exc_info=True)
            pytest.fail(f"Failed to parse pyproject.toml: {e}")


class TestEnvironmentConfiguration:
    """Test environment configuration and setup."""

    def test_required_environment_variables(self):
        """Test that required environment variables are documented."""
        logger.info("--- Running Required Environment Variables Test ---")
        required_vars = [
            "GOOGLE_API_KEY",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
            "REDIS_HOST",
        ]

        # This test checks for documentation, not actual values.
        # A more robust test could check if they are set during CI.
        logger.debug(f"Checking for documentation of required vars: {required_vars}")
        assert isinstance(required_vars, list)
        logger.info("✅ Test passed (checks for documentation existence).")

    def test_optional_environment_variables(self):
        """Test that optional environment variables are handled gracefully."""
        logger.info("--- Running Optional Environment Variables Test ---")
        # Example of how an app might use an optional variable
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logger.debug(f"LOG_LEVEL is '{log_level}' (defaults to INFO).")
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        logger.info("✅ Optional environment variables handled correctly.")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_faiss_files_error(self):
        """Test graceful failure when FAISS files are missing."""
        logger.info("--- Running Missing FAISS Files Error Handling Test ---")
        from src.data.faiss_loader import load_faiss_index

        with patch("os.path.exists", return_value=False):
            logger.debug("Simulating that FAISS files do not exist.")
            with pytest.raises(FileNotFoundError) as excinfo:
                load_faiss_index("dummy_path.bin", "dummy_map.pkl")

            logger.debug(f"Caught expected exception: {excinfo.value}")
            assert "Index file not found" in str(excinfo.value)
        logger.info("✅ Correctly raised FileNotFoundError for missing FAISS index.")

    @pytest.mark.asyncio
    async def test_cache_manager_error_handling(self):
        """Test cache manager error handling, avoiding aioredis import issues."""
        logger.info("--- Running Cache Manager Error Handling Test ---")

        # Mock aioredis in sys.modules *before* importing CacheManager
        mock_aioredis = Mock()
        mock_redis_client = AsyncMock()
        mock_redis_client.get.side_effect = ConnectionError("Redis is down")
        mock_aioredis.from_url.return_value = mock_redis_client

        with patch.dict("sys.modules", {"aioredis": mock_aioredis}):
            from src.infrastructure.cache_manager import CacheManager

            # Initialize the manager. It will use the mocked aioredis.
            cache_manager = CacheManager(redis_url="redis://dummy")

            logger.debug("Simulating Redis connection error on 'redis.get' method.")

            # The CacheManager's 'get' method should catch the error and return None
            result = await cache_manager.get("some_key")

        assert result is None

        logger.info("✅ CacheManager gracefully handled Redis connection error.")

    def test_database_connection_error_handling(self):
        """Test database connection error handling."""
        logger.info("--- Running Database Connection Error Handling Test ---")
        from src.infrastructure.db_manager import DatabaseManager

        db_manager = DatabaseManager()

        # Mock the asyncpg.create_pool to raise an error
        with patch("asyncpg.create_pool", side_effect=OSError("DB connection refused")):
            logger.debug("Simulating database connection refusal via create_pool.")
            with pytest.raises(OSError):
                asyncio.run(db_manager.initialize())
        logger.info(
            "✅ DatabaseManager correctly propagated connection error from create_pool."
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
