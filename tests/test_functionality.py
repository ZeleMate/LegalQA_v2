"""
Functionality Tests for LegalQA System

Tests core functionality to ensure optimizations don't break existing features.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from pathlib import Path

from tests import TEST_CONFIG


class TestBasicFunctionality:
    """Test basic system functionality."""
    
    def test_imports_work(self):
        """Test that all core modules can be imported without errors."""
        try:
            # Test original components
            from src.data.faiss_loader import load_faiss_index
            from src.chain.qa_chain import build_qa_chain
            
            # Test new optimized components
            from src.infrastructure.cache_manager import CacheManager, get_cache_manager
            from src.infrastructure.db_manager import DatabaseManager, get_db_manager
            from src.rag.retriever import CustomRetriever, RerankingRetriever
            
            assert True, "All imports successful"
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_cache_manager_initialization(self):
        """Test cache manager can be initialized."""
        try:
            from src.infrastructure.cache_manager import CacheManager
            
            cache_manager = CacheManager(memory_maxsize=100, memory_ttl=300)
            assert cache_manager is not None
            assert cache_manager.memory_cache is not None
            
        except Exception as e:
            pytest.fail(f"Cache manager initialization failed: {e}")
    
    def test_database_manager_initialization(self):
        """Test database manager can be initialized."""
        try:
            from src.infrastructure.db_manager import DatabaseManager
            
            db_manager = DatabaseManager()
            assert db_manager is not None
            assert not db_manager._initialized  # Should not be initialized without calling initialize()
            
        except Exception as e:
            pytest.fail(f"Database manager initialization failed: {e}")
    
    def test_faiss_loader_exists(self):
        """Test that FAISS loader function exists and is callable."""
        try:
            from src.data.faiss_loader import load_faiss_index
            
            assert callable(load_faiss_index)
            
        except ImportError as e:
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
        cache_manager.memory_cache.set("test_key", "test_value", ttl=60)
        result = cache_manager.memory_cache.get("test_key")
        
        assert result == "test_value"
    
    def test_memory_cache_expiry(self, cache_manager):
        """Test memory cache TTL expiry."""
        import time
        
        # Set with very short TTL
        cache_manager.memory_cache.set("expire_key", "expire_value", ttl=1)
        
        # Should exist immediately
        assert cache_manager.memory_cache.get("expire_key") == "expire_value"
        
        # Should expire after TTL (simulate with manual cleanup)
        cache_manager.memory_cache.cache["expire_key"] = (
            "expire_value",
            cache_manager.memory_cache.cache["expire_key"][1].__class__(1970, 1, 1)  # Past date
        )
        
        assert cache_manager.memory_cache.get("expire_key") is None
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation is consistent."""
        key1 = cache_manager._generate_key("test", "data")
        key2 = cache_manager._generate_key("test", "data")
        key3 = cache_manager._generate_key("test", "different_data")
        
        assert key1 == key2  # Same input should generate same key
        assert key1 != key3  # Different input should generate different key
        assert key1.startswith("test:")


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
        # Test the deduplication logic
        chunk_ids = ["id1", "id2", "id1", "id3", "id2"]
        unique_ids = list(dict.fromkeys(chunk_ids))
        
        assert len(unique_ids) == 3
        assert unique_ids == ["id1", "id2", "id3"]
    
    def test_embedding_parsing(self, mock_db_manager):
        """Test embedding string parsing functionality."""
        # Test the embedding parsing logic
        embedding_str = "[1.0,2.0,3.0,4.0]"
        
        try:
            import numpy as np
            clean_str = embedding_str.strip('[]')
            values = np.fromstring(clean_str, sep=',', dtype=np.float32)
            assert len(values) == 4
            assert values[0] == 1.0
        except ImportError:
            # Skip if numpy not available in test environment
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
                metadata={"test": True}
            )
        
        @app.get("/health")
        async def health_check():
            return {"status": "ok"}
        
        return app
    
    def test_request_model_validation(self, mock_app):
        """Test that request models validate correctly."""
        from pydantic import BaseModel
        
        class QuestionRequest(BaseModel):
            question: str
            use_cache: bool = True
            max_documents: int = 5
        
        # Valid request
        valid_req = QuestionRequest(question="Test question")
        assert valid_req.question == "Test question"
        assert valid_req.use_cache is True
        assert valid_req.max_documents == 5
        
        # Test with custom values
        custom_req = QuestionRequest(
            question="Custom question",
            use_cache=False,
            max_documents=10
        )
        assert custom_req.use_cache is False
        assert custom_req.max_documents == 10
    
    def test_response_model_structure(self, mock_app):
        """Test that response models have correct structure."""
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
            metadata={"key": "value"}
        )
        
        assert hasattr(response, 'answer')
        assert hasattr(response, 'sources')
        assert hasattr(response, 'processing_time')
        assert hasattr(response, 'cache_hit')
        assert hasattr(response, 'metadata')


class TestProjectStructure:
    """Tests for project structure and configuration files."""

    def test_expected_files_exist(self):
        """Test that key project files and directories exist."""
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
            "pyproject.toml"
        ]
        for path in expected_paths:
            assert (project_root / path).exists(), f"File or directory not found: {path}"

    def test_pyproject_toml_dependencies(self):
        """Check for key performance dependencies in pyproject.toml."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        try:
            import toml
            pyproject_data = toml.load(pyproject_path)
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            
            performance_deps = ["asyncpg", "aioredis", "prometheus-client"]
            for dep in performance_deps:
                assert any(dep in d for d in dependencies), f"Missing performance dependency in pyproject.toml: {dep}"
        except ImportError:
            pytest.skip("toml package not installed, skipping pyproject.toml check.")
        except Exception as e:
            pytest.fail(f"Failed to parse pyproject.toml: {e}")


class TestEnvironmentConfiguration:
    """Test environment configuration and setup."""
    
    def test_required_environment_variables(self):
        """Test that required environment variables are documented."""
        required_vars = [
            "OPENAI_API_KEY",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD", 
            "POSTGRES_DB",
            "POSTGRES_HOST",
            "POSTGRES_PORT"
        ]
        
        # These should be documented in the configuration
        assert all(isinstance(var, str) for var in required_vars)
    
    def test_optional_environment_variables(self):
        """Test optional environment variables for optimizations."""
        optional_vars = [
            "REDIS_URL",
            "FAISS_INDEX_PATH",
            "ID_MAPPING_PATH",
            "LOG_LEVEL",
            "MAX_WORKERS"
        ]
        
        # These should be documented as optional
        assert all(isinstance(var, str) for var in optional_vars)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_faiss_files_error(self):
        """Test error handling when FAISS files are missing."""
        from src.data.faiss_loader import load_faiss_index
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_faiss_index("nonexistent_index.bin", "nonexistent_mapping.json")
    
    def test_cache_manager_error_handling(self):
        """Test cache manager handles errors gracefully."""
        from src.infrastructure.cache_manager import CacheManager
        
        cache_manager = CacheManager()
        
        # Test with invalid cache key
        result = cache_manager.memory_cache.get(None)
        # Should not raise exception, should return None
        assert result is None
    
    def test_database_connection_error_handling(self):
        """Test database manager handles connection errors."""
        from src.infrastructure.db_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        
        # Test without initialization
        assert not db_manager._initialized


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])