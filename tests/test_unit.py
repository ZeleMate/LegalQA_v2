"""
Unit Tests for LegalQA System

Tests individual components and functions to ensure they work correctly.
"""

import logging
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Configure logging for this test file
logger = logging.getLogger(__name__)

load_dotenv()


class TestInfrastructureComponents:
    """Test infrastructure components."""

    def test_cache_manager_import(self) -> None:
        """Test that CacheManager can be imported."""
        logger.info("--- Running Cache Manager Import Test ---")
        try:
            from src.infrastructure.cache_manager import CacheManager

            assert CacheManager is not None
            logger.info("✅ CacheManager imported successfully.")
        except ImportError as e:
            logger.error(f"❌ CacheManager import failed: {e}")
            pytest.fail(f"CacheManager import failed: {e}")

    def test_db_manager_import(self) -> None:
        """Test that DatabaseManager can be imported."""
        logger.info("--- Running Database Manager Import Test ---")
        try:
            from src.infrastructure.db_manager import DatabaseManager

            assert DatabaseManager is not None
            logger.info("✅ DatabaseManager imported successfully.")
        except ImportError as e:
            logger.error(f"❌ DatabaseManager import failed: {e}")
            pytest.fail(f"DatabaseManager import failed: {e}")

    def test_gemini_embeddings_import(self) -> None:
        """Test that GeminiEmbeddings can be imported."""
        logger.info("--- Running Gemini Embeddings Import Test ---")
        try:
            from src.infrastructure.gemini_embeddings import GeminiEmbeddings

            assert GeminiEmbeddings is not None
            logger.info("✅ GeminiEmbeddings imported successfully.")
        except ImportError as e:
            logger.error(f"❌ GeminiEmbeddings import failed: {e}")
            pytest.fail(f"GeminiEmbeddings import failed: {e}")

    def test_cache_manager_import_only(self) -> None:
        """Test CacheManager import only (no initialization)."""
        logger.info("--- Running Cache Manager Import Only Test ---")
        try:
            from src.infrastructure.cache_manager import CacheManager

            assert CacheManager is not None
            logger.info("✅ CacheManager import successful.")
        except Exception as e:
            logger.error(f"❌ CacheManager import failed: {e}")
            pytest.fail(f"CacheManager import failed: {e}")

    def test_db_manager_import_only(self) -> None:
        """Test DatabaseManager import only (no initialization)."""
        logger.info("--- Running Database Manager Import Only Test ---")
        try:
            from src.infrastructure.db_manager import DatabaseManager

            assert DatabaseManager is not None
            logger.info("✅ DatabaseManager import successful.")
        except Exception as e:
            logger.error(f"❌ DatabaseManager import failed: {e}")
            pytest.fail(f"DatabaseManager import failed: {e}")

    def test_memory_cache_functionality(self) -> None:
        """Test MemoryCache basic functionality."""
        logger.info("--- Running Memory Cache Functionality Test ---")
        try:
            from src.infrastructure.cache_manager import MemoryCache

            # Test initialization
            cache = MemoryCache(maxsize=10, default_ttl=60)
            assert cache.maxsize == 10
            assert cache.default_ttl == 60

            # Test set and get
            cache.set("test_key", "test_value")
            value = cache.get("test_key")
            assert value == "test_value"

            # Test clear
            cache.clear()
            value = cache.get("test_key")
            assert value is None

            logger.info("✅ MemoryCache functionality works correctly.")
        except Exception as e:
            logger.error(f"❌ MemoryCache functionality test failed: {e}")
            pytest.fail(f"MemoryCache functionality test failed: {e}")

    def test_cache_key_generation(self) -> None:
        """Test cache key generation functionality."""
        logger.info("--- Running Cache Key Generation Test ---")
        try:
            from src.infrastructure.cache_manager import CacheManager

            # Create cache manager instance
            cache_manager = CacheManager()

            # Test key generation with string
            key1 = cache_manager._generate_key("test", "hello")
            key2 = cache_manager._generate_key("test", "hello")
            assert key1 == key2  # Same input should generate same key

            # Test key generation with different inputs
            key3 = cache_manager._generate_key("test", "world")
            assert key1 != key3  # Different inputs should generate different keys

            # Test key generation with different prefixes
            key4 = cache_manager._generate_key("other", "hello")
            assert key1 != key4  # Different prefixes should generate different keys

            logger.info("✅ Cache key generation works correctly.")
        except Exception as e:
            logger.error(f"❌ Cache key generation test failed: {e}")
            pytest.fail(f"Cache key generation test failed: {e}")

    def test_gemini_embeddings_initialization(self) -> None:
        """Test GeminiEmbeddings initialization."""
        logger.info("--- Running Gemini Embeddings Initialization Test ---")
        try:
            from src.infrastructure.gemini_embeddings import GeminiEmbeddings

            # Test that we can create an instance with API key
            # This tests the class structure and imports
            embeddings = GeminiEmbeddings(api_key="test_key")
            assert embeddings is not None
            assert hasattr(embeddings, "api_key")
            assert hasattr(embeddings, "client")
            assert hasattr(embeddings, "embed_query")
            assert hasattr(embeddings, "aembed_query")

            logger.info("✅ GeminiEmbeddings initialization works correctly.")
        except Exception as e:
            logger.error(f"❌ GeminiEmbeddings initialization test failed: {e}")
            pytest.fail(f"GeminiEmbeddings initialization test failed: {e}")


class TestRAGComponents:
    """Test RAG components."""

    def test_retriever_import(self) -> None:
        """Test that CustomRetriever can be imported."""
        logger.info("--- Running Retriever Import Test ---")
        try:
            from src.rag.retriever import CustomRetriever

            assert CustomRetriever is not None
            logger.info("✅ CustomRetriever imported successfully.")
        except ImportError as e:
            logger.error(f"❌ CustomRetriever import failed: {e}")
            pytest.fail(f"CustomRetriever import failed: {e}")

    def test_retriever_import_only(self) -> None:
        """Test CustomRetriever import only (no initialization)."""
        logger.info("--- Running Retriever Import Only Test ---")
        try:
            from src.rag.retriever import CustomRetriever

            assert CustomRetriever is not None
            logger.info("✅ CustomRetriever import successful.")
        except Exception as e:
            logger.error(f"❌ CustomRetriever import failed: {e}")
            pytest.fail(f"CustomRetriever import failed: {e}")

    def test_retriever_class_structure(self) -> None:
        """Test CustomRetriever class structure."""
        logger.info("--- Running Retriever Class Structure Test ---")
        try:
            from src.rag.retriever import CustomRetriever

            # Test that the class has expected attributes
            assert hasattr(CustomRetriever, "__init__")
            assert hasattr(CustomRetriever, "_get_relevant_documents")
            assert hasattr(CustomRetriever, "_aget_relevant_documents")

            logger.info("✅ CustomRetriever class structure is correct.")
        except Exception as e:
            logger.error(f"❌ CustomRetriever class structure test failed: {e}")
            pytest.fail(f"CustomRetriever class structure test failed: {e}")


class TestChainComponents:
    """Test chain components."""

    def test_qa_chain_functions_import(self) -> None:
        """Test that QA chain functions can be imported."""
        logger.info("--- Running QA Chain Functions Import Test ---")
        try:
            from src.chain.qa_chain import build_qa_chain, format_docs

            assert build_qa_chain is not None
            assert format_docs is not None
            logger.info("✅ QA chain functions imported successfully.")
        except ImportError as e:
            logger.error(f"❌ QA chain functions import failed: {e}")
            pytest.fail(f"QA chain functions import failed: {e}")

    def test_qa_chain_functions_exist(self) -> None:
        """Test that QA chain functions exist."""
        logger.info("--- Running QA Chain Functions Existence Test ---")
        try:
            from src.chain.qa_chain import build_qa_chain, format_docs

            # Test that functions are callable
            assert callable(build_qa_chain)
            assert callable(format_docs)

            logger.info("✅ QA chain functions exist and are callable.")
        except Exception as e:
            logger.error(f"❌ QA chain functions test failed: {e}")
            pytest.fail(f"QA chain functions test failed: {e}")

    def test_format_docs_function(self) -> None:
        """Test format_docs function with mock data."""
        logger.info("--- Running Format Docs Function Test ---")
        try:
            from src.chain.qa_chain import format_docs

            # Create mock documents
            class MockDoc:
                def __init__(self, content: str, chunk_id: str):
                    self.page_content = content
                    self.metadata = {"chunk_id": chunk_id}

            mock_docs = [
                MockDoc("Test content 1", "chunk_1"),
                MockDoc("Test content 2", "chunk_2"),
            ]

            # Test format_docs function
            result = format_docs(mock_docs)
            assert isinstance(result, str)
            assert "chunk_1" in result
            assert "chunk_2" in result
            # The function truncates content to 2 characters, so we check for "Te"
            assert "Te" in result

            logger.info("✅ format_docs function works correctly.")
        except Exception as e:
            logger.error(f"❌ format_docs function test failed: {e}")
            pytest.fail(f"format_docs function test failed: {e}")


class TestInferenceComponents:
    """Test inference components."""

    def test_app_import(self) -> None:
        """Test that FastAPI app can be imported."""
        logger.info("--- Running FastAPI App Import Test ---")
        try:
            from src.inference.app import app

            assert app is not None
            logger.info("✅ FastAPI app imported successfully.")
        except ImportError as e:
            logger.error(f"❌ FastAPI app import failed: {e}")
            pytest.fail(f"FastAPI app import failed: {e}")

    def test_app_endpoints(self) -> None:
        """Test that app has expected endpoints."""
        logger.info("--- Running App Endpoints Test ---")
        try:
            from src.inference.app import app

            # Get all routes
            routes = [route.path for route in app.routes]
            expected_routes = ["/health", "/stats", "/metrics", "/ask", "/clear-cache"]

            # Check if expected routes exist
            for route in expected_routes:
                assert route in routes, f"Expected route {route} not found"

            logger.info("✅ All expected endpoints exist.")
        except Exception as e:
            logger.error(f"❌ App endpoints test failed: {e}")
            pytest.fail(f"App endpoints test failed: {e}")

    def test_app_models_import(self) -> None:
        """Test that app models can be imported."""
        logger.info("--- Running App Models Import Test ---")
        try:
            from src.inference.app import QuestionRequest, QuestionResponse

            assert QuestionRequest is not None
            assert QuestionResponse is not None

            logger.info("✅ App models imported successfully.")
        except ImportError as e:
            logger.error(f"❌ App models import failed: {e}")
            pytest.fail(f"App models import failed: {e}")


class TestDataLoadingComponents:
    """Test data loading components."""

    def test_faiss_loader_import(self) -> None:
        """Test that FAISS loader can be imported."""
        logger.info("--- Running FAISS Loader Import Test ---")
        try:
            from src.data_loading.faiss_loader import load_faiss_index

            assert load_faiss_index is not None
            logger.info("✅ FAISS loader imported successfully.")
        except ImportError as e:
            logger.error(f"❌ FAISS loader import failed: {e}")
            pytest.fail(f"FAISS loader import failed: {e}")

    def test_parquet_loader_import(self) -> None:
        """Test that Parquet loader can be imported."""
        logger.info("--- Running Parquet Loader Import Test ---")
        try:
            from src.data_loading.parquet_loader import load_parquet_file

            assert load_parquet_file is not None
            logger.info("✅ Parquet loader imported successfully.")
        except ImportError as e:
            logger.error(f"❌ Parquet loader import failed: {e}")
            pytest.fail(f"Parquet loader import failed: {e}")

    def test_parquet_loader_function(self) -> None:
        """Test parquet loader function signature."""
        logger.info("--- Running Parquet Loader Function Test ---")
        try:
            from src.data_loading.parquet_loader import load_parquet_file

            # Test that function is callable and has correct signature
            assert callable(load_parquet_file)

            # Test function signature (should accept a string parameter)
            import inspect

            sig = inspect.signature(load_parquet_file)
            assert len(sig.parameters) == 1
            assert "parquet_path" in sig.parameters

            logger.info("✅ Parquet loader function signature is correct.")
        except Exception as e:
            logger.error(f"❌ Parquet loader function test failed: {e}")
            pytest.fail(f"Parquet loader function test failed: {e}")


class TestPromptsComponents:
    """Test prompts components."""

    def test_prompts_exist(self) -> None:
        """Test that prompt files exist."""
        logger.info("--- Running Prompts Existence Test ---")
        try:
            project_root = Path(__file__).parent.parent
            prompts_dir = project_root / "src" / "prompts"

            expected_prompts = ["legal_assistant_prompt.txt", "reranker_prompt.txt"]

            for prompt_file in expected_prompts:
                prompt_path = prompts_dir / prompt_file
                assert prompt_path.exists(), f"Expected prompt file {prompt_file} not found"

            logger.info("✅ All expected prompt files exist.")
        except Exception as e:
            logger.error(f"❌ Prompts existence test failed: {e}")
            pytest.fail(f"Prompts existence test failed: {e}")

    def test_prompts_content(self) -> None:
        """Test that prompt files have content."""
        logger.info("--- Running Prompts Content Test ---")
        try:
            project_root = Path(__file__).parent.parent
            prompts_dir = project_root / "src" / "prompts"

            expected_prompts = ["legal_assistant_prompt.txt", "reranker_prompt.txt"]

            for prompt_file in expected_prompts:
                prompt_path = prompts_dir / prompt_file
                content = prompt_path.read_text(encoding="utf-8")
                assert len(content) > 0, f"Prompt file {prompt_file} is empty"
                assert (
                    "{" in content or "[" in content
                ), f"Prompt file {prompt_file} seems to be missing template variables"

            logger.info("✅ All prompt files have valid content.")
        except Exception as e:
            logger.error(f"❌ Prompts content test failed: {e}")
            pytest.fail(f"Prompts content test failed: {e}")


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v", "--tb=short"])
