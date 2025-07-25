"""
Infrastructure module for caching, database, and other core services.

This module contains:
- CacheManager: Multi-level caching for embeddings and query results
- DatabaseManager: Optimized database connections with pooling
- Performance monitoring utilities
"""

from .cache_manager import (
    CacheManager,
    cache_embedding_query,
    get_cache_manager,
)
from .db_manager import (
    DatabaseManager,
    ensure_database_setup,
    fetch_chunks,
    get_db_manager,
)

__all__ = [
    "CacheManager",
    "get_cache_manager",
    "cache_embedding_query",
    "DatabaseManager",
    "get_db_manager",
    "fetch_chunks",
    "ensure_database_setup",
]
