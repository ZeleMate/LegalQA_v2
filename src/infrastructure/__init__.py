"""
Infrastructure module for caching, database, and other core services.

This module contains:
- CacheManager: Multi-level caching for embeddings and query results
- DatabaseManager: Optimized database connections with pooling
- Performance monitoring utilities
"""

from .cache_manager import CacheManager, get_cache_manager, cache_embedding_query
from .db_manager import DatabaseManager, get_db_manager, fetch_chunks, ensure_database_setup

__all__ = [
    'CacheManager',
    'get_cache_manager', 
    'cache_embedding_query',
    'DatabaseManager',
    'get_db_manager',
    'fetch_chunks',
    'ensure_database_setup'
]