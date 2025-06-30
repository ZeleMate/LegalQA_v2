"""
Infrastructure module for performance optimizations.

This module contains:
- CacheManager: Multi-level caching for embeddings and query results
- DatabaseManager: Optimized database connections with pooling
- Performance monitoring utilities
"""

from .cache_manager import CacheManager, get_cache_manager, cache_embedding_query, cache_document_chunks
from .db_manager import DatabaseManager, get_db_manager, fetch_chunks_optimized, ensure_database_optimized

__all__ = [
    'CacheManager',
    'get_cache_manager', 
    'cache_embedding_query',
    'cache_document_chunks',
    'DatabaseManager',
    'get_db_manager',
    'fetch_chunks_optimized',
    'ensure_database_optimized'
]