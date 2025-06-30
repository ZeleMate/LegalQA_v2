import asyncio
import pickle
import hashlib
import logging
from typing import Any, Optional, Callable, Dict
from functools import wraps
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, maxsize: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, tuple] = {}
        self.maxsize = maxsize
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self.cache) >= self.maxsize:
            # Simple LRU eviction - remove oldest
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)
    
    def clear(self) -> None:
        self.cache.clear()


class CacheManager:
    """Multi-level cache manager for embeddings and query results."""
    
    def __init__(self, 
                 memory_maxsize: int = 1000,
                 memory_ttl: int = 300,
                 redis_url: Optional[str] = None):
        self.memory_cache = MemoryCache(memory_maxsize, memory_ttl)
        self.redis = None
        
        # Initialize Redis if URL provided
        if redis_url:
            try:
                import aioredis
                self.redis = aioredis.from_url(redis_url)
                logger.info("Redis cache initialized")
            except ImportError:
                logger.warning("aioredis not installed, falling back to memory cache only")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a consistent cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        else:
            content = str(data)
        
        hash_object = hashlib.md5(content.encode())
        return f"{prefix}:{hash_object.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)."""
        # L1: Memory cache
        value = self.memory_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit (memory): {key}")
            return value
        
        # L2: Redis cache
        if self.redis:
            try:
                cached_data = await self.redis.get(key)
                if cached_data:
                    value = pickle.loads(cached_data)
                    # Store in memory cache for faster future access
                    self.memory_cache.set(key, value)
                    logger.debug(f"Cache hit (redis): {key}")
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in all cache levels."""
        # L1: Memory cache
        self.memory_cache.set(key, value, ttl)
        
        # L2: Redis cache
        if self.redis:
            try:
                await self.redis.setex(key, ttl, pickle.dumps(value))
                logger.debug(f"Cache set: {key}")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
    
    async def get_or_compute(self, 
                           key: str, 
                           compute_func: Callable,
                           ttl: int = 300,
                           *args, **kwargs) -> Any:
        """Get value from cache or compute and cache it."""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute value
        if asyncio.iscoroutinefunction(compute_func):
            value = await compute_func(*args, **kwargs)
        else:
            value = compute_func(*args, **kwargs)
        
        # Cache the result
        await self.set(key, value, ttl)
        return value
    
    def cache_embedding(self, ttl: int = 3600):
        """Decorator for caching embedding computations."""
        def decorator(func):
            @wraps(func)
            async def wrapper(self_obj, text: str, *args, **kwargs):
                cache_key = self._generate_key("embedding", text)
                return await self.get_or_compute(
                    cache_key, 
                    func, 
                    ttl,
                    self_obj, 
                    text, 
                    *args, 
                    **kwargs
                )
            return wrapper
        return decorator
    
    def cache_query_results(self, ttl: int = 600):
        """Decorator for caching query results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(self_obj, query: str, *args, **kwargs):
                # Create cache key from query and relevant parameters
                key_data = f"{query}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = self._generate_key("query", key_data)
                return await self.get_or_compute(
                    cache_key,
                    func,
                    ttl,
                    self_obj,
                    query,
                    *args,
                    **kwargs
                )
            return wrapper
        return decorator
    
    async def clear_all(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.redis:
            try:
                await self.redis.flushdb()
                logger.info("All caches cleared")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")


# Global cache manager instance
cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        import os
        redis_url = os.getenv("REDIS_URL")
        cache_manager = CacheManager(redis_url=redis_url)
    return cache_manager


# Utility functions for common caching patterns
async def cache_embedding_query(text: str, embeddings_model) -> np.ndarray:
    """Cache embedding computation for a text query."""
    cache = get_cache_manager()
    cache_key = cache._generate_key("embedding", text)
    
    async def compute_embedding():
        if hasattr(embeddings_model, 'aembed_query'):
            return await embeddings_model.aembed_query(text)
        else:
            return embeddings_model.embed_query(text)
    
    return await cache.get_or_compute(cache_key, compute_embedding, ttl=3600)


async def cache_document_chunks(doc_id: str, chunk_func, *args) -> list:
    """Cache document chunking results."""
    cache = get_cache_manager()
    cache_key = cache._generate_key("chunks", doc_id)
    return await cache.get_or_compute(cache_key, chunk_func, ttl=1800, *args)