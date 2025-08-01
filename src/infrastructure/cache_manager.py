import asyncio
import hashlib
import json
import logging
import pickle  # nosec
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional

import numpy as np

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

    def __init__(
        self,
        memory_maxsize: int = 1000,
        memory_ttl: int = 300,
        redis_url: Optional[str] = None,
    ):
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
            content = data.encode()
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        else:
            content = str(data).encode()

        hash_object = hashlib.sha256(content)
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
                    # Handle both JSON and pickle formats
                    if cached_data.startswith(b"json:"):
                        # JSON format
                        json_data = cached_data[5:].decode("utf-8")
                        value = json.loads(json_data)
                    elif cached_data.startswith(b"pickle:"):
                        # Pickle format (backward compatibility)
                        pickle_data = cached_data[7:]
                        value = pickle.loads(pickle_data)  # nosec
                    else:
                        # Legacy format (no prefix) - assume pickle
                        value = pickle.loads(cached_data)  # nosec

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
                # Use JSON for new cache entries, pickle for backward compatibility
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    # JSON-serializable types
                    serialized_value = json.dumps(value).encode("utf-8")
                    await self.redis.setex(key, ttl, b"json:" + serialized_value)
                else:
                    # Non-JSON types (like numpy arrays) - use pickle with nosec
                    serialized_value = pickle.dumps(value)  # nosec
                    await self.redis.setex(key, ttl, b"pickle:" + serialized_value)
                logger.debug(f"Cache set: {key}")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

    async def get_or_compute(
        self, key: str, compute_func: Callable[..., Any], ttl: int = 300, *args: Any, **kwargs: Any
    ) -> Any:
        """Get value from cache or compute and cache it."""
        # Remove ttl from kwargs if present to avoid duplicate argument
        kwargs.pop("ttl", None)
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

    def cache_embedding(
        self, ttl: int = 3600
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for caching embedding computations."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(self_obj: Any, text: str, *args: Any, **kwargs: Any) -> Any:
                cache_key = self._generate_key("embedding", text)
                if "ttl" in kwargs:
                    kwargs.pop("ttl")
                return await self.get_or_compute(
                    cache_key, func, ttl, self_obj, text, *args, **kwargs
                )

            return wrapper

        return decorator

    def cache_query_results(
        self, ttl: int = 600
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for caching query results."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(self_obj: Any, query: str, *args: Any, **kwargs: Any) -> Any:
                # Create cache key from query and relevant parameters
                key_data = f"{query}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = self._generate_key("query", key_data)
                if "ttl" in kwargs:
                    kwargs.pop("ttl")
                return await self.get_or_compute(
                    cache_key, func, ttl, self_obj, query, *args, **kwargs
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

    async def close(self) -> None:
        """Close the cache manager."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis cache closed")


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
async def cache_embedding_query(text: str, embeddings_model: Any) -> np.ndarray:
    """Cache embedding computation for a text query."""
    cache = get_cache_manager()
    cache_key = cache._generate_key("embedding", text)

    async def compute_embedding() -> np.ndarray:
        if hasattr(embeddings_model, "aembed_query"):
            result = await embeddings_model.aembed_query(text)
            return np.array(result, dtype=np.float32)
        else:
            result = embeddings_model.embed_query(text)
            return np.array(result, dtype=np.float32)

    result = await cache.get_or_compute(cache_key, compute_embedding, ttl=3600)
    return result if isinstance(result, np.ndarray) else np.array(result, dtype=np.float32)


async def cache_document_chunks(
    doc_id: str, chunk_func: Callable[..., Any], *args: Any
) -> list[Any]:
    """Cache document chunking results."""
    cache = get_cache_manager()
    cache_key = cache._generate_key("chunks", doc_id)
    result = await cache.get_or_compute(cache_key, chunk_func, 1800, *args)
    return result if isinstance(result, list) else []
