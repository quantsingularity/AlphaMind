"""
Caching module for AlphaMind data processing.

Provides various caching mechanisms for performance optimization.
"""

import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CachePolicy(ABC):
    """Base class for cache policies."""

    @abstractmethod
    def should_cache(self, key: str, value: Any) -> bool:
        """Determine if a value should be cached."""

    @abstractmethod
    def is_valid(self, cached_time: float) -> bool:
        """Check if cached value is still valid."""


class TTLCachePolicy(CachePolicy):
    """Time-to-live cache policy."""

    def __init__(self, ttl_seconds: float = 3600) -> None:
        """
        Initialize TTL cache policy.

        Args:
            ttl_seconds: Time to live in seconds (default: 1 hour)
        """
        self.ttl_seconds = ttl_seconds

    def should_cache(self, key: str, value: Any) -> bool:
        """Always cache non-None values."""
        return value is not None

    def is_valid(self, cached_time: float) -> bool:
        """Check if cache entry hasn't expired."""
        return (time.time() - cached_time) < self.ttl_seconds


class SizeCachePolicy(CachePolicy):
    """Size-based cache policy with LRU eviction."""

    def __init__(self, max_size_mb: float = 100) -> None:
        """
        Initialize size cache policy.

        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_mb = max_size_mb
        self.current_size = 0

    def should_cache(self, key: str, value: Any) -> bool:
        """Cache if size limit not exceeded."""
        try:
            value_size = len(pickle.dumps(value)) / (1024 * 1024)  # Convert to MB
            return value_size < self.max_size_mb
        except Exception:
            return False

    def is_valid(self, cached_time: float) -> bool:
        """Always valid for size-based policy."""
        return True


class CacheManager:
    """Manages caching with configurable policies and backends."""

    def __init__(
        self, policy: Optional[CachePolicy] = None, namespace: str = "default"
    ) -> None:
        """
        Initialize cache manager.

        Args:
            policy: Cache policy to use (default: TTL with 1 hour)
            namespace: Namespace for cache keys
        """
        self.policy = policy or TTLCachePolicy()
        self.namespace = namespace
        self._cache: Dict[str, tuple[Any, float]] = {}
        logger.info(f"CacheManager initialized with namespace: {namespace}")

    def _make_key(self, key: Union[str, tuple]) -> str:
        """Create a cache key with namespace."""
        if isinstance(key, tuple):
            key = str(key)
        return f"{self.namespace}:{key}"

    def get(self, key: Union[str, tuple], default: Optional[Any] = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            value, cached_time = self._cache[cache_key]
            if self.policy.is_valid(cached_time):
                logger.debug(f"Cache hit for key: {cache_key}")
                return value
            else:
                logger.debug(f"Cache expired for key: {cache_key}")
                del self._cache[cache_key]
        return default

    def set(self, key: Union[str, tuple], value: Any) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if cached successfully, False otherwise
        """
        cache_key = self._make_key(key)
        if self.policy.should_cache(cache_key, value):
            self._cache[cache_key] = (value, time.time())
            logger.debug(f"Cached value for key: {cache_key}")
            return True
        return False

    def delete(self, key: Union[str, tuple]) -> bool:
        """Delete value from cache."""
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Deleted cache key: {cache_key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries in namespace."""
        keys_to_delete = [
            k for k in self._cache.keys() if k.startswith(f"{self.namespace}:")
        ]
        for key in keys_to_delete:
            del self._cache[key]
        logger.info(f"Cleared cache namespace: {self.namespace}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        namespace_keys = [
            k for k in self._cache.keys() if k.startswith(f"{self.namespace}:")
        ]
        return {
            "namespace": self.namespace,
            "entries": len(namespace_keys),
            "total_entries": len(self._cache),
        }


def cache_function(ttl_seconds: float = 3600, namespace: str = "function"):
    """
    Decorator for caching function results.

    Args:
        ttl_seconds: Time to live in seconds
        namespace: Cache namespace

    Usage:
        @cache_function(ttl_seconds=300)
        def expensive_computation(x, y):
            return x * y
    """
    cache_mgr = CacheManager(policy=TTLCachePolicy(ttl_seconds), namespace=namespace)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": sorted(kwargs.items()),
            }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()

            # Try to get from cache
            result = cache_mgr.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache_mgr.set(cache_key, result)
            return result

        return wrapper

    return decorator
