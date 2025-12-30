"""Rate limiting for GlueLLM using throttled-py.

This module provides rate limiting functionality to prevent excessive API calls
and ensure compliance with provider rate limits.
"""

import asyncio
from typing import Literal

from throttled import Throttled, per_min
from throttled.exceptions import LimitedError
from throttled.store import MemoryStore, RedisStore

from gluellm.config import settings
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)

# Global rate limiter instances cache
_rate_limiters: dict[str, Throttled] = {}


def _get_store(backend: Literal["memory", "redis"] = "memory", redis_url: str | None = None):
    """Get the appropriate store backend for rate limiting.

    Args:
        backend: Storage backend type ("memory" or "redis")
        redis_url: Redis connection URL (required if backend is "redis")

    Returns:
        Store instance (MemoryStore or RedisStore)

    Raises:
        ValueError: If backend is "redis" but redis_url is not provided
    """
    if backend == "redis":
        if not redis_url:
            raise ValueError("redis_url is required when backend is 'redis'")
        return RedisStore(redis_url=redis_url)
    return MemoryStore()


def get_rate_limiter(
    key: str,
    requests_per_minute: int | None = None,
    burst: int | None = None,
    backend: Literal["memory", "redis"] | None = None,
    redis_url: str | None = None,
) -> Throttled:
    """Get or create a rate limiter for a specific key.

    Rate limiters are cached per key to avoid creating multiple instances
    for the same rate limit configuration.

    Args:
        key: Unique identifier for this rate limiter (e.g., "global" or API key hash)
        requests_per_minute: Maximum requests per minute (defaults to settings.rate_limit_requests_per_minute)
        burst: Burst capacity (not used directly, but kept for API compatibility)
        backend: Storage backend type (defaults to settings.rate_limit_backend)
        redis_url: Redis connection URL (defaults to settings.rate_limit_redis_url)

    Returns:
        Configured Throttled instance
    """
    # Use settings defaults if not provided
    requests_per_minute = requests_per_minute or settings.rate_limit_requests_per_minute
    backend = backend or settings.rate_limit_backend
    redis_url = redis_url or settings.rate_limit_redis_url

    # Create cache key
    cache_key = f"{key}:{requests_per_minute}:{backend}:{redis_url or ''}"

    # Return cached instance if available
    if cache_key in _rate_limiters:
        return _rate_limiters[cache_key]

    # Create new rate limiter
    store = _get_store(backend, redis_url)
    quota = per_min(requests_per_minute)
    rate_limiter = Throttled(
        key=key,
        using="sliding_window",  # Use string value instead of enum
        quota=quota,
        store=store,
        timeout=-1,  # Non-blocking mode, we handle waiting ourselves
    )

    # Cache and return
    _rate_limiters[cache_key] = rate_limiter
    logger.debug(f"Created rate limiter: key={key}, requests_per_minute={requests_per_minute}, backend={backend}")

    return rate_limiter


async def acquire_rate_limit(
    key: str,
    rate_limiter: Throttled | None = None,
    requests_per_minute: int | None = None,
    burst: int | None = None,
) -> None:
    """Acquire a rate limit permit, waiting if necessary.

    This function will block until a permit is available. It automatically
    handles waiting when rate limits are hit, making rate limiting transparent
    to the caller.

    Args:
        key: Unique identifier for rate limiting (e.g., "global" or API key hash)
        rate_limiter: Optional pre-configured Throttled instance
        requests_per_minute: Maximum requests per minute (used if rate_limiter not provided)
        burst: Burst capacity (not used, kept for API compatibility)

    Raises:
        LimitedError: If rate limit cannot be acquired (should not happen with auto-wait)
    """
    if not settings.rate_limit_enabled:
        return

    # Get rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = get_rate_limiter(
            key=key,
            requests_per_minute=requests_per_minute,
            burst=burst,
        )

    # Try to acquire permit
    while True:
        try:
            # Use limit() method which returns RateLimitResult
            result = rate_limiter.limit(key=key)
            if result.limited:
                # Extract retry_after from the result's state
                retry_after = (
                    result.state.retry_after / 1000.0 if result.state and result.state.retry_after else 1.0
                )  # Convert ms to seconds
                logger.debug(f"Rate limit hit for key={key}, waiting {retry_after:.2f}s")
                await asyncio.sleep(retry_after)
                continue
            logger.debug(f"Rate limit permit acquired: key={key}")
            return
        except LimitedError as e:
            # Extract retry_after from the exception
            retry_after = 1.0
            if hasattr(e, "rate_limit_result") and e.rate_limit_result and e.rate_limit_result.state:
                retry_after = (
                    e.rate_limit_result.state.retry_after / 1000.0 if e.rate_limit_result.state.retry_after else 1.0
                )
            logger.debug(f"Rate limit hit for key={key}, waiting {retry_after:.2f}s")
            await asyncio.sleep(retry_after)


def clear_rate_limiter_cache() -> None:
    """Clear the rate limiter cache.

    Useful for testing or when configuration changes.
    """
    global _rate_limiters
    _rate_limiters.clear()
    logger.debug("Rate limiter cache cleared")
