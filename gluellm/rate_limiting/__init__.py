"""Rate limiting module for API access control.

This module provides rate limiting and API key pool management
to ensure compliance with provider limits and distribute load.

Components:
    - rate_limiter: Token bucket rate limiting with Redis support
    - api_key_pool: Multi-key rotation and load balancing
"""

from gluellm.rate_limiting.api_key_pool import (
    APIKeyPool,
    extract_provider_from_model,
    get_api_key_env_var,
)
from gluellm.rate_limiting.rate_limiter import (
    acquire_rate_limit,
    get_rate_limiter,
)

__all__ = [
    # Rate limiter
    "get_rate_limiter",
    "acquire_rate_limit",
    # API key pool
    "APIKeyPool",
    "extract_provider_from_model",
    "get_api_key_env_var",
]
