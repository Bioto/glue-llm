"""Rate limit algorithm type.

Kept in a separate module to avoid circular imports with config.
"""

from enum import Enum


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithm type (matches throttled-py's RateLimiterType)."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    LEAKING_BUCKET = "leaking_bucket"
    TOKEN_BUCKET = "token_bucket"
    GCRA = "gcra"
