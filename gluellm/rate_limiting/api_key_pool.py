"""API key pool management for distributed rate limiting.

This module provides functionality to manage multiple API keys per provider,
automatically rotating keys when rate limits are hit and tracking usage per key.
"""

import hashlib
import os
from collections import defaultdict

from gluellm.config import settings
from gluellm.models.batch import APIKeyConfig as BatchAPIKeyConfig
from gluellm.observability.logging_config import get_logger
from gluellm.rate_limiting.rate_limiter import acquire_rate_limit, get_rate_limiter

logger = get_logger(__name__)


def extract_provider_from_model(model: str) -> str:
    """Extract provider name from model string.

    Args:
        model: Model string in format "provider:model_name" (e.g., "openai:gpt-4o")

    Returns:
        Provider name (e.g., "openai", "anthropic", "xai")
    """
    if ":" in model:
        return model.split(":")[0].lower()
    # Default to openai if no provider specified
    return "openai"


def get_api_key_env_var(provider: str) -> str:
    """Get the environment variable name for a provider's API key.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")

    Returns:
        Environment variable name (e.g., "OPENAI_API_KEY")
    """
    provider_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
    }
    return provider_map.get(provider.lower(), f"{provider.upper()}_API_KEY")


class APIKeyConfig:
    """Configuration for a single API key.

    This is a wrapper around the Pydantic model that adds runtime functionality.

    Attributes:
        key: The API key value
        provider: Provider name (e.g., "openai", "anthropic")
        requests_per_minute: Optional per-key rate limit override
        burst: Optional per-key burst capacity override
        key_hash: Hash of the key for identification
    """

    def __init__(
        self,
        key: str,
        provider: str,
        requests_per_minute: int | None = None,
        burst: int | None = None,
    ):
        """Initialize API key configuration.

        Args:
            key: The API key value
            provider: Provider name
            requests_per_minute: Optional per-key rate limit
            burst: Optional per-key burst capacity
        """
        self.key = key
        self.provider = provider.lower()
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        # Create a hash for the key (for logging/identification without exposing the key)
        self.key_hash = hashlib.sha256(key.encode()).hexdigest()[:8]

    @classmethod
    def from_batch_config(cls, batch_config: BatchAPIKeyConfig) -> "APIKeyConfig":
        """Create APIKeyConfig from BatchAPIKeyConfig model.

        Args:
            batch_config: BatchAPIKeyConfig instance

        Returns:
            APIKeyConfig instance
        """
        return cls(
            key=batch_config.key,
            provider=batch_config.provider,
            requests_per_minute=batch_config.requests_per_minute,
            burst=batch_config.burst,
        )

    def __repr__(self) -> str:
        return f"APIKeyConfig(provider={self.provider}, key_hash={self.key_hash})"


class APIKeyPool:
    """Manages a pool of API keys with per-key rate limiting.

    This class organizes API keys by provider and automatically manages
    rate limits per key, rotating keys when rate limits are hit.
    """

    def __init__(
        self,
        keys: list[APIKeyConfig] | list[BatchAPIKeyConfig] | None = None,
        default_requests_per_minute: int | None = None,
        default_burst: int | None = None,
    ):
        """Initialize API key pool.

        Args:
            keys: Optional list of API key configurations (APIKeyConfig or BatchAPIKeyConfig)
            default_requests_per_minute: Default rate limit if not specified per key
            default_burst: Default burst capacity if not specified per key
        """
        # Organize keys by provider
        self._keys_by_provider: dict[str, list[APIKeyConfig]] = defaultdict(list)
        self._key_index: dict[str, int] = {}  # Current index per provider for round-robin

        # Default rate limits
        self.default_requests_per_minute = default_requests_per_minute or settings.rate_limit_requests_per_minute
        self.default_burst = default_burst or settings.rate_limit_burst

        # Add provided keys
        if keys is not None and len(keys) > 0:
            for key_config in keys:
                if isinstance(key_config, BatchAPIKeyConfig):
                    key_config = APIKeyConfig.from_batch_config(key_config)
                self.add_key(key_config)
        elif keys is None:
            # Also load keys from environment variables only if keys param is None (not empty list)
            self._load_keys_from_env()

    def _load_keys_from_env(self) -> None:
        """Load API keys from environment variables."""
        providers = ["openai", "anthropic", "xai"]
        for provider in providers:
            env_var = get_api_key_env_var(provider)
            api_key = os.getenv(env_var) or getattr(settings, f"{provider}_api_key", None)
            if api_key:
                key_config = APIKeyConfig(key=api_key, provider=provider)
                self.add_key(key_config)
                logger.debug(f"Loaded API key for {provider} from environment")

    def add_key(self, key_config: APIKeyConfig) -> None:
        """Add an API key to the pool.

        Args:
            key_config: API key configuration
        """
        self._keys_by_provider[key_config.provider].append(key_config)
        logger.info(f"Added API key to pool: provider={key_config.provider}, key_hash={key_config.key_hash}")

    def get_key(self, provider: str, model: str | None = None) -> str | None:
        """Get an API key for a provider (non-blocking, for synchronous use).

        This method returns a key without waiting for rate limits.
        For async code, use acquire_key() instead.

        Args:
            provider: Provider name (e.g., "openai")
            model: Optional model string (provider will be extracted if not provided)

        Returns:
            API key string, or None if no keys available
        """
        # Extract provider from model if needed
        if model and not provider:
            provider = extract_provider_from_model(model)
        provider = provider.lower()

        # Get keys for this provider
        keys = self._keys_by_provider.get(provider, [])
        if not keys:
            # Fallback to environment variable
            env_var = get_api_key_env_var(provider)
            api_key = os.getenv(env_var) or getattr(settings, f"{provider}_api_key", None)
            if api_key:
                logger.debug(f"Using API key from environment for {provider}")
                return api_key
            logger.warning(f"No API keys available for provider: {provider}")
            return None

        # Round-robin through keys
        start_index = self._key_index.get(provider, 0)
        index = start_index % len(keys)
        key_config = keys[index]
        self._key_index[provider] = (index + 1) % len(keys)
        logger.debug(f"Selected API key: provider={provider}, key_hash={key_config.key_hash}")
        return key_config.key

    async def acquire_key(self, provider: str = "", model: str | None = None) -> str | None:
        """Acquire an API key with rate limiting (async version).

        This is the async version that properly waits for rate limits.

        Args:
            provider: Provider name (e.g., "openai")
            model: Optional model string (provider will be extracted if not provided)

        Returns:
            API key string, or None if no keys available
        """
        # Extract provider from model if needed
        if model and not provider:
            provider = extract_provider_from_model(model)
        provider = provider.lower()

        # Get keys for this provider
        keys = self._keys_by_provider.get(provider, [])
        if not keys:
            # Fallback to environment variable
            env_var = get_api_key_env_var(provider)
            api_key = os.getenv(env_var) or getattr(settings, f"{provider}_api_key", None)
            if api_key:
                logger.debug(f"Using API key from environment for {provider}")
                # Still apply rate limiting to env key
                await acquire_rate_limit(f"env_key:{provider}")
                return api_key
            logger.warning(f"No API keys available for provider: {provider}")
            return None

        # Round-robin through keys, checking rate limits
        start_index = self._key_index.get(provider, 0)
        for i in range(len(keys)):
            index = (start_index + i) % len(keys)
            key_config = keys[index]

            # Get rate limiter for this key
            rate_limiter = get_rate_limiter(
                key=f"api_key:{key_config.key_hash}",
                requests_per_minute=key_config.requests_per_minute or self.default_requests_per_minute,
                burst=key_config.burst or self.default_burst,
            )

            # Try to acquire rate limit (non-blocking first)
            try:
                rate_limiter.acquire(f"api_key:{key_config.key_hash}")
                # Success - update index for next time
                self._key_index[provider] = (index + 1) % len(keys)
                logger.debug(f"Selected API key: provider={provider}, key_hash={key_config.key_hash}")
                return key_config.key
            except Exception:
                # Rate limit hit, try next key
                continue

        # All keys are rate-limited, use the first one and wait
        key_config = keys[0]
        rate_limiter = get_rate_limiter(
            key=f"api_key:{key_config.key_hash}",
            requests_per_minute=key_config.requests_per_minute or self.default_requests_per_minute,
            burst=key_config.burst or self.default_burst,
        )
        # This will wait until rate limit is available
        await acquire_rate_limit(f"api_key:{key_config.key_hash}", rate_limiter=rate_limiter)
        return key_config.key

    def get_provider_keys(self, provider: str) -> list[APIKeyConfig]:
        """Get all keys for a specific provider.

        Args:
            provider: Provider name

        Returns:
            List of API key configurations
        """
        return self._keys_by_provider.get(provider.lower(), [])

    def has_keys(self, provider: str) -> bool:
        """Check if pool has keys for a provider.

        Args:
            provider: Provider name

        Returns:
            True if keys are available
        """
        return len(self._keys_by_provider.get(provider.lower(), [])) > 0
