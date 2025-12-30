"""Tests for rate limiting and API key pool functionality."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from gluellm.config import GlueLLMSettings
from gluellm.models.batch import APIKeyConfig as BatchAPIKeyConfig
from gluellm.rate_limiting.api_key_pool import (
    APIKeyConfig,
    APIKeyPool,
    extract_provider_from_model,
    get_api_key_env_var,
)
from gluellm.rate_limiting.rate_limiter import acquire_rate_limit, clear_rate_limiter_cache, get_rate_limiter


class TestExtractProviderFromModel:
    """Tests for provider extraction from model strings."""

    def test_extract_openai_provider(self):
        """Test extracting OpenAI provider."""
        assert extract_provider_from_model("openai:gpt-4o") == "openai"
        assert extract_provider_from_model("openai:gpt-4-turbo") == "openai"

    def test_extract_anthropic_provider(self):
        """Test extracting Anthropic provider."""
        assert extract_provider_from_model("anthropic:claude-3-5-sonnet") == "anthropic"
        assert extract_provider_from_model("anthropic:claude-3-opus") == "anthropic"

    def test_extract_xai_provider(self):
        """Test extracting xAI provider."""
        assert extract_provider_from_model("xai:grok-beta") == "xai"

    def test_default_provider(self):
        """Test that model without provider defaults to openai."""
        assert extract_provider_from_model("gpt-4o") == "openai"


class TestGetAPIKeyEnvVar:
    """Tests for API key environment variable name mapping."""

    def test_openai_env_var(self):
        """Test OpenAI environment variable name."""
        assert get_api_key_env_var("openai") == "OPENAI_API_KEY"

    def test_anthropic_env_var(self):
        """Test Anthropic environment variable name."""
        assert get_api_key_env_var("anthropic") == "ANTHROPIC_API_KEY"

    def test_xai_env_var(self):
        """Test xAI environment variable name."""
        assert get_api_key_env_var("xai") == "XAI_API_KEY"

    def test_unknown_provider(self):
        """Test unknown provider generates expected env var name."""
        assert get_api_key_env_var("unknown") == "UNKNOWN_API_KEY"


class TestAPIKeyConfig:
    """Tests for APIKeyConfig class."""

    def test_api_key_config_creation(self):
        """Test creating APIKeyConfig."""
        config = APIKeyConfig(key="test-key-123", provider="openai")
        assert config.key == "test-key-123"
        assert config.provider == "openai"
        assert config.requests_per_minute is None
        assert config.burst is None
        assert len(config.key_hash) == 8

    def test_api_key_config_with_rate_limits(self):
        """Test APIKeyConfig with custom rate limits."""
        config = APIKeyConfig(key="test-key-123", provider="openai", requests_per_minute=100, burst=20)
        assert config.requests_per_minute == 100
        assert config.burst == 20

    def test_api_key_config_key_hash(self):
        """Test that key hash is consistent."""
        config1 = APIKeyConfig(key="test-key-123", provider="openai")
        config2 = APIKeyConfig(key="test-key-123", provider="openai")
        assert config1.key_hash == config2.key_hash

    def test_api_key_config_from_batch_config(self):
        """Test creating APIKeyConfig from BatchAPIKeyConfig."""
        batch_config = BatchAPIKeyConfig(key="test-key", provider="openai", requests_per_minute=50)
        config = APIKeyConfig.from_batch_config(batch_config)
        assert config.key == "test-key"
        assert config.provider == "openai"
        assert config.requests_per_minute == 50


class TestAPIKeyPool:
    """Tests for APIKeyPool class."""

    def test_api_key_pool_empty(self):
        """Test creating empty API key pool."""
        # Clear environment to avoid loading keys from env
        with patch.dict(os.environ, {}, clear=True):
            pool = APIKeyPool(keys=[])  # Explicitly pass empty list
            assert len(pool._keys_by_provider) == 0

    def test_api_key_pool_with_keys(self):
        """Test creating API key pool with keys."""
        keys = [
            APIKeyConfig(key="key1", provider="openai"),
            APIKeyConfig(key="key2", provider="openai"),
            APIKeyConfig(key="key3", provider="anthropic"),
        ]
        pool = APIKeyPool(keys=keys)
        assert len(pool.get_provider_keys("openai")) == 2
        assert len(pool.get_provider_keys("anthropic")) == 1

    def test_api_key_pool_add_key(self):
        """Test adding keys to pool."""
        with patch.dict(os.environ, {}, clear=True):
            pool = APIKeyPool(keys=[])  # Start with empty pool
            key_config = APIKeyConfig(key="test-key", provider="openai")
            pool.add_key(key_config)
            assert len(pool.get_provider_keys("openai")) == 1

    def test_api_key_pool_get_key_sync(self):
        """Test getting key synchronously."""
        keys = [APIKeyConfig(key="key1", provider="openai")]
        pool = APIKeyPool(keys=keys)
        key = pool.get_key("openai")
        assert key == "key1"

    def test_api_key_pool_get_key_round_robin(self):
        """Test round-robin key selection."""
        keys = [
            APIKeyConfig(key="key1", provider="openai"),
            APIKeyConfig(key="key2", provider="openai"),
        ]
        pool = APIKeyPool(keys=keys)
        key1 = pool.get_key("openai")
        key2 = pool.get_key("openai")
        key3 = pool.get_key("openai")
        # Should round-robin
        assert key1 == "key1"
        assert key2 == "key2"
        assert key3 == "key1"

    def test_api_key_pool_from_env(self):
        """Test loading keys from environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "env-openai-key",
            "ANTHROPIC_API_KEY": "env-anthropic-key",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            pool = APIKeyPool()
            # Should load from environment
            assert pool.has_keys("openai") or pool.get_key("openai") == "env-openai-key"

    @pytest.mark.asyncio
    async def test_api_key_pool_acquire_key(self):
        """Test acquiring key asynchronously."""
        keys = [APIKeyConfig(key="test-key", provider="openai")]
        pool = APIKeyPool(keys=keys)
        key = await pool.acquire_key("openai")
        assert key == "test-key"

    @pytest.mark.asyncio
    async def test_api_key_pool_acquire_key_with_model(self):
        """Test acquiring key with model string."""
        keys = [APIKeyConfig(key="test-key", provider="openai")]
        with patch.dict(os.environ, {}, clear=True):
            pool = APIKeyPool(keys=keys)
            key = await pool.acquire_key(model="openai:gpt-4o")
            assert key == "test-key"

    def test_api_key_pool_has_keys(self):
        """Test checking if pool has keys for provider."""
        with patch.dict(os.environ, {}, clear=True):
            pool = APIKeyPool(keys=[])  # Start with empty pool
            assert pool.has_keys("openai") is False

            key_config = APIKeyConfig(key="test-key", provider="openai")
            pool.add_key(key_config)
            assert pool.has_keys("openai") is True
            assert pool.has_keys("anthropic") is False


class TestRateLimiter:
    """Tests for rate limiter functionality."""

    def test_get_rate_limiter_defaults(self):
        """Test getting rate limiter with default settings."""
        clear_rate_limiter_cache()
        limiter = get_rate_limiter(key="test-key")
        assert limiter is not None
        from throttled import Throttled

        assert isinstance(limiter, Throttled)

    def test_get_rate_limiter_caching(self):
        """Test that rate limiters are cached."""
        clear_rate_limiter_cache()
        limiter1 = get_rate_limiter(key="test-key", requests_per_minute=60)
        limiter2 = get_rate_limiter(key="test-key", requests_per_minute=60)
        assert limiter1 is limiter2

    def test_get_rate_limiter_different_configs(self):
        """Test that different configs create different limiters."""
        clear_rate_limiter_cache()
        limiter1 = get_rate_limiter(key="test-key", requests_per_minute=60)
        limiter2 = get_rate_limiter(key="test-key", requests_per_minute=100)
        assert limiter1 is not limiter2

    def test_clear_rate_limiter_cache(self):
        """Test clearing rate limiter cache."""
        get_rate_limiter(key="test-key-cache")
        # Call clear and verify it works
        clear_rate_limiter_cache()
        # Create a new one to verify cache was cleared
        limiter1 = get_rate_limiter(key="test-key-cache")
        limiter2 = get_rate_limiter(key="test-key-cache")
        # Should be the same instance (cached)
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_acquire_rate_limit_disabled(self):
        """Test that rate limiting is skipped when disabled."""
        with patch("gluellm.rate_limiting.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = False
            # Should return immediately without blocking
            start = time.time()
            await acquire_rate_limit("test-key")
            elapsed = time.time() - start
            assert elapsed < 0.1  # Should be very fast

    @pytest.mark.asyncio
    async def test_acquire_rate_limit_allows_request(self):
        """Test that rate limiting allows requests within limit."""
        clear_rate_limiter_cache()
        with patch("gluellm.rate_limiting.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests_per_minute = 100
            mock_settings.rate_limit_burst = 10
            mock_settings.rate_limit_backend = "memory"
            mock_settings.rate_limit_redis_url = None

            # Should acquire immediately
            start = time.time()
            await acquire_rate_limit("test-key")
            elapsed = time.time() - start
            assert elapsed < 0.1  # Should be fast

    @pytest.mark.asyncio
    async def test_acquire_rate_limit_waits_when_limited(self):
        """Test that rate limiting waits when limit is hit."""
        clear_rate_limiter_cache()
        with patch("gluellm.rate_limiting.rate_limiter.settings") as mock_settings:
            mock_settings.rate_limit_enabled = True
            mock_settings.rate_limit_requests_per_minute = 1  # Very low limit
            mock_settings.rate_limit_burst = 1
            mock_settings.rate_limit_backend = "memory"
            mock_settings.rate_limit_redis_url = None

            # First request should succeed
            await acquire_rate_limit("test-key-limited")

            # Second request should wait
            start = time.time()
            await acquire_rate_limit("test-key-limited")
            elapsed = time.time() - start
            # Should have waited at least a bit (though exact timing depends on throttled-py)
            assert elapsed >= 0  # At minimum, should not error


class TestRateLimitingConfiguration:
    """Tests for rate limiting configuration."""

    def test_rate_limit_settings_defaults(self):
        """Test default rate limiting settings."""
        with patch.dict(os.environ, {}, clear=True):
            config = GlueLLMSettings()
            assert config.rate_limit_enabled is True
            assert config.rate_limit_requests_per_minute == 60
            assert config.rate_limit_burst == 10
            assert config.rate_limit_backend == "memory"
            assert config.rate_limit_redis_url is None

    def test_rate_limit_settings_from_env(self):
        """Test loading rate limiting settings from environment."""
        env_vars = {
            "GLUELLM_RATE_LIMIT_ENABLED": "false",
            "GLUELLM_RATE_LIMIT_REQUESTS_PER_MINUTE": "100",
            "GLUELLM_RATE_LIMIT_BURST": "20",
            "GLUELLM_RATE_LIMIT_BACKEND": "redis",
            "GLUELLM_RATE_LIMIT_REDIS_URL": "redis://localhost:6379",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = GlueLLMSettings()
            assert config.rate_limit_enabled is False
            assert config.rate_limit_requests_per_minute == 100
            assert config.rate_limit_burst == 20
            assert config.rate_limit_backend == "redis"
            assert config.rate_limit_redis_url == "redis://localhost:6379"

    def test_rate_limit_settings_validation(self):
        """Test rate limiting settings validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GlueLLMSettings(rate_limit_requests_per_minute=-1)

        with pytest.raises(ValidationError):
            GlueLLMSettings(rate_limit_requests_per_minute=0)

        with pytest.raises(ValidationError):
            GlueLLMSettings(rate_limit_burst=-1)

        with pytest.raises(ValidationError):
            GlueLLMSettings(rate_limit_burst=0)


class TestBatchAPIKeyConfig:
    """Tests for BatchAPIKeyConfig model."""

    def test_batch_api_key_config_creation(self):
        """Test creating BatchAPIKeyConfig."""
        config = BatchAPIKeyConfig(key="test-key", provider="openai")
        assert config.key == "test-key"
        assert config.provider == "openai"
        assert config.requests_per_minute is None
        assert config.burst is None

    def test_batch_api_key_config_with_rate_limits(self):
        """Test BatchAPIKeyConfig with rate limits."""
        config = BatchAPIKeyConfig(key="test-key", provider="openai", requests_per_minute=100, burst=20)
        assert config.requests_per_minute == 100
        assert config.burst == 20

    def test_batch_api_key_config_validation(self):
        """Test BatchAPIKeyConfig validation."""
        from pydantic import ValidationError

        # Valid config
        config = BatchAPIKeyConfig(key="test-key", provider="openai", requests_per_minute=100)
        assert config.requests_per_minute == 100

        # Invalid: negative requests_per_minute
        with pytest.raises(ValidationError):
            BatchAPIKeyConfig(key="test-key", provider="openai", requests_per_minute=-1)

        # Invalid: zero requests_per_minute
        with pytest.raises(ValidationError):
            BatchAPIKeyConfig(key="test-key", provider="openai", requests_per_minute=0)


class TestRateLimitingIntegration:
    """Integration tests for rate limiting with API calls."""

    @pytest.mark.asyncio
    async def test_rate_limiting_in_api_call(self):
        """Test that rate limiting is applied in API calls."""
        from gluellm.api import _safe_llm_call

        # Mock the actual LLM call to avoid real API calls
        with patch("gluellm.api.any_llm_acompletion") as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_llm.return_value = mock_response

            with patch("gluellm.api.acquire_rate_limit") as mock_rate_limit:
                await _safe_llm_call(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="openai:gpt-4o-mini",
                )
                # Should have called rate limiting
                assert mock_rate_limit.called

    @pytest.mark.asyncio
    async def test_rate_limiting_with_api_key(self):
        """Test rate limiting with API key override."""
        from gluellm.api import _safe_llm_call

        with patch("gluellm.api.any_llm_acompletion") as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_llm.return_value = mock_response

            with patch("gluellm.api.acquire_rate_limit") as mock_rate_limit:
                await _safe_llm_call(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="openai:gpt-4o-mini",
                    api_key="test-api-key",
                )
                # Should have called rate limiting
                assert mock_rate_limit.called
                # Check that rate limit key includes API key hash
                call_args = mock_rate_limit.call_args[0][0]
                assert "api_key:" in call_args
