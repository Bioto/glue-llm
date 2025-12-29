"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from gluellm.config import GlueLLMSettings, get_settings, reload_settings


class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        # Create a fresh instance without env vars
        with patch.dict(os.environ, {}, clear=True):
            config = GlueLLMSettings()
            assert config.default_model == "openai:gpt-4o-mini"
            assert config.default_system_prompt == "You are a helpful assistant."
            assert config.max_tool_iterations == 10
            assert config.retry_max_attempts == 3
            assert config.retry_min_wait == 2
            assert config.retry_max_wait == 30
            assert config.retry_multiplier == 1
            assert config.log_level == "INFO"

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "GLUELLM_DEFAULT_MODEL": "anthropic:claude-3-sonnet",
            "GLUELLM_DEFAULT_SYSTEM_PROMPT": "Custom system prompt",
            "GLUELLM_MAX_TOOL_ITERATIONS": "5",
            "GLUELLM_RETRY_MAX_ATTEMPTS": "5",
            "GLUELLM_RETRY_MIN_WAIT": "1",
            "GLUELLM_RETRY_MAX_WAIT": "60",
            "GLUELLM_RETRY_MULTIPLIER": "2",
            "GLUELLM_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = GlueLLMSettings()
            assert config.default_model == "anthropic:claude-3-sonnet"
            assert config.default_system_prompt == "Custom system prompt"
            assert config.max_tool_iterations == 5
            assert config.retry_max_attempts == 5
            assert config.retry_min_wait == 1
            assert config.retry_max_wait == 60
            assert config.retry_multiplier == 2
            assert config.log_level == "DEBUG"

    def test_env_prefix_required(self):
        """Test that GLUELLM_ prefix is required for env vars."""
        # Set env var without prefix - should not be loaded
        with patch.dict(os.environ, {"DEFAULT_MODEL": "should-not-load"}, clear=False):
            config = GlueLLMSettings()
            assert config.default_model == "openai:gpt-4o-mini"  # Still default

    def test_case_insensitive_env_vars(self):
        """Test that environment variables are case-insensitive."""
        with patch.dict(os.environ, {"gluellm_default_model": "test:model"}, clear=False):
            config = GlueLLMSettings()
            assert config.default_model == "test:model"

    def test_api_key_loading(self):
        """Test API key loading from environment."""
        env_vars = {
            "GLUELLM_OPENAI_API_KEY": "test-openai-key",
            "GLUELLM_ANTHROPIC_API_KEY": "test-anthropic-key",
            "GLUELLM_XAI_API_KEY": "test-xai-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = GlueLLMSettings()
            assert config.openai_api_key == "test-openai-key"
            assert config.anthropic_api_key == "test-anthropic-key"
            assert config.xai_api_key == "test-xai-key"

    def test_api_key_none_by_default(self):
        """Test that API keys are None by default."""
        with patch.dict(os.environ, {}, clear=True):
            config = GlueLLMSettings()
            assert config.openai_api_key is None
            assert config.anthropic_api_key is None
            assert config.xai_api_key is None


class TestConfigurationValidation:
    """Test configuration value validation."""

    def test_invalid_retry_max_attempts_negative(self):
        """Test that negative retry_max_attempts raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(retry_max_attempts=-1)

    def test_invalid_retry_max_attempts_zero(self):
        """Test that zero retry_max_attempts raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(retry_max_attempts=0)

    def test_invalid_max_tool_iterations_negative(self):
        """Test that negative max_tool_iterations raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(max_tool_iterations=-1)

    def test_invalid_max_tool_iterations_zero(self):
        """Test that zero max_tool_iterations raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(max_tool_iterations=0)

    def test_invalid_retry_min_wait_negative(self):
        """Test that negative retry_min_wait raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(retry_min_wait=-1)

    def test_invalid_retry_max_wait_negative(self):
        """Test that negative retry_max_wait raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(retry_max_wait=-1)

    def test_invalid_retry_multiplier_negative(self):
        """Test that negative retry_multiplier raises validation error."""
        with pytest.raises(ValidationError):
            GlueLLMSettings(retry_multiplier=-1)

    def test_valid_configuration(self):
        """Test that valid configuration values are accepted."""
        config = GlueLLMSettings(
            default_model="test:model",
            max_tool_iterations=20,
            retry_max_attempts=5,
            retry_min_wait=1,
            retry_max_wait=60,
            retry_multiplier=2,
        )
        assert config.default_model == "test:model"
        assert config.max_tool_iterations == 20
        assert config.retry_max_attempts == 5


class TestReloadSettings:
    """Test settings reload functionality."""

    def test_reload_settings_creates_new_instance(self):
        """Test that reload_settings creates a new settings instance."""
        original_settings = get_settings()
        original_model = original_settings.default_model

        # Change env var
        with patch.dict(os.environ, {"GLUELLM_DEFAULT_MODEL": "reloaded:model"}, clear=False):
            reloaded_settings = reload_settings()
            assert reloaded_settings.default_model == "reloaded:model"
            # Global settings should be updated
            assert get_settings().default_model == "reloaded:model"

    def test_reload_settings_picks_up_env_changes(self):
        """Test that reload_settings picks up environment variable changes."""
        # Set initial env var
        with patch.dict(os.environ, {"GLUELLM_DEFAULT_MODEL": "initial:model"}, clear=False):
            initial_settings = reload_settings()
            assert initial_settings.default_model == "initial:model"

        # Change env var and reload
        with patch.dict(os.environ, {"GLUELLM_DEFAULT_MODEL": "updated:model"}, clear=False):
            updated_settings = reload_settings()
            assert updated_settings.default_model == "updated:model"
            assert get_settings().default_model == "updated:model"

    def test_get_settings_returns_global_instance(self):
        """Test that get_settings returns the global settings instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        # Both should return the same global instance
        assert get_settings() is settings1


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored (extra='ignore' config)."""
        # This should not raise an error due to extra='ignore'
        config = GlueLLMSettings()
        # Try to set an invalid field via env var
        with patch.dict(os.environ, {"GLUELLM_INVALID_FIELD": "value"}, clear=False):
            # Should not raise error, just ignore
            config = GlueLLMSettings()
            assert not hasattr(config, "invalid_field")

    def test_empty_string_env_vars(self):
        """Test handling of empty string environment variables."""
        with patch.dict(os.environ, {"GLUELLM_DEFAULT_MODEL": ""}, clear=False):
            config = GlueLLMSettings()
            # Empty string should be treated as valid (though not recommended)
            assert config.default_model == ""

    def test_string_to_int_conversion(self):
        """Test that string env vars are converted to int correctly."""
        with patch.dict(os.environ, {"GLUELLM_MAX_TOOL_ITERATIONS": "15"}, clear=False):
            config = GlueLLMSettings()
            assert config.max_tool_iterations == 15
            assert isinstance(config.max_tool_iterations, int)
