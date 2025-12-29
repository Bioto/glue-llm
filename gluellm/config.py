"""Configuration management using pydantic-settings.

This module provides configuration management for GlueLLM using Pydantic
settings, with support for environment variables and .env files.

Configuration Sources (in order of precedence):
    1. Direct instantiation parameters
    2. Environment variables (prefixed with GLUELLM_)
    3. .env file in project root

Available Settings:
    - Model Configuration: default_model, default_system_prompt
    - Tool Execution: max_tool_iterations
    - Retry Behavior: retry_max_attempts, retry_min_wait, retry_max_wait, retry_multiplier
    - Logging: log_level
    - API Keys: openai_api_key, anthropic_api_key, xai_api_key

Example:
    >>> from gluellm.config import settings, reload_settings
    >>>
    >>> # Access settings
    >>> print(settings.default_model)
    'openai:gpt-4o-mini'
    >>>
    >>> # Reload after changing .env
    >>> settings = reload_settings()
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find the project root (where .env file is located)
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"


class GlueLLMSettings(BaseSettings):
    """Global settings for GlueLLM.

    Configuration values can be set via:
    1. Environment variables (e.g., GLUELLM_DEFAULT_MODEL)
    2. .env file in the project root
    3. Direct instantiation with parameters
    """

    model_config = SettingsConfigDict(
        env_prefix="GLUELLM_",
        env_file=_env_file,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Default model settings
    default_model: str = "openai:gpt-4o-mini"
    default_system_prompt: str = "You are a helpful assistant."

    # Tool execution settings
    max_tool_iterations: Annotated[int, Field(gt=0)] = 10

    # Retry settings
    retry_max_attempts: Annotated[int, Field(gt=0)] = 3
    retry_min_wait: Annotated[int, Field(ge=0)] = 2
    retry_max_wait: Annotated[int, Field(ge=0)] = 30
    retry_multiplier: Annotated[int, Field(ge=0)] = 1

    # Logging settings
    log_level: str = "INFO"

    # Optional API keys (can also be set via provider-specific env vars)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    xai_api_key: str | None = None


# Global settings instance
settings = GlueLLMSettings()


def get_settings() -> GlueLLMSettings:
    """Get the global settings instance.

    Returns:
        GlueLLMSettings: The global settings instance
    """
    return settings


def reload_settings() -> GlueLLMSettings:
    """Reload settings from environment and .env file.

    Returns:
        GlueLLMSettings: A new settings instance
    """
    global settings
    settings = GlueLLMSettings()
    return settings
