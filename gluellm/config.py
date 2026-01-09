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
    - Request Timeout: default_request_timeout, max_request_timeout
    - Logging: log_level, log_file_level, log_dir, log_file_name, log_json_format, log_max_bytes, log_backup_count, log_console_output
    - API Keys: openai_api_key, anthropic_api_key, xai_api_key
    - Tracing: enable_tracing, mlflow_tracking_uri, mlflow_experiment_name, otel_exporter_endpoint

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
from typing import Annotated, Literal

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
    default_embedding_model: str = "openai/text-embedding-3-small"
    default_system_prompt: str = "You are a helpful assistant."

    # Tool execution settings
    max_tool_iterations: Annotated[int, Field(gt=0)] = 10

    # Retry settings
    retry_max_attempts: Annotated[int, Field(gt=0)] = 3
    retry_min_wait: Annotated[int, Field(ge=0)] = 2
    retry_max_wait: Annotated[int, Field(ge=0)] = 30
    retry_multiplier: Annotated[int, Field(ge=0)] = 1

    # Request timeout settings (in seconds)
    default_request_timeout: Annotated[float, Field(gt=0)] = 60.0  # Default 60 seconds
    max_request_timeout: Annotated[float, Field(gt=0)] = 300.0  # Maximum 5 minutes

    # Logging settings
    log_level: str = "INFO"
    log_file_level: str = "DEBUG"
    log_dir: Path | None = None  # None means use default 'logs' directory
    log_file_name: str = "gluellm.log"
    log_json_format: bool = False
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5
    log_console_output: bool = False  # Disabled by default for library usage

    # Optional API keys (can also be set via provider-specific env vars)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    xai_api_key: str | None = None

    # OpenTelemetry and MLflow tracing settings
    enable_tracing: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "gluellm"
    otel_exporter_endpoint: str | None = None

    # Rate limiting settings
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: Annotated[int, Field(gt=0)] = 60
    rate_limit_burst: Annotated[int, Field(gt=0)] = 10
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    rate_limit_redis_url: str | None = None

    # Cost tracking settings
    track_costs: bool = True  # Enable cost tracking and include in responses
    print_session_summary_on_exit: bool = True  # Print token/cost summary when program exits

    # Evaluation recording settings
    eval_recording_enabled: bool = False  # Enable evaluation data recording
    eval_recording_path: Path | None = None  # Path to JSONL file (defaults to logs/eval_records.jsonl)


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
