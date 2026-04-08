"""Configuration management using pydantic-settings.

This module provides configuration management for GlueLLM using Pydantic
settings, with support for environment variables and .env files.

Configuration Sources (in order of precedence):
    1. Direct instantiation parameters
    2. Environment variables (prefixed with GLUELLM_)
    3. .env file in project root

Available Settings:
    - Model Configuration: default_model, default_embedding_model, default_embedding_dimensions, default_system_prompt, default_max_tokens
    - Tool Execution: max_tool_iterations
    - Retry Behavior: retry_max_attempts, retry_min_wait, retry_max_wait, retry_multiplier
    - Request Timeout: default_request_timeout, max_request_timeout
    - Connection Timeout: default_connect_timeout, max_connect_timeout
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

from gluellm.rate_limit_types import RateLimitAlgorithm

ToolExecutionOrder = Literal["sequential", "parallel"]
from pydantic_settings import BaseSettings, SettingsConfigDict

# When used as an installed library, Path(__file__) points into site-packages,
# so an absolute path derived from it would miss the consumer's .env.
# A plain relative path lets pydantic-settings resolve against CWD instead.
_env_file = Path(".env")


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
    default_model: str = "openai:gpt-5.4-nano"
    default_embedding_model: str = "openai/text-embedding-3-small"
    default_embedding_dimensions: int | None = None  # e.g., 512 for text-embedding-3-small
    default_system_prompt: str = "You are a helpful assistant."
    default_max_tokens: int | None = None  # Global default; overridable per client and per call (e.g. 8192 for Anthropic)
    default_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "auto"] | None = None
    default_parallel_tool_calls: bool | None = None

    # Tool execution settings
    max_tool_iterations: Annotated[int, Field(gt=0)] = 10
    default_tool_mode: Literal["standard", "dynamic"] = "standard"
    default_tool_execution_order: Literal["sequential", "parallel"] = "sequential"
    tool_route_model: str = "openai:gpt-5.4-nano"
    default_condense_tool_messages: bool = False

    # Retry settings
    retry_max_attempts: Annotated[int, Field(gt=0)] = 3
    retry_min_wait: Annotated[int, Field(ge=0)] = 2
    retry_max_wait: Annotated[int, Field(ge=0)] = 30
    retry_multiplier: Annotated[int, Field(ge=0)] = 1

    # Request timeout settings (in seconds)
    default_request_timeout: Annotated[float, Field(gt=0)] = 300.0  # Default 5 minutes
    max_request_timeout: Annotated[float, Field(gt=0)] = 1800.0  # Maximum 30 minutes

    # Connection timeout settings (in seconds)
    default_connect_timeout: Annotated[float, Field(gt=0)] = 10.0  # Default 10 seconds
    max_connect_timeout: Annotated[float, Field(gt=0)] = 60.0  # Maximum 60 seconds

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
    rate_limit_algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW

    # Context summarization settings
    default_summarize_context: bool = False
    default_summarize_context_threshold: Annotated[int, Field(gt=0)] = 20
    default_summarize_context_keep_recent: Annotated[int, Field(gt=0)] = 6

    # AAAK lossless shorthand (used when summarization triggers; optional tool-round encoding)
    aaak_compression_enabled: bool = True
    aaak_compression_model: str | None = None  # None: use summarize_context_model / primary model
    aaak_tool_condensing: bool = True

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


def configure(**kwargs) -> GlueLLMSettings:
    """Configure GlueLLM settings programmatically.

    Merges the provided keyword arguments on top of the current settings
    (which still respect environment variables and .env). Call this once at
    application startup before making any LLM calls.

    Args:
        **kwargs: Any GlueLLMSettings field and value.

    Returns:
        GlueLLMSettings: The updated global settings instance.

    Example:
        >>> import gluellm
        >>> from gluellm import RateLimitAlgorithm
        >>>
        >>> gluellm.configure(
        ...     rate_limit_backend="redis",
        ...     rate_limit_redis_url="redis://localhost:6379",
        ...     rate_limit_algorithm=RateLimitAlgorithm.LEAKING_BUCKET,
        ...     default_model="anthropic:claude-3-5-sonnet-20241022",
        ... )
    """
    for key, value in kwargs.items():
        setattr(settings, key, value)
    return settings
