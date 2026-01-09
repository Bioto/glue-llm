"""GlueLLM - A high-level LLM SDK with automatic tool execution and structured outputs.

GlueLLM is a Python SDK that simplifies working with Large Language Models by providing:
- Automatic tool/function calling with execution loops
- Structured outputs using Pydantic models
- Multi-turn conversations with memory
- Batch processing with configurable concurrency
- Automatic retry with exponential backoff
- Comprehensive error handling
- Provider-agnostic interface (OpenAI, Anthropic, xAI, etc.)

Quick Start:
    >>> import asyncio
    >>> from gluellm import complete, structured_complete, GlueLLM
    >>> from pydantic import BaseModel
    >>>
    >>> async def main():
    ...     # Simple completion
    ...     result = await complete("What is the capital of France?")
    ...     print(result.final_response)
    ...
    ...     # With tools
    ...     def get_weather(city: str) -> str:
    ...         '''Get weather for a city.'''
    ...         return f"Sunny in {city}"
    ...
    ...     result = await complete(
    ...         "What's the weather in Paris?",
    ...         tools=[get_weather]
    ...     )
    ...     print(result.final_response)
    ...
    ...     # Structured output
    ...     class City(BaseModel):
    ...         name: str
    ...         country: str
    ...
    ...     city = await structured_complete(
    ...         "Extract: Paris, France",
    ...         response_format=City
    ...     )
    ...     print(f"{city.name}, {city.country}")
    ...
    ...     # Multi-turn conversation
    ...     client = GlueLLM()
    ...     await client.complete("My name is Alice")
    ...     response = await client.complete("What's my name?")
    ...     print(response.final_response)
    >>>
    >>> asyncio.run(main())

Main Components:
    - GlueLLM: Main client class for LLM interactions
    - complete: Quick completion function
    - structured_complete: Quick structured output function
    - Conversation: Conversation history manager
    - Message: Individual message model
    - Role: Message role enumeration
    - SystemPrompt: System prompt with tool integration
    - RequestConfig: Request configuration model
    - GlueLLMSettings: Global settings manager

Exceptions:
    - LLMError: Base exception
    - TokenLimitError: Token limit exceeded
    - RateLimitError: Rate limit hit
    - APIConnectionError: Connection/network error
    - InvalidRequestError: Invalid request parameters
    - AuthenticationError: Authentication failed
"""

from gluellm.api import (
    APIConnectionError,
    AuthenticationError,
    ExecutionResult,
    GlueLLM,
    InvalidRequestError,
    # Exceptions
    LLMError,
    RateLimitError,
    StreamingChunk,
    TokenLimitError,
    complete,
    embed,
    get_session_summary,
    reset_session_tracker,
    stream_complete,
    structured_complete,
)
from gluellm.batch import (
    BatchProcessor,
    batch_complete,
    batch_complete_simple,
)
from gluellm.config import GlueLLMSettings, get_settings, reload_settings, settings
from gluellm.eval import (
    CallbackStore,
    EvalRecord,
    EvalStore,
    JSONLFileStore,
    MultiStore,
    enable_callback_recording,
    enable_file_recording,
    get_global_eval_store,
    set_global_eval_store,
)
from gluellm.models.batch import (
    APIKeyConfig,
    BatchConfig,
    BatchErrorStrategy,
    BatchRequest,
    BatchResponse,
    BatchResult,
)
from gluellm.models.config import RequestConfig
from gluellm.models.conversation import Conversation, Message, Role
from gluellm.models.embedding import EmbeddingResult
from gluellm.models.prompt import Prompt, SystemPrompt
from gluellm.observability.logging_config import setup_logging
from gluellm.rate_limiting.api_key_pool import APIKeyPool
from gluellm.runtime.context import (
    clear_correlation_id,
    clear_request_metadata,
    get_context_dict,
    get_correlation_id,
    get_request_metadata,
    set_correlation_id,
    set_request_metadata,
    with_correlation_id,
)
from gluellm.runtime.shutdown import (
    ShutdownContext,
    execute_shutdown_callbacks,
    get_in_flight_count,
    graceful_shutdown,
    is_shutting_down,
    register_shutdown_callback,
    setup_signal_handlers,
    unregister_shutdown_callback,
    wait_for_shutdown,
)
from gluellm.schema import (
    create_normalized_model,
    create_openai_response_format,
    normalize_schema_for_openai,
)

# Initialize logging on package import
_setup_logging_called = False


def _initialize_logging() -> None:
    """Initialize logging configuration from settings."""
    global _setup_logging_called
    if not _setup_logging_called:
        setup_logging(
            log_level=settings.log_level,
            log_file_level=settings.log_file_level,
            log_dir=settings.log_dir,
            log_file_name=settings.log_file_name,
            log_json_format=settings.log_json_format,
            log_max_bytes=settings.log_max_bytes,
            log_backup_count=settings.log_backup_count,
            console_output=settings.log_console_output,
        )
        _setup_logging_called = True


# Initialize logging when package is imported
_initialize_logging()

__all__ = [
    # High-level API
    "GlueLLM",
    "complete",
    "embed",
    "stream_complete",
    "structured_complete",
    "ExecutionResult",
    "EmbeddingResult",
    "StreamingChunk",
    # Session Tracking
    "get_session_summary",
    "reset_session_tracker",
    # Batch Processing
    "BatchProcessor",
    "batch_complete",
    "batch_complete_simple",
    "BatchRequest",
    "BatchResult",
    "BatchResponse",
    "BatchConfig",
    "BatchErrorStrategy",
    "APIKeyConfig",
    # Rate Limiting
    "APIKeyPool",
    # Exceptions
    "LLMError",
    "TokenLimitError",
    "RateLimitError",
    "APIConnectionError",
    "InvalidRequestError",
    "AuthenticationError",
    # Models
    "RequestConfig",
    "Conversation",
    "Message",
    "Role",
    "SystemPrompt",
    "Prompt",
    "EmbeddingResult",
    "EvalRecord",
    # Evaluation Recording
    "EvalStore",
    "JSONLFileStore",
    "CallbackStore",
    "MultiStore",
    "enable_file_recording",
    "enable_callback_recording",
    "set_global_eval_store",
    "get_global_eval_store",
    # Configuration
    "GlueLLMSettings",
    "settings",
    "get_settings",
    "reload_settings",
    # Context and Correlation IDs
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "with_correlation_id",
    "get_request_metadata",
    "set_request_metadata",
    "clear_request_metadata",
    "get_context_dict",
    # Graceful Shutdown
    "ShutdownContext",
    "is_shutting_down",
    "setup_signal_handlers",
    "graceful_shutdown",
    "register_shutdown_callback",
    "unregister_shutdown_callback",
    "get_in_flight_count",
    "wait_for_shutdown",
    "execute_shutdown_callbacks",
    # Schema Utilities
    "normalize_schema_for_openai",
    "create_normalized_model",
    "create_openai_response_format",
]

__version__ = "0.1.0"
