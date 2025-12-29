"""GlueLLM - A high-level LLM SDK with automatic tool execution and structured outputs.

GlueLLM is a Python SDK that simplifies working with Large Language Models by providing:
- Automatic tool/function calling with execution loops
- Structured outputs using Pydantic models
- Multi-turn conversations with memory
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
    GlueLLM,
    InvalidRequestError,
    # Exceptions
    LLMError,
    RateLimitError,
    StreamingChunk,
    TokenLimitError,
    ToolExecutionResult,
    complete,
    stream_complete,
    structured_complete,
)
from gluellm.config import GlueLLMSettings, get_settings, reload_settings, settings
from gluellm.logging_config import setup_logging
from gluellm.models.config import RequestConfig
from gluellm.models.conversation import Conversation, Message, Role
from gluellm.models.prompt import Prompt, SystemPrompt

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
        )
        _setup_logging_called = True


# Initialize logging when package is imported
_initialize_logging()

__all__ = [
    # High-level API
    "GlueLLM",
    "complete",
    "stream_complete",
    "structured_complete",
    "ToolExecutionResult",
    "StreamingChunk",
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
    # Configuration
    "GlueLLMSettings",
    "settings",
    "get_settings",
    "reload_settings",
]

__version__ = "0.1.0"
