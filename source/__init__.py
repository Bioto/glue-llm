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
    >>> from source import complete, structured_complete, GlueLLM
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

from source.api import (
    APIConnectionError,
    AuthenticationError,
    GlueLLM,
    InvalidRequestError,
    # Exceptions
    LLMError,
    RateLimitError,
    TokenLimitError,
    ToolExecutionResult,
    complete,
    structured_complete,
)
from source.config import GlueLLMSettings, get_settings, reload_settings, settings
from source.models.config import RequestConfig
from source.models.conversation import Conversation, Message, Role
from source.models.prompt import Prompt, SystemPrompt

__all__ = [
    # High-level API
    "GlueLLM",
    "complete",
    "structured_complete",
    "ToolExecutionResult",
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
