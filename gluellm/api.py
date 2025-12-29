"""GlueLLM Python API - High-level interface for LLM interactions.

This module provides the main API for interacting with Large Language Models,
including automatic tool execution, structured outputs, and comprehensive error
handling with automatic retries.

Core Components:
    - GlueLLM: Main client class for LLM interactions
    - complete: Quick completion function with tool execution
    - structured_complete: Quick structured output function
    - ToolExecutionResult: Result container for tool execution

Exception Hierarchy:
    - LLMError (base)
        - TokenLimitError: Token/context length exceeded
        - RateLimitError: Rate limit hit
        - APIConnectionError: Network/connection issues
        - InvalidRequestError: Bad request parameters
        - AuthenticationError: Authentication failed

Features:
    - Automatic tool execution with configurable iterations
    - Structured output using Pydantic models
    - Multi-turn conversations with memory
    - Automatic retry with exponential backoff
    - Comprehensive error classification and handling

Example:
    >>> import asyncio
    >>> from gluellm.api import complete, structured_complete
    >>> from pydantic import BaseModel
    >>>
    >>> async def main():
    ...     # Simple completion
    ...     result = await complete("What is 2+2?")
    ...     print(result.final_response)
    ...
    ...     # Structured output
    ...     class Answer(BaseModel):
    ...         number: int
    ...
    ...     answer = await structured_complete(
    ...         "What is 2+2?",
    ...         response_format=Answer
    ...     )
    ...     print(answer.number)
    >>>
    >>> asyncio.run(main())
"""

import json
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from any_llm import acompletion as any_llm_acompletion
from any_llm.types.completion import ChatCompletion
from pydantic import BaseModel, Field
from pydantic.functional_validators import SkipValidation
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gluellm.config import settings
from gluellm.models.conversation import Conversation, Role

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Exception Classes
# ============================================================================


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""

    pass


class RateLimitError(LLMError):
    """Raised when rate limit is hit."""

    pass


class APIConnectionError(LLMError):
    """Raised when there's a connection issue with the API."""

    pass


class InvalidRequestError(LLMError):
    """Raised when the request is invalid (bad params, etc)."""

    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails."""

    pass


# ============================================================================
# Error Classification
# ============================================================================


def classify_llm_error(error: Exception) -> Exception:
    """Classify an error from any_llm into our custom exception types.

    This function examines the error message and type to determine what kind
    of error occurred, making it easier to handle specific cases.
    """
    error_msg = str(error).lower()
    error_type = type(error).__name__

    # Token/context length errors
    if any(
        keyword in error_msg
        for keyword in [
            "context length",
            "token limit",
            "maximum context",
            "too many tokens",
            "context_length_exceeded",
            "max_tokens",
        ]
    ):
        return TokenLimitError(f"Token limit exceeded: {error}")

    # Rate limiting errors
    if any(
        keyword in error_msg
        for keyword in [
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota exceeded",
            "resource exhausted",
            "throttled",
            "429",
        ]
    ):
        return RateLimitError(f"Rate limit hit: {error}")

    # Connection/network errors
    if any(
        keyword in error_msg
        for keyword in [
            "connection",
            "timeout",
            "network",
            "unreachable",
            "503",
            "502",
            "504",
        ]
    ):
        return APIConnectionError(f"API connection error: {error}")

    # Authentication errors
    if any(
        keyword in error_msg
        for keyword in [
            "unauthorized",
            "invalid api key",
            "authentication",
            "auth",
            "401",
            "403",
        ]
    ):
        return AuthenticationError(f"Authentication failed: {error}")

    # Invalid request errors
    if any(
        keyword in error_msg
        for keyword in [
            "invalid",
            "bad request",
            "400",
            "validation",
        ]
    ):
        return InvalidRequestError(f"Invalid request: {error}")

    # Default to generic LLM error
    return LLMError(f"LLM error ({error_type}): {error}")


def should_retry_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry.

    Retryable errors:
    - RateLimitError (wait and retry)
    - APIConnectionError (transient network issues)

    Non-retryable errors:
    - TokenLimitError (need to reduce input)
    - AuthenticationError (bad credentials)
    - InvalidRequestError (bad parameters)
    """
    return isinstance(error, (RateLimitError, APIConnectionError))


# ============================================================================
# Retry-wrapped LLM Completion
# ============================================================================


async def _safe_llm_call(
    messages: list[dict],
    model: str,
    tools: list[Callable] | None = None,
    response_format: type[BaseModel] | None = None,
) -> ChatCompletion:
    """Make an LLM call with error classification.

    This wraps the any_llm_acompletion call to catch and classify errors.
    Raises our custom exception types for better error handling.
    """
    try:
        return await any_llm_acompletion(
            messages=messages,
            model=model,
            tools=tools if tools else None,
            response_format=response_format,
        )
    except Exception as e:
        # Classify the error and raise the appropriate exception
        classified_error = classify_llm_error(e)
        logger.error(f"LLM call failed: {classified_error}")
        raise classified_error from e


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(settings.retry_max_attempts),
    wait=wait_exponential(
        multiplier=settings.retry_multiplier, min=settings.retry_min_wait, max=settings.retry_max_wait
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _llm_call_with_retry(
    messages: list[dict],
    model: str,
    tools: list[Callable] | None = None,
    response_format: type[BaseModel] | None = None,
) -> ChatCompletion:
    """Make an LLM call with automatic retry on transient errors.

    Retries up to 3 times with exponential backoff for:
    - Rate limit errors (429)
    - Connection errors (5xx)

    Does NOT retry for:
    - Token limit errors (need to reduce input)
    - Authentication errors (bad credentials)
    - Invalid request errors (bad parameters)
    """
    return await _safe_llm_call(
        messages=messages,
        model=model,
        tools=tools,
        response_format=response_format,
    )


class ToolExecutionResult(BaseModel):
    """Result of a tool execution loop."""

    final_response: Annotated[str, Field(description="The final text response from the model")]
    tool_calls_made: Annotated[int, Field(description="Number of tool calls made")]
    tool_execution_history: Annotated[list[dict[str, Any]], Field(description="History of tool calls and results")]
    raw_response: Annotated[
        SkipValidation[ChatCompletion] | None, Field(description="The raw final response from the LLM", default=None)
    ]


class GlueLLM:
    """High-level API for LLM interactions with automatic tool execution."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        max_tool_iterations: int | None = None,
    ):
        """Initialize GlueLLM client.

        Args:
            model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
            system_prompt: System prompt content (defaults to settings.default_system_prompt)
            tools: List of callable functions to use as tools
            max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)
        """
        self.model = model or settings.default_model
        self.system_prompt = system_prompt or settings.default_system_prompt
        self.tools = tools or []
        self.max_tool_iterations = max_tool_iterations or settings.max_tool_iterations
        self._conversation = Conversation()

    async def complete(
        self,
        user_message: str,
        execute_tools: bool = True,
    ) -> ToolExecutionResult:
        """Complete a request with automatic tool execution loop.

        Args:
            user_message: The user's message/request
            execute_tools: Whether to automatically execute tools and loop

        Returns:
            ToolExecutionResult with final response and execution history

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
        """
        # Add user message to conversation
        self._conversation.add_message(Role.USER, user_message)

        # Build initial messages
        system_message = {
            "role": "system",
            "content": self._format_system_prompt(),
        }
        messages = [system_message] + self._conversation.messages_dict

        tool_execution_history = []
        tool_calls_made = 0

        # Tool execution loop
        for iteration in range(self.max_tool_iterations):
            try:
                response = await _llm_call_with_retry(
                    messages=messages,
                    model=self.model,
                    tools=self.tools if self.tools else None,
                )
            except LLMError as e:
                # Log the error and re-raise with context
                logger.error(f"LLM call failed on iteration {iteration + 1}: {e}")
                # Add error context to the exception
                error_msg = f"Failed during tool execution loop (iteration {iteration + 1}/{self.max_tool_iterations})"
                raise type(e)(f"{error_msg}: {e}") from e

            # Check if model wants to call tools
            if execute_tools and self.tools and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls

                # Add assistant message with tool calls to history
                messages.append(response.choices[0].message)

                # Execute each tool call
                for tool_call in tool_calls:
                    tool_calls_made += 1
                    tool_name = tool_call.function.name

                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        # Handle malformed JSON from model
                        error_msg = f"Invalid JSON in tool arguments: {str(e)}"
                        logger.warning(f"Tool {tool_name} - {error_msg}")
                        tool_execution_history.append(
                            {
                                "tool_name": tool_name,
                                "arguments": tool_call.function.arguments,
                                "result": error_msg,
                                "error": True,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg,
                            }
                        )
                        continue

                    # Find and execute the tool
                    tool_func = self._find_tool(tool_name)
                    if tool_func:
                        try:
                            tool_result = tool_func(**tool_args)
                            tool_result_str = str(tool_result)

                            # Record in history
                            tool_execution_history.append(
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": tool_result_str,
                                    "error": False,
                                }
                            )
                        except Exception as e:
                            # Tool execution error
                            tool_result_str = f"Error executing tool: {type(e).__name__}: {str(e)}"
                            logger.warning(f"Tool {tool_name} execution failed: {e}")

                            tool_execution_history.append(
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": tool_result_str,
                                    "error": True,
                                }
                            )

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result_str,
                            }
                        )
                    else:
                        # Tool not found
                        error_msg = f"Tool '{tool_name}' not found in available tools"
                        logger.warning(error_msg)
                        tool_execution_history.append(
                            {
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result": error_msg,
                                "error": True,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg,
                            }
                        )

                # Continue loop to get next response
                continue

            # No more tool calls, we have final response
            final_content = response.choices[0].message.content or ""

            # Add assistant response to conversation
            self._conversation.add_message(Role.ASSISTANT, final_content)

            return ToolExecutionResult(
                final_response=final_content,
                tool_calls_made=tool_calls_made,
                tool_execution_history=tool_execution_history,
                raw_response=response,
            )

        # Max iterations reached
        logger.warning(f"Max tool execution iterations ({self.max_tool_iterations}) reached")
        final_content = "Maximum tool execution iterations reached."
        return ToolExecutionResult(
            final_response=final_content,
            tool_calls_made=tool_calls_made,
            tool_execution_history=tool_execution_history,
            raw_response=response,
        )

    async def structured_complete(
        self,
        user_message: str,
        response_format: type[T],
    ) -> T:
        """Complete a request and return structured output.

        Args:
            user_message: The user's message/request
            response_format: Pydantic model class for structured output

        Returns:
            Instance of response_format with parsed data

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
        """
        # Add user message to conversation
        self._conversation.add_message(Role.USER, user_message)

        # Build messages
        system_message = {
            "role": "system",
            "content": self._format_system_prompt(),
        }
        messages = [system_message] + self._conversation.messages_dict

        # Call with response_format (no tools for structured output)
        try:
            response = await _llm_call_with_retry(
                messages=messages,
                model=self.model,
                response_format=response_format,
            )
        except LLMError as e:
            logger.error(f"Structured completion failed: {e}")
            raise

        # Parse the response
        parsed = response.choices[0].message.parsed
        content = response.choices[0].message.content

        # Add assistant response to conversation
        if content:
            self._conversation.add_message(Role.ASSISTANT, content)

        # If parsed is already a Pydantic instance, return it
        # Otherwise, instantiate the model from the parsed dict/JSON
        if isinstance(parsed, response_format):
            return parsed

        # Handle case where parsed is a dict
        if isinstance(parsed, dict):
            return response_format(**parsed)

        # Fallback: try to parse from JSON string in content
        if content:
            try:
                data = json.loads(content)
                return response_format(**data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse structured response from content: {e}")

        # Last resort: return parsed as-is and hope for the best
        logger.warning(f"Returning parsed response as-is (type: {type(parsed)})")
        return parsed

    def _format_system_prompt(self) -> str:
        """Format system prompt with tools if available."""
        from gluellm.models.prompt import BASE_SYSTEM_PROMPT

        return BASE_SYSTEM_PROMPT.render(
            instructions=self.system_prompt,
            tools=self.tools,
        ).strip()

    def _find_tool(self, tool_name: str) -> Callable | None:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.__name__ == tool_name:
                return tool
        return None

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self._conversation = Conversation()


# Convenience functions for one-off requests


async def complete(
    user_message: str,
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
) -> ToolExecutionResult:
    """Quick completion with automatic tool execution.

    Args:
        user_message: The user's message/request
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)
        tools: List of callable functions to use as tools
        execute_tools: Whether to automatically execute tools
        max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)

    Returns:
        ToolExecutionResult with final response and execution history
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
    )
    return await client.complete(user_message, execute_tools=execute_tools)


async def structured_complete(
    user_message: str,
    response_format: type[T],
    model: str | None = None,
    system_prompt: str | None = None,
) -> T:
    """Quick structured completion.

    Args:
        user_message: The user's message/request
        response_format: Pydantic model class for structured output
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)

    Returns:
        Instance of response_format with parsed data
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
    )
    return await client.structured_complete(user_message, response_format)
