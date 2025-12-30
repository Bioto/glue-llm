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

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
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

from gluellm.api_key_pool import extract_provider_from_model
from gluellm.config import settings
from gluellm.context import clear_correlation_id, get_correlation_id, set_correlation_id
from gluellm.logging_config import get_logger
from gluellm.models.conversation import Conversation, Role
from gluellm.rate_limiter import acquire_rate_limit
from gluellm.shutdown import ShutdownContext, is_shutting_down
from gluellm.telemetry import (
    is_tracing_enabled,
    log_llm_metrics,
    record_token_usage,
    set_span_attributes,
    trace_llm_call,
)

# Configure logging
logger = get_logger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Mapping of provider names to their API key environment variables
PROVIDER_ENV_VAR_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_token_usage(response: ChatCompletion) -> dict[str, int] | None:
    """Extract token usage from a ChatCompletion response safely.

    Handles various response formats and ensures token counts are integers.

    Args:
        response: The ChatCompletion response object from the LLM

    Returns:
        Dictionary with 'prompt', 'completion', and 'total' token counts,
        or None if usage information is not available.

    Example:
        >>> tokens = _extract_token_usage(response)
        >>> if tokens:
        ...     print(f"Total tokens: {tokens['total']}")
    """
    if not hasattr(response, "usage") or not response.usage:
        return None

    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    return {
        "prompt": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0,
        "completion": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0,
        "total": int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
    }


@contextmanager
def _temporary_api_key(model: str, api_key: str | None):
    """Context manager for temporarily setting an API key in the environment.

    Temporarily sets the appropriate environment variable for the given provider,
    and restores the original value (or removes it) when the context exits.

    Args:
        model: Model identifier in format "provider:model_name"
        api_key: The API key to set temporarily, or None to skip

    Yields:
        None

    Example:
        >>> with _temporary_api_key("openai:gpt-4", "sk-test-key"):
        ...     # OPENAI_API_KEY is set to "sk-test-key"
        ...     await make_api_call()
        ... # Original value is restored
    """
    if not api_key:
        yield
        return

    provider = extract_provider_from_model(model)
    env_var = PROVIDER_ENV_VAR_MAP.get(provider.lower())

    if not env_var:
        yield
        return

    original_value = os.environ.get(env_var)
    os.environ[env_var] = api_key
    logger.debug(f"Temporarily set {env_var} for this request")

    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = original_value
        logger.debug(f"Restored {env_var} to original value")


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
    stream: bool = False,
    timeout: float | None = None,
    api_key: str | None = None,
) -> ChatCompletion | AsyncIterator[ChatCompletion]:
    """Make an LLM call with error classification and tracing.

    This wraps the any_llm_acompletion call to catch and classify errors,
    and optionally trace the call with OpenTelemetry.
    Raises our custom exception types for better error handling.

    Args:
        messages: List of message dictionaries
        model: Model identifier
        tools: Optional list of tools
        response_format: Optional Pydantic model for structured output
        stream: Whether to stream the response
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        api_key: Optional API key override (for key pool usage)

    Returns:
        ChatCompletion if stream=False, AsyncIterator[ChatCompletion] if stream=True

    Raises:
        asyncio.TimeoutError: If the request exceeds the timeout
    """
    correlation_id = get_correlation_id()
    timeout = timeout or settings.default_request_timeout
    timeout = min(timeout, settings.max_request_timeout)  # Enforce max timeout

    # Apply rate limiting before making the call
    provider = extract_provider_from_model(model)
    rate_limit_key = (
        f"global:{provider}" if not api_key else f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"
    )
    await acquire_rate_limit(rate_limit_key)

    start_time = time.time()
    logger.debug(
        f"Making LLM call: model={model}, stream={stream}, has_tools={bool(tools)}, "
        f"response_format={response_format.__name__ if response_format else None}, "
        f"message_count={len(messages)}, timeout={timeout}s, correlation_id={correlation_id}"
    )

    try:
        # Use tracing context if enabled
        with trace_llm_call(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            response_format=response_format.__name__ if response_format else None,
            correlation_id=correlation_id,
        ) as span:
            # Add correlation ID to span attributes
            if correlation_id:
                set_span_attributes(span, correlation_id=correlation_id)

            # Use context manager for temporary API key
            with _temporary_api_key(model, api_key):
                # Make LLM call with timeout
                response = await asyncio.wait_for(
                    any_llm_acompletion(
                        messages=messages,
                        model=model,
                        tools=tools if tools else None,
                        response_format=response_format,
                        stream=stream,
                    ),
                    timeout=timeout,
                )

            elapsed_time = time.time() - start_time

            # For non-streaming responses, record token usage
            tokens_used = None
            finish_reason = None
            has_tool_calls = False

            if not stream:
                tokens_used = _extract_token_usage(response)
                if tokens_used:
                    record_token_usage(span, tokens_used)
                    logger.info(
                        f"LLM call completed: model={model}, latency={elapsed_time:.3f}s, "
                        f"tokens={tokens_used['total']} (prompt={tokens_used['prompt']}, "
                        f"completion={tokens_used['completion']})"
                    )

            # Record response metadata
            if not stream and hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", "unknown")
                has_tool_calls = bool(getattr(choice.message, "tool_calls", None))
                set_span_attributes(
                    span,
                    **{
                        "llm.response.finish_reason": finish_reason,
                        "llm.response.has_tool_calls": has_tool_calls,
                    },
                )
                logger.debug(f"Response metadata: finish_reason={finish_reason}, has_tool_calls={has_tool_calls}")
            elif stream:
                logger.debug(f"LLM call streaming started: model={model}, latency={elapsed_time:.3f}s")

            # Log metrics to MLflow
            log_llm_metrics(
                model=model,
                latency=elapsed_time,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                has_tool_calls=has_tool_calls,
                error=False,
            )

            return response

    except TimeoutError:
        elapsed_time = time.time() - start_time
        logger.error(
            f"LLM call timed out after {elapsed_time:.3f}s (timeout={timeout}s): model={model}, "
            f"correlation_id={correlation_id}",
            exc_info=True,
        )
        # Log timeout metrics
        log_llm_metrics(
            model=model,
            latency=elapsed_time,
            tokens_used=None,
            finish_reason=None,
            has_tool_calls=False,
            error=True,
            error_type="TimeoutError",
        )
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        # Classify the error and raise the appropriate exception
        classified_error = classify_llm_error(e)
        error_type = type(classified_error).__name__
        logger.error(
            f"LLM call failed after {elapsed_time:.3f}s: model={model}, error={classified_error}, "
            f"error_type={error_type}, correlation_id={correlation_id}",
            exc_info=True,
        )

        # Log error metrics to MLflow
        log_llm_metrics(
            model=model,
            latency=elapsed_time,
            tokens_used=None,
            finish_reason=None,
            has_tool_calls=False,
            error=True,
            error_type=error_type,
        )

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
    timeout: float | None = None,
    api_key: str | None = None,
) -> ChatCompletion:
    """Make an LLM call with automatic retry on transient errors.

    Retries up to 3 times with exponential backoff for:
    - Rate limit errors (429)
    - Connection errors (5xx)

    Does NOT retry for:
    - Token limit errors (need to reduce input)
    - Authentication errors (bad credentials)
    - Invalid request errors (bad parameters)
    - Timeout errors

    Args:
        messages: List of message dictionaries
        model: Model identifier
        tools: Optional list of tools
        response_format: Optional Pydantic model for structured output
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        api_key: Optional API key override (for key pool usage)
    """
    return await _safe_llm_call(
        messages=messages,
        model=model,
        tools=tools,
        response_format=response_format,
        timeout=timeout,
        api_key=api_key,
    )


class ToolExecutionResult(BaseModel):
    """Result of a tool execution loop."""

    final_response: Annotated[str, Field(description="The final text response from the model")]
    tool_calls_made: Annotated[int, Field(description="Number of tool calls made")]
    tool_execution_history: Annotated[list[dict[str, Any]], Field(description="History of tool calls and results")]
    raw_response: Annotated[
        SkipValidation[ChatCompletion] | None, Field(description="The raw final response from the LLM", default=None)
    ]
    tokens_used: Annotated[
        dict[str, int] | None,
        Field(
            description="Token usage information with 'prompt', 'completion', and 'total' keys",
            default=None,
        ),
    ]


class StreamingChunk(BaseModel):
    """A chunk of streaming response."""

    content: Annotated[str, Field(description="The content chunk")]
    done: Annotated[bool, Field(description="Whether this is the final chunk")]
    tool_calls_made: Annotated[int, Field(description="Number of tool calls made so far", default=0)]


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
        correlation_id: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> ToolExecutionResult:
        """Complete a request with automatic tool execution loop.

        Args:
            user_message: The user's message/request
            execute_tools: Whether to automatically execute tools and loop
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            api_key: Optional API key override (for key pool usage)

        Returns:
            ToolExecutionResult with final response and execution history

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
            asyncio.TimeoutError: If request exceeds timeout
            RuntimeError: If shutdown is in progress
        """
        # Check for shutdown
        if is_shutting_down():
            raise RuntimeError("Cannot process request: shutdown in progress")

        # Set correlation ID if provided
        if correlation_id:
            set_correlation_id(correlation_id)
        elif not get_correlation_id():
            # Auto-generate if not set
            set_correlation_id()

        correlation_id = get_correlation_id()
        logger.info(f"Starting completion request: correlation_id={correlation_id}, message_length={len(user_message)}")

        # Use shutdown context to track in-flight requests for entire execution
        try:
            with ShutdownContext():
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
                logger.debug(
                    f"Starting tool execution loop: max_iterations={self.max_tool_iterations}, "
                    f"tools_available={len(self.tools) if self.tools else 0}"
                )
                for iteration in range(self.max_tool_iterations):
                    logger.debug(f"Tool execution iteration {iteration + 1}/{self.max_tool_iterations}")
                    try:
                        response = await _llm_call_with_retry(
                            messages=messages,
                            model=self.model,
                            tools=self.tools if self.tools else None,
                            timeout=timeout,
                            api_key=api_key,
                        )
                    except LLMError as e:
                        # Log the error and re-raise with context
                        logger.error(f"LLM call failed on iteration {iteration + 1}/{self.max_tool_iterations}: {e}")
                        # Add error context to the exception
                        error_msg = (
                            f"Failed during tool execution loop (iteration {iteration + 1}/{self.max_tool_iterations})"
                        )
                        raise type(e)(f"{error_msg}: {e}") from e

                    # Validate response has choices
                    if not response.choices:
                        raise InvalidRequestError("Empty response from LLM provider")

                    # Check if model wants to call tools
                    if execute_tools and self.tools and response.choices[0].message.tool_calls:
                        tool_calls = response.choices[0].message.tool_calls
                        logger.info(f"Iteration {iteration + 1}: Model requested {len(tool_calls)} tool call(s)")

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
                                tool_start_time = time.time()
                                try:
                                    # Support both sync and async tools
                                    if asyncio.iscoroutinefunction(tool_func):
                                        logger.debug(f"Executing async tool: {tool_name}")
                                        tool_result = await tool_func(**tool_args)
                                    else:
                                        logger.debug(f"Executing sync tool: {tool_name}")
                                        tool_result = tool_func(**tool_args)
                                    tool_result_str = str(tool_result)
                                    tool_elapsed = time.time() - tool_start_time
                                    logger.info(f"Tool {tool_name} executed successfully in {tool_elapsed:.3f}s")

                                    # Record in history
                                    tool_execution_history.append(
                                        {
                                            "tool_name": tool_name,
                                            "arguments": tool_args,
                                            "result": tool_result_str,
                                            "error": False,
                                        }
                                    )

                                    # Record tool execution in trace if enabled
                                    if is_tracing_enabled():
                                        from gluellm.telemetry import _tracer

                                        if _tracer is not None:
                                            with _tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
                                                set_span_attributes(
                                                    tool_span,
                                                    **{
                                                        "tool.name": tool_name,
                                                        "tool.arg_count": len(tool_args),
                                                        "tool.success": True,
                                                    },
                                                )
                                except Exception as e:
                                    # Tool execution error
                                    tool_elapsed = time.time() - tool_start_time
                                    tool_result_str = f"Error executing tool: {type(e).__name__}: {str(e)}"
                                    logger.warning(
                                        f"Tool {tool_name} execution failed after {tool_elapsed:.3f}s: {e}",
                                        exc_info=True,
                                    )

                                    tool_execution_history.append(
                                        {
                                            "tool_name": tool_name,
                                            "arguments": tool_args,
                                            "result": tool_result_str,
                                            "error": True,
                                        }
                                    )

                                    # Record tool execution error in trace if enabled
                                    if is_tracing_enabled():
                                        from gluellm.telemetry import _tracer

                                        if _tracer is not None:
                                            with _tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
                                                set_span_attributes(
                                                    tool_span,
                                                    **{
                                                        "tool.name": tool_name,
                                                        "tool.arg_count": len(tool_args),
                                                        "tool.success": False,
                                                        "tool.error": str(e),
                                                    },
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

                    # Validate response has choices
                    if not response.choices:
                        raise InvalidRequestError("Empty response from LLM provider")

                    # No more tool calls, we have final response
                    final_content = response.choices[0].message.content or ""
                    logger.info(
                        f"Tool execution completed: total_tool_calls={tool_calls_made}, "
                        f"final_response_length={len(final_content)}"
                    )

                    # Add assistant response to conversation
                    self._conversation.add_message(Role.ASSISTANT, final_content)

                    # Extract token usage if available
                    tokens_used = _extract_token_usage(response)
                    if tokens_used:
                        logger.debug(f"Token usage: {tokens_used}")

                    return ToolExecutionResult(
                        final_response=final_content,
                        tool_calls_made=tool_calls_made,
                        tool_execution_history=tool_execution_history,
                        raw_response=response,
                        tokens_used=tokens_used,
                    )

                # Max iterations reached
                logger.warning(f"Max tool execution iterations ({self.max_tool_iterations}) reached")
                final_content = "Maximum tool execution iterations reached."

                # Extract token usage if available
                tokens_used = _extract_token_usage(response)

                return ToolExecutionResult(
                    final_response=final_content,
                    tool_calls_made=tool_calls_made,
                    tool_execution_history=tool_execution_history,
                    raw_response=response,
                    tokens_used=tokens_used,
                )
        finally:
            # Clear correlation ID after request completes
            clear_correlation_id()

    async def structured_complete(
        self,
        user_message: str,
        response_format: type[T],
        correlation_id: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> T:
        """Complete a request and return structured output.

        Args:
            user_message: The user's message/request
            response_format: Pydantic model class for structured output
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            api_key: Optional API key override (for key pool usage)

        Returns:
            Instance of response_format with parsed data

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
            asyncio.TimeoutError: If request exceeds timeout
            RuntimeError: If shutdown is in progress
        """
        # Check for shutdown
        if is_shutting_down():
            raise RuntimeError("Cannot process request: shutdown in progress")

        # Set correlation ID if provided
        if correlation_id:
            set_correlation_id(correlation_id)
        elif not get_correlation_id():
            # Auto-generate if not set
            set_correlation_id()

        correlation_id = get_correlation_id()
        logger.info(
            f"Starting structured completion: correlation_id={correlation_id}, "
            f"response_format={response_format.__name__}, message_length={len(user_message)}"
        )

        # Use shutdown context to track in-flight requests for entire execution
        try:
            with ShutdownContext():
                # Add user message to conversation
                self._conversation.add_message(Role.USER, user_message)

                # Build messages
                system_message = {
                    "role": "system",
                    "content": self._format_system_prompt(),
                }
                messages = [system_message] + self._conversation.messages_dict

                # Call with response_format (no tools for structured output)
                logger.debug(
                    f"Starting structured completion: model={self.model}, response_format={response_format.__name__}"
                )
                try:
                    response = await _llm_call_with_retry(
                        messages=messages,
                        model=self.model,
                        response_format=response_format,
                        timeout=timeout,
                        api_key=api_key,
                    )
                except LLMError as e:
                    logger.error(
                        f"Structured completion failed: model={self.model}, "
                        f"response_format={response_format.__name__}, error={e}"
                    )
                    raise

                # Validate response has choices
                if not response.choices:
                    raise InvalidRequestError("Empty response from LLM provider")

                # Parse the response
                parsed = response.choices[0].message.parsed
                content = response.choices[0].message.content
                logger.debug(
                    f"Structured response received: parsed_type={type(parsed)}, content_length={len(content) if content else 0}"
                )

                # Add assistant response to conversation
                if content:
                    self._conversation.add_message(Role.ASSISTANT, content)

                # If parsed is already a Pydantic instance, return it
                # Otherwise, instantiate the model from the parsed dict/JSON
                if isinstance(parsed, response_format):
                    logger.debug(f"Returning parsed Pydantic instance: {response_format.__name__}")
                    return parsed

                # Handle case where parsed is a dict
                if isinstance(parsed, dict):
                    logger.debug(f"Instantiating {response_format.__name__} from dict")
                    return response_format(**parsed)

                # Fallback: try to parse from JSON string in content
                if content:
                    try:
                        data = json.loads(content)
                        logger.debug(f"Parsed JSON from content, instantiating {response_format.__name__}")
                        return response_format(**data)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse structured response from content: {e}")

                # Last resort: return parsed as-is and hope for the best
                logger.warning(f"Returning parsed response as-is (type: {type(parsed)})")
                return parsed
        finally:
            # Clear correlation ID after request completes
            clear_correlation_id()

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

    async def stream_complete(
        self,
        user_message: str,
        execute_tools: bool = True,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream completion with automatic tool execution.

        Yields chunks of the response as they arrive. When tools are called,
        streaming pauses and tool execution occurs, then streaming resumes.

        Note:
            **Streaming Limitation:** When tools are enabled (`execute_tools=True`),
            the final response after tool execution is NOT streamed token-by-token.
            Instead, it's returned as a single chunk. This is because we need to
            check if the model wants to call more tools before streaming.

            True streaming only occurs when:
            - No tools are provided (`tools=[]` or `tools=None`), OR
            - Tool execution is disabled (`execute_tools=False`)

            This is a known limitation and may be improved in future versions.

        Args:
            user_message: The user's message/request
            execute_tools: Whether to automatically execute tools

        Yields:
            StreamingChunk objects with content and metadata

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
        accumulated_content = ""

        # Tool execution loop
        for iteration in range(self.max_tool_iterations):
            try:
                # Try streaming first (if no tools or tools disabled, stream directly)
                if not execute_tools or not self.tools:
                    # Simple streaming without tool execution
                    async for chunk_response in await _safe_llm_call(
                        messages=messages, model=self.model, tools=None, stream=True
                    ):
                        if hasattr(chunk_response, "choices") and chunk_response.choices:
                            delta = chunk_response.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                accumulated_content += delta.content
                                yield StreamingChunk(
                                    content=delta.content,
                                    done=False,
                                    tool_calls_made=tool_calls_made,
                                )
                    # Final chunk
                    if accumulated_content:
                        self._conversation.add_message(Role.ASSISTANT, accumulated_content)
                        yield StreamingChunk(content="", done=True, tool_calls_made=tool_calls_made)
                    return

                # With tools: get full response first to check for tool calls
                response = await _llm_call_with_retry(
                    messages=messages,
                    model=self.model,
                    tools=self.tools if self.tools else None,
                )
            except LLMError as e:
                logger.error(f"LLM call failed on iteration {iteration + 1}: {e}")
                error_msg = f"Failed during tool execution loop (iteration {iteration + 1}/{self.max_tool_iterations})"
                raise type(e)(f"{error_msg}: {e}") from e

            # Validate response has choices
            if not response.choices:
                raise InvalidRequestError("Empty response from LLM provider")

            # Check if model wants to call tools
            if execute_tools and self.tools and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                logger.info(f"Stream iteration {iteration + 1}: Model requested {len(tool_calls)} tool call(s)")

                # Yield a chunk indicating tool execution is happening
                yield StreamingChunk(
                    content="[Executing tools...]",
                    done=False,
                    tool_calls_made=tool_calls_made,
                )

                # Add assistant message with tool calls to history
                messages.append(response.choices[0].message)

                # Execute each tool call
                for tool_call in tool_calls:
                    tool_calls_made += 1
                    tool_name = tool_call.function.name
                    logger.debug(f"Executing tool call {tool_calls_made}: {tool_name}")

                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        logger.debug(f"Tool {tool_name} arguments: {tool_args}")
                    except json.JSONDecodeError as e:
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
                        tool_start_time = time.time()
                        try:
                            # Check if tool is async
                            if asyncio.iscoroutinefunction(tool_func):
                                logger.debug(f"Executing async tool: {tool_name}")
                                tool_result = await tool_func(**tool_args)
                            else:
                                logger.debug(f"Executing sync tool: {tool_name}")
                                tool_result = tool_func(**tool_args)
                            tool_result_str = str(tool_result)
                            tool_elapsed = time.time() - tool_start_time
                            logger.info(f"Tool {tool_name} executed successfully in {tool_elapsed:.3f}s")

                            tool_execution_history.append(
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": tool_result_str,
                                    "error": False,
                                }
                            )
                        except Exception as e:
                            tool_elapsed = time.time() - tool_start_time
                            tool_result_str = f"Error executing tool: {type(e).__name__}: {str(e)}"
                            logger.warning(
                                f"Tool {tool_name} execution failed after {tool_elapsed:.3f}s: {e}", exc_info=True
                            )
                            tool_execution_history.append(
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": tool_result_str,
                                    "error": True,
                                }
                            )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result_str,
                            }
                        )
                    else:
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

                # Continue loop to get next response (may stream this time)
                continue

            # Validate response has choices
            if not response.choices:
                raise InvalidRequestError("Empty response from LLM provider")

            # No more tool calls, stream the final response
            final_content = response.choices[0].message.content or ""
            accumulated_content = ""

            # Stream the final response character by character (simulated streaming)
            # In a real implementation, you'd stream from the API
            if final_content:
                # For now, yield the full content as a single chunk
                # In production, this would be actual streaming chunks from the API
                self._conversation.add_message(Role.ASSISTANT, final_content)
                yield StreamingChunk(content=final_content, done=True, tool_calls_made=tool_calls_made)
            else:
                yield StreamingChunk(content="", done=True, tool_calls_made=tool_calls_made)

            return

        # Max iterations reached
        logger.warning(f"Max tool execution iterations ({self.max_tool_iterations}) reached")
        yield StreamingChunk(
            content="Maximum tool execution iterations reached.",
            done=True,
            tool_calls_made=tool_calls_made,
        )

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
    correlation_id: str | None = None,
    timeout: float | None = None,
) -> ToolExecutionResult:
    """Quick completion with automatic tool execution.

    Args:
        user_message: The user's message/request
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)
        tools: List of callable functions to use as tools
        execute_tools: Whether to automatically execute tools
        max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)
        correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)

    Returns:
        ToolExecutionResult with final response and execution history
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
    )
    return await client.complete(
        user_message, execute_tools=execute_tools, correlation_id=correlation_id, timeout=timeout
    )


async def structured_complete(
    user_message: str,
    response_format: type[T],
    model: str | None = None,
    system_prompt: str | None = None,
    correlation_id: str | None = None,
    timeout: float | None = None,
) -> T:
    """Quick structured completion.

    Args:
        user_message: The user's message/request
        response_format: Pydantic model class for structured output
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)
        correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)

    Returns:
        Instance of response_format with parsed data
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
    )
    return await client.structured_complete(
        user_message, response_format, correlation_id=correlation_id, timeout=timeout
    )


async def stream_complete(
    user_message: str,
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
) -> AsyncIterator[StreamingChunk]:
    """Stream completion with automatic tool execution.

    Yields chunks of the response as they arrive. Note: tool execution
    interrupts streaming - when tools are called, streaming pauses until
    tool results are processed.

    Note:
        **Streaming Limitation:** When tools are enabled (`execute_tools=True`),
        the final response after tool execution is NOT streamed token-by-token.
        Instead, it's returned as a single chunk. True streaming only occurs
        when no tools are provided or when `execute_tools=False`.
        See GlueLLM.stream_complete() for details.

    Args:
        user_message: The user's message/request
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)
        tools: List of callable functions to use as tools
        execute_tools: Whether to automatically execute tools
        max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)

    Yields:
        StreamingChunk objects with content and metadata

    Example:
        >>> async for chunk in stream_complete("Tell me a story"):
        ...     print(chunk.content, end="", flush=True)
        ...     if chunk.done:
        ...         print(f"\\nTool calls: {chunk.tool_calls_made}")
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
    )
    async for chunk in client.stream_complete(user_message, execute_tools=execute_tools):
        yield chunk
