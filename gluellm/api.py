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
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
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
from gluellm.logging_config import get_logger
from gluellm.models.conversation import Conversation, Role
from gluellm.telemetry import (
    is_tracing_enabled,
    record_token_usage,
    set_span_attributes,
    trace_llm_call,
)

# Configure logging
logger = get_logger(__name__)

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

    Returns:
        ChatCompletion if stream=False, AsyncIterator[ChatCompletion] if stream=True
    """
    start_time = time.time()
    logger.debug(
        f"Making LLM call: model={model}, stream={stream}, has_tools={bool(tools)}, "
        f"response_format={response_format.__name__ if response_format else None}, "
        f"message_count={len(messages)}"
    )

    try:
        # Use tracing context if enabled
        with trace_llm_call(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            response_format=response_format.__name__ if response_format else None,
        ) as span:
            response = await any_llm_acompletion(
                messages=messages,
                model=model,
                tools=tools if tools else None,
                response_format=response_format,
                stream=stream,
            )

            elapsed_time = time.time() - start_time

            # For non-streaming responses, record token usage
            if not stream and hasattr(response, "usage") and response.usage:
                usage = response.usage
                # Safely extract token counts, ensuring they're integers
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                tokens_used = {
                    "prompt": int(prompt_tokens)
                    if prompt_tokens is not None and isinstance(prompt_tokens, (int, float))
                    else 0,
                    "completion": int(completion_tokens)
                    if completion_tokens is not None and isinstance(completion_tokens, (int, float))
                    else 0,
                    "total": int(total_tokens)
                    if total_tokens is not None and isinstance(total_tokens, (int, float))
                    else 0,
                }
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

            return response

    except Exception as e:
        elapsed_time = time.time() - start_time
        # Classify the error and raise the appropriate exception
        classified_error = classify_llm_error(e)
        logger.error(
            f"LLM call failed after {elapsed_time:.3f}s: model={model}, error={classified_error}, "
            f"error_type={type(classified_error).__name__}",
            exc_info=True,
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
                )
            except LLMError as e:
                # Log the error and re-raise with context
                logger.error(f"LLM call failed on iteration {iteration + 1}/{self.max_tool_iterations}: {e}")
                # Add error context to the exception
                error_msg = f"Failed during tool execution loop (iteration {iteration + 1}/{self.max_tool_iterations})"
                raise type(e)(f"{error_msg}: {e}") from e

            # Check if model wants to call tools
            if execute_tools and self.tools and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                logger.info(f"Iteration {iteration + 1}: Model requested {len(tool_calls)} tool call(s)")
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

            # No more tool calls, we have final response
            final_content = response.choices[0].message.content or ""
            logger.info(
                f"Tool execution completed: total_tool_calls={tool_calls_made}, "
                f"final_response_length={len(final_content)}"
            )

            # Add assistant response to conversation
            self._conversation.add_message(Role.ASSISTANT, final_content)

            # Extract token usage if available
            tokens_used = None
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                # Safely extract token counts, ensuring they're integers
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                tokens_used = {
                    "prompt": int(prompt_tokens)
                    if prompt_tokens is not None and isinstance(prompt_tokens, (int, float))
                    else 0,
                    "completion": int(completion_tokens)
                    if completion_tokens is not None and isinstance(completion_tokens, (int, float))
                    else 0,
                    "total": int(total_tokens)
                    if total_tokens is not None and isinstance(total_tokens, (int, float))
                    else 0,
                }
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
        tokens_used = None
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            # Safely extract token counts, ensuring they're integers
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            tokens_used = {
                "prompt": int(prompt_tokens)
                if prompt_tokens is not None and isinstance(prompt_tokens, (int, float))
                else 0,
                "completion": int(completion_tokens)
                if completion_tokens is not None and isinstance(completion_tokens, (int, float))
                else 0,
                "total": int(total_tokens)
                if total_tokens is not None and isinstance(total_tokens, (int, float))
                else 0,
            }

        return ToolExecutionResult(
            final_response=final_content,
            tool_calls_made=tool_calls_made,
            tool_execution_history=tool_execution_history,
            raw_response=response,
            tokens_used=tokens_used,
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
        logger.debug(f"Starting structured completion: model={self.model}, response_format={response_format.__name__}")
        try:
            response = await _llm_call_with_retry(
                messages=messages,
                model=self.model,
                response_format=response_format,
            )
        except LLMError as e:
            logger.error(
                f"Structured completion failed: model={self.model}, "
                f"response_format={response_format.__name__}, error={e}"
            )
            raise

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
