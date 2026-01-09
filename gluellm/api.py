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
    ...     result = await structured_complete(
    ...         "What is 2+2?",
    ...         response_format=Answer
    ...     )
    ...     print(result.structured_output.number)
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
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

if TYPE_CHECKING:
    from gluellm.models.embedding import EmbeddingResult

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
from gluellm.costing.pricing_data import calculate_cost
from gluellm.eval import get_global_eval_store
from gluellm.eval.store import EvalStore
from gluellm.models.conversation import Conversation, Role
from gluellm.models.eval import EvalRecord
from gluellm.observability.logging_config import get_logger
from gluellm.rate_limiting.api_key_pool import extract_provider_from_model
from gluellm.rate_limiting.rate_limiter import acquire_rate_limit
from gluellm.runtime.context import clear_correlation_id, get_correlation_id, set_correlation_id
from gluellm.runtime.shutdown import ShutdownContext, is_shutting_down, register_shutdown_callback
from gluellm.schema import create_normalized_model
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
# Session Cost Tracker
# ============================================================================


class _SessionCostTracker:
    """Tracks token usage and costs for the current session.

    This is a lightweight in-memory tracker that accumulates usage across
    all API calls and can print a summary on exit.
    """

    def __init__(self):
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._request_count: int = 0
        self._cost_by_model: dict[str, float] = {}
        self._tokens_by_model: dict[str, dict[str, int]] = {}
        self._shutdown_registered: bool = False

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None,
    ) -> None:
        """Record usage from an API call."""
        if not settings.track_costs:
            return

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._request_count += 1

        if cost_usd is not None:
            self._total_cost_usd += cost_usd
            self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + cost_usd

        if model not in self._tokens_by_model:
            self._tokens_by_model[model] = {"prompt": 0, "completion": 0}
        self._tokens_by_model[model]["prompt"] += prompt_tokens
        self._tokens_by_model[model]["completion"] += completion_tokens

        # Register shutdown callback on first usage
        if not self._shutdown_registered and settings.print_session_summary_on_exit:
            register_shutdown_callback(self._print_summary)
            self._shutdown_registered = True

    def _print_summary(self) -> None:
        """Print session summary on exit."""
        if self._request_count == 0:
            return

        total_tokens = self._total_prompt_tokens + self._total_completion_tokens

        logger.info("=" * 60)
        logger.info("GlueLLM Session Summary")
        logger.info("=" * 60)
        logger.info(f"Total Requests: {self._request_count}")
        logger.info(f"Total Tokens: {total_tokens:,}")
        logger.info(f"  - Prompt: {self._total_prompt_tokens:,}")
        logger.info(f"  - Completion: {self._total_completion_tokens:,}")
        logger.info(f"Estimated Total Cost: ${self._total_cost_usd:.6f}")
        logger.info("-" * 60)

        if self._cost_by_model:
            logger.info("Breakdown by Model:")
            for model, cost in sorted(self._cost_by_model.items(), key=lambda x: -x[1]):
                tokens = self._tokens_by_model.get(model, {})
                model_tokens = tokens.get("prompt", 0) + tokens.get("completion", 0)
                logger.info(f"  {model}: ${cost:.6f} ({model_tokens:,} tokens)")

        logger.info("=" * 60)

    def get_summary(self) -> dict[str, Any]:
        """Get session summary as a dictionary."""
        return {
            "request_count": self._request_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_cost_usd": self._total_cost_usd,
            "cost_by_model": self._cost_by_model.copy(),
            "tokens_by_model": {k: v.copy() for k, v in self._tokens_by_model.items()},
        }

    def reset(self) -> dict[str, Any]:
        """Reset the tracker and return the final summary."""
        summary = self.get_summary()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._request_count = 0
        self._cost_by_model.clear()
        self._tokens_by_model.clear()
        return summary


# Global session tracker instance
_session_tracker = _SessionCostTracker()


def get_session_summary() -> dict[str, Any]:
    """Get the current session cost/token summary.

    Returns:
        Dictionary with session statistics including total tokens, cost, and breakdowns.

    Example:
        >>> summary = get_session_summary()
        >>> print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    """
    return _session_tracker.get_summary()


def reset_session_tracker() -> dict[str, Any]:
    """Reset the session tracker and return the final summary.

    Returns:
        Dictionary with the final session statistics before reset.
    """
    return _session_tracker.reset()


async def _record_eval_data(
    eval_store: EvalStore | None,
    user_message: str,
    system_prompt: str,
    model: str,
    messages_snapshot: list[dict],
    start_time: float,
    result: "ExecutionResult | None" = None,
    error: Exception | None = None,
    tools_available: list[Callable] | None = None,
) -> None:
    """Record evaluation data to the eval store.

    Args:
        eval_store: The evaluation store to record to (None = no recording)
        user_message: The user's input message
        system_prompt: System prompt used
        model: Model identifier
        messages_snapshot: Full conversation state
        start_time: Request start time (from time.time())
        result: ExecutionResult if successful
        error: Exception if request failed
        tools_available: List of available tools
    """
    if not eval_store:
        return

    try:
        latency_ms = (time.time() - start_time) * 1000.0

        # Extract tool names
        tools_available_names = [tool.__name__ for tool in (tools_available or [])]

        # Build EvalRecord
        if result:
            # Success case
            # Serialize raw_response if present
            raw_response_dict = None
            if result.raw_response:
                try:
                    # Convert ChatCompletion to dict
                    raw_response_dict = {
                        "id": getattr(result.raw_response, "id", None),
                        "model": getattr(result.raw_response, "model", None),
                        "choices": [
                            {
                                "index": getattr(choice, "index", None),
                                "message": {
                                    "role": getattr(choice.message, "role", None),
                                    "content": getattr(choice.message, "content", None),
                                    "tool_calls": [
                                        {
                                            "id": getattr(tc, "id", None),
                                            "type": getattr(tc, "type", None),
                                            "function": {
                                                "name": getattr(tc.function, "name", None),
                                                "arguments": getattr(tc.function, "arguments", None),
                                            },
                                        }
                                        for tc in (getattr(choice.message, "tool_calls", None) or [])
                                    ],
                                },
                                "finish_reason": getattr(choice, "finish_reason", None),
                            }
                            for choice in (getattr(result.raw_response, "choices", None) or [])
                        ],
                        "usage": {
                            "prompt_tokens": getattr(result.raw_response.usage, "prompt_tokens", None)
                            if hasattr(result.raw_response, "usage") and result.raw_response.usage
                            else None,
                            "completion_tokens": getattr(result.raw_response.usage, "completion_tokens", None)
                            if hasattr(result.raw_response, "usage") and result.raw_response.usage
                            else None,
                            "total_tokens": getattr(result.raw_response.usage, "total_tokens", None)
                            if hasattr(result.raw_response, "usage") and result.raw_response.usage
                            else None,
                        }
                        if hasattr(result.raw_response, "usage") and result.raw_response.usage
                        else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to serialize raw_response: {e}")

            # Serialize structured_output if present
            structured_output_serialized = None
            if result.structured_output:
                try:
                    if hasattr(result.structured_output, "model_dump"):
                        structured_output_serialized = result.structured_output.model_dump()
                    elif hasattr(result.structured_output, "dict"):
                        structured_output_serialized = result.structured_output.dict()
                    else:
                        structured_output_serialized = str(result.structured_output)
                except Exception as e:
                    logger.debug(f"Failed to serialize structured_output: {e}")
                    structured_output_serialized = str(result.structured_output)

            record = EvalRecord(
                correlation_id=get_correlation_id(),
                user_message=user_message,
                system_prompt=system_prompt,
                model=model,
                messages_snapshot=messages_snapshot,
                final_response=result.final_response,
                structured_output=structured_output_serialized,
                raw_response=raw_response_dict,
                tool_calls_made=result.tool_calls_made,
                tool_execution_history=result.tool_execution_history,
                tools_available=tools_available_names,
                latency_ms=latency_ms,
                tokens_used=result.tokens_used,
                estimated_cost_usd=result.estimated_cost_usd,
                success=True,
            )
        else:
            # Error case
            record = EvalRecord(
                correlation_id=get_correlation_id(),
                user_message=user_message,
                system_prompt=system_prompt,
                model=model,
                messages_snapshot=messages_snapshot,
                final_response="",
                tool_calls_made=0,
                tool_execution_history=[],
                tools_available=tools_available_names,
                latency_ms=latency_ms,
                success=False,
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
            )

        # Record asynchronously (fire and forget)
        await eval_store.record(record)

    except Exception as e:
        # Log but don't raise - recording failures shouldn't break completions
        logger.error(f"Failed to record evaluation data: {e}", exc_info=True)


def _calculate_and_record_cost(
    model: str,
    tokens_used: dict[str, int] | None,
    correlation_id: str | None = None,
) -> float | None:
    """Calculate cost from token usage and record to session tracker.

    Args:
        model: Model identifier (e.g., "openai:gpt-4o-mini")
        tokens_used: Token usage dictionary with 'prompt', 'completion', 'total'
        correlation_id: Optional correlation ID for logging

    Returns:
        Estimated cost in USD, or None if cost cannot be calculated
    """
    if not tokens_used:
        return None

    prompt_tokens = tokens_used.get("prompt", 0)
    completion_tokens = tokens_used.get("completion", 0)

    # Calculate cost
    provider = extract_provider_from_model(model)
    model_name = model.split(":", 1)[1] if ":" in model else model

    cost = calculate_cost(
        provider=provider,
        model_name=model_name,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )

    # Record to session tracker
    _session_tracker.record_usage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
    )

    if cost is not None:
        logger.debug(
            f"Cost calculated: ${cost:.6f} for {prompt_tokens}+{completion_tokens} tokens "
            f"(model={model}, correlation_id={correlation_id})"
        )

    return cost


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
StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel | None)


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

    # Normalize Pydantic model schema for OpenAI compatibility
    # This fixes issues with union types, additionalProperties, etc. that cause
    # "True is not of type 'array'" and similar schema validation errors
    # We create a subclass that overrides model_json_schema() so OpenAI's .parse()
    # method gets the normalized schema when it calls model_json_schema()
    normalized_response_format: type[BaseModel] | None = None
    if response_format is not None:
        try:
            normalized_response_format = create_normalized_model(response_format)
            # Verify the normalization worked by checking the schema
            test_schema = normalized_response_format.model_json_schema()
            if test_schema.get("additionalProperties") is True:
                logger.error(
                    f"Schema normalization failed for {response_format.__name__}: "
                    "additionalProperties is still True. Falling back to original model."
                )
                normalized_response_format = None
            else:
                logger.debug(
                    f"Created normalized model class for {response_format.__name__}: "
                    f"strict={test_schema.get('strict')}, "
                    f"additionalProperties={test_schema.get('additionalProperties')}"
                )
        except Exception as e:
            # Fall back to passing the Pydantic model directly if normalization fails
            logger.warning(
                f"Schema normalization failed for {response_format.__name__}: {e}",
                exc_info=True,
            )
            normalized_response_format = None

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
                # Use normalized model class if available, otherwise fall back to original Pydantic model
                # The normalized class is a subclass, so response parsing still works correctly
                response = await asyncio.wait_for(
                    any_llm_acompletion(
                        messages=messages,
                        model=model,
                        tools=tools if tools else None,
                        response_format=normalized_response_format if normalized_response_format else response_format,
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
                    # Calculate cost for this LLM call
                    provider = extract_provider_from_model(model)
                    model_name = model.split(":", 1)[1] if ":" in model else model
                    call_cost = calculate_cost(
                        provider=provider,
                        model_name=model_name,
                        input_tokens=tokens_used.get("prompt", 0),
                        output_tokens=tokens_used.get("completion", 0),
                    )

                    # Record tokens and cost to span
                    record_token_usage(span, tokens_used, cost_usd=call_cost)

                    cost_str = f", cost=${call_cost:.6f}" if call_cost else ""
                    logger.info(
                        f"LLM call completed: model={model}, latency={elapsed_time:.3f}s, "
                        f"tokens={tokens_used['total']} (prompt={tokens_used['prompt']}, "
                        f"completion={tokens_used['completion']}){cost_str}"
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


class ExecutionResult(BaseModel):
    """Result of a tool execution loop.

    Generic type parameter allows proper typing for structured outputs.
    Use ExecutionResult[YourModel] for structured completions.
    """

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
    estimated_cost_usd: Annotated[
        float | None,
        Field(
            description="Estimated cost in USD based on token usage and model pricing",
            default=None,
        ),
    ]
    model: Annotated[
        str | None,
        Field(
            description="The model used for this completion",
            default=None,
        ),
    ]
    structured_output: Annotated[
        Any | None,
        Field(
            description="Parsed structured output (Pydantic model instance) for structured completions",
            default=None,
        ),
    ]

    def __len__(self) -> int:
        """Return the length of the final response or tool execution history."""
        if hasattr(self, "final_response") and self.final_response:
            return len(str(self.final_response))
        if hasattr(self, "tool_execution_history") and self.tool_execution_history:
            return len(self.tool_execution_history)
        return 0


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
        embedding_model: str | None = None,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        max_tool_iterations: int | None = None,
        eval_store: EvalStore | None = None,
    ):
        """Initialize GlueLLM client.

        Args:
            model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
            embedding_model: Embedding model identifier in format "provider/model_name" (defaults to settings.default_embedding_model)
            system_prompt: System prompt content (defaults to settings.default_system_prompt)
            tools: List of callable functions to use as tools
            max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)
            eval_store: Optional evaluation store for recording request/response data (defaults to global store if set)
        """
        self.model = model or settings.default_model
        self.embedding_model = embedding_model or settings.default_embedding_model
        self.system_prompt = system_prompt or settings.default_system_prompt
        self.tools = tools or []
        self.max_tool_iterations = max_tool_iterations or settings.max_tool_iterations
        self._conversation = Conversation()
        self.eval_store = eval_store or get_global_eval_store()

    async def complete(
        self,
        user_message: str,
        model: str | None = None,
        execute_tools: bool = True,
        correlation_id: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> ExecutionResult:
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

        # Capture start time for evaluation recording
        start_time = time.time()
        system_prompt_content = self._format_system_prompt()
        messages_snapshot: list[dict] = []
        result: ExecutionResult | None = None
        error: Exception | None = None

        # Use shutdown context to track in-flight requests for entire execution
        try:
            with ShutdownContext():
                # Add user message to conversation
                self._conversation.add_message(Role.USER, user_message)

                # Build initial messages
                system_message = {
                    "role": "system",
                    "content": system_prompt_content,
                }
                messages = [system_message] + self._conversation.messages_dict
                messages_snapshot = messages.copy()

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
                            model=model or self.model,
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

                    # Calculate cost and record to session tracker
                    estimated_cost = _calculate_and_record_cost(
                        model=model or self.model,
                        tokens_used=tokens_used,
                        correlation_id=correlation_id,
                    )

                    result = ExecutionResult(
                        final_response=final_content,
                        tool_calls_made=tool_calls_made,
                        tool_execution_history=tool_execution_history,
                        raw_response=response,
                        tokens_used=tokens_used,
                        estimated_cost_usd=estimated_cost,
                        model=self.model,
                    )

                    # Record evaluation data
                    await _record_eval_data(
                        eval_store=self.eval_store,
                        user_message=user_message,
                        system_prompt=system_prompt_content,
                        model=model or self.model,
                        messages_snapshot=messages_snapshot,
                        start_time=start_time,
                        result=result,
                        tools_available=self.tools,
                    )

                    return result

                # Max iterations reached
                logger.warning(f"Max tool execution iterations ({self.max_tool_iterations}) reached")
                final_content = "Maximum tool execution iterations reached."

                # Extract token usage if available
                tokens_used = _extract_token_usage(response)

                # Calculate cost and record to session tracker
                estimated_cost = _calculate_and_record_cost(
                    model=self.model,
                    tokens_used=tokens_used,
                    correlation_id=correlation_id,
                )

                result = ExecutionResult(
                    final_response=final_content,
                    tool_calls_made=tool_calls_made,
                    tool_execution_history=tool_execution_history,
                    raw_response=response,
                    tokens_used=tokens_used,
                    estimated_cost_usd=estimated_cost,
                    model=self.model,
                )

                # Record evaluation data
                await _record_eval_data(
                    eval_store=self.eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    result=result,
                    tools_available=self.tools,
                )

                return result
        except Exception as e:
            error = e
            raise
        finally:
            # Record evaluation data on error if not already recorded
            if error and not result:
                await _record_eval_data(
                    eval_store=self.eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    error=error,
                    tools_available=self.tools,
                )
            # Clear correlation ID after request completes
            clear_correlation_id()

    async def structured_complete(
        self,
        user_message: str,
        response_format: type[T],
        model: str | None = None,
        tools: list[Callable] | None = None,
        execute_tools: bool = True,
        correlation_id: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> ExecutionResult:
        """Complete a request and return structured output.

        The LLM can optionally use tools to gather information before returning
        the final structured output. Tools will be executed in a loop until the
        LLM returns the structured response.

        Args:
            user_message: The user's message/request
            response_format: Pydantic model class for structured output
            model: Model identifier override (defaults to instance model)
            tools: List of callable functions to use as tools (defaults to instance tools)
            execute_tools: Whether to automatically execute tools and loop
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            api_key: Optional API key override (for key pool usage)

        Returns:
            ExecutionResult with structured_output field containing instance of response_format

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

        # Capture start time for evaluation recording
        start_time = time.time()
        system_prompt_content = self._format_system_prompt()
        messages_snapshot: list[dict] = []
        result: ExecutionResult | None = None
        error: Exception | None = None

        # Determine which tools to use: parameter overrides instance tools
        tools_to_use = tools if tools is not None else self.tools

        # Use shutdown context to track in-flight requests for entire execution
        try:
            with ShutdownContext():
                # Add user message to conversation
                self._conversation.add_message(Role.USER, user_message)

                # Build initial messages
                system_message = {
                    "role": "system",
                    "content": system_prompt_content,
                }
                messages = [system_message] + self._conversation.messages_dict
                messages_snapshot = messages.copy()

                tool_execution_history = []
                tool_calls_made = 0
                total_tokens_used = None
                total_cost = 0.0

                # Helper to track token usage and cost
                def _track_usage(resp):
                    nonlocal total_tokens_used, total_cost
                    iteration_tokens = _extract_token_usage(resp)
                    if iteration_tokens:
                        if total_tokens_used is None:
                            total_tokens_used = iteration_tokens.copy()
                        else:
                            total_tokens_used["prompt_tokens"] = total_tokens_used.get(
                                "prompt_tokens", 0
                            ) + iteration_tokens.get("prompt_tokens", 0)
                            total_tokens_used["completion_tokens"] = total_tokens_used.get(
                                "completion_tokens", 0
                            ) + iteration_tokens.get("completion_tokens", 0)
                            total_tokens_used["total_tokens"] = total_tokens_used.get(
                                "total_tokens", 0
                            ) + iteration_tokens.get("total_tokens", 0)
                    iteration_cost = _calculate_and_record_cost(
                        model=model or self.model,
                        tokens_used=iteration_tokens,
                        correlation_id=correlation_id,
                    )
                    if iteration_cost is not None:
                        total_cost += iteration_cost

                # PHASE 1: Tool execution loop (if tools are provided and execute_tools is True)
                if tools_to_use and execute_tools:
                    logger.debug(
                        f"Starting tool execution phase: max_iterations={self.max_tool_iterations}, "
                        f"tools_available={len(tools_to_use)}"
                    )
                    for iteration in range(self.max_tool_iterations):
                        logger.debug(f"Tool execution iteration {iteration + 1}/{self.max_tool_iterations}")

                        try:
                            response = await _llm_call_with_retry(
                                messages=messages,
                                model=model or self.model,
                                tools=tools_to_use,
                                # No response_format during tool phase
                                timeout=timeout,
                                api_key=api_key,
                            )
                        except LLMError as e:
                            logger.error(f"LLM call failed on iteration {iteration + 1}: {e}")
                            raise type(e)(f"Failed during tool execution (iteration {iteration + 1}): {e}") from e

                        _track_usage(response)

                        if not response.choices:
                            raise InvalidRequestError("Empty response from LLM provider")

                        # Check if model wants to call tools
                        if response.choices[0].message.tool_calls:
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
                                tool_func = self._find_tool(tool_name, tools_to_use)
                                if tool_func:
                                    tool_start_time = time.time()
                                    try:
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

                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "content": tool_result_str,
                                        }
                                    )
                                else:
                                    error_msg = f"Tool '{tool_name}' not found"
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
                            # Continue to next iteration
                        else:
                            # LLM didn't call any tools - it has enough info, break out of tool loop
                            logger.debug(f"LLM finished with tools after {iteration + 1} iteration(s)")
                            # Add the assistant's response to messages so structured output call has context
                            if response.choices[0].message.content:
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": response.choices[0].message.content,
                                    }
                                )
                            break
                    else:
                        # Exhausted all iterations with tool calls - still need structured output
                        logger.debug(f"Reached max tool iterations ({self.max_tool_iterations})")

                # PHASE 2: Final structured output call
                logger.debug(f"Requesting structured output: response_format={response_format.__name__}")
                try:
                    response = await _llm_call_with_retry(
                        messages=messages,
                        model=model or self.model,
                        response_format=response_format,
                        # No tools during structured output phase
                        timeout=timeout,
                        api_key=api_key,
                    )
                except LLMError as e:
                    logger.error(f"Structured output call failed: {e}")
                    raise type(e)(f"Failed during structured output request: {e}") from e

                _track_usage(response)

                if not response.choices:
                    raise InvalidRequestError("Empty response from LLM provider")

                # Parse the response
                parsed = getattr(response.choices[0].message, "parsed", None)
                content = response.choices[0].message.content
                logger.debug(
                    f"Structured response received: parsed_type={type(parsed)}, "
                    f"content_length={len(content) if content else 0}"
                )

                # Add assistant response to conversation
                if content:
                    self._conversation.add_message(Role.ASSISTANT, content)

                # Parse the structured output
                structured_output = None
                if isinstance(parsed, response_format):
                    logger.debug(f"Using parsed Pydantic instance: {response_format.__name__}")
                    structured_output = parsed
                elif isinstance(parsed, dict):
                    logger.debug(f"Instantiating {response_format.__name__} from dict")
                    structured_output = response_format(**parsed)
                elif content:
                    # Fallback: try to parse from JSON string in content
                    try:
                        data = json.loads(content)
                        logger.debug(f"Parsed JSON from content, instantiating {response_format.__name__}")
                        structured_output = response_format(**data)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse structured response from content: {e}")
                        structured_output = parsed
                else:
                    logger.warning(f"Using parsed response as-is (type: {type(parsed)})")
                    structured_output = parsed

                logger.info(f"Structured completion finished: tool_calls={tool_calls_made}, cost=${total_cost:.6f}")

                result = ExecutionResult(
                    final_response=content or "",
                    tool_calls_made=tool_calls_made,
                    tool_execution_history=tool_execution_history,
                    raw_response=response,
                    tokens_used=total_tokens_used,
                    estimated_cost_usd=total_cost if total_cost > 0 else None,
                    model=model or self.model,
                    structured_output=structured_output,
                )

                # Record evaluation data
                await _record_eval_data(
                    eval_store=self.eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    result=result,
                    tools_available=tools_to_use,
                )

                return result
        except Exception as e:
            error = e
            raise
        finally:
            # Record evaluation data on error if not already recorded
            if error and not result:
                await _record_eval_data(
                    eval_store=self.eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    error=error,
                    tools_available=tools_to_use,
                )
            # Clear correlation ID after request completes
            clear_correlation_id()

    def _format_system_prompt(self) -> str:
        """Format system prompt with tools if available."""
        from gluellm.models.prompt import BASE_SYSTEM_PROMPT

        return BASE_SYSTEM_PROMPT.render(
            instructions=self.system_prompt,
            tools=self.tools,
        ).strip()

    def _find_tool(self, tool_name: str, tools: list[Callable] | None = None) -> Callable | None:
        """Find a tool by name.

        Args:
            tool_name: Name of the tool to find
            tools: Optional list of tools to search (defaults to self.tools)

        Returns:
            The tool function if found, None otherwise
        """
        tools_to_search = tools if tools is not None else self.tools
        for tool in tools_to_search:
            if tool.__name__ == tool_name:
                return tool
        return None

    async def stream_complete(
        self,
        user_message: str,
        execute_tools: bool = True,
        model: str | None = None,
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
                    model=model or self.model,
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

    async def embed(
        self,
        texts: str | list[str],
        model: str | None = None,
        correlation_id: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> "EmbeddingResult":
        """Generate embeddings for the given text(s).

        Args:
            texts: Single text string or list of text strings to embed
            model: Model identifier (defaults to self.embedding_model)
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            api_key: Optional API key override (for key pool usage)

        Returns:
            EmbeddingResult with embeddings, model, token usage, and cost

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
            asyncio.TimeoutError: If request exceeds timeout
            RuntimeError: If shutdown is in progress

        Example:
            >>> import asyncio
            >>> from gluellm import GlueLLM
            >>>
            >>> async def main():
            ...     client = GlueLLM()
            ...     result = await client.complete("Hello")
            ...     embedding = await client.embed("Hello")
            ...     print(f"Embedding dimension: {embedding.dimension}")
            >>>
            >>> asyncio.run(main())
        """
        from gluellm.embeddings import embed as embed_func

        # Use instance embedding model if no override provided
        model = model or self.embedding_model

        return await embed_func(
            texts=texts,
            model=model,
            correlation_id=correlation_id,
            timeout=timeout,
            api_key=api_key,
        )


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
) -> ExecutionResult:
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
    tools: list[Callable] | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    correlation_id: str | None = None,
    timeout: float | None = None,
) -> ExecutionResult:
    """Quick structured completion with optional tool support.

    The LLM can optionally use tools to gather information before returning
    the final structured output. Tools will be executed in a loop until the
    LLM returns the structured response.

    Args:
        user_message: The user's message/request
        response_format: Pydantic model class for structured output
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)
        tools: List of callable functions to use as tools
        execute_tools: Whether to automatically execute tools and loop
        max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)
        correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)

    Returns:
        ExecutionResult with structured_output field containing instance of response_format

    Example:
        >>> import asyncio
        >>> from gluellm.api import structured_complete
        >>> from pydantic import BaseModel
        >>>
        >>> class Answer(BaseModel):
        ...     number: int
        ...     reasoning: str
        >>>
        >>> def get_calculator_result(a: int, b: int) -> int:
        ...     '''Add two numbers together.'''
        ...     return a + b
        >>>
        >>> async def main():
        ...     # Example 1: Without tools
        ...     result = await structured_complete(
        ...         "What is 2+2?",
        ...         response_format=Answer
        ...     )
        ...     print(f"Answer: {result.structured_output.number}")
        ...     print(f"Reasoning: {result.structured_output.reasoning}")
        ...
        ...     # Example 2: With tools - LLM can gather data before returning structured output
        ...     result = await structured_complete(
        ...         "Calculate 2+2 using the calculator tool and explain your answer",
        ...         response_format=Answer,
        ...         tools=[get_calculator_result]
        ...     )
        ...     print(f"Answer: {result.structured_output.number}")
        ...     print(f"Tools used: {result.tool_calls_made}")
        ...     print(f"Cost: ${result.estimated_cost_usd:.6f}")
        >>>
        >>> asyncio.run(main())
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
    )
    return await client.structured_complete(
        user_message,
        response_format,
        tools=tools,
        execute_tools=execute_tools,
        correlation_id=correlation_id,
        timeout=timeout,
    )


async def embed(
    texts: str | list[str],
    model: str | None = None,
    correlation_id: str | None = None,
    timeout: float | None = None,
) -> "EmbeddingResult":
    """Quick embedding generation.

    Args:
        texts: Single text string or list of text strings to embed
        model: Model identifier (defaults to settings.default_embedding_model)
        correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)

    Returns:
        EmbeddingResult with embeddings, model, token usage, and cost

    Example:
        >>> import asyncio
        >>> from gluellm.api import embed
        >>>
        >>> async def main():
        ...     result = await embed("Hello, world!")
        ...     print(f"Embedding dimension: {result.dimension}")
        >>>
        >>> asyncio.run(main())
    """
    from gluellm.embeddings import embed as embed_func

    return await embed_func(
        texts=texts,
        model=model,
        correlation_id=correlation_id,
        timeout=timeout,
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
