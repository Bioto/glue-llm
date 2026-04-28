"""GlueLLM Python API - High-level interface for LLM interactions.

This module provides the main API for interacting with Large Language Models,
including automatic tool execution, structured outputs, and comprehensive error
handling with automatic retries.

Core Components:
    - GlueLLM: Main client class for LLM interactions
    - complete: Quick completion function with tool execution
    - structured_complete: Quick structured output via Chat Completions
    - structured_response: Quick structured output via the OpenAI Responses API
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

import httpx
import importlib
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar, Union, get_args, get_origin, get_type_hints, overload

if TYPE_CHECKING:
    from gluellm.models.agent import Agent
    from gluellm.models.embedding import EmbeddingResult

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion
from pydantic import BaseModel, Field, ValidationError, field_validator, field_serializer
from pydantic.functional_validators import SkipValidation
from gluellm.config import settings
from gluellm.config import ToolExecutionOrder
from gluellm.tool_router import (
    ToolMode,
    build_router_tool,
    is_router_call,
    is_static_tool,
    resolve_tool_route,
)
from gluellm.costing.pricing_data import calculate_cost
from gluellm.eval import get_global_eval_store
from gluellm.eval.store import EvalStore
from gluellm.events import ProcessEvent, Sink, emit_status
from gluellm.guardrails import GuardrailBlockedError, GuardrailRejectedError, GuardrailsConfig
from gluellm.hooks.manager import HookManager, _get_global_registry
from gluellm.models.hook import HookRegistry, HookStage
from gluellm.guardrails.runner import run_input_guardrails, run_output_guardrails
from gluellm.models.conversation import Conversation, Role
from gluellm.models.eval import EvalRecord
from gluellm.observability.logging_config import get_logger
from gluellm.provider_params import normalize_model_params
from gluellm.rate_limiting.api_key_pool import extract_provider_from_model
from gluellm.rate_limiting.key_fingerprint import api_key_hmac_fingerprint
from gluellm.rate_limiting.rate_limiter import acquire_rate_limit
from gluellm.rate_limit_types import RateLimitAlgorithm
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

# Callback for process status events (sync or async)
type OnStatusCallback = Callable[[ProcessEvent], None] | Callable[[ProcessEvent], Awaitable[None]] | None

# Reasoning effort for o3, o4-mini, Claude thinking models (any_llm 1.11.0)
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "auto"]

# GlueLLM call-level options that must never be forwarded to provider.acompletion (OpenAI, etc.).
_PROVIDER_ACOMPLETION_SKIP_KEYS = frozenset({"max_tool_iterations", "execute_tools", "rate_limit_algorithm"})


# ============================================================================
# Retry Configuration
# ============================================================================


# Callback invoked on retryable error: (error, attempt) -> (should_retry, next_params | None)
# next_params is merged into model kwargs for the next attempt (e.g. {"temperature": 0.8})
type RetryCallback = (
    Callable[[Exception, int], tuple[bool, dict[str, Any] | None]]
    | Callable[[Exception, int], Awaitable[tuple[bool, dict[str, Any] | None]]]
    | None
)


class RetryConfig(BaseModel):
    """Configuration for retry behavior on LLM calls.

    Attributes:
        retry_enabled: If False, no retries are performed (single attempt only).
        max_attempts: Maximum number of attempts including the first call.
        min_wait: Minimum wait time in seconds between retries.
        max_wait: Maximum wait time in seconds between retries.
        multiplier: Exponential backoff multiplier.
        retry_on: Optional list of exception types to retry on. When set,
            only errors that are instances of one of these types trigger a
            retry; all others are re-raised immediately. Takes precedence
            over the default (RateLimitError + APIConnectionError) but is
            ignored when callback is also set.
        callback: Optional function called on each retryable error.
            Signature: (error, attempt) -> (should_retry, next_params | None)
            - should_retry: if False, stop retrying and re-raise the error.
            - next_params: dict merged into model kwargs for the next attempt
              (e.g. {"temperature": 0.0}). Pass None to keep current params.
            Can be sync or async. When set, retry_on is ignored.
    """

    retry_enabled: bool = True
    max_attempts: Annotated[int, Field(gt=0)] = 3
    min_wait: Annotated[float, Field(ge=0)] = 2.0
    max_wait: Annotated[float, Field(ge=0)] = 30.0
    multiplier: Annotated[float, Field(ge=0)] = 1.0
    retry_on: list[type[Exception]] | None = None
    callback: RetryCallback = None

    model_config = {"arbitrary_types_allowed": True}


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting on LLM and embedding calls.

    Attributes:
        algorithm: Rate limiting algorithm (RateLimitAlgorithm enum or string).
            When None, uses settings.rate_limit_algorithm (GLUELLM_RATE_LIMIT_ALGORITHM).
    """

    algorithm: RateLimitAlgorithm | str | None = None

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, v: RateLimitAlgorithm | str | None) -> RateLimitAlgorithm | str | None:
        if v is None or isinstance(v, RateLimitAlgorithm):
            return v
        if v not in [e.value for e in RateLimitAlgorithm]:
            raise ValueError(
                f"Invalid rate limit algorithm: {v!r}. "
                f"Must be one of: {', '.join(e.value for e in RateLimitAlgorithm)}"
            )
        return v


class SummarizeContextConfig(BaseModel):
    """Configuration for automatic context summarization."""

    enabled: bool = False
    threshold: int | None = None
    model: str | None = None
    keep_recent: int | None = None

    @field_validator("threshold", "keep_recent")
    @classmethod
    def _positive_optional(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("must be > 0")
        return v


def _default_summarize_context_config() -> SummarizeContextConfig:
    return SummarizeContextConfig(
        enabled=settings.default_summarize_context,
        threshold=settings.default_summarize_context_threshold,
        model=None,
        keep_recent=settings.default_summarize_context_keep_recent,
    )


def _normalize_summarize_context_init(value: SummarizeContextConfig | bool | None) -> SummarizeContextConfig:
    """Build instance-level summarization config from ctor argument."""
    base = _default_summarize_context_config()
    if value is None:
        return base
    if isinstance(value, bool):
        return base.model_copy(update={"enabled": value})
    merged = base.model_copy(update=value.model_dump(exclude_unset=True))
    if merged.threshold is None:
        merged = merged.model_copy(update={"threshold": settings.default_summarize_context_threshold})
    if merged.keep_recent is None:
        merged = merged.model_copy(update={"keep_recent": settings.default_summarize_context_keep_recent})
    return merged


def _merge_summarize_context_for_call(
    base: SummarizeContextConfig,
    override: SummarizeContextConfig | bool | None,
) -> SummarizeContextConfig:
    """Merge per-call override with instance-level config."""
    if override is None:
        return base
    if isinstance(override, bool):
        return base.model_copy(update={"enabled": override})
    merged = base.model_copy(update=override.model_dump(exclude_unset=True))
    if merged.threshold is None:
        merged = merged.model_copy(update={"threshold": base.threshold})
    if merged.keep_recent is None:
        merged = merged.model_copy(update={"keep_recent": base.keep_recent})
    return merged


def _summarize_model_for_call(cfg: SummarizeContextConfig, primary_model: str) -> str:
    """Resolve model for summarization (explicit config, then primary)."""
    return cfg.model or primary_model


# Configure logging
logger = get_logger(__name__)

# Context variable to store current agent during executor execution
# This allows _record_eval_data to automatically capture agent information
# when AgentExecutor is used, without requiring API changes
_current_agent: ContextVar["Agent | None"] = ContextVar("_current_agent", default=None)
_any_llm_openai_patch_applied = False
_any_llm_openai_embedding_dimensions_patch_applied = False


def _convert_openai_chat_completion_without_parsed(response: Any) -> ChatCompletion:
    """Convert OpenAI chat completion to any_llm ChatCompletion without serializing `message.parsed`.

    OpenAI's parsed completion objects may carry a user model instance in
    `choices[*].message.parsed`. Serializing that field through a base schema that
    expects `None` triggers Pydantic serializer warnings. We exclude it at source.
    """
    openai_utils = importlib.import_module("any_llm.providers.openai.utils")
    normalize_openai_dict_response = getattr(openai_utils, "_normalize_openai_dict_response")

    if getattr(response, "object", None) != "chat.completion":
        logger.warning(
            "API returned an unexpected object type: %s. Setting to 'chat.completion'.",
            getattr(response, "object", None),
        )
        response.object = "chat.completion"
    if not isinstance(getattr(response, "created", None), int):
        logger.warning(
            "API returned an unexpected created type: %s. Setting to int.",
            type(getattr(response, "created", None)),
        )
        response.created = int(getattr(response, "created", 0))

    parsed_by_index: dict[int, Any] = {}
    for i, choice in enumerate(getattr(response, "choices", []) or []):
        message = getattr(choice, "message", None)
        parsed_value = getattr(message, "parsed", None)
        if parsed_value is not None:
            parsed_by_index[i] = parsed_value

    normalized = normalize_openai_dict_response(
        response.model_dump(exclude={"choices": {"__all__": {"message": {"parsed"}}}})
    )
    converted = ChatCompletion.model_validate(normalized)

    # Preserve runtime parsed objects for structured output consumers while keeping
    # serialization warning-free by excluding parsed from model_dump conversion.
    for i, parsed_value in parsed_by_index.items():
        if i < len(converted.choices):
            converted.choices[i].message.parsed = parsed_value

    return converted


def _patch_any_llm_openai_converter() -> None:
    """Patch any_llm OpenAI conversion to avoid serializing `message.parsed`."""
    global _any_llm_openai_patch_applied
    if _any_llm_openai_patch_applied:
        return

    try:
        openai_base = importlib.import_module("any_llm.providers.openai.base")
        openai_utils = importlib.import_module("any_llm.providers.openai.utils")
    except Exception:
        return

    openai_base._convert_chat_completion = _convert_openai_chat_completion_without_parsed
    openai_utils._convert_chat_completion = _convert_openai_chat_completion_without_parsed
    _any_llm_openai_patch_applied = True
    _patch_any_llm_openai_embedding_dimensions()


def _patch_any_llm_openai_embedding_dimensions() -> None:
    """Patch any_llm OpenAI provider so dimensions is only passed once to embeddings.create().

    any_llm's _aembedding calls _convert_embedding_params(inputs, **kwargs) which merges
    kwargs (including dimensions) into embedding_kwargs, then calls
    client.embeddings.create(model=..., dimensions=kwargs.get(...), **embedding_kwargs).
    That passes dimensions twice → TypeError. This patch pops dimensions from kwargs
    before _convert_embedding_params so it's only passed as the explicit argument.
    """
    global _any_llm_openai_embedding_dimensions_patch_applied
    if _any_llm_openai_embedding_dimensions_patch_applied:
        return

    try:
        from openai._types import NOT_GIVEN

        openai_base = importlib.import_module("any_llm.providers.openai.base")
        BaseOpenAIProvider = openai_base.BaseOpenAIProvider
        original_aembedding = BaseOpenAIProvider._aembedding
    except Exception:
        return

    async def _patched_aembedding(
        self: Any,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> Any:
        dimensions = kwargs.pop("dimensions", NOT_GIVEN)
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        return self._convert_embedding_response(
            await self.client.embeddings.create(
                model=model,
                dimensions=dimensions,
                **embedding_kwargs,
            )
        )

    BaseOpenAIProvider._aembedding = _patched_aembedding
    _any_llm_openai_embedding_dimensions_patch_applied = True


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
# Provider Cache
# ============================================================================


class _ProviderCache:
    """Module-level cache of AnyLLM provider instances.

    Each unique (provider_name, api_key) pair maps to a single AnyLLM instance
    that owns an httpx AsyncClient. Reusing instances means the underlying HTTP
    connection pool is shared across requests, which prevents the
    'RuntimeError: Event loop is closed' error that occurs when abandoned
    AsyncOpenAI clients are garbage-collected after the event loop exits.
    """

    def __init__(self) -> None:
        self._providers: dict[tuple[str, str | None], AnyLLM] = {}
        self._lock = threading.Lock()

    def get_provider(self, model: str, api_key: str | None) -> tuple[AnyLLM, str]:
        """Return a cached (provider, model_id) pair, creating one if needed.

        Args:
            model: Full model string in "provider:model_name" or "provider/model_name" format
            api_key: Explicit API key, or None to resolve from env at first use

        Returns:
            Tuple of (provider_instance, model_id) ready for acompletion()/_aembedding()
        """
        if ":" in model:
            provider_name, model_id = model.split(":", 1)
        elif "/" in model:
            provider_name, model_id = model.split("/", 1)
        else:
            provider_name, model_id = model, model

        provider_name = provider_name.lower()

        # Resolve the key that will actually be used so the cache key is stable
        resolved_key = api_key
        if resolved_key is None:
            env_var = PROVIDER_ENV_VAR_MAP.get(provider_name)
            if env_var:
                resolved_key = os.environ.get(env_var)

        cache_key = (provider_name, resolved_key)
        with self._lock:
            if cache_key not in self._providers:
                self._providers[cache_key] = AnyLLM.create(
                    provider_name,
                    api_key=resolved_key,
                )
            provider = self._providers[cache_key]

        return provider, model_id

    async def close_all(self) -> None:
        """Close all cached provider HTTP clients gracefully.

        Call this during application shutdown to ensure httpx connections are
        cleanly closed before the event loop exits, preventing the
        'RuntimeError: Event loop is closed' warning from the GC.
        """
        with self._lock:
            providers = list(self._providers.values())
            self._providers.clear()

        for provider in providers:
            client = getattr(provider, "client", None)
            if client is None:
                continue
            try:
                aclose = getattr(client, "aclose", None)
                if aclose is not None:
                    await aclose()
                else:
                    close = getattr(client, "close", None)
                    if close is not None:
                        if asyncio.iscoroutinefunction(close):
                            await close()
                        else:
                            close()
            except Exception:
                logger.debug("Error closing provider client during shutdown", exc_info=True)


_provider_cache = _ProviderCache()


async def close_providers() -> None:
    """Close all cached LLM provider HTTP clients.

    Call this during application shutdown before the event loop closes.
    GlueLLM registers this automatically when :func:`graceful_shutdown` is used,
    but you should call it manually if you manage the event loop directly::

        async def main():
            try:
                await my_app()
            finally:
                await close_providers()

        asyncio.run(main())
    """
    await _provider_cache.close_all()


# Register provider cleanup with the graceful shutdown system so that
# close_providers() is called automatically when graceful_shutdown() runs.
# This ensures httpx clients are closed before the event loop exits,
# preventing 'RuntimeError: Event loop is closed' from the GC.
register_shutdown_callback(close_providers)


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
    on_eval_record_hooks: "list | None" = None,
    hook_manager: "HookManager | None" = None,
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

    # Get agent from context (set by AgentExecutor)
    agent = _current_agent.get()

    # Extract agent information if available
    agent_name = None
    agent_description = None
    agent_model = None
    agent_system_prompt = None
    agent_tools = None
    agent_max_tool_iterations = None

    if agent:
        agent_name = agent.name
        agent_description = agent.description
        agent_model = agent.model
        agent_system_prompt = agent.system_prompt.content if agent.system_prompt else None
        agent_tools = [tool.__name__ for tool in agent.tools] if agent.tools else []
        agent_max_tool_iterations = agent.max_tool_iterations

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
                    raw_response_dict = _serialize_chat_completion_to_dict(result.raw_response)
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
                agent_name=agent_name,
                agent_description=agent_description,
                agent_model=agent_model,
                agent_system_prompt=agent_system_prompt,
                agent_tools=agent_tools,
                agent_max_tool_iterations=agent_max_tool_iterations,
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
                agent_name=agent_name,
                agent_description=agent_description,
                agent_model=agent_model,
                agent_system_prompt=agent_system_prompt,
                agent_tools=agent_tools,
                agent_max_tool_iterations=agent_max_tool_iterations,
            )

        # PRE_EVAL_RECORD hook: allow modifying user_message before writing (e.g. PII scrubbing)
        if on_eval_record_hooks and hook_manager:
            scrubbed = await hook_manager.execute_hooks(
                content=record.user_message,
                stage=HookStage.PRE_EVAL_RECORD,
                metadata={
                    "correlation_id": record.correlation_id,
                    "success": record.success,
                    "model": record.model,
                },
                hooks=on_eval_record_hooks,
            )
            record = record.model_copy(update={"user_message": scrubbed})

        # Record asynchronously (fire and forget)
        await eval_store.record(record)

    except Exception as e:
        # Log but don't raise - recording failures shouldn't break completions
        logger.error(f"Failed to record evaluation data: {e}", exc_info=True)


def _calculate_and_record_cost(
    model: str,
    tokens_used: dict[str, int] | None,
    correlation_id: str | None = None,
    track_costs: bool | None = None,
) -> float | None:
    """Calculate cost from token usage and record to session tracker.

    Args:
        model: Model identifier (e.g., "openai:gpt-5.4-2026-03-05")
        tokens_used: Token usage dictionary with 'prompt', 'completion', 'total'
        correlation_id: Optional correlation ID for logging
        track_costs: If False, skip recording and return None. If None, use settings.track_costs.

    Returns:
        Estimated cost in USD, or None if cost cannot be calculated or track_costs is False
    """
    if not tokens_used:
        return None

    effective_track = track_costs if track_costs is not None else settings.track_costs
    if not effective_track:
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


T = TypeVar("T", bound=BaseModel)


def _parse_structured_content(content: str, response_format: type[T]) -> T | None:
    """Parse accumulated stream/content into response_format. Returns None on failure."""
    if not content or not content.strip():
        return None
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return response_format(**data)
        return None
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.debug(f"Could not parse structured content: {e}")
        return None


def _build_message_from_stream(
    accumulated_content: str,
    tool_calls_accumulator: dict[int, dict[str, Any]],
) -> SimpleNamespace | None:
    """Build an assistant message from streamed content and/or accumulated tool_calls.

    Returns a message-like object with .content and .tool_calls (list of objects with
    .id, .function.name, .function.arguments) for use when appending to messages.
    Returns None if there are no tool_calls (caller uses content only).
    """
    if not tool_calls_accumulator:
        return None
    # Build tool_calls list in index order; each entry is compatible with attribute access.
    sorted_indices = sorted(tool_calls_accumulator.keys())
    tool_calls_list = []
    for idx in sorted_indices:
        acc = tool_calls_accumulator[idx]
        fid = acc.get("id") or ""
        fname = acc.get("function", {}).get("name") or ""
        fargs = acc.get("function", {}).get("arguments") or ""
        tool_calls_list.append(
            SimpleNamespace(
                id=fid,
                type="function",
                function=SimpleNamespace(name=fname, arguments=fargs),
            )
        )
    return SimpleNamespace(
        role="assistant",
        content=accumulated_content or None,
        tool_calls=tool_calls_list,
    )


def _streamed_assistant_message_to_dict(msg: SimpleNamespace | None) -> dict[str, Any] | None:
    """Convert assistant message from _build_message_from_stream to dict for API (messages list).

    any_llm validates messages as a list of dicts; the stream path produces SimpleNamespace
    objects, so we must convert before appending to messages for the next _safe_llm_call.
    """
    if msg is None:
        return None
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None),
        "tool_calls": [
            {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                },
            }
            for tc in tool_calls
        ],
    }


def _normalize_tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Convert a single tool call (dict or object) to the canonical dict shape for messages."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            name = fn.get("name", "")
            args = fn.get("arguments", "")
        else:
            name = getattr(fn, "name", "")
            args = getattr(fn, "arguments", "")
        return {
            "id": tc.get("id", ""),
            "type": tc.get("type", "function"),
            "function": {"name": name, "arguments": args},
        }
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", ""),
        "type": getattr(tc, "type", "function"),
        "function": {
            "name": getattr(fn, "name", "") if fn is not None else "",
            "arguments": getattr(fn, "arguments", "") if fn is not None else "",
        },
    }


def _tool_name_from_call(tool_call: Any) -> str:
    """Extract tool name from a tool call object; always return a string (for ProcessEvent)."""
    name = getattr(getattr(tool_call, "function", None), "name", None)
    return name if isinstance(name, str) else (str(name) if name is not None else "")


def _response_message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a provider response message to a dict for appending to messages.

    Providers may return a Pydantic model or an object (e.g. OpenAI ChatCompletionMessage).
    any_llm expects messages to be a list of dicts, so we normalize before appending.

    We build the dict from role/content/tool_calls only, and do not use model_dump()
    on the message. With structured output the message can have a `parsed` field
    holding the user's Pydantic model; the provider's schema often expects `parsed`
    to be None, so serializing it triggers Pydantic serializer warnings.
    """
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    return {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None),
        "tool_calls": [_normalize_tool_call_to_dict(tc) for tc in tool_calls_raw],
    }


def _serialize_chat_completion_to_dict(completion: Any) -> dict[str, Any]:
    """Serialize a ChatCompletion object to a plain dict, omitting the `parsed` field.

    The OpenAI SDK's `ParsedChatCompletionMessage` adds a `parsed` field typed as
    `Optional[ContentType]`. When Pydantic serializes this through the base schema
    (which declares `parsed: None`), it emits a `PydanticSerializationUnexpectedValue`
    warning because the runtime value is a user Pydantic model, not `None`.

    This helper extracts only the safe, schema-stable fields so that serialization
    (via `model_dump` / `model_dump_json`) is always warning-free.
    """
    usage = getattr(completion, "usage", None)
    return {
        "id": getattr(completion, "id", None),
        "model": getattr(completion, "model", None),
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
            for choice in (getattr(completion, "choices", None) or [])
        ],
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        if usage
        else None,
    }
def _condense_tool_round(
    messages: list[dict[str, Any]],
    *,
    aaak_tool_condensing: bool = False,
) -> None:
    """Replace the last tool-call round in messages with a single condensed summary.

    Finds the trailing sequence of ``role: "tool"`` messages preceded by a
    ``role: "assistant"`` message that contains ``tool_calls``. If the assistant
    message had no text content (a pure tool-call round), those N+1 messages are
    replaced with one ``role: "user"`` message whose content summarises each
    call and its result. User role keeps the conversation in a state the LLM
    naturally continues from.

    Rounds where the assistant produced visible text content alongside tool calls
    are left untouched, because that text carries meaning the model needs to see.

    Args:
        messages: Conversation messages (mutated in place).
        aaak_tool_condensing: If True, encode the round as an ``[AT]`` AAAK block
            and ensure the system message has the AAAK decoding preamble.
    """
    # Collect trailing tool-response messages
    tool_response_indices: list[int] = []
    idx = len(messages) - 1
    while idx >= 0 and messages[idx].get("role") == "tool":
        tool_response_indices.append(idx)
        idx -= 1
    tool_response_indices.reverse()

    if not tool_response_indices:
        return

    assistant_idx = idx
    if assistant_idx < 0:
        return

    assistant_msg = messages[assistant_idx]
    if assistant_msg.get("role") != "assistant":
        return

    tool_calls = assistant_msg.get("tool_calls") or []
    if not tool_calls:
        return

    # Only condense pure tool-call rounds (no visible assistant text)
    if assistant_msg.get("content"):
        return

    # Build a lookup from tool_call_id -> tool name for readable output
    id_to_name: dict[str, str] = {}
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        fn = tc.get("function") or {}
        name = fn.get("name", "unknown") if isinstance(fn, dict) else getattr(fn, "name", "unknown")
        id_to_name[tc_id] = name

    if aaak_tool_condensing:
        from gluellm.compression.aaak import AAAKCompressor

        if messages and messages[0].get("role") == "system":
            AAAKCompressor.ensure_preamble_in_system(messages[0])
        tool_messages = [messages[i] for i in tool_response_indices]
        condensed_content = AAAKCompressor.encode_tool_round(
            list(tool_calls),
            tool_messages,
            id_to_name,
        )
        del messages[assistant_idx:]
        messages.append({"role": "user", "content": condensed_content})
        return

    lines = ["[Tool Results]"]
    for tool_msg in (messages[i] for i in tool_response_indices):
        tc_id = tool_msg.get("tool_call_id", "")
        name = id_to_name.get(tc_id, tc_id)
        result = tool_msg.get("content", "")
        lines.append(f"- {name} -> {result}")

    condensed_content = "\n".join(lines)

    # Replace the N+1 messages with a single condensed user message (ends on user so LLM continues)
    del messages[assistant_idx:]
    messages.append({"role": "user", "content": condensed_content})


_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Produce a concise, factual summary of the "
    "conversation history provided. Preserve all key facts, decisions, and context "
    "that would be needed to continue the conversation coherently. Do not add "
    "opinions or filler — just the essential content, in plain prose."
)

_SUMMARIZE_USER_PREFIX = (
    "Summarize the following conversation history concisely so it can be used as "
    "context for continuing the conversation:\n\n"
)


async def _summarize_old_messages(
    messages: list[dict[str, Any]],
    keep_recent: int,
    model: str,
    *,
    use_aaak: bool = False,
    aaak_model: str | None = None,
    completion_extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compress older conversation messages into a single summary message.

    Preserves the system prompt (index 0) and the most recent ``keep_recent``
    messages verbatim. Everything in between is replaced with one ``role: "user"``
    summary message: prose ``[Conversation Summary]`` by default, or AAAK
    ``[AAAK CTX]`` blocks when ``use_aaak`` is True.

    Returns the original list unchanged when there is nothing old enough to
    summarize (i.e. the non-system messages number ``keep_recent`` or fewer).

    Args:
        messages: Full message list including the system prompt at index 0.
        keep_recent: Number of trailing messages to keep verbatim.
        model: Model identifier used for prose summarization and default AAAK model.
        use_aaak: If True, run AAAK compression on old messages instead of prose summary.
        aaak_model: Model for AAAK compression when ``use_aaak`` (defaults to ``model``).
        completion_extra: Optional kwargs merged into provider ``acompletion`` calls.

    Returns:
        A shortened message list: [system] + [summary] + recent_messages,
        or the original list if no summarization was needed (or AAAK returned empty).
    """
    if len(messages) < 2:
        return messages

    system_msg = messages[0]
    non_system = messages[1:]

    if len(non_system) <= keep_recent:
        return messages

    old_messages = non_system[: len(non_system) - keep_recent]
    recent_messages = non_system[len(non_system) - keep_recent :]

    if use_aaak:
        from gluellm.compression.aaak import AAAKCompressor

        sys_out: dict[str, Any] = {**system_msg}
        AAAKCompressor.ensure_preamble_in_system(sys_out)
        compress_model = aaak_model or model
        try:
            encoded = await AAAKCompressor.compress_messages(
                old_messages,
                model=compress_model,
                completion_extra=completion_extra,
            )
        except Exception:
            logger.warning("AAAK context compression failed; continuing with original messages.", exc_info=True)
            return messages

        if not (encoded or "").strip():
            return messages

        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": f"[AAAK CTX]\n{encoded}\n[/AAAK CTX]",
        }
        logger.debug(
            f"Context AAAK-compressed: {len(old_messages)} old messages → 1 summary message "
            f"({len(recent_messages)} recent messages kept verbatim)"
        )
        return [sys_out, summary_msg, *recent_messages]

    # Prose summarization (default)
    transcript_lines: list[str] = []
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        transcript_lines.append(f"{role.upper()}: {content}")

    transcript = "\n".join(transcript_lines)

    summarize_messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": _SUMMARIZE_USER_PREFIX + transcript},
    ]

    try:
        provider_cache = _ProviderCache()
        provider, model_id = provider_cache.get_provider(model, api_key=None)
        extra = dict(completion_extra or {})
        response = await provider.acompletion(model=model_id, messages=summarize_messages, **extra)
        summary_text = response.choices[0].message.content or ""
    except Exception:
        logger.warning("Context summarization failed; continuing with original messages.", exc_info=True)
        return messages

    summary_msg = {
        "role": "user",
        "content": f"[Conversation Summary]\n{summary_text}",
    }

    logger.debug(
        f"Context summarized: {len(old_messages)} old messages → 1 summary message "
        f"({len(recent_messages)} recent messages kept verbatim)"
    )

    return [system_msg, summary_msg, *recent_messages]


async def _consume_stream_with_tools(
    stream_iter: AsyncIterator[Any],
) -> AsyncIterator[tuple[bool, str, SimpleNamespace | None]]:
    """Consume a streaming LLM response that may include content and/or tool_calls.

    Yields (True, content_delta) for each content chunk, then (False, accumulated_content, assistant_message)
    at the end. assistant_message is non-None only if tool_calls were present (caller appends to messages
    and executes tools).
    """
    accumulated_content = ""
    tool_calls_accumulator: dict[int, dict[str, Any]] = {}

    async for chunk in stream_iter:
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue
        delta = getattr(choice, "delta", None)
        if not delta:
            continue

        # Content delta: forward immediately
        content = getattr(delta, "content", None)
        if content:
            accumulated_content += content
            yield (True, content, None)

        # Tool call deltas: accumulate by index (arguments may arrive in multiple chunks)
        tool_calls_delta = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls_delta:
            idx = getattr(tc, "index", None)
            if idx is None:
                continue
            if idx not in tool_calls_accumulator:
                tool_calls_accumulator[idx] = {"id": None, "function": {"name": "", "arguments": ""}}
            acc = tool_calls_accumulator[idx]
            if getattr(tc, "id", None):
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn:
                if getattr(fn, "name", None):
                    acc["function"]["name"] = fn.name
                if getattr(fn, "arguments", None):
                    acc["function"]["arguments"] = acc["function"]["arguments"] + fn.arguments

    message = _build_message_from_stream(accumulated_content, tool_calls_accumulator)
    yield (False, accumulated_content, message)


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


class APITimeoutError(APIConnectionError):
    """Raised when the API request times out."""

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


def _build_cause_chain(error: Exception) -> str:
    """Build a dotted type chain from an exception's cause hierarchy.

    Walks __cause__ (explicit `raise X from Y`) then falls back to
    __context__ (implicit chaining), stopping when suppress_context is set.
    Deduplicates consecutive identical type names that some libraries repeat.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: Exception | None = error
    while current is not None:
        if id(current) in seen:
            break
        seen.add(id(current))
        parts.append(type(current).__qualname__)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        else:
            break
    return " -> ".join(parts)


def classify_llm_error(error: Exception) -> Exception:
    """Classify an error from any_llm into our custom exception types.

    This function examines the error message and type to determine what kind
    of error occurred, making it easier to handle specific cases.
    """
    error_msg = str(error).lower()
    error_type = type(error).__name__

    # Invalid param involving max_tokens — check before generic TokenLimitError to avoid false matches:
    # 1. max_tokens required but missing (e.g. Anthropic)
    # 2. max_tokens not supported by model, use max_completion_tokens instead (e.g. OpenAI o-series)
    if "max_tokens" in error_msg and any(
        phrase in error_msg
        for phrase in [
            "required",
            "cannot be left empty",
            "must be",
            "must be provided",
            "not supported",
            "unsupported parameter",
            "max_completion_tokens",
        ]
    ):
        return InvalidRequestError(f"Invalid request: {error}")

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

    # Timeout errors
    if (
        any(k in error_msg for k in ["timeout", "timed out"])
        or error_type == "APITimeoutError"
    ):
        return APITimeoutError(f"API request timed out: {error}")

    # Connection/network errors
    if any(
        keyword in error_msg
        for keyword in [
            "connection",
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


def _trim_tool_docstrings(tools: list[Callable]) -> list[Callable]:
    """Return copies of tools with __doc__ trimmed to the first non-empty line.

    This reduces token overhead since any_llm sends the full docstring as the
    function description in the OpenAI tools schema.
    """
    import functools

    trimmed: list[Callable] = []
    for tool in tools:
        doc = (tool.__doc__ or "").strip()
        first_line = next((line.strip() for line in doc.split("\n") if line.strip()), "")
        if first_line == doc:
            trimmed.append(tool)
            continue
        wrapper = functools.wraps(tool)(lambda *a, _t=tool, **kw: _t(*a, **kw))
        wrapper.__doc__ = first_line
        trimmed.append(wrapper)
    return trimmed


async def _safe_llm_call(
    messages: list[dict],
    model: str,
    tools: list[Callable] | None = None,
    response_format: type[BaseModel] | None = None,
    stream: bool = False,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **model_kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletion]:
    """Make an LLM call with error classification and tracing.

    Wraps provider.acompletion() (obtained from the module-level provider cache)
    to catch and classify errors, and optionally trace the call with OpenTelemetry.
    Raises our custom exception types for better error handling.

    Args:
        messages: List of message dictionaries
        model: Model identifier
        tools: Optional list of tools
        response_format: Optional Pydantic model for structured output
        stream: Whether to stream the response
        request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
        api_key: Optional API key override (for key pool usage)
        max_tokens: Maximum number of tokens to generate. Required for Anthropic models.
        **model_kwargs: Additional parameters passed to provider.acompletion (e.g. temperature, top_p).

    Returns:
        ChatCompletion if stream=False, AsyncIterator[ChatCompletion] if stream=True

    Raises:
        asyncio.TimeoutError: If the request exceeds the timeout
    """
    correlation_id = get_correlation_id()
    request_timeout = request_timeout or settings.default_request_timeout
    request_timeout = min(request_timeout, settings.max_request_timeout)  # Enforce max timeout
    connect_timeout = connect_timeout or settings.default_connect_timeout
    connect_timeout = min(connect_timeout, settings.max_connect_timeout)  # Enforce max connect timeout
    _patch_any_llm_openai_converter()

    model_kwargs = dict(model_kwargs)
    for _k in _PROVIDER_ACOMPLETION_SKIP_KEYS:
        model_kwargs.pop(_k, None)

    # Inject httpx.Timeout into model_kwargs if caller hasn't set it (provider SDKs accept this)
    if "timeout" not in model_kwargs:
        model_kwargs["timeout"] = httpx.Timeout(
            connect=connect_timeout,
            read=request_timeout,
            write=request_timeout,
            pool=connect_timeout,
        )

    # Normalize provider-specific params (e.g. Anthropic max_tokens, OpenAI o-series max_completion_tokens)
    max_tokens, model_kwargs = normalize_model_params(model, max_tokens, model_kwargs)

    # Apply rate limiting before making the call
    provider = extract_provider_from_model(model)
    rate_limit_key = (
        f"global:{provider}" if not api_key else f"api_key:{api_key_hmac_fingerprint(api_key)}"
    )
    rate_limit_algorithm = rate_limit_config.algorithm if rate_limit_config else None
    await acquire_rate_limit(rate_limit_key, algorithm=rate_limit_algorithm)

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

    if tools:
        tools = _trim_tool_docstrings(tools)

    start_time = time.time()
    logger.debug(
        f"Making LLM call: model={model}, stream={stream}, has_tools={bool(tools)}, "
        f"response_format={response_format.__name__ if response_format else None}, "
        f"message_count={len(messages)}, request_timeout={request_timeout}s, "
        f"connect_timeout={connect_timeout}s, correlation_id={correlation_id}"
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

            # Resolve cached provider (reuses the same AsyncOpenAI/httpx client
            # across calls, preventing 'Event loop is closed' on GC cleanup).
            provider, model_id = _provider_cache.get_provider(model, api_key)

            # Make LLM call with timeout.
            # Use normalized model class if available, otherwise fall back to original Pydantic model.
            # The normalized class is a subclass, so response parsing still works correctly.
            response = await asyncio.wait_for(
                provider.acompletion(
                    model=model_id,
                    messages=messages,
                    tools=tools if tools else None,
                    response_format=normalized_response_format if normalized_response_format else response_format,
                    stream=stream,
                    max_tokens=max_tokens,
                    **model_kwargs,
                ),
                timeout=request_timeout,
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
                        f"completion={tokens_used['completion']}){cost_str}, correlation_id={correlation_id}"
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
                logger.debug(
                    f"LLM call streaming started: model={model}, latency={elapsed_time:.3f}s, "
                    f"correlation_id={correlation_id}"
                )

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
            f"LLM call timed out after {elapsed_time:.3f}s (request_timeout={request_timeout}s): model={model}, "
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
        cause_chain = _build_cause_chain(e)
        logger.error(
            f"LLM call failed after {elapsed_time:.3f}s: model={model}, error={classified_error}, "
            f"error_type={error_type}, cause_chain={cause_chain}, correlation_id={correlation_id}",
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


# ============================================================================
# OpenAI Responses API helpers
# ============================================================================
#
# The Responses API uses a different protocol than Chat Completions:
#   - ``input`` items (string OR list of typed dicts) instead of ``messages``
#   - ``instructions`` parameter instead of a system role message
#   - Tools are flat dicts (``{"type": "function", "name": ..., ...}``) rather
#     than the nested chat-completions shape (``{"type": "function", "function": {...}}``)
#   - Tool calls come back as ``function_call`` output items with ``call_id``
#     instead of ``message.tool_calls``; tool results are sent back as
#     ``function_call_output`` input items.
#   - Structured output is requested via ``text={"format": {"type": "json_schema",
#     "name": ..., "strict": True, "schema": ...}}`` rather than ``response_format``.
#   - Token usage uses ``input_tokens``/``output_tokens`` rather than
#     ``prompt_tokens``/``completion_tokens``.
#
# To preserve full feature parity with the chat-completions code path
# (summarisation, AAAK condensing, eval recording, hooks, validation retries,
# tool loop, ...) we keep the **internal** representation in chat-completions
# message/tool-call shape and translate at the API boundary in both directions.

# Chat-completions/GlueLLM kwargs that must not be forwarded to ``aresponses``:
# ``logprobs``/``top_logprobs`` are not supported, ``response_format`` is
# replaced by ``text``, and ``session_label`` is OpenAI-platform specific.
_PROVIDER_ARESPONSES_SKIP_KEYS = frozenset(
    {
        "max_tool_iterations",
        "execute_tools",
        "rate_limit_algorithm",
        "logprobs",
        "top_logprobs",
        "response_format",
        "session_label",
        "n",
        "stop",
    }
)


def _callable_to_responses_tool(fn: Callable) -> dict[str, Any]:
    """Convert a Python callable to the Responses API flat tool format.

    The Responses API expects ``{"type": "function", "name": ..., "description":
    ..., "parameters": ...}`` whereas chat completions uses the nested
    ``{"type": "function", "function": {...}}`` shape. We reuse any_llm's
    schema generation and then flatten.
    """
    from any_llm.tools import callable_to_tool

    chat_tool = callable_to_tool(fn)
    function = chat_tool.get("function") or {}
    return {
        "type": "function",
        "name": function.get("name", ""),
        "description": function.get("description"),
        "parameters": function.get("parameters") or {},
    }


def _responses_tools_format(
    tools: list[Callable | dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert mixed callables / pre-formatted dicts into Responses tool format.

    Pre-formatted dicts may already be flat (built-in tools like
    ``{"type": "web_search_preview"}``) or in chat-completions nested form;
    we flatten the nested form and pass everything else through.
    """
    if not tools:
        return None

    out: list[dict[str, Any]] = []
    for tool in tools:
        if callable(tool):
            out.append(_callable_to_responses_tool(tool))
            continue
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters") or {},
                }
            )
        else:
            out.append(tool)
    return out


def _messages_to_response_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate chat-completions messages into (instructions, input_items).

    The Responses API accepts a single optional ``instructions`` string and a
    list of typed input items. We collapse all ``role: system`` messages into
    one ``instructions`` string (joined by blank lines) and convert the rest:

    - ``user`` / ``assistant`` (with content) → message item with role+content
    - ``assistant`` with ``tool_calls`` → one ``function_call`` item per tool call
    - ``tool`` → ``function_call_output`` item using ``tool_call_id`` as ``call_id``

    Empty assistant messages (pure tool-call rounds) are not emitted as
    standalone messages; only their tool calls show up.
    """
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                instructions_parts.append(content)
            continue

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
            continue

        if role == "assistant":
            if isinstance(content, str) and content.strip():
                input_items.append({"role": "assistant", "content": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    name = fn.get("name", "")
                    arguments = fn.get("arguments", "")
                else:
                    name = getattr(fn, "name", "")
                    arguments = getattr(fn, "arguments", "")
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": name,
                        "arguments": arguments or "",
                    }
                )
            continue

        if role == "user":
            if isinstance(content, list):
                input_items.append({"role": "user", "content": content})
            else:
                input_items.append({"role": "user", "content": str(content or "")})
            continue

        # Unknown role: pass through as a user message containing JSON
        input_items.append({"role": "user", "content": json.dumps(msg)})

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, input_items


def _stringify_response_input_for_log(input_param: list[dict[str, Any]]) -> str:
    """Best-effort plain-text rendering of Responses input items for logs/eval.

    Concatenates any text content found across messages and tool outputs so
    callers (eval store, conversation tracking) get something meaningful when
    a user submits a structured ``ResponseInputParam`` rather than a string.
    """
    parts: list[str] = []
    for item in input_param:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        item_type = item.get("type")
        if item_type == "function_call_output":
            parts.append(f"[tool_result:{item.get('call_id', '')}] {item.get('output', '')}")
            continue
        if item_type == "function_call":
            parts.append(f"[tool_call:{item.get('name', '')}] {item.get('arguments', '')}")
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("input_text") or ""
                    if text:
                        parts.append(text)
                else:
                    parts.append(str(part))
        else:
            parts.append(json.dumps(item, default=str))
    return "\n".join(p for p in parts if p)


def _coerce_response_input_to_messages(input_param: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-effort conversion of Responses input items into chat-completions messages.

    Used so a list ``user_input`` can flow through the same internal
    ``messages`` plumbing (summarisation, conversation tracking, eval) as a
    string. Multimodal content lists are preserved verbatim on user/assistant
    messages so they round-trip back through ``_messages_to_response_input``.
    """
    out: list[dict[str, Any]] = []
    for item in input_param:
        if not isinstance(item, dict):
            out.append({"role": "user", "content": str(item)})
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            out.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            output = item.get("output", "")
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": output if isinstance(output, str) else json.dumps(output),
                }
            )
            continue

        # Message-shaped item (with role/content). 'developer' role maps to system.
        role = item.get("role")
        if role:
            mapped_role = "system" if role == "developer" else role
            out.append({"role": mapped_role, "content": item.get("content", "")})
            continue

        # Fallback: stringify unknown items as a user message
        out.append({"role": "user", "content": json.dumps(item)})

    return out


def _adapt_response_function_call(rfc: Any) -> SimpleNamespace:
    """Wrap a ``ResponseFunctionToolCall`` in the chat-completions tool_call shape.

    This lets ``_execute_tool_calls_round`` consume responses-API tool calls
    without modification. The synthetic ``id`` field stores the responses
    ``call_id`` so the resulting ``{"role": "tool", "tool_call_id": ...}``
    message round-trips back through ``_messages_to_response_input`` into a
    matching ``function_call_output`` item.
    """
    return SimpleNamespace(
        id=getattr(rfc, "call_id", "") or "",
        type="function",
        function=SimpleNamespace(
            name=getattr(rfc, "name", "") or "",
            arguments=getattr(rfc, "arguments", "") or "",
        ),
    )


def _extract_response_function_calls(resp: Any) -> list[SimpleNamespace]:
    """Return adapted function-call objects from a Responses API output.

    Skips built-in tool calls (web_search, code_interpreter, ...) - those run
    server-side and don't need local execution.
    """
    output = getattr(resp, "output", None) or []
    calls: list[SimpleNamespace] = []
    for item in output:
        item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if item_type != "function_call":
            continue
        if isinstance(item, dict):
            calls.append(
                _adapt_response_function_call(
                    SimpleNamespace(
                        call_id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        arguments=item.get("arguments", ""),
                    )
                )
            )
        else:
            calls.append(_adapt_response_function_call(item))
    return calls


def _extract_response_text(resp: Any) -> str:
    """Extract concatenated assistant text from a Responses API response.

    Prefers the convenience ``output_text`` field when present, then falls
    back to walking ``output`` items collecting ``output_text`` parts.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text

    output = getattr(resp, "output", None) or []
    parts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            if item.get("type") == "message":
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
            continue
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                parts.append(getattr(c, "text", "") or "")
            elif isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "".join(parts)


def _extract_response_token_usage(resp: Any) -> dict[str, int] | None:
    """Extract token usage from a Response, mapped onto the chat-completions schema.

    Returns ``{"prompt": int, "completion": int, "total": int}`` so existing
    cost tracking and ``_track_usage`` aggregation work without modification.
    """
    usage = getattr(resp, "usage", None)
    if not usage:
        return None

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", 0)
    else:
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", 0)

    return {
        "prompt": int(input_tokens) if isinstance(input_tokens, (int, float)) else 0,
        "completion": int(output_tokens) if isinstance(output_tokens, (int, float)) else 0,
        "total": int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
    }


def _extract_response_parsed(resp: Any, response_format: type[BaseModel]) -> Any:
    """Best-effort extraction of an already-parsed structured output from a Response.

    Returns either an instance of ``response_format``, a dict, or None.
    The OpenAI Responses SDK exposes parsed structured output through
    ``output_parsed`` when ``text.format`` is a json_schema, but the field
    is not always populated; callers fall back to JSON-decoding the text.
    """
    parsed = getattr(resp, "output_parsed", None)
    if parsed is not None:
        return parsed

    output = getattr(resp, "output", None) or []
    for item in output:
        contents = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        for c in contents or []:
            parsed_part = getattr(c, "parsed", None) if not isinstance(c, dict) else c.get("parsed")
            if parsed_part is not None:
                return parsed_part
    return None


def _serialize_response_to_dict(response: Any) -> dict[str, Any]:
    """Serialize a Responses API ``Response`` to a plain dict for eval recording.

    Mirrors ``_serialize_chat_completion_to_dict`` in spirit: we extract only
    the safe, schema-stable fields and avoid invoking ``model_dump`` on the
    full object so that user-defined parsed structures do not produce
    ``PydanticSerializationUnexpectedValue`` warnings.
    """
    usage = getattr(response, "usage", None)
    output_items: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if item_type == "function_call":
            if isinstance(item, dict):
                output_items.append(
                    {
                        "type": "function_call",
                        "call_id": item.get("call_id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    }
                )
            else:
                output_items.append(
                    {
                        "type": "function_call",
                        "call_id": getattr(item, "call_id", None),
                        "name": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", None),
                    }
                )
        elif item_type == "message":
            content_out: list[dict[str, Any]] = []
            contents = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
            for c in contents or []:
                ctype = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
                ctext = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                content_out.append({"type": ctype, "text": ctext})
            output_items.append(
                {
                    "type": "message",
                    "role": getattr(item, "role", None) if not isinstance(item, dict) else item.get("role"),
                    "content": content_out,
                }
            )
        else:
            output_items.append({"type": item_type})

    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "object": getattr(response, "object", None),
        "output_text": getattr(response, "output_text", None),
        "output": output_items,
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        },
    }


def _build_responses_text_format(response_format: type[BaseModel] | None) -> dict[str, Any] | None:
    """Construct the ``text.format`` payload for a json_schema-based structured output.

    Uses :func:`create_normalized_model` so the schema is OpenAI-strict-mode
    compatible (no ``additionalProperties: True``, no unsupported union forms,
    consistent ``$defs``).
    """
    if response_format is None:
        return None

    try:
        normalized = create_normalized_model(response_format)
        schema = normalized.model_json_schema()
        if schema.get("additionalProperties") is True:
            logger.warning(
                f"Schema normalization for {response_format.__name__} left "
                "additionalProperties=True; falling back to raw schema."
            )
            schema = response_format.model_json_schema()
    except Exception as e:
        logger.warning(
            f"Schema normalization failed for {response_format.__name__}: {e}; "
            "using raw schema",
            exc_info=True,
        )
        schema = response_format.model_json_schema()

    return {
        "format": {
            "type": "json_schema",
            "name": response_format.__name__,
            "strict": True,
            "schema": schema,
        }
    }


async def _safe_responses_call(
    input_data: str | list[dict[str, Any]],
    model: str,
    *,
    instructions: str | None = None,
    tools: list[Callable | dict[str, Any]] | None = None,
    response_format: type[BaseModel] | None = None,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **model_kwargs: Any,
) -> Any:
    """Call ``provider.aresponses`` with error classification and tracing.

    The Responses API analogue of :func:`_safe_llm_call`. Returns the raw
    provider response (``Response`` or ``ResponseResource``); structured
    output extraction and tool-call adaptation happen in higher layers.

    Notes on parameter translation:

    * ``max_tokens`` is forwarded as ``max_output_tokens`` (the Responses
      equivalent), unless the caller already passed ``max_output_tokens``.
    * ``response_format`` is converted to ``text={"format": json_schema}``
      via :func:`_build_responses_text_format`.
    * ``reasoning_effort`` (chat-completions style) is wrapped into
      ``reasoning={"effort": ...}`` because the Responses API expects a dict.
    * ``parallel_tool_calls`` is coerced to ``int`` to match the
      ``aresponses`` signature.
    """
    correlation_id = get_correlation_id()
    request_timeout = request_timeout or settings.default_request_timeout
    request_timeout = min(request_timeout, settings.max_request_timeout)
    connect_timeout = connect_timeout or settings.default_connect_timeout
    connect_timeout = min(connect_timeout, settings.max_connect_timeout)
    _patch_any_llm_openai_converter()

    # Filter and translate kwargs for the Responses surface
    kwargs = {
        k: v
        for k, v in model_kwargs.items()
        if k not in _PROVIDER_ARESPONSES_SKIP_KEYS
    }

    # max_tokens -> max_output_tokens (caller-provided max_output_tokens wins)
    max_output_tokens = kwargs.pop("max_output_tokens", None) or max_tokens
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    # reasoning_effort: str -> reasoning: {"effort": str}
    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort is not None and "reasoning" not in kwargs:
        if reasoning_effort != "auto" and reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}
    if isinstance(kwargs.get("reasoning"), str):
        kwargs["reasoning"] = {"effort": kwargs["reasoning"]}

    # parallel_tool_calls: aresponses signature is int | None
    if "parallel_tool_calls" in kwargs and kwargs["parallel_tool_calls"] is not None:
        kwargs["parallel_tool_calls"] = int(bool(kwargs["parallel_tool_calls"]))

    # Inject httpx.Timeout if caller hasn't set one
    if "timeout" not in kwargs:
        kwargs["timeout"] = httpx.Timeout(
            connect=connect_timeout,
            read=request_timeout,
            write=request_timeout,
            pool=connect_timeout,
        )

    # Tools: convert callables/nested dicts into Responses flat format
    prepared_tools = _responses_tools_format(tools) if tools else None

    # Structured output via text.format
    text_param = _build_responses_text_format(response_format)
    if text_param is not None:
        kwargs.setdefault("text", text_param)

    if instructions is not None:
        kwargs.setdefault("instructions", instructions)

    # Rate limiting
    provider_name = extract_provider_from_model(model)
    rate_limit_key = (
        f"global:{provider_name}"
        if not api_key
        else f"api_key:{api_key_hmac_fingerprint(api_key)}"
    )
    rate_limit_algorithm = rate_limit_config.algorithm if rate_limit_config else None
    await acquire_rate_limit(rate_limit_key, algorithm=rate_limit_algorithm)

    start_time = time.time()
    logger.debug(
        f"Making Responses API call: model={model}, has_tools={bool(prepared_tools)}, "
        f"has_text_format={text_param is not None}, "
        f"has_input_list={isinstance(input_data, list)}, "
        f"request_timeout={request_timeout}s, correlation_id={correlation_id}"
    )

    try:
        with trace_llm_call(
            model=model,
            messages=input_data if isinstance(input_data, list) else [{"role": "user", "content": input_data}],
            tools=prepared_tools,
            stream=False,
            response_format=response_format.__name__ if response_format else None,
            correlation_id=correlation_id,
        ) as span:
            if correlation_id:
                set_span_attributes(span, correlation_id=correlation_id)

            provider, model_id = _provider_cache.get_provider(model, api_key)

            response = await asyncio.wait_for(
                provider.aresponses(
                    model=model_id,
                    input_data=input_data,
                    tools=prepared_tools if prepared_tools else None,
                    **kwargs,
                ),
                timeout=request_timeout,
            )

            elapsed_time = time.time() - start_time
            tokens_used = _extract_response_token_usage(response)
            if tokens_used:
                model_name = model.split(":", 1)[1] if ":" in model else model
                call_cost = calculate_cost(
                    provider=provider_name,
                    model_name=model_name,
                    input_tokens=tokens_used.get("prompt", 0),
                    output_tokens=tokens_used.get("completion", 0),
                )
                record_token_usage(span, tokens_used, cost_usd=call_cost)
                cost_str = f", cost=${call_cost:.6f}" if call_cost else ""
                logger.info(
                    f"Responses API call completed: model={model}, latency={elapsed_time:.3f}s, "
                    f"tokens={tokens_used['total']} (input={tokens_used['prompt']}, "
                    f"output={tokens_used['completion']}){cost_str}, correlation_id={correlation_id}"
                )

            has_tool_calls = bool(_extract_response_function_calls(response))
            set_span_attributes(
                span,
                **{
                    "llm.response.has_tool_calls": has_tool_calls,
                    "llm.response.kind": "responses",
                },
            )

            log_llm_metrics(
                model=model,
                latency=elapsed_time,
                tokens_used=tokens_used,
                finish_reason=getattr(response, "status", None),
                has_tool_calls=has_tool_calls,
                error=False,
            )

            return response

    except TimeoutError:
        elapsed_time = time.time() - start_time
        logger.error(
            f"Responses API call timed out after {elapsed_time:.3f}s "
            f"(request_timeout={request_timeout}s): model={model}, correlation_id={correlation_id}",
            exc_info=True,
        )
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
        classified_error = classify_llm_error(e)
        error_type = type(classified_error).__name__
        cause_chain = _build_cause_chain(e)
        logger.error(
            f"Responses API call failed after {elapsed_time:.3f}s: model={model}, "
            f"error={classified_error}, error_type={error_type}, "
            f"cause_chain={cause_chain}, correlation_id={correlation_id}",
            exc_info=True,
        )
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


async def _responses_call_with_retry(
    input_data: str | list[dict[str, Any]],
    model: str,
    *,
    instructions: str | None = None,
    tools: list[Callable | dict[str, Any]] | None = None,
    response_format: type[BaseModel] | None = None,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    retry_config: RetryConfig | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **model_kwargs: Any,
) -> Any:
    """Make a Responses API call with retry semantics matching :func:`_llm_call_with_retry`."""
    cfg = retry_config or RetryConfig(
        retry_enabled=True,
        max_attempts=settings.retry_max_attempts,
        min_wait=float(settings.retry_min_wait),
        max_wait=float(settings.retry_max_wait),
        multiplier=float(settings.retry_multiplier),
    )
    effective_max = 1 if not cfg.retry_enabled else cfg.max_attempts
    kwargs = dict(model_kwargs)

    for attempt in range(effective_max):
        try:
            final_max_tokens = kwargs.pop("max_tokens", None) or max_tokens
            return await _safe_responses_call(
                input_data=input_data,
                model=model,
                instructions=instructions,
                tools=tools,
                response_format=response_format,
                request_timeout=request_timeout,
                connect_timeout=connect_timeout,
                api_key=api_key,
                max_tokens=final_max_tokens,
                rate_limit_config=rate_limit_config,
                **kwargs,
            )
        except Exception as e:
            if attempt + 1 >= effective_max:
                raise

            if cfg.callback is not None:
                if asyncio.iscoroutinefunction(cfg.callback):
                    should_retry, next_params = await cfg.callback(e, attempt + 1)
                else:
                    should_retry, next_params = cfg.callback(e, attempt + 1)
                if not should_retry:
                    raise
                if next_params:
                    kwargs.update(next_params)
            elif cfg.retry_on is not None:
                if not isinstance(e, tuple(cfg.retry_on)):
                    raise
            else:
                if not should_retry_error(e):
                    raise

            wait_time = min(cfg.max_wait, cfg.min_wait * (cfg.multiplier ** attempt))
            logger.warning(
                f"Responses API call failed (attempt {attempt + 1}/{effective_max}), "
                f"retrying in {wait_time:.2f}s: {e}"
            )
            await asyncio.sleep(wait_time)


async def _llm_call_with_retry(
    messages: list[dict],
    model: str,
    tools: list[Callable] | None = None,
    response_format: type[BaseModel] | None = None,
    stream: bool = False,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    retry_config: RetryConfig | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **model_kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletion]:
    """Make an LLM call with configurable retry on transient errors.

    When retry_config.retry_enabled is False, performs a single attempt.
    Otherwise retries according to retry_config.callback (or default behavior
    for RateLimitError/APIConnectionError when no callback is set).

    Args:
        messages: List of message dictionaries
        model: Model identifier
        tools: Optional list of tools
        response_format: Optional Pydantic model for structured output
        request_timeout: Request timeout in seconds
        connect_timeout: Connection timeout in seconds
        api_key: Optional API key override
        max_tokens: Maximum number of tokens to generate
        retry_config: Retry configuration including optional callback
        **model_kwargs: Additional params for acompletion (e.g. temperature)
    """
    cfg = retry_config or RetryConfig(
        retry_enabled=True,
        max_attempts=settings.retry_max_attempts,
        min_wait=float(settings.retry_min_wait),
        max_wait=float(settings.retry_max_wait),
        multiplier=float(settings.retry_multiplier),
    )
    effective_max = 1 if not cfg.retry_enabled else cfg.max_attempts
    kwargs = dict(model_kwargs)

    for attempt in range(effective_max):
        try:
            final_max_tokens = kwargs.pop("max_tokens", None) or max_tokens
            return await _safe_llm_call(
                messages=messages,
                model=model,
                tools=tools,
                response_format=response_format,
                stream=stream,
                request_timeout=request_timeout,
                connect_timeout=connect_timeout,
                api_key=api_key,
                max_tokens=final_max_tokens,
                rate_limit_config=rate_limit_config,
                **kwargs,
            )
        except Exception as e:
            if attempt + 1 >= effective_max:
                raise

            if cfg.callback is not None:
                if asyncio.iscoroutinefunction(cfg.callback):
                    should_retry, next_params = await cfg.callback(e, attempt + 1)
                else:
                    should_retry, next_params = cfg.callback(e, attempt + 1)
                if not should_retry:
                    raise
                if next_params:
                    kwargs.update(next_params)
            elif cfg.retry_on is not None:
                if not isinstance(e, tuple(cfg.retry_on)):
                    raise
            else:
                if not should_retry_error(e):
                    raise

            wait_time = min(
                cfg.max_wait,
                cfg.min_wait * (cfg.multiplier**attempt),
            )
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{effective_max}), retrying in {wait_time:.2f}s: {e}"
            )
            await asyncio.sleep(wait_time)


class ExecutionResult(BaseModel, Generic[T]):
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
        T | None,
        Field(
            description="Parsed structured output (Pydantic model instance) for structured completions",
            default=None,
        ),
    ]

    @field_serializer("raw_response")
    @staticmethod
    def serialize_raw_response(value: Any, _info: Any) -> dict[str, Any] | None:
        """Serialize raw_response to a plain dict.

        Dispatches on the response shape so both Chat Completions
        (``choices``) and Responses API (``output``) objects round-trip
        without triggering ``PydanticSerializationUnexpectedValue`` warnings
        from user-defined parsed structures.
        """
        if value is None:
            return None
        if hasattr(value, "choices"):
            return _serialize_chat_completion_to_dict(value)
        return _serialize_response_to_dict(value)

    def __len__(self) -> int:
        """Return the length of the final response or tool execution history."""
        if hasattr(self, "final_response") and self.final_response:
            return len(str(self.final_response))
        if hasattr(self, "tool_execution_history") and self.tool_execution_history:
            return len(self.tool_execution_history)
        return 0


class StreamingChunk(BaseModel, Generic[T]):
    """A chunk of streaming response."""

    content: Annotated[str, Field(description="The content chunk")]
    done: Annotated[bool, Field(description="Whether this is the final chunk")]
    tool_calls_made: Annotated[int, Field(description="Number of tool calls made so far", default=0)]
    structured_output: Annotated[
        T | None,
        Field(
            description="Parsed structured output (when response_format was set); set on the final chunk only",
            default=None,
        ),
    ] = None


def _coerce_pydantic_value(value: Any, annotation: Any) -> Any:
    """Recursively coerce *value* toward *annotation*.

    Handles the following annotation forms so that tool functions always
    receive proper Pydantic model instances rather than raw dicts:

    * ``MyModel`` — bare BaseModel subclass
    * ``list[MyModel]`` — list of models (each element coerced)
    * ``Union[A, B]`` / ``A | B`` — tries each concrete type in order
    * ``Optional[MyModel]`` — Union[MyModel, None], None arm skipped

    Coercion is best-effort: if every attempt raises the original value
    is returned unchanged.
    """
    import types as _types

    origin = get_origin(annotation)

    # bare BaseModel subclass
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(value, dict):
            return annotation.model_validate(value)
        return value

    # list[X]
    if origin is list:
        args = get_args(annotation)
        if args and isinstance(value, list):
            return [_coerce_pydantic_value(item, args[0]) for item in value]
        return value

    # Union[X, Y] / X | Y (Python 3.10+)
    is_union = origin is Union or (
        hasattr(_types, "UnionType") and isinstance(annotation, _types.UnionType)
    )
    if is_union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            try:
                return _coerce_pydantic_value(value, arg)
            except Exception:
                pass
        return value

    return value


class GlueLLM:
    """High-level API for LLM interactions with automatic tool execution."""

    def __init__(
        self,
        # -- Core --
        embedding_model: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        # -- Model generation --
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        top_logprobs: int | None = None,
        # -- Tools --
        condense_tool_messages: bool | None = None,
        max_tool_iterations: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_execution_order: ToolExecutionOrder | None = None,
        tool_mode: ToolMode = "standard",
        tool_route_model: str | None = None,
        tools: list[Callable] | None = None,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
        # -- Context management --
        aaak_compression_enabled: bool | None = None,
        aaak_compression_model: str | None = None,
        aaak_tool_condensing: bool | None = None,
        summarize_context: SummarizeContextConfig | bool | None = None,
        # -- Observability / behavior --
        eval_store: EvalStore | None = None,
        guardrails: GuardrailsConfig | None = None,
        hook_registry: HookRegistry | None = None,
        session_label: str | None = None,
    ):
        """Initialize GlueLLM client.

        Args:
            model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
            embedding_model: Embedding model identifier in format "provider/model_name" (defaults to settings.default_embedding_model)
            system_prompt: System prompt content (defaults to settings.default_system_prompt)
            tools: List of callable functions to use as tools
            max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)
            eval_store: Optional evaluation store for recording request/response data (defaults to global store if set)
            guardrails: Optional guardrails configuration for input/output validation
            max_tokens: Maximum completion tokens per LLM call (defaults to settings.default_max_tokens; overridable per complete() call).
            condense_tool_messages: If True, each completed tool-call round is condensed into a
                single assistant message summarising the calls and results, reducing context size
                across multi-iteration tool loops. Defaults to False.
            tool_mode: "standard" (all tools in system prompt) or "dynamic" (router tool discovers tools on demand)
            tool_execution_order: "sequential" (default) or "parallel" - how to run multiple tool calls in a round
            tool_route_model: Fast model used for tool routing when tool_mode="dynamic" (defaults to settings.tool_route_model)
            retry_config: Optional retry configuration. Set retry_config.callback to customise retry
                decisions and modify model params per attempt. Set retry_config.retry_enabled=False
                to disable retries entirely.
            rate_limit_config: Optional rate limit configuration. Set algorithm to override
                the default (from GLUELLM_RATE_LIMIT_ALGORITHM). E.g. RateLimitConfig(algorithm="leaking_bucket").
            model_kwargs: Optional dict of extra params for acompletion (e.g. temperature, top_p).
            reasoning_effort: For o3, o4-mini, Claude thinking models: "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"auto".
            logprobs: Include log probabilities in the response (eval/confidence scoring).
            top_logprobs: Number of top log probs to return when logprobs=True.
            session_label: Observability metadata for gateway/mzai traces.
            parallel_tool_calls: Allow parallel tool calls when multiple tools are requested.
            hook_registry: Optional hook registry for pre/post tool and iteration hooks. Merged
                with the global registry at call time so global hooks always apply.
            summarize_context: ``True``/``False``, or ``SummarizeContextConfig`` for automatic summarization
                of older messages when the context exceeds the threshold. Defaults from settings.
            aaak_compression_enabled: Use AAAK LLM compression when summarizing context (defaults to settings).
            aaak_compression_model: Model for AAAK compression when enabled (defaults to settings).
            aaak_tool_condensing: Encode condensed tool rounds as AAAK ``[AT]`` blocks (defaults to settings).
        """
        self.model = model or settings.default_model
        self.embedding_model = embedding_model or settings.default_embedding_model
        self.system_prompt = system_prompt or settings.default_system_prompt
        self.tools = tools or []
        self.max_tool_iterations = max_tool_iterations or settings.max_tool_iterations
        self._conversation = Conversation()
        self.eval_store = eval_store or get_global_eval_store()
        self.guardrails = guardrails
        self.max_tokens = max_tokens if max_tokens is not None else settings.default_max_tokens
        self.condense_tool_messages = (
            condense_tool_messages if condense_tool_messages is not None else settings.default_condense_tool_messages
        )
        self.tool_mode = tool_mode
        self.tool_execution_order = tool_execution_order if tool_execution_order is not None else settings.default_tool_execution_order
        self.tool_route_model = tool_route_model or settings.tool_route_model
        self.retry_config = retry_config
        self.rate_limit_config = rate_limit_config
        self.model_kwargs = model_kwargs or {}
        # Merge explicit LLM params (from config when set)
        effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else settings.default_reasoning_effort
        effective_parallel_tool_calls = parallel_tool_calls if parallel_tool_calls is not None else settings.default_parallel_tool_calls
        if effective_reasoning_effort is not None:
            self.model_kwargs["reasoning_effort"] = effective_reasoning_effort
        if logprobs is not None:
            self.model_kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            self.model_kwargs["top_logprobs"] = top_logprobs
        if session_label is not None:
            self.model_kwargs["session_label"] = session_label
        if effective_parallel_tool_calls is not None:
            self.model_kwargs["parallel_tool_calls"] = effective_parallel_tool_calls
        self._hook_registry = hook_registry or HookRegistry()
        self._hook_manager = HookManager()
        self._summarize_context_cfg = _normalize_summarize_context_init(summarize_context)
        self.aaak_compression_enabled = (
            aaak_compression_enabled
            if aaak_compression_enabled is not None
            else settings.aaak_compression_enabled
        )
        self.aaak_compression_model = (
            aaak_compression_model if aaak_compression_model is not None else settings.aaak_compression_model
        )
        self.aaak_tool_condensing = (
            aaak_tool_condensing if aaak_tool_condensing is not None else settings.aaak_tool_condensing
        )

    @property
    def summarize_context(self) -> bool:
        return self._summarize_context_cfg.enabled

    @property
    def summarize_context_threshold(self) -> int:
        t = self._summarize_context_cfg.threshold
        assert t is not None  # normalized at init
        return t

    @property
    def summarize_context_model(self) -> str | None:
        return self._summarize_context_cfg.model

    @property
    def summarize_context_keep_recent(self) -> int:
        k = self._summarize_context_cfg.keep_recent
        assert k is not None  # normalized at init
        return k

    def _get_merged_hook_registry(self) -> HookRegistry:
        """Return the global hook registry merged with this instance's registry."""
        return _get_global_registry().merge(self._hook_registry)

    async def _llm_call(
        self,
        merged_registry: HookRegistry,
        **kwargs: Any,
    ) -> Any:
        """Wrap _llm_call_with_retry, firing ON_LLM_RETRY hooks before each retry sleep.

        ON_LLM_RETRY content is the stringified exception; metadata includes
        attempt, max_attempts, wait_seconds, and exception_type.
        """
        retry_hooks = merged_registry.get_hooks(HookStage.ON_LLM_RETRY)

        if not retry_hooks:
            return await _llm_call_with_retry(**kwargs)

        # Inject a wrapper RetryConfig that fires hooks then delegates to the user's callback.
        original_retry_config: RetryConfig | None = kwargs.pop("retry_config", None)
        cfg = original_retry_config or RetryConfig(
            retry_enabled=True,
            max_attempts=settings.retry_max_attempts,
            min_wait=float(settings.retry_min_wait),
            max_wait=float(settings.retry_max_wait),
            multiplier=float(settings.retry_multiplier),
        )
        effective_max = 1 if not cfg.retry_enabled else cfg.max_attempts

        async def hooked_callback(exc: Exception, attempt: int) -> tuple[bool, dict]:
            wait_time = min(cfg.max_wait, cfg.min_wait * (cfg.multiplier ** (attempt - 1)))
            await self._hook_manager.execute_hooks(
                content=str(exc),
                stage=HookStage.ON_LLM_RETRY,
                metadata={
                    "attempt": attempt,
                    "max_attempts": effective_max,
                    "wait_seconds": wait_time,
                    "exception_type": type(exc).__name__,
                },
                hooks=retry_hooks,
            )
            if cfg.callback is not None:
                if asyncio.iscoroutinefunction(cfg.callback):
                    return await cfg.callback(exc, attempt)
                return cfg.callback(exc, attempt)
            return should_retry_error(exc), {}

        hooked_config = RetryConfig(
            retry_enabled=cfg.retry_enabled,
            max_attempts=cfg.max_attempts,
            min_wait=cfg.min_wait,
            max_wait=cfg.max_wait,
            multiplier=cfg.multiplier,
            callback=hooked_callback,
        )
        return await _llm_call_with_retry(retry_config=hooked_config, **kwargs)

    async def _run_guardrails(
        self,
        content: str,
        config: Any,
        direction: str,
        merged_registry: HookRegistry,
        attempt: int = 0,
    ) -> str:
        """Wrap run_input/output_guardrails with PRE_GUARDRAIL / POST_GUARDRAIL hooks.

        PRE_GUARDRAIL content is the text entering the chain (modifiable).
        POST_GUARDRAIL content is the text exiting the chain (modifiable).
        metadata includes direction ("input"|"output") and attempt number.
        """
        meta = {"direction": direction, "attempt": attempt}
        pre_hooks = merged_registry.get_hooks(HookStage.PRE_GUARDRAIL)
        post_hooks = merged_registry.get_hooks(HookStage.POST_GUARDRAIL)

        if pre_hooks:
            content = await self._hook_manager.execute_hooks(
                content=content,
                stage=HookStage.PRE_GUARDRAIL,
                metadata=meta,
                hooks=pre_hooks,
            )

        if direction == "input":
            result = run_input_guardrails(content, config)
        else:
            result = run_output_guardrails(content, config)

        if post_hooks:
            result = await self._hook_manager.execute_hooks(
                content=result,
                stage=HookStage.POST_GUARDRAIL,
                metadata=meta,
                hooks=post_hooks,
            )

        return result

    async def _route_tools(
        self,
        user_context: str,
        dynamic_tools: list[Callable],
        merged_registry: HookRegistry,
        model: str,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> list[Callable]:
        """Wrap resolve_tool_route with PRE_TOOL_ROUTE / POST_TOOL_ROUTE hooks.

        PRE_TOOL_ROUTE content is the user context string (observational).
        POST_TOOL_ROUTE content is a JSON array of matched tool names; returning
        a modified JSON array overrides which tools become active.
        """
        pre_hooks = merged_registry.get_hooks(HookStage.PRE_TOOL_ROUTE)
        post_hooks = merged_registry.get_hooks(HookStage.POST_TOOL_ROUTE)
        base_meta = {"route_query": user_context, "available_tool_count": len(dynamic_tools)}

        if pre_hooks:
            await self._hook_manager.execute_hooks(
                content=user_context,
                stage=HookStage.PRE_TOOL_ROUTE,
                metadata=base_meta,
                hooks=pre_hooks,
            )

        matched = await resolve_tool_route(
            user_context,
            dynamic_tools,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        fallback_to_all = len(matched) == len(dynamic_tools)

        if post_hooks:
            name_to_fn = {getattr(fn, "__name__", str(fn)): fn for fn in dynamic_tools}
            matched_names = [getattr(fn, "__name__", str(fn)) for fn in matched]
            result_json = await self._hook_manager.execute_hooks(
                content=json.dumps(matched_names),
                stage=HookStage.POST_TOOL_ROUTE,
                metadata={
                    **base_meta,
                    "matched_tool_names": matched_names,
                    "fallback_to_all": fallback_to_all,
                },
                hooks=post_hooks,
            )
            try:
                override_names = json.loads(result_json)
                if isinstance(override_names, list):
                    matched = [name_to_fn[n] for n in override_names if n in name_to_fn]
            except (json.JSONDecodeError, TypeError):
                logger.warning("POST_TOOL_ROUTE hook returned invalid JSON; using original matched tools")

        return matched

    async def complete(
        self,
        user_message: str,
        # -- Core --
        api_key: str | None = None,
        model: str | None = None,
        # -- Model generation --
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        top_logprobs: int | None = None,
        # -- Tools --
        condense_tool_messages: bool | None = None,
        execute_tools: bool = True,
        parallel_tool_calls: bool | None = None,
        tool_execution_order: ToolExecutionOrder | None = None,
        tool_mode: ToolMode | None = None,
        # -- Timeouts --
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
        retry_enabled: bool | None = None,
        # -- Context management --
        summarize_context: SummarizeContextConfig | bool | None = None,
        # -- Observability / behavior --
        correlation_id: str | None = None,
        enable_eval_recording: bool | None = None,
        guardrails: GuardrailsConfig | None = None,
        on_status: OnStatusCallback = None,
        session_label: str | None = None,
        sinks: list[Sink] | None = None,
        track_costs: bool | None = None,
        **model_kwargs: Any,
    ) -> ExecutionResult:
        """Complete a request with automatic tool execution loop.

        Args:
            user_message: The user's message/request
            api_key: Optional API key override (for key pool usage)
            execute_tools: Whether to automatically execute tools and loop
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
            guardrails: Optional guardrails configuration (overrides instance guardrails if provided)
            on_status: Optional callback for process status events (LLM call start/end, tool start/end, complete)
            max_tokens: Maximum number of tokens to generate. Overrides instance-level max_tokens if provided.
            condense_tool_messages: Override the instance-level condense_tool_messages setting for this call.
            tool_mode: Override the instance-level tool_mode for this call ("standard" or "dynamic").
            retry_enabled: If False, disables retries for this call (shorthand for retry_config.retry_enabled=False).
            retry_config: Per-call retry configuration override (includes optional callback).
            rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
            track_costs: If False, skip cost tracking for this call (defaults to settings.track_costs).
            enable_eval_recording: If False, skip eval recording for this call (defaults to using instance eval_store).
            reasoning_effort: For o3/o4-mini/Claude: "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"auto".
            logprobs: Include log probabilities (eval/confidence scoring).
            top_logprobs: Number of top log probs when logprobs=True.
            session_label: Observability metadata for gateway traces.
            parallel_tool_calls: Allow parallel tool calls.
            summarize_context: Override instance summarization (bool or ``SummarizeContextConfig``).
            **model_kwargs: Extra params for acompletion (e.g. temperature, top_p).

        Returns:
            ExecutionResult with final response and execution history

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
            asyncio.TimeoutError: If request exceeds timeout
            RuntimeError: If shutdown is in progress
            GuardrailBlockedError: If input guardrails block the request or output guardrails fail after max retries
        """
        # Check for shutdown
        if is_shutting_down():
            raise RuntimeError("Cannot process request: shutdown in progress")

        # Resolve guardrails config: per-call overrides instance
        effective_guardrails = guardrails if guardrails is not None else self.guardrails
        # Per-call max_tokens takes precedence over instance-level setting
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_condense = (
            condense_tool_messages if condense_tool_messages is not None else self.condense_tool_messages
        )
        effective_tool_mode: ToolMode = (
            tool_mode if tool_mode is not None else self.tool_mode
        )
        effective_tool_execution_order: ToolExecutionOrder = (
            tool_execution_order if tool_execution_order is not None else self.tool_execution_order
        )
        effective_retry_config = retry_config if retry_config is not None else self.retry_config
        if retry_enabled is not None and not retry_enabled:
            cfg = effective_retry_config
            effective_retry_config = (
                RetryConfig(retry_enabled=False)
                if cfg is None
                else RetryConfig(**{**cfg.model_dump(exclude={"callback"}), "retry_enabled": False, "callback": cfg.callback})
            )
        # Per-call rate_limit_config override; else client-level; else None (use settings)
        effective_rate_limit_config = rate_limit_config if rate_limit_config is not None else self.rate_limit_config
        effective_eval_store = None if enable_eval_recording is False else self.eval_store
        call_model_kwargs = dict(model_kwargs)
        override_max_tool_iterations = call_model_kwargs.pop("max_tool_iterations", None)
        effective_max_tool_iterations = (
            override_max_tool_iterations
            if override_max_tool_iterations is not None
            else self.max_tool_iterations
        )
        effective_model_kwargs = {**self.model_kwargs, **call_model_kwargs}
        if reasoning_effort is not None:
            effective_model_kwargs["reasoning_effort"] = reasoning_effort
        if logprobs is not None:
            effective_model_kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            effective_model_kwargs["top_logprobs"] = top_logprobs
        if session_label is not None:
            effective_model_kwargs["session_label"] = session_label
        if parallel_tool_calls is not None:
            effective_model_kwargs["parallel_tool_calls"] = parallel_tool_calls
        effective_summarize_cfg = _merge_summarize_context_for_call(self._summarize_context_cfg, summarize_context)
        effective_summarize_context_model = _summarize_model_for_call(effective_summarize_cfg, model or self.model)

        # Set correlation ID if provided
        if correlation_id:
            set_correlation_id(correlation_id)
        elif not get_correlation_id():
            # Auto-generate if not set
            set_correlation_id()

        correlation_id = get_correlation_id()
        logger.info(f"Starting completion request: correlation_id={correlation_id}, message_length={len(user_message)}")

        # Run input guardrails before processing
        _merged_guardrail_registry = self._get_merged_hook_registry()
        if effective_guardrails:
            try:
                user_message = await self._run_guardrails(
                    user_message, effective_guardrails, "input", _merged_guardrail_registry
                )
            except GuardrailBlockedError:
                raise  # Re-raise as-is (no retry for input)

        # Capture start time for evaluation recording
        start_time = time.time()
        # Resolve active_tools for dynamic vs standard mode
        all_tools = self.tools
        static_tools = [t for t in all_tools if is_static_tool(t)]
        dynamic_tools = [t for t in all_tools if not is_static_tool(t)]
        router_tool = None
        if effective_tool_mode == "dynamic" and all_tools:
            router_tool = build_router_tool(dynamic_tools)
            active_tools: list[Callable] = [router_tool] + static_tools
        else:
            active_tools = all_tools
        system_prompt_content = self._format_system_prompt(tools=active_tools)
        messages_snapshot: list[dict] = []
        result: ExecutionResult | None = None
        error: Exception | None = None

        # Use shutdown context to track in-flight requests for entire execution
        try:
            with ShutdownContext():
                # Add user message to conversation (after input guardrails)
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

                # Resolve iteration hooks once for the entire call
                _merged_registry = self._get_merged_hook_registry()
                _pre_iteration_hooks = _merged_registry.get_hooks(HookStage.PRE_ITERATION)
                _post_iteration_hooks = _merged_registry.get_hooks(HookStage.POST_ITERATION)

                # Tool execution loop
                logger.debug(
                    f"Starting tool execution loop: max_iterations={effective_max_tool_iterations}, "
                    f"tools_available={len(active_tools) if active_tools else 0}, tool_mode={effective_tool_mode}"
                )
                for iteration in range(effective_max_tool_iterations):
                    logger.debug(f"Tool execution iteration {iteration + 1}/{effective_max_tool_iterations}")
                    try:
                        # Context summarization: compress old messages when threshold is exceeded
                        _thr = effective_summarize_cfg.threshold
                        _kr = effective_summarize_cfg.keep_recent
                        if effective_summarize_cfg.enabled and _thr is not None and len(messages) > _thr:
                            messages = await _summarize_old_messages(
                                messages,
                                keep_recent=_kr if _kr is not None else settings.default_summarize_context_keep_recent,
                                model=effective_summarize_context_model,
                                use_aaak=self.aaak_compression_enabled,
                                aaak_model=self.aaak_compression_model,
                                completion_extra=effective_model_kwargs,
                            )

                        # PRE_ITERATION hook: content is the last user message in the conversation
                        if _pre_iteration_hooks:
                            last_user_msg = next(
                                (m.get("content", "") or "" for m in reversed(messages) if m.get("role") == "user"),
                                "",
                            )
                            if isinstance(last_user_msg, list):
                                last_user_msg = ""
                            await self._hook_manager.execute_hooks(
                                content=last_user_msg,
                                stage=HookStage.PRE_ITERATION,
                                metadata={"iteration": iteration + 1, "tool_count": len(active_tools)},
                                hooks=_pre_iteration_hooks,
                            )

                        await emit_status(
                            ProcessEvent(
                                kind="llm_call_start",
                                correlation_id=correlation_id,
                                timestamp=time.time(),
                                iteration=iteration + 1,
                                model=model or self.model,
                                message_count=len(messages),
                            ),
                            on_status,
                            sinks=sinks,
                        )
                        response = await self._llm_call(
                            _merged_guardrail_registry,
                            messages=messages,
                            model=model or self.model,
                            tools=active_tools if active_tools else None,
                            request_timeout=request_timeout,
                            connect_timeout=connect_timeout,
                            api_key=api_key,
                            max_tokens=effective_max_tokens,
                            retry_config=effective_retry_config,
                            rate_limit_config=effective_rate_limit_config,
                            **effective_model_kwargs,
                        )
                    except LLMError as e:
                        # Log the error and re-raise with context
                        cause_chain = _build_cause_chain(e)
                        logger.error(
                            f"LLM call failed on iteration {iteration + 1}/{effective_max_tool_iterations}: {e}, "
                            f"error_type={type(e).__name__}, cause_chain={cause_chain}"
                        )
                        # Add error context to the exception
                        error_msg = (
                            f"Failed during tool execution loop (iteration {iteration + 1}/{effective_max_tool_iterations})"
                        )
                        raise type(e)(f"{error_msg}: {e}") from e

                    # Validate response has choices
                    if not response.choices:
                        raise InvalidRequestError("Empty response from LLM provider")

                    tokens_used = _extract_token_usage(response)
                    _has_tool_calls = bool(execute_tools and active_tools and response.choices[0].message.tool_calls)
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_end",
                            correlation_id=correlation_id,
                            timestamp=time.time(),
                            iteration=iteration + 1,
                            model=model or self.model,
                            has_tool_calls=_has_tool_calls,
                            token_usage=tokens_used,
                        ),
                        on_status,
                        sinks=sinks,
                    )

                    # POST_ITERATION hook: content is the LLM response text before tool dispatch
                    if _post_iteration_hooks:
                        _response_text = response.choices[0].message.content or ""
                        await self._hook_manager.execute_hooks(
                            content=_response_text,
                            stage=HookStage.POST_ITERATION,
                            metadata={
                                "iteration": iteration + 1,
                                "has_tool_calls": _has_tool_calls,
                                "tool_count": len(active_tools),
                            },
                            hooks=_post_iteration_hooks,
                        )

                    # Check if model wants to call tools
                    if execute_tools and active_tools and response.choices[0].message.tool_calls:
                        tool_calls = response.choices[0].message.tool_calls
                        logger.info(f"Iteration {iteration + 1}: Model requested {len(tool_calls)} tool call(s)")

                        # Handle router call: resolve tools, do NOT append messages, continue
                        if router_tool is not None and is_router_call(tool_calls):
                            query = json.loads(tool_calls[0].function.arguments)["query"]
                            user_context = next(
                                (m.get("content", "") or "" for m in messages if m.get("role") == "user"),
                                query,
                            )
                            if isinstance(user_context, list):
                                user_context = query
                            matched = await self._route_tools(
                                user_context,
                                dynamic_tools,
                                _merged_guardrail_registry,
                                model=self.tool_route_model,
                                api_key=api_key,
                                timeout=request_timeout,
                            )
                            active_tools = matched + static_tools
                            logger.debug(
                                f"[{correlation_id}] Dynamic tool routing complete: "
                                f"matched={len(matched)}, static={len(static_tools)}, total_active={len(active_tools)}"
                            )
                            await emit_status(
                                ProcessEvent(
                                    kind="tool_route",
                                    correlation_id=correlation_id,
                                    timestamp=time.time(),
                                    route_query=query,
                                    matched_tools=[t.__name__ for t in matched],
                                ),
                                on_status,
                                sinks=sinks,
                            )
                            # Update system prompt to reflect matched tools (no longer router)
                            self._apply_system_prompt_after_tool_route(messages, active_tools)
                            continue

                        # Add assistant message with tool calls to history (dict for any_llm validation)
                        messages.append(_response_message_to_dict(response.choices[0].message))

                        # Execute tool calls (sequential or parallel per effective_tool_execution_order)
                        round_results = await self._execute_tool_calls_round(
                            tool_calls=tool_calls,
                            active_tools=active_tools,
                            parallel=(effective_tool_execution_order == "parallel"),
                            iteration=iteration + 1,
                            correlation_id=correlation_id,
                            on_status=on_status,
                            sinks=sinks,
                        )
                        tool_calls_made += len(round_results)
                        for r in round_results:
                            tool_execution_history.append(r["history"])
                            messages.append(r["message"])

                        if effective_condense:
                            _condense_tool_round(messages, aaak_tool_condensing=self.aaak_tool_condensing)

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

                    # Output guardrails with retry loop
                    if effective_guardrails:
                        max_retries = effective_guardrails.max_output_guardrail_retries
                        output_retry_count = 0
                        while output_retry_count <= max_retries:
                            try:
                                # Run output guardrails
                                final_content = await self._run_guardrails(
                                    final_content, effective_guardrails, "output",
                                    _merged_guardrail_registry, attempt=output_retry_count,
                                )
                                # Guardrails passed, break out of retry loop
                                break
                            except GuardrailRejectedError as e:
                                output_retry_count += 1
                                if output_retry_count > max_retries:
                                    # Max retries exceeded, raise blocked error
                                    logger.warning(f"Output guardrails failed after {max_retries} retries: {e.reason}")
                                    raise GuardrailBlockedError(
                                        f"Output guardrails failed after {max_retries} retries: {e.reason}",
                                        guardrail_name=e.guardrail_name,
                                    ) from e

                                # Add rejected response to conversation (for context)
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": final_content,
                                    }
                                )

                                # Append feedback message requesting revised response
                                feedback_message = (
                                    f"Your previous response was rejected: {e.reason}. "
                                    "Please provide a revised response that addresses this issue."
                                )
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": feedback_message,
                                    }
                                )
                                logger.info(
                                    f"Output guardrail rejected response (attempt {output_retry_count}/{max_retries}): "
                                    f"{e.reason}. Requesting revised response."
                                )

                                # Call LLM again for revised response (no tools, just text response)
                                try:
                                    response = await self._llm_call(
                                        _merged_guardrail_registry,
                                        messages=messages,
                                        model=model or self.model,
                                        tools=None,  # No tools on retry
                                        request_timeout=request_timeout,
                                        connect_timeout=connect_timeout,
                                        api_key=api_key,
                                        max_tokens=effective_max_tokens,
                                        retry_config=effective_retry_config,
                                        rate_limit_config=effective_rate_limit_config,
                                        **effective_model_kwargs,
                                    )
                                except LLMError as llm_error:
                                    # LLM call failed during retry, raise blocked error
                                    raise GuardrailBlockedError(
                                        f"Failed to get revised response after guardrail rejection: {llm_error}",
                                        guardrail_name=e.guardrail_name,
                                    ) from llm_error

                                if not response.choices:
                                    raise InvalidRequestError(
                                        "Empty response from LLM provider during guardrail retry"
                                    ) from None

                                # Get the revised response
                                final_content = response.choices[0].message.content or ""
                                logger.debug(
                                    f"Received revised response (length={len(final_content)}), "
                                    f"re-running output guardrails"
                                )
                                # Continue loop to re-check guardrails

                    # Add assistant response to conversation (after guardrails pass)
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
                        track_costs=track_costs,
                    )

                    await emit_status(
                        ProcessEvent(
                            kind="complete",
                            correlation_id=correlation_id,
                            timestamp=time.time(),
                            tool_calls_made=tool_calls_made,
                            response_length=len(final_content),
                        ),
                        on_status,
                        sinks=sinks,
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
                        eval_store=effective_eval_store,
                        user_message=user_message,
                        system_prompt=system_prompt_content,
                        model=model or self.model,
                        messages_snapshot=messages_snapshot,
                        start_time=start_time,
                        result=result,
                        tools_available=self.tools,
                        on_eval_record_hooks=_merged_guardrail_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                        hook_manager=self._hook_manager,
                    )

                    return result

                # Max iterations reached
                logger.warning(f"Max tool execution iterations ({effective_max_tool_iterations}) reached")
                final_content = "Maximum tool execution iterations reached."

                # Extract token usage if available
                tokens_used = _extract_token_usage(response)

                # Calculate cost and record to session tracker
                estimated_cost = _calculate_and_record_cost(
                    model=self.model,
                    tokens_used=tokens_used,
                    correlation_id=correlation_id,
                    track_costs=track_costs,
                )

                await emit_status(
                    ProcessEvent(
                        kind="complete",
                        correlation_id=correlation_id,
                        timestamp=time.time(),
                        tool_calls_made=tool_calls_made,
                        response_length=len(final_content),
                    ),
                    on_status,
                    sinks=sinks,
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
                    eval_store=effective_eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    result=result,
                    tools_available=self.tools,
                    on_eval_record_hooks=_merged_guardrail_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )

                return result
        except Exception as e:
            error = e
            raise
        finally:
            # Record evaluation data on error if not already recorded
            if error and not result:
                await _record_eval_data(
                    eval_store=effective_eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    error=error,
                    tools_available=self.tools,
                    on_eval_record_hooks=_merged_guardrail_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )
            # Clear correlation ID after request completes
            clear_correlation_id()

    async def structured_complete(
        self,
        user_message: str,
        response_format: type[T],
        # -- Core --
        api_key: str | None = None,
        model: str | None = None,
        tools: list[Callable] | None = None,
        # -- Model generation --
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        top_logprobs: int | None = None,
        # -- Tools --
        condense_tool_messages: bool | None = None,
        execute_tools: bool = True,
        parallel_tool_calls: bool | None = None,
        tool_execution_order: ToolExecutionOrder | None = None,
        tool_mode: ToolMode | None = None,
        # -- Timeouts --
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
        retry_enabled: bool | None = None,
        # -- Context management --
        summarize_context: SummarizeContextConfig | bool | None = None,
        # -- Observability / behavior --
        correlation_id: str | None = None,
        enable_eval_recording: bool | None = None,
        guardrails: GuardrailsConfig | None = None,
        max_validation_retries: int | None = None,
        on_status: OnStatusCallback = None,
        session_label: str | None = None,
        sinks: list[Sink] | None = None,
        track_costs: bool | None = None,
        **model_kwargs: Any,
    ) -> ExecutionResult[T]:
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
            request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
            api_key: Optional API key override (for key pool usage)
            guardrails: Optional guardrails configuration (overrides instance guardrails if provided)
            on_status: Optional callback for process status events
            max_tokens: Maximum number of tokens to generate. Overrides instance-level max_tokens if provided.
            condense_tool_messages: Override the instance-level condense_tool_messages setting for this call.
            tool_mode: Override the instance-level tool_mode for this call ("standard" or "dynamic").
            retry_enabled: If False, disables retries for this call (shorthand for retry_config.retry_enabled=False).
            retry_config: Per-call retry configuration override (includes optional callback).
            rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
            track_costs: If False, skip cost tracking for this call (defaults to settings.track_costs).
            enable_eval_recording: If False, skip eval recording for this call (defaults to using instance eval_store).
            reasoning_effort: For o3/o4-mini/Claude: "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"auto".
            logprobs: Include log probabilities (eval/confidence scoring).
            top_logprobs: Number of top log probs when logprobs=True.
            session_label: Observability metadata for gateway traces.
            parallel_tool_calls: Allow parallel tool calls.
            max_validation_retries: Max retries when Pydantic validation fails (default 3).
                On validation error, the error is fed back to the model for self-correction.
            summarize_context: Override instance summarization (bool or ``SummarizeContextConfig``).
            **model_kwargs: Extra params for acompletion (e.g. temperature, top_p).

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
            GuardrailBlockedError: If input guardrails block the request or output guardrails fail after max retries
        """
        # Check for shutdown
        if is_shutting_down():
            raise RuntimeError("Cannot process request: shutdown in progress")

        # Resolve guardrails config: per-call overrides instance
        effective_guardrails = guardrails if guardrails is not None else self.guardrails
        # Per-call max_tokens takes precedence over instance-level setting
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_condense = (
            condense_tool_messages if condense_tool_messages is not None else self.condense_tool_messages
        )
        effective_tool_mode: ToolMode = (
            tool_mode if tool_mode is not None else self.tool_mode
        )
        effective_tool_execution_order: ToolExecutionOrder = (
            tool_execution_order if tool_execution_order is not None else self.tool_execution_order
        )
        effective_retry_config = retry_config if retry_config is not None else self.retry_config
        if retry_enabled is not None and not retry_enabled:
            cfg = effective_retry_config
            effective_retry_config = (
                RetryConfig(retry_enabled=False)
                if cfg is None
                else RetryConfig(**{**cfg.model_dump(exclude={"callback"}), "retry_enabled": False, "callback": cfg.callback})
            )
        effective_rate_limit_config = rate_limit_config if rate_limit_config is not None else self.rate_limit_config
        effective_eval_store = None if enable_eval_recording is False else self.eval_store
        call_model_kwargs = dict(model_kwargs)
        override_max_tool_iterations = call_model_kwargs.pop("max_tool_iterations", None)
        effective_max_tool_iterations = (
            override_max_tool_iterations
            if override_max_tool_iterations is not None
            else self.max_tool_iterations
        )
        effective_model_kwargs = {**self.model_kwargs, **call_model_kwargs}
        if reasoning_effort is not None:
            effective_model_kwargs["reasoning_effort"] = reasoning_effort
        if logprobs is not None:
            effective_model_kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            effective_model_kwargs["top_logprobs"] = top_logprobs
        if session_label is not None:
            effective_model_kwargs["session_label"] = session_label
        if parallel_tool_calls is not None:
            effective_model_kwargs["parallel_tool_calls"] = parallel_tool_calls
        effective_summarize_cfg = _merge_summarize_context_for_call(self._summarize_context_cfg, summarize_context)
        effective_summarize_context_model = _summarize_model_for_call(effective_summarize_cfg, model or self.model)

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

        # Run input guardrails before processing
        _sc_merged_registry = self._get_merged_hook_registry()
        if effective_guardrails:
            try:
                user_message = await self._run_guardrails(
                    user_message, effective_guardrails, "input", _sc_merged_registry
                )
            except GuardrailBlockedError:
                raise  # Re-raise as-is (no retry for input)

        # Capture start time for evaluation recording
        start_time = time.time()
        # Determine which tools to use: parameter overrides instance tools
        tools_to_use = tools if tools is not None else self.tools
        # Resolve active_tools for dynamic vs standard mode
        all_tools = tools_to_use
        static_tools = [t for t in all_tools if is_static_tool(t)]
        dynamic_tools = [t for t in all_tools if not is_static_tool(t)]
        router_tool = None
        if effective_tool_mode == "dynamic" and all_tools:
            router_tool = build_router_tool(dynamic_tools)
            active_tools: list[Callable] = [router_tool] + static_tools
        else:
            active_tools = all_tools
        system_prompt_content = self._format_system_prompt(tools=active_tools)
        messages_snapshot: list[dict] = []
        result: ExecutionResult | None = None
        error: Exception | None = None

        # Use shutdown context to track in-flight requests for entire execution
        try:
            with ShutdownContext():
                # Add user message to conversation (after input guardrails)
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
                        track_costs=track_costs,
                    )
                    if iteration_cost is not None:
                        total_cost += iteration_cost

                # PHASE 1: Tool execution loop (if tools are provided and execute_tools is True)
                if active_tools and execute_tools:
                    logger.debug(
                        f"Starting tool execution phase: max_iterations={effective_max_tool_iterations}, "
                        f"tools_available={len(active_tools)}, tool_mode={effective_tool_mode}"
                    )
                    _sc_pre_iter_hooks = _sc_merged_registry.get_hooks(HookStage.PRE_ITERATION)
                    _sc_post_iter_hooks = _sc_merged_registry.get_hooks(HookStage.POST_ITERATION)
                    for iteration in range(effective_max_tool_iterations):
                        logger.debug(f"Tool execution iteration {iteration + 1}/{effective_max_tool_iterations}")

                        try:
                            # Context summarization: compress old messages when threshold is exceeded
                            _thr_sc = effective_summarize_cfg.threshold
                            _kr_sc = effective_summarize_cfg.keep_recent
                            if effective_summarize_cfg.enabled and _thr_sc is not None and len(messages) > _thr_sc:
                                messages = await _summarize_old_messages(
                                    messages,
                                    keep_recent=_kr_sc
                                    if _kr_sc is not None
                                    else settings.default_summarize_context_keep_recent,
                                    model=effective_summarize_context_model,
                                    use_aaak=self.aaak_compression_enabled,
                                    aaak_model=self.aaak_compression_model,
                                    completion_extra=effective_model_kwargs,
                                )

                            if _sc_pre_iter_hooks:
                                _last_user_msg = next(
                                    (m.get("content", "") or "" for m in reversed(messages) if m.get("role") == "user"),
                                    "",
                                )
                                if isinstance(_last_user_msg, list):
                                    _last_user_msg = ""
                                await self._hook_manager.execute_hooks(
                                    content=_last_user_msg,
                                    stage=HookStage.PRE_ITERATION,
                                    metadata={"iteration": iteration + 1, "tool_count": len(active_tools)},
                                    hooks=_sc_pre_iter_hooks,
                                )

                            await emit_status(
                                ProcessEvent(
                                    kind="llm_call_start",
                                    correlation_id=correlation_id,
                                    timestamp=time.time(),
                                    iteration=iteration + 1,
                                    model=model or self.model,
                                    message_count=len(messages),
                                ),
                                on_status,
                                sinks=sinks,
                            )
                            response = await self._llm_call(
                                _sc_merged_registry,
                                messages=messages,
                                model=model or self.model,
                                tools=active_tools,
                                # No response_format during tool phase
                                request_timeout=request_timeout,
                                connect_timeout=connect_timeout,
                                api_key=api_key,
                                max_tokens=effective_max_tokens,
                                retry_config=effective_retry_config,
                                rate_limit_config=effective_rate_limit_config,
                                **effective_model_kwargs,
                            )
                            _sc_has_tool_calls = bool(
                                response.choices and response.choices[0].message.tool_calls
                            )
                            await emit_status(
                                ProcessEvent(
                                    kind="llm_call_end",
                                    correlation_id=correlation_id,
                                    timestamp=time.time(),
                                    iteration=iteration + 1,
                                    model=model or self.model,
                                    has_tool_calls=_sc_has_tool_calls,
                                    token_usage=_extract_token_usage(response),
                                ),
                                on_status,
                                sinks=sinks,
                            )
                        except LLMError as e:
                            cause_chain = _build_cause_chain(e)
                            logger.error(
                                f"LLM call failed on iteration {iteration + 1}: {e}, "
                                f"error_type={type(e).__name__}, cause_chain={cause_chain}"
                            )
                            raise type(e)(f"Failed during tool execution (iteration {iteration + 1}): {e}") from e

                        _track_usage(response)

                        if not response.choices:
                            raise InvalidRequestError("Empty response from LLM provider")

                        if _sc_post_iter_hooks:
                            await self._hook_manager.execute_hooks(
                                content=response.choices[0].message.content or "",
                                stage=HookStage.POST_ITERATION,
                                metadata={
                                    "iteration": iteration + 1,
                                    "has_tool_calls": _sc_has_tool_calls,
                                    "tool_count": len(active_tools),
                                },
                                hooks=_sc_post_iter_hooks,
                            )

                        # Check if model wants to call tools
                        if response.choices[0].message.tool_calls:
                            tool_calls = response.choices[0].message.tool_calls
                            logger.info(f"Iteration {iteration + 1}: Model requested {len(tool_calls)} tool call(s)")

                            # Handle router call: resolve tools, do NOT append messages, continue
                            if router_tool is not None and is_router_call(tool_calls):
                                query = json.loads(tool_calls[0].function.arguments)["query"]
                                user_context = next(
                                    (m.get("content", "") or "" for m in messages if m.get("role") == "user"),
                                    query,
                                )
                                if isinstance(user_context, list):
                                    user_context = query
                                matched = await self._route_tools(
                                    user_context,
                                    dynamic_tools,
                                    _sc_merged_registry,
                                    model=self.tool_route_model,
                                    api_key=api_key,
                                    timeout=request_timeout,
                                )
                                active_tools = matched + static_tools
                                logger.debug(
                                    f"[{correlation_id}] Dynamic tool routing complete: "
                                    f"matched={len(matched)}, static={len(static_tools)}, total_active={len(active_tools)}"
                                )
                                await emit_status(
                                    ProcessEvent(
                                        kind="tool_route",
                                        correlation_id=correlation_id,
                                        timestamp=time.time(),
                                        route_query=query,
                                        matched_tools=[t.__name__ for t in matched],
                                    ),
                                    on_status,
                                    sinks=sinks,
                                )
                                # Update system prompt to reflect matched tools (no longer router)
                                self._apply_system_prompt_after_tool_route(messages, active_tools)
                                continue

                            # Add assistant message with tool calls to history (dict for any_llm validation)
                            messages.append(_response_message_to_dict(response.choices[0].message))

                            # Execute tool calls (sequential or parallel per effective_tool_execution_order)
                            round_results = await self._execute_tool_calls_round(
                                tool_calls=tool_calls,
                                active_tools=active_tools,
                                parallel=(effective_tool_execution_order == "parallel"),
                                iteration=iteration + 1,
                                correlation_id=correlation_id,
                                on_status=on_status,
                                sinks=sinks,
                            )
                            tool_calls_made += len(round_results)
                            for r in round_results:
                                tool_execution_history.append(r["history"])
                                messages.append(r["message"])

                            if effective_condense:
                                _condense_tool_round(messages, aaak_tool_condensing=self.aaak_tool_condensing)
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
                        logger.debug(f"Reached max tool iterations ({effective_max_tool_iterations})")

                # PHASE 2: Final structured output call
                logger.debug(f"Requesting structured output: response_format={response_format.__name__}")
                try:
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_start",
                            correlation_id=correlation_id,
                            timestamp=time.time(),
                            iteration=None,
                            model=model or self.model,
                            message_count=len(messages),
                        ),
                        on_status,
                        sinks=sinks,
                    )
                    response = await self._llm_call(
                        _sc_merged_registry,
                        messages=messages,
                        model=model or self.model,
                        response_format=response_format,
                        # No tools during structured output phase
                        request_timeout=request_timeout,
                        connect_timeout=connect_timeout,
                        api_key=api_key,
                        max_tokens=effective_max_tokens,
                        retry_config=effective_retry_config,
                        rate_limit_config=effective_rate_limit_config,
                        **effective_model_kwargs,
                    )
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_end",
                            correlation_id=correlation_id,
                            timestamp=time.time(),
                            model=model or self.model,
                            has_tool_calls=False,
                            token_usage=_extract_token_usage(response),
                        ),
                        on_status,
                        sinks=sinks,
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

                # Output guardrails with retry loop
                if effective_guardrails and content:
                    max_retries = effective_guardrails.max_output_guardrail_retries
                    output_retry_count = 0
                    while output_retry_count <= max_retries:
                        try:
                            # Run output guardrails
                            content = await self._run_guardrails(
                                content, effective_guardrails, "output",
                                _sc_merged_registry, attempt=output_retry_count,
                            )
                            # Guardrails passed, break out of retry loop
                            break
                        except GuardrailRejectedError as e:
                            output_retry_count += 1
                            if output_retry_count > max_retries:
                                # Max retries exceeded, raise blocked error
                                logger.warning(f"Output guardrails failed after {max_retries} retries: {e.reason}")
                                raise GuardrailBlockedError(
                                    f"Output guardrails failed after {max_retries} retries: {e.reason}",
                                    guardrail_name=e.guardrail_name,
                                ) from e

                            # Add rejected response to conversation (for context)
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": content,
                                }
                            )

                            # Append feedback message requesting revised response
                            feedback_message = (
                                f"Your previous response was rejected: {e.reason}. "
                                "Please provide a revised structured response that addresses this issue."
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": feedback_message,
                                }
                            )
                            logger.info(
                                f"Output guardrail rejected structured response "
                                f"(attempt {output_retry_count}/{max_retries}): {e.reason}. "
                                "Requesting revised response."
                            )

                            # Call LLM again for revised structured response
                            try:
                                response = await self._llm_call(
                                    _sc_merged_registry,
                                    messages=messages,
                                    model=model or self.model,
                                    response_format=response_format,  # Still request structured output
                                    request_timeout=request_timeout,
                                    connect_timeout=connect_timeout,
                                    api_key=api_key,
                                    max_tokens=effective_max_tokens,
                                    retry_config=effective_retry_config,
                                    rate_limit_config=effective_rate_limit_config,
                                    **effective_model_kwargs,
                                )
                            except LLMError as llm_error:
                                # LLM call failed during retry, raise blocked error
                                raise GuardrailBlockedError(
                                    f"Failed to get revised structured response after guardrail rejection: {llm_error}",
                                    guardrail_name=e.guardrail_name,
                                ) from llm_error

                            _track_usage(response)

                            if not response.choices:
                                raise InvalidRequestError(
                                    "Empty response from LLM provider during guardrail retry"
                                ) from None

                            # Get the revised response
                            parsed = getattr(response.choices[0].message, "parsed", None)
                            content = response.choices[0].message.content
                            logger.debug(
                                f"Received revised structured response (length={len(content) if content else 0}), "
                                f"re-running output guardrails"
                            )
                            # Continue loop to re-check guardrails

                # Add assistant response to conversation (after guardrails pass)
                if content:
                    self._conversation.add_message(Role.ASSISTANT, content)

                # Parse the structured output with validation-aware retry
                max_val_retries = max_validation_retries if max_validation_retries is not None else 3
                validation_attempt = 0
                structured_output = None
                while True:
                    try:
                        if isinstance(parsed, response_format):
                            logger.debug(f"Using parsed Pydantic instance: {response_format.__name__}")
                            structured_output = parsed
                        elif isinstance(parsed, dict):
                            logger.debug(f"Instantiating {response_format.__name__} from dict")
                            structured_output = response_format(**parsed)
                        elif content:
                            data = json.loads(content)
                            logger.debug(f"Parsed JSON from content, instantiating {response_format.__name__}")
                            structured_output = response_format(**data)
                        else:
                            logger.warning(f"Using parsed response as-is (type: {type(parsed)})")
                            structured_output = parsed
                        break
                    except (ValidationError, json.JSONDecodeError, TypeError) as e:
                        validation_attempt += 1
                        if validation_attempt > max_val_retries:
                            raise
                        messages.append({"role": "assistant", "content": content or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your response failed validation: {e}\n"
                                    "Please return a corrected JSON response."
                                ),
                            },
                        )
                        logger.info(
                            f"Validation retry {validation_attempt}/{max_val_retries}: {e}"
                        )
                        _val_retry_hooks = _sc_merged_registry.get_hooks(HookStage.ON_VALIDATION_RETRY)
                        if _val_retry_hooks:
                            await self._hook_manager.execute_hooks(
                                content=content or "",
                                stage=HookStage.ON_VALIDATION_RETRY,
                                metadata={
                                    "validation_attempt": validation_attempt,
                                    "max_validation_retries": max_val_retries,
                                    "error": str(e),
                                    "response_format_name": response_format.__name__,
                                },
                                hooks=_val_retry_hooks,
                            )
                        response = await self._llm_call(
                            _sc_merged_registry,
                            messages=messages,
                            model=model or self.model,
                            response_format=response_format,
                            request_timeout=request_timeout,
                            connect_timeout=connect_timeout,
                            api_key=api_key,
                            max_tokens=effective_max_tokens,
                            retry_config=effective_retry_config,
                            rate_limit_config=effective_rate_limit_config,
                            **effective_model_kwargs,
                        )
                        _track_usage(response)
                        if not response.choices:
                            raise InvalidRequestError(
                                "Empty response from LLM provider during validation retry"
                            ) from None
                        parsed = getattr(response.choices[0].message, "parsed", None)
                        content = response.choices[0].message.content

                logger.info(f"Structured completion finished: tool_calls={tool_calls_made}, cost=${total_cost:.6f}")

                await emit_status(
                    ProcessEvent(
                        kind="complete",
                        correlation_id=correlation_id,
                        timestamp=time.time(),
                        tool_calls_made=tool_calls_made,
                        response_length=len(content) if content else 0,
                    ),
                    on_status,
                    sinks=sinks,
                )

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
                    eval_store=effective_eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    result=result,
                    tools_available=tools_to_use,
                    on_eval_record_hooks=_sc_merged_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )

                return result
        except Exception as e:
            error = e
            raise
        finally:
            # Record evaluation data on error if not already recorded
            if error and not result:
                await _record_eval_data(
                    eval_store=effective_eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    error=error,
                    tools_available=tools_to_use,
                    on_eval_record_hooks=_sc_merged_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )
            # Clear correlation ID after request completes
            clear_correlation_id()

    async def _llm_responses_call(
        self,
        merged_registry: HookRegistry,
        **kwargs: Any,
    ) -> Any:
        """Wrap :func:`_responses_call_with_retry`, firing ON_LLM_RETRY hooks.

        Mirrors :meth:`_llm_call` but for the OpenAI Responses API.
        """
        retry_hooks = merged_registry.get_hooks(HookStage.ON_LLM_RETRY)

        if not retry_hooks:
            return await _responses_call_with_retry(**kwargs)

        original_retry_config: RetryConfig | None = kwargs.pop("retry_config", None)
        cfg = original_retry_config or RetryConfig(
            retry_enabled=True,
            max_attempts=settings.retry_max_attempts,
            min_wait=float(settings.retry_min_wait),
            max_wait=float(settings.retry_max_wait),
            multiplier=float(settings.retry_multiplier),
        )
        effective_max = 1 if not cfg.retry_enabled else cfg.max_attempts

        async def hooked_callback(exc: Exception, attempt: int) -> tuple[bool, dict]:
            wait_time = min(cfg.max_wait, cfg.min_wait * (cfg.multiplier ** (attempt - 1)))
            await self._hook_manager.execute_hooks(
                content=str(exc),
                stage=HookStage.ON_LLM_RETRY,
                metadata={
                    "attempt": attempt,
                    "max_attempts": effective_max,
                    "wait_seconds": wait_time,
                    "exception_type": type(exc).__name__,
                    "endpoint": "responses",
                },
                hooks=retry_hooks,
            )
            if cfg.callback is not None:
                if asyncio.iscoroutinefunction(cfg.callback):
                    return await cfg.callback(exc, attempt)
                return cfg.callback(exc, attempt)
            return should_retry_error(exc), {}

        hooked_config = RetryConfig(
            retry_enabled=cfg.retry_enabled,
            max_attempts=cfg.max_attempts,
            min_wait=cfg.min_wait,
            max_wait=cfg.max_wait,
            multiplier=cfg.multiplier,
            callback=hooked_callback,
        )
        return await _responses_call_with_retry(retry_config=hooked_config, **kwargs)

    async def structured_response(
        self,
        user_input: str | list[dict[str, Any]],
        response_format: type[T],
        # -- Core --
        api_key: str | None = None,
        model: str | None = None,
        tools: list[Callable] | None = None,
        # -- Model generation --
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        # -- Tools --
        condense_tool_messages: bool | None = None,
        execute_tools: bool = True,
        parallel_tool_calls: bool | None = None,
        tool_execution_order: ToolExecutionOrder | None = None,
        tool_mode: ToolMode | None = None,
        # -- Timeouts --
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
        retry_enabled: bool | None = None,
        # -- Context management --
        summarize_context: SummarizeContextConfig | bool | None = None,
        # -- Observability / behavior --
        correlation_id: str | None = None,
        enable_eval_recording: bool | None = None,
        guardrails: GuardrailsConfig | None = None,
        max_validation_retries: int | None = None,
        on_status: OnStatusCallback = None,
        sinks: list[Sink] | None = None,
        track_costs: bool | None = None,
        **model_kwargs: Any,
    ) -> ExecutionResult[T]:
        """Structured completion using the OpenAI Responses API.

        Behavioural twin of :meth:`structured_complete` — supports the same
        tool-execution loop, dynamic tool routing, guardrails (input + output
        with retry), validation-aware retries, conversation summarisation,
        AAAK compression and condensing, hook stages, eval recording, cost
        tracking, status events and sinks. Differences are confined to the
        wire protocol:

        * Calls ``provider.aresponses`` instead of ``provider.acompletion``.
        * ``user_input`` may be a plain string OR a Responses
          ``ResponseInputParam`` list (multimodal, prefilled tool history,
          etc.). Lists are coerced into the internal message log so all
          context-management features still apply.
        * Structured output is enforced via ``text={"format": json_schema}``.
        * Tools are flattened to the Responses tool shape; tool calls are
          adapted to the chat-completions tool_call shape so the existing
          ``_execute_tool_calls_round`` plumbing is reused unchanged.

        Args:
            user_input: User prompt as a string, or a list of typed Responses
                input items for multimodal/prefilled-history use cases.
            response_format: Pydantic model class for structured output.
            (all other args mirror :meth:`structured_complete`)

        Returns:
            ExecutionResult with ``structured_output`` populated.

        Raises:
            Same exception hierarchy as :meth:`structured_complete`.
        """
        if is_shutting_down():
            raise RuntimeError("Cannot process request: shutdown in progress")

        effective_guardrails = guardrails if guardrails is not None else self.guardrails
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_condense = (
            condense_tool_messages if condense_tool_messages is not None else self.condense_tool_messages
        )
        effective_tool_mode: ToolMode = (
            tool_mode if tool_mode is not None else self.tool_mode
        )
        effective_tool_execution_order: ToolExecutionOrder = (
            tool_execution_order if tool_execution_order is not None else self.tool_execution_order
        )
        effective_retry_config = retry_config if retry_config is not None else self.retry_config
        if retry_enabled is not None and not retry_enabled:
            cfg = effective_retry_config
            effective_retry_config = (
                RetryConfig(retry_enabled=False)
                if cfg is None
                else RetryConfig(
                    **{**cfg.model_dump(exclude={"callback"}), "retry_enabled": False, "callback": cfg.callback}
                )
            )
        effective_rate_limit_config = rate_limit_config if rate_limit_config is not None else self.rate_limit_config
        effective_eval_store = None if enable_eval_recording is False else self.eval_store
        call_model_kwargs = dict(model_kwargs)
        override_max_tool_iterations = call_model_kwargs.pop("max_tool_iterations", None)
        effective_max_tool_iterations = (
            override_max_tool_iterations
            if override_max_tool_iterations is not None
            else self.max_tool_iterations
        )
        effective_model_kwargs = {**self.model_kwargs, **call_model_kwargs}
        if reasoning_effort is not None:
            effective_model_kwargs["reasoning_effort"] = reasoning_effort
        if parallel_tool_calls is not None:
            effective_model_kwargs["parallel_tool_calls"] = parallel_tool_calls
        effective_summarize_cfg = _merge_summarize_context_for_call(self._summarize_context_cfg, summarize_context)
        effective_summarize_context_model = _summarize_model_for_call(effective_summarize_cfg, model or self.model)

        if correlation_id:
            set_correlation_id(correlation_id)
        elif not get_correlation_id():
            set_correlation_id()
        correlation_id = get_correlation_id()

        # Normalise user_input -> a single textual user_message (for guardrails,
        # logs, conversation tracking) and a parallel chat-completions message
        # list (for tool-loop bookkeeping, summarisation, AAAK condensing).
        if isinstance(user_input, str):
            user_message_text = user_input
            extra_input_messages: list[dict[str, Any]] = []
        else:
            user_message_text = _stringify_response_input_for_log(user_input)
            extra_input_messages = _coerce_response_input_to_messages(user_input)

        logger.info(
            f"Starting structured response: correlation_id={correlation_id}, "
            f"response_format={response_format.__name__}, "
            f"input_kind={'string' if isinstance(user_input, str) else 'list'}, "
            f"message_length={len(user_message_text)}"
        )

        _sr_merged_registry = self._get_merged_hook_registry()
        if effective_guardrails:
            try:
                user_message_text = await self._run_guardrails(
                    user_message_text, effective_guardrails, "input", _sr_merged_registry
                )
            except GuardrailBlockedError:
                raise

        start_time = time.time()
        tools_to_use = tools if tools is not None else self.tools
        all_tools = tools_to_use
        static_tools = [t for t in all_tools if is_static_tool(t)]
        dynamic_tools = [t for t in all_tools if not is_static_tool(t)]
        router_tool = None
        if effective_tool_mode == "dynamic" and all_tools:
            router_tool = build_router_tool(dynamic_tools)
            active_tools: list[Callable] = [router_tool] + static_tools
        else:
            active_tools = all_tools
        system_prompt_content = self._format_system_prompt(tools=active_tools)
        messages_snapshot: list[dict] = []
        result: ExecutionResult | None = None
        error: Exception | None = None

        try:
            with ShutdownContext():
                # The conversation always sees the textual user message so
                # downstream consumers (eval store, multi-turn replay) can
                # treat structured_response and structured_complete uniformly.
                self._conversation.add_message(Role.USER, user_message_text)

                system_message = {"role": "system", "content": system_prompt_content}
                # Insert any list-form extras AFTER the existing conversation
                # but BEFORE the auto-added USER message, so the chronological
                # order matches what the user constructed.
                if extra_input_messages:
                    history = self._conversation.messages_dict
                    base = history[:-1]  # all but the just-added user msg
                    last_user = history[-1:]
                    messages = [system_message] + base + extra_input_messages + last_user
                else:
                    messages = [system_message] + self._conversation.messages_dict
                messages_snapshot = messages.copy()

                tool_execution_history: list[dict[str, Any]] = []
                tool_calls_made = 0
                total_tokens_used: dict[str, int] | None = None
                total_cost = 0.0

                def _track_usage(resp: Any) -> None:
                    nonlocal total_tokens_used, total_cost
                    iteration_tokens = _extract_response_token_usage(resp)
                    if iteration_tokens:
                        if total_tokens_used is None:
                            total_tokens_used = iteration_tokens.copy()
                        else:
                            for key in ("prompt", "completion", "total"):
                                total_tokens_used[key] = total_tokens_used.get(key, 0) + iteration_tokens.get(key, 0)
                    iteration_cost = _calculate_and_record_cost(
                        model=model or self.model,
                        tokens_used=iteration_tokens,
                        correlation_id=correlation_id,
                        track_costs=track_costs,
                    )
                    if iteration_cost is not None:
                        total_cost += iteration_cost

                # PHASE 1: Tool execution loop
                if active_tools and execute_tools:
                    logger.debug(
                        f"Starting Responses tool phase: max_iterations={effective_max_tool_iterations}, "
                        f"tools_available={len(active_tools)}, tool_mode={effective_tool_mode}"
                    )
                    _sr_pre_iter_hooks = _sr_merged_registry.get_hooks(HookStage.PRE_ITERATION)
                    _sr_post_iter_hooks = _sr_merged_registry.get_hooks(HookStage.POST_ITERATION)
                    for iteration in range(effective_max_tool_iterations):
                        logger.debug(
                            f"Responses tool iteration {iteration + 1}/{effective_max_tool_iterations}"
                        )

                        try:
                            _thr_sr = effective_summarize_cfg.threshold
                            _kr_sr = effective_summarize_cfg.keep_recent
                            if (
                                effective_summarize_cfg.enabled
                                and _thr_sr is not None
                                and len(messages) > _thr_sr
                            ):
                                messages = await _summarize_old_messages(
                                    messages,
                                    keep_recent=_kr_sr
                                    if _kr_sr is not None
                                    else settings.default_summarize_context_keep_recent,
                                    model=effective_summarize_context_model,
                                    use_aaak=self.aaak_compression_enabled,
                                    aaak_model=self.aaak_compression_model,
                                    completion_extra=effective_model_kwargs,
                                )

                            if _sr_pre_iter_hooks:
                                _last_user_msg = next(
                                    (
                                        m.get("content", "") or ""
                                        for m in reversed(messages)
                                        if m.get("role") == "user"
                                    ),
                                    "",
                                )
                                if isinstance(_last_user_msg, list):
                                    _last_user_msg = ""
                                await self._hook_manager.execute_hooks(
                                    content=_last_user_msg,
                                    stage=HookStage.PRE_ITERATION,
                                    metadata={
                                        "iteration": iteration + 1,
                                        "tool_count": len(active_tools),
                                        "endpoint": "responses",
                                    },
                                    hooks=_sr_pre_iter_hooks,
                                )

                            instructions, input_items = _messages_to_response_input(messages)

                            await emit_status(
                                ProcessEvent(
                                    kind="llm_call_start",
                                    correlation_id=correlation_id,
                                    timestamp=time.time(),
                                    iteration=iteration + 1,
                                    model=model or self.model,
                                    message_count=len(messages),
                                ),
                                on_status,
                                sinks=sinks,
                            )
                            response = await self._llm_responses_call(
                                _sr_merged_registry,
                                input_data=input_items,
                                model=model or self.model,
                                instructions=instructions,
                                tools=active_tools,
                                # No structured-output schema during the tool phase
                                request_timeout=request_timeout,
                                connect_timeout=connect_timeout,
                                api_key=api_key,
                                max_tokens=effective_max_tokens,
                                retry_config=effective_retry_config,
                                rate_limit_config=effective_rate_limit_config,
                                **effective_model_kwargs,
                            )
                            response_tool_calls = _extract_response_function_calls(response)
                            _sr_has_tool_calls = bool(response_tool_calls)
                            await emit_status(
                                ProcessEvent(
                                    kind="llm_call_end",
                                    correlation_id=correlation_id,
                                    timestamp=time.time(),
                                    iteration=iteration + 1,
                                    model=model or self.model,
                                    has_tool_calls=_sr_has_tool_calls,
                                    token_usage=_extract_response_token_usage(response),
                                ),
                                on_status,
                                sinks=sinks,
                            )
                        except LLMError as e:
                            cause_chain = _build_cause_chain(e)
                            logger.error(
                                f"Responses call failed on iteration {iteration + 1}: {e}, "
                                f"error_type={type(e).__name__}, cause_chain={cause_chain}"
                            )
                            raise type(e)(
                                f"Failed during tool execution (iteration {iteration + 1}): {e}"
                            ) from e

                        _track_usage(response)

                        response_text = _extract_response_text(response)

                        if _sr_post_iter_hooks:
                            await self._hook_manager.execute_hooks(
                                content=response_text,
                                stage=HookStage.POST_ITERATION,
                                metadata={
                                    "iteration": iteration + 1,
                                    "has_tool_calls": _sr_has_tool_calls,
                                    "tool_count": len(active_tools),
                                    "endpoint": "responses",
                                },
                                hooks=_sr_post_iter_hooks,
                            )

                        if response_tool_calls:
                            logger.info(
                                f"Iteration {iteration + 1}: Model requested "
                                f"{len(response_tool_calls)} tool call(s)"
                            )

                            # Dynamic routing: resolve tools, do not append, retry
                            if router_tool is not None and is_router_call(response_tool_calls):
                                query = json.loads(response_tool_calls[0].function.arguments)["query"]
                                user_context = next(
                                    (m.get("content", "") or "" for m in messages if m.get("role") == "user"),
                                    query,
                                )
                                if isinstance(user_context, list):
                                    user_context = query
                                matched = await self._route_tools(
                                    user_context,
                                    dynamic_tools,
                                    _sr_merged_registry,
                                    model=self.tool_route_model,
                                    api_key=api_key,
                                    timeout=request_timeout,
                                )
                                active_tools = matched + static_tools
                                logger.debug(
                                    f"[{correlation_id}] Dynamic routing: matched={len(matched)}, "
                                    f"static={len(static_tools)}, total_active={len(active_tools)}"
                                )
                                await emit_status(
                                    ProcessEvent(
                                        kind="tool_route",
                                        correlation_id=correlation_id,
                                        timestamp=time.time(),
                                        route_query=query,
                                        matched_tools=[t.__name__ for t in matched],
                                    ),
                                    on_status,
                                    sinks=sinks,
                                )
                                self._apply_system_prompt_after_tool_route(messages, active_tools)
                                continue

                            # Append assistant tool-call message in chat shape so
                            # the next iteration's input rebuild yields proper
                            # function_call items for the Responses API.
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": response_text or None,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments,
                                            },
                                        }
                                        for tc in response_tool_calls
                                    ],
                                }
                            )

                            round_results = await self._execute_tool_calls_round(
                                tool_calls=response_tool_calls,
                                active_tools=active_tools,
                                parallel=(effective_tool_execution_order == "parallel"),
                                iteration=iteration + 1,
                                correlation_id=correlation_id,
                                on_status=on_status,
                                sinks=sinks,
                            )
                            tool_calls_made += len(round_results)
                            for r in round_results:
                                tool_execution_history.append(r["history"])
                                messages.append(r["message"])

                            if effective_condense:
                                _condense_tool_round(messages, aaak_tool_condensing=self.aaak_tool_condensing)
                        else:
                            logger.debug(f"Model finished tools after {iteration + 1} iteration(s)")
                            if response_text:
                                messages.append({"role": "assistant", "content": response_text})
                            break
                    else:
                        logger.debug(
                            f"Reached max tool iterations ({effective_max_tool_iterations})"
                        )

                # PHASE 2: Final structured-output call
                logger.debug(
                    f"Requesting structured Responses output: response_format={response_format.__name__}"
                )
                try:
                    instructions, input_items = _messages_to_response_input(messages)
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_start",
                            correlation_id=correlation_id,
                            timestamp=time.time(),
                            iteration=None,
                            model=model or self.model,
                            message_count=len(messages),
                        ),
                        on_status,
                        sinks=sinks,
                    )
                    response = await self._llm_responses_call(
                        _sr_merged_registry,
                        input_data=input_items,
                        model=model or self.model,
                        instructions=instructions,
                        response_format=response_format,
                        request_timeout=request_timeout,
                        connect_timeout=connect_timeout,
                        api_key=api_key,
                        max_tokens=effective_max_tokens,
                        retry_config=effective_retry_config,
                        rate_limit_config=effective_rate_limit_config,
                        **effective_model_kwargs,
                    )
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_end",
                            correlation_id=correlation_id,
                            timestamp=time.time(),
                            model=model or self.model,
                            has_tool_calls=False,
                            token_usage=_extract_response_token_usage(response),
                        ),
                        on_status,
                        sinks=sinks,
                    )
                except LLMError as e:
                    logger.error(f"Structured Responses output call failed: {e}")
                    raise type(e)(f"Failed during structured output request: {e}") from e

                _track_usage(response)

                parsed = _extract_response_parsed(response, response_format)
                content = _extract_response_text(response)
                logger.debug(
                    f"Structured Responses received: parsed_type={type(parsed)}, "
                    f"content_length={len(content) if content else 0}"
                )

                # Output guardrails with retry loop (mirrors structured_complete)
                if effective_guardrails and content:
                    max_retries = effective_guardrails.max_output_guardrail_retries
                    output_retry_count = 0
                    while output_retry_count <= max_retries:
                        try:
                            content = await self._run_guardrails(
                                content,
                                effective_guardrails,
                                "output",
                                _sr_merged_registry,
                                attempt=output_retry_count,
                            )
                            break
                        except GuardrailRejectedError as e:
                            output_retry_count += 1
                            if output_retry_count > max_retries:
                                logger.warning(
                                    f"Output guardrails failed after {max_retries} retries: {e.reason}"
                                )
                                raise GuardrailBlockedError(
                                    f"Output guardrails failed after {max_retries} retries: {e.reason}",
                                    guardrail_name=e.guardrail_name,
                                ) from e

                            messages.append({"role": "assistant", "content": content})
                            feedback_message = (
                                f"Your previous response was rejected: {e.reason}. "
                                "Please provide a revised structured response that addresses this issue."
                            )
                            messages.append({"role": "user", "content": feedback_message})
                            logger.info(
                                f"Output guardrail rejected structured response "
                                f"(attempt {output_retry_count}/{max_retries}): {e.reason}. "
                                "Requesting revised response."
                            )

                            try:
                                instructions, input_items = _messages_to_response_input(messages)
                                response = await self._llm_responses_call(
                                    _sr_merged_registry,
                                    input_data=input_items,
                                    model=model or self.model,
                                    instructions=instructions,
                                    response_format=response_format,
                                    request_timeout=request_timeout,
                                    connect_timeout=connect_timeout,
                                    api_key=api_key,
                                    max_tokens=effective_max_tokens,
                                    retry_config=effective_retry_config,
                                    rate_limit_config=effective_rate_limit_config,
                                    **effective_model_kwargs,
                                )
                            except LLMError as llm_error:
                                raise GuardrailBlockedError(
                                    f"Failed to get revised structured response after guardrail rejection: {llm_error}",
                                    guardrail_name=e.guardrail_name,
                                ) from llm_error

                            _track_usage(response)
                            parsed = _extract_response_parsed(response, response_format)
                            content = _extract_response_text(response)
                            logger.debug(
                                f"Received revised structured Responses output "
                                f"(length={len(content) if content else 0})"
                            )

                if content:
                    self._conversation.add_message(Role.ASSISTANT, content)

                # Validation-aware retry loop (mirrors structured_complete)
                max_val_retries = max_validation_retries if max_validation_retries is not None else 3
                validation_attempt = 0
                structured_output: T | None = None
                while True:
                    try:
                        if isinstance(parsed, response_format):
                            logger.debug(f"Using parsed Pydantic instance: {response_format.__name__}")
                            structured_output = parsed
                        elif isinstance(parsed, dict):
                            logger.debug(f"Instantiating {response_format.__name__} from dict")
                            structured_output = response_format(**parsed)
                        elif content:
                            data = json.loads(content)
                            logger.debug(
                                f"Parsed JSON from content, instantiating {response_format.__name__}"
                            )
                            structured_output = response_format(**data)
                        else:
                            logger.warning(f"Using parsed response as-is (type: {type(parsed)})")
                            structured_output = parsed
                        break
                    except (ValidationError, json.JSONDecodeError, TypeError) as e:
                        validation_attempt += 1
                        if validation_attempt > max_val_retries:
                            raise
                        messages.append({"role": "assistant", "content": content or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your response failed validation: {e}\n"
                                    "Please return a corrected JSON response."
                                ),
                            }
                        )
                        logger.info(
                            f"Validation retry {validation_attempt}/{max_val_retries}: {e}"
                        )
                        _val_retry_hooks = _sr_merged_registry.get_hooks(HookStage.ON_VALIDATION_RETRY)
                        if _val_retry_hooks:
                            await self._hook_manager.execute_hooks(
                                content=content or "",
                                stage=HookStage.ON_VALIDATION_RETRY,
                                metadata={
                                    "validation_attempt": validation_attempt,
                                    "max_validation_retries": max_val_retries,
                                    "error": str(e),
                                    "response_format_name": response_format.__name__,
                                    "endpoint": "responses",
                                },
                                hooks=_val_retry_hooks,
                            )
                        instructions, input_items = _messages_to_response_input(messages)
                        response = await self._llm_responses_call(
                            _sr_merged_registry,
                            input_data=input_items,
                            model=model or self.model,
                            instructions=instructions,
                            response_format=response_format,
                            request_timeout=request_timeout,
                            connect_timeout=connect_timeout,
                            api_key=api_key,
                            max_tokens=effective_max_tokens,
                            retry_config=effective_retry_config,
                            rate_limit_config=effective_rate_limit_config,
                            **effective_model_kwargs,
                        )
                        _track_usage(response)
                        parsed = _extract_response_parsed(response, response_format)
                        content = _extract_response_text(response)

                logger.info(
                    f"Structured response finished: tool_calls={tool_calls_made}, cost=${total_cost:.6f}"
                )

                await emit_status(
                    ProcessEvent(
                        kind="complete",
                        correlation_id=correlation_id,
                        timestamp=time.time(),
                        tool_calls_made=tool_calls_made,
                        response_length=len(content) if content else 0,
                    ),
                    on_status,
                    sinks=sinks,
                )

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

                await _record_eval_data(
                    eval_store=effective_eval_store,
                    user_message=user_message_text,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    result=result,
                    tools_available=tools_to_use,
                    on_eval_record_hooks=_sr_merged_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )

                return result
        except Exception as e:
            error = e
            raise
        finally:
            if error and not result:
                await _record_eval_data(
                    eval_store=effective_eval_store,
                    user_message=user_message_text,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    error=error,
                    tools_available=tools_to_use,
                    on_eval_record_hooks=_sr_merged_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )
            clear_correlation_id()

    def _format_system_prompt(self, tools: list[Callable] | None = None) -> str:
        """Format system prompt. Tool schemas are sent via the API tools parameter, not here."""
        from gluellm.models.prompt import BASE_SYSTEM_PROMPT

        return BASE_SYSTEM_PROMPT.render(
            instructions=self.system_prompt,
        ).strip()

    def _apply_system_prompt_after_tool_route(
        self,
        messages: list[dict[str, Any]],
        active_tools: list[Callable],
    ) -> None:
        """Set system message content after dynamic routing; keep AAAK decode preamble if enabled."""
        messages[0]["content"] = self._format_system_prompt(tools=active_tools)
        if self.aaak_compression_enabled or self.aaak_tool_condensing:
            from gluellm.compression.aaak import AAAKCompressor

            AAAKCompressor.ensure_preamble_in_system(messages[0])

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

    @staticmethod
    def _coerce_args_to_pydantic(tool_func: Callable, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Coerce dict arguments to Pydantic model instances where the annotation demands it.

        When the LLM produces tool arguments they arrive as plain dicts from
        ``json.loads``.  If the tool function declares a parameter with a
        ``BaseModel`` subclass annotation we automatically coerce the dict to
        the correct model via ``model_validate``, giving tool authors natural
        attribute access (``arg.field``) without any boilerplate inside the tool.

        Coercion handles bare ``BaseModel`` subclasses, ``list[Model]``,
        ``Union[A, B]`` / ``A | B``, and ``Optional[Model]`` annotations.

        Coercion is best-effort: if the annotation cannot be resolved or
        ``model_validate`` raises, the original value is passed through unchanged
        so existing error handling continues to work.
        """
        try:
            hints = get_type_hints(tool_func)
        except Exception:
            return tool_args

        if not hints:
            return tool_args

        coerced = dict(tool_args)
        for param_name, value in tool_args.items():
            annotation = hints.get(param_name)
            if annotation is None:
                continue
            try:
                coerced[param_name] = _coerce_pydantic_value(value, annotation)
            except Exception as coerce_err:
                logger.debug(
                    f"Pydantic coercion failed for param '{param_name}' "
                    f"({type(coerce_err).__name__}: {coerce_err}); passing raw value"
                )
        return coerced

    async def _execute_tool_calls_round(
        self,
        tool_calls: list[Any],
        active_tools: list[Callable],
        parallel: bool,
        iteration: int,
        correlation_id: str,
        on_status: OnStatusCallback,
        sinks: list[Sink] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a round of tool calls (sequential or parallel).

        Returns a list of dicts, each with "history" and "message" keys for
        appending to tool_execution_history and messages.
        """
        mode = "parallel" if parallel else "sequential"
        logger.debug(
            f"[{correlation_id}] Starting tool execution round: iteration={iteration}, "
            f"tool_count={len(tool_calls)}, mode={mode}"
        )
        merged_registry = self._get_merged_hook_registry()
        pre_tool_hooks = merged_registry.get_hooks(HookStage.PRE_TOOL)
        post_tool_hooks = merged_registry.get_hooks(HookStage.POST_TOOL)

        # Phase 1: parse and resolve each tool call
        Parsed = tuple[int, Any, str, dict[str, Any] | None, Callable | None, str | None]
        parsed: list[Parsed] = []
        for call_index, tool_call in enumerate(tool_calls, 1):
            tool_name = _tool_name_from_call(tool_call)
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[{correlation_id}] Tool call {call_index} '{tool_name}' has invalid JSON arguments: {e}"
                )
                parsed.append((call_index, tool_call, tool_name, None, None, f"Invalid JSON in tool arguments: {str(e)}"))
                continue
            tool_func = self._find_tool(tool_name, tools=active_tools)
            if not tool_func:
                logger.warning(
                    f"[{correlation_id}] Tool call {call_index} '{tool_name}' not found in available tools"
                )
                parsed.append((call_index, tool_call, tool_name, tool_args, None, f"Tool '{tool_name}' not found in available tools"))
                continue
            parsed.append((call_index, tool_call, tool_name, tool_args, tool_func, None))

        def _truncate_for_log(value: Any, max_len: int = 200) -> str:
            """Truncate a value for logging purposes."""
            s = str(value)
            return s if len(s) <= max_len else s[:max_len] + "..."

        async def run_one(call_index: int, tool_call: Any, tool_name: str, tool_args: dict[str, Any], tool_func: Callable) -> tuple[str, bool, float]:
            """Execute a single tool, running pre/post hooks around the invocation."""
            logger.debug(
                f"[{correlation_id}] Executing tool '{tool_name}' (call {call_index}): "
                f"args={_truncate_for_log(tool_args)}"
            )
            # PRE_TOOL hooks: content is JSON-serialised args; hook may return modified JSON
            if pre_tool_hooks:
                args_content = await self._hook_manager.execute_hooks(
                    content=json.dumps(tool_args),
                    stage=HookStage.PRE_TOOL,
                    metadata={"tool_name": tool_name, "call_index": call_index, "iteration": iteration},
                    hooks=pre_tool_hooks,
                )
                try:
                    tool_args = json.loads(args_content)
                except json.JSONDecodeError:
                    logger.warning(
                        f"PRE_TOOL hook for '{tool_name}' returned invalid JSON; using original args"
                    )

            start = time.time()
            try:
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
                tool_args = self._coerce_args_to_pydantic(tool_func, tool_args)
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**tool_args)
                else:
                    result = await asyncio.to_thread(tool_func, **tool_args)
                elapsed = time.time() - start
                logger.debug(
                    f"[{correlation_id}] Tool '{tool_name}' (call {call_index}) completed successfully "
                    f"in {elapsed:.3f}s: result={_truncate_for_log(result)}"
                )
                result_str = result.model_dump_json() if isinstance(result, BaseModel) else str(result)
                had_error = False
            except Exception as e:
                elapsed = time.time() - start
                logger.warning(
                    f"[{correlation_id}] Tool '{tool_name}' (call {call_index}) failed "
                    f"in {elapsed:.3f}s: {type(e).__name__}: {e}"
                )
                result_str = f"Error executing tool: {type(e).__name__}: {str(e)}"
                had_error = True

            # POST_TOOL hooks: content is result string; hook may return modified result
            if post_tool_hooks:
                result_str = await self._hook_manager.execute_hooks(
                    content=result_str,
                    stage=HookStage.POST_TOOL,
                    metadata={
                        "tool_name": tool_name,
                        "call_index": call_index,
                        "iteration": iteration,
                        "duration_seconds": elapsed,
                        "error": had_error,
                    },
                    hooks=post_tool_hooks,
                )

            return (result_str, had_error, elapsed)

        results: list[dict[str, Any]] = []
        if parallel:
            # Emit all starts first
            for call_index, tool_call, tool_name, _a, _f, err in parsed:
                await emit_status(
                    ProcessEvent(
                        kind="tool_call_start",
                        correlation_id=correlation_id,
                        timestamp=time.time(),
                        iteration=iteration,
                        tool_name=tool_name,
                        call_index=call_index,
                    ),
                    on_status,
                    sinks=sinks,
                )
            # Build ordered results: errors first, then runnables
            ordered: list[dict[str, Any] | None] = [None] * len(parsed)
            runnable_items: list[tuple[int, Any, str, dict[str, Any], Callable]] = []
            for i, (call_index, tool_call, tool_name, tool_args, tool_func, err) in enumerate(parsed):
                if err is not None:
                    await emit_status(
                        ProcessEvent(kind="tool_call_end", correlation_id=correlation_id, timestamp=time.time(), tool_name=tool_name, call_index=call_index, success=False, duration_seconds=0, error=err),
                        on_status,
                        sinks=sinks,
                    )
                    args_str = getattr(tool_call.function, "arguments", "{}")
                    ordered[i] = {"history": {"tool_name": tool_name, "arguments": args_str, "result": err, "error": True}, "message": {"role": "tool", "tool_call_id": tool_call.id, "content": err}}
                else:
                    runnable_items.append((call_index, tool_call, tool_name, tool_args or {}, tool_func))
            if runnable_items:
                tasks = [run_one(ci, tc, tn, ta, tf) for ci, tc, tn, ta, tf in runnable_items]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                runnable_idx = 0
                for i, (call_index, tool_call, tool_name, tool_args, tool_func, err) in enumerate(parsed):
                    if err is not None:
                        continue
                    g = gathered[runnable_idx]
                    runnable_idx += 1
                    if isinstance(g, Exception):
                        result_str, error, duration = f"Error executing tool: {type(g).__name__}: {str(g)}", True, 0.0
                    else:
                        result_str, error, duration = g
                    await emit_status(
                        ProcessEvent(kind="tool_call_end", correlation_id=correlation_id, timestamp=time.time(), tool_name=tool_name, call_index=call_index, success=not error, duration_seconds=duration, error=result_str if error else None),
                        on_status,
                        sinks=sinks,
                    )
                    ordered[i] = {"history": {"tool_name": tool_name, "arguments": tool_args or {}, "result": result_str, "error": error}, "message": {"role": "tool", "tool_call_id": tool_call.id, "content": result_str}}
            results = [r for r in ordered if r is not None]
        else:
            # Sequential
            for call_index, tool_call, tool_name, tool_args, tool_func, err in parsed:
                await emit_status(
                    ProcessEvent(kind="tool_call_start", correlation_id=correlation_id, timestamp=time.time(), iteration=iteration, tool_name=tool_name, call_index=call_index),
                    on_status,
                    sinks=sinks,
                )
                if err is not None:
                    await emit_status(
                        ProcessEvent(kind="tool_call_end", correlation_id=correlation_id, timestamp=time.time(), tool_name=tool_name, call_index=call_index, success=False, duration_seconds=0, error=err),
                        on_status,
                        sinks=sinks,
                    )
                    results.append({"history": {"tool_name": tool_name, "arguments": getattr(tool_call.function, "arguments", "{}"), "result": err, "error": True}, "message": {"role": "tool", "tool_call_id": tool_call.id, "content": err}})
                    continue
                result_str, error, duration = await run_one(call_index, tool_call, tool_name, tool_args or {}, tool_func)
                await emit_status(
                    ProcessEvent(kind="tool_call_end", correlation_id=correlation_id, timestamp=time.time(), tool_name=tool_name, call_index=call_index, success=not error, duration_seconds=duration, error=result_str if error else None),
                    on_status,
                    sinks=sinks,
                )
                results.append({"history": {"tool_name": tool_name, "arguments": tool_args or {}, "result": result_str, "error": error}, "message": {"role": "tool", "tool_call_id": tool_call.id, "content": result_str}})

        # Log summary of tool execution round
        successful = sum(1 for r in results if not r["history"].get("error"))
        failed = len(results) - successful
        logger.info(
            f"[{correlation_id}] Tool execution round {iteration} complete: "
            f"{successful} succeeded, {failed} failed, mode={mode}"
        )
        return results

    @overload
    async def stream_complete(
        self,
        user_message: str,
        # -- Core --
        model: str | None = ...,
        guardrails: GuardrailsConfig | None = ...,
        response_format: type[T] = ...,
        # -- Model generation --
        max_tokens: int | None = ...,
        # -- Tools --
        condense_tool_messages: bool | None = ...,
        execute_tools: bool = ...,
        tool_execution_order: ToolExecutionOrder | None = ...,
        tool_mode: ToolMode | None = ...,
        # -- Timeouts --
        connect_timeout: float | None = ...,
        request_timeout: float | None = ...,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = ...,
        retry_config: RetryConfig | None = ...,
        retry_enabled: bool | None = ...,
        # -- Context management --
        summarize_context: SummarizeContextConfig | bool | None = ...,
        # -- Observability / behavior --
        correlation_id: str | None = ...,
        enable_eval_recording: bool | None = ...,
        on_status: OnStatusCallback = ...,
        sinks: list[Sink] | None = ...,
        track_costs: bool | None = ...,
        **model_kwargs: Any,
    ) -> AsyncIterator[StreamingChunk[T]]: ...

    @overload
    async def stream_complete(
        self,
        user_message: str,
        # -- Core --
        model: str | None = ...,
        guardrails: GuardrailsConfig | None = ...,
        response_format: None = ...,
        # -- Model generation --
        max_tokens: int | None = ...,
        # -- Tools --
        condense_tool_messages: bool | None = ...,
        execute_tools: bool = ...,
        tool_execution_order: ToolExecutionOrder | None = ...,
        tool_mode: ToolMode | None = ...,
        # -- Timeouts --
        connect_timeout: float | None = ...,
        request_timeout: float | None = ...,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = ...,
        retry_config: RetryConfig | None = ...,
        retry_enabled: bool | None = ...,
        # -- Context management --
        summarize_context: SummarizeContextConfig | bool | None = ...,
        # -- Observability / behavior --
        correlation_id: str | None = ...,
        enable_eval_recording: bool | None = ...,
        on_status: OnStatusCallback = ...,
        sinks: list[Sink] | None = ...,
        track_costs: bool | None = ...,
        **model_kwargs: Any,
    ) -> AsyncIterator[StreamingChunk[Any]]: ...

    async def stream_complete(
        self,
        user_message: str,
        # -- Core --
        model: str | None = None,
        guardrails: GuardrailsConfig | None = None,
        response_format: type[T] | None = None,
        # -- Model generation --
        max_tokens: int | None = None,
        # -- Tools --
        condense_tool_messages: bool | None = None,
        execute_tools: bool = True,
        tool_execution_order: ToolExecutionOrder | None = None,
        tool_mode: ToolMode | None = None,
        # -- Timeouts --
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        # -- Resilience --
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
        retry_enabled: bool | None = None,
        # -- Context management --
        summarize_context: SummarizeContextConfig | bool | None = None,
        # -- Observability / behavior --
        correlation_id: str | None = None,
        enable_eval_recording: bool | None = None,
        on_status: OnStatusCallback = None,
        sinks: list[Sink] | None = None,
        track_costs: bool | None = None,
        **model_kwargs: Any,
    ) -> AsyncIterator[StreamingChunk[Any]]:
        """Stream completion with automatic tool execution.

        Yields chunks of the response as they arrive. When tools are called,
        streaming pauses and tool execution occurs, then streaming resumes.

        When response_format is set, the model is asked to return JSON matching
        that schema; the final chunk will have structured_output set to the
        parsed Pydantic instance (when parsing succeeds).

        Note:
            When tools are enabled, the LLM is called with streaming. Content
            deltas are yielded and emitted via ``on_status`` (``stream_chunk``) as
            they arrive. If the model requests tool calls, they are accumulated
            from the stream, tools are executed, and the loop continues with
            another streaming call. Token-by-token streaming therefore applies to
            both tool and non-tool turns.

        Args:
            user_message: The user's message/request
            execute_tools: Whether to automatically execute tools
            model: Model identifier override (defaults to instance model)
            guardrails: Optional guardrails configuration (overrides instance guardrails if provided)
            response_format: Optional Pydantic model; if set, the final chunk may include structured_output
            on_status: Optional callback for process status events
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
            connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
            max_tokens: Maximum number of tokens to generate. Overrides instance-level max_tokens if provided.
            condense_tool_messages: Override the instance-level condense_tool_messages setting for this call.
            tool_mode: Override the instance-level tool_mode for this call ("standard" or "dynamic").
            retry_enabled: If False, disables retries for this call (shorthand for retry_config.retry_enabled=False).
            retry_config: Per-call retry configuration override (includes optional callback).
            rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
            track_costs: If False, skip cost tracking for this call (defaults to settings.track_costs).
            enable_eval_recording: If False, skip eval recording for this call (defaults to using instance eval_store).
            summarize_context: Override instance summarization (bool or ``SummarizeContextConfig``).
            **model_kwargs: Extra params for acompletion (e.g. temperature, top_p), merged over instance defaults.

        Yields:
            StreamingChunk objects with content and metadata (and optional structured_output on the final chunk)

        Raises:
            TokenLimitError: If token limit is exceeded
            RateLimitError: If rate limit persists after retries
            APIConnectionError: If connection fails after retries
            AuthenticationError: If authentication fails
            InvalidRequestError: If request parameters are invalid
            GuardrailBlockedError: If input guardrails block the request or output guardrails fail after max retries
        """
        # Resolve guardrails config: per-call overrides instance
        effective_guardrails = guardrails if guardrails is not None else self.guardrails
        # Per-call max_tokens takes precedence over instance-level setting
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_condense = (
            condense_tool_messages if condense_tool_messages is not None else self.condense_tool_messages
        )
        effective_tool_mode: ToolMode = (
            tool_mode if tool_mode is not None else self.tool_mode
        )
        effective_tool_execution_order: ToolExecutionOrder = (
            tool_execution_order if tool_execution_order is not None else self.tool_execution_order
        )
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_retry_config = retry_config if retry_config is not None else self.retry_config
        if retry_enabled is not None and not retry_enabled:
            cfg = effective_retry_config
            effective_retry_config = (
                RetryConfig(retry_enabled=False)
                if cfg is None
                else RetryConfig(**{**cfg.model_dump(exclude={"callback"}), "retry_enabled": False, "callback": cfg.callback})
            )
        effective_rate_limit_config = rate_limit_config if rate_limit_config is not None else self.rate_limit_config
        effective_summarize_cfg = _merge_summarize_context_for_call(self._summarize_context_cfg, summarize_context)
        effective_summarize_context_model = _summarize_model_for_call(effective_summarize_cfg, model or self.model)

        # Instance sampling / provider extras (parity with complete() → _llm_call)
        stream_call_kwargs = dict(model_kwargs)
        override_stream_max_tool_iterations = stream_call_kwargs.pop("max_tool_iterations", None)
        effective_stream_max_tool_iterations = (
            override_stream_max_tool_iterations
            if override_stream_max_tool_iterations is not None
            else self.max_tool_iterations
        )
        effective_stream_model_kwargs = {**self.model_kwargs, **stream_call_kwargs}
        effective_eval_store = None if enable_eval_recording is False else self.eval_store

        # Set correlation ID if provided
        if correlation_id:
            set_correlation_id(correlation_id)
        elif not get_correlation_id():
            set_correlation_id()

        # Resolve active_tools for dynamic vs standard mode
        all_tools = self.tools
        static_tools = [t for t in all_tools if is_static_tool(t)]
        dynamic_tools = [t for t in all_tools if not is_static_tool(t)]
        router_tool = None
        if effective_tool_mode == "dynamic" and all_tools:
            router_tool = build_router_tool(dynamic_tools)
            active_tools: list[Callable] = [router_tool] + static_tools
        else:
            active_tools = all_tools

        # Run input guardrails before processing
        _stream_merged_registry = self._get_merged_hook_registry()
        if effective_guardrails:
            try:
                user_message = await self._run_guardrails(
                    user_message, effective_guardrails, "input", _stream_merged_registry
                )
            except GuardrailBlockedError:
                raise  # Re-raise as-is (no retry for input)

        # Add user message to conversation (after input guardrails)
        self._conversation.add_message(Role.USER, user_message)

        # Build initial messages (snapshot for eval recording matches complete())
        system_prompt_content = self._format_system_prompt(tools=active_tools)
        system_message = {
            "role": "system",
            "content": system_prompt_content,
        }
        messages = [system_message] + self._conversation.messages_dict
        messages_snapshot = messages.copy()
        start_time = time.time()

        tool_execution_history = []
        tool_calls_made = 0
        accumulated_content = ""

        _stream_pre_iter_hooks = _stream_merged_registry.get_hooks(HookStage.PRE_ITERATION)
        _stream_post_iter_hooks = _stream_merged_registry.get_hooks(HookStage.POST_ITERATION)

        error: Exception | None = None
        eval_recorded = False

        async def record_stream_eval_success(final_text: str, *, structured_out: Any = None) -> None:
            nonlocal eval_recorded
            if not effective_eval_store:
                return
            _eval_result = ExecutionResult(
                final_response=final_text,
                tool_calls_made=tool_calls_made,
                tool_execution_history=tool_execution_history,
                raw_response=None,
                tokens_used=None,
                estimated_cost_usd=None,
                model=model or self.model,
                structured_output=structured_out,
            )
            await _record_eval_data(
                eval_store=effective_eval_store,
                user_message=user_message,
                system_prompt=system_prompt_content,
                model=model or self.model,
                messages_snapshot=messages_snapshot,
                start_time=start_time,
                result=_eval_result,
                tools_available=self.tools,
                on_eval_record_hooks=_stream_merged_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                hook_manager=self._hook_manager,
            )
            eval_recorded = True

        # Tool execution loop
        try:
            for iteration in range(effective_stream_max_tool_iterations):
                assistant_message = None
                try:
                    # Context summarization: compress old messages when threshold is exceeded
                    _thr_st = effective_summarize_cfg.threshold
                    _kr_st = effective_summarize_cfg.keep_recent
                    if effective_summarize_cfg.enabled and _thr_st is not None and len(messages) > _thr_st:
                        messages = await _summarize_old_messages(
                            messages,
                            keep_recent=_kr_st
                            if _kr_st is not None
                            else settings.default_summarize_context_keep_recent,
                            model=effective_summarize_context_model,
                            use_aaak=self.aaak_compression_enabled,
                            aaak_model=self.aaak_compression_model,
                            completion_extra=effective_stream_model_kwargs,
                        )

                    if _stream_pre_iter_hooks:
                        _stream_last_user = next(
                            (m.get("content", "") or "" for m in reversed(messages) if m.get("role") == "user"),
                            "",
                        )
                        if isinstance(_stream_last_user, list):
                            _stream_last_user = ""
                        await self._hook_manager.execute_hooks(
                            content=_stream_last_user,
                            stage=HookStage.PRE_ITERATION,
                            metadata={"iteration": iteration + 1, "tool_count": len(active_tools)},
                            hooks=_stream_pre_iter_hooks,
                        )

                    # Try streaming first (if no tools or tools disabled, stream directly)
                    if not execute_tools or not active_tools:
                        # Simple streaming without tool execution
                        await emit_status(
                            ProcessEvent(
                                kind="stream_start",
                                correlation_id=get_correlation_id(),
                                timestamp=time.time(),
                            ),
                            on_status,
                            sinks=sinks,
                        )
                        # Providers (e.g. OpenAI) do not support response_format with stream=True;
                        # we stream plain text and parse into response_format when the stream ends.
                        stream_iter = await self._llm_call(
                            _stream_merged_registry,
                            messages=messages,
                            model=model or self.model,
                            tools=None,
                            response_format=None,
                            stream=True,
                            request_timeout=request_timeout,
                            connect_timeout=connect_timeout,
                            max_tokens=effective_max_tokens,
                            retry_config=effective_retry_config,
                            rate_limit_config=effective_rate_limit_config,
                            **effective_stream_model_kwargs,
                        )
                        async for chunk_response in stream_iter:
                            if hasattr(chunk_response, "choices") and chunk_response.choices:
                                delta = chunk_response.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    accumulated_content += delta.content
                                    chunk = StreamingChunk(
                                        content=delta.content,
                                        done=False,
                                        tool_calls_made=tool_calls_made,
                                    )
                                    await emit_status(
                                        ProcessEvent(
                                            kind="stream_chunk",
                                            correlation_id=get_correlation_id(),
                                            timestamp=time.time(),
                                            content=delta.content,
                                            done=False,
                                        ),
                                        on_status,
                                        sinks=sinks,
                                    )
                                    yield chunk
                        # Final chunk - run output guardrails on accumulated content
                        await emit_status(
                            ProcessEvent(
                                kind="stream_end",
                                correlation_id=get_correlation_id(),
                                timestamp=time.time(),
                            ),
                            on_status,
                            sinks=sinks,
                        )
                        structured_output = None
                        if response_format and accumulated_content:
                            structured_output = _parse_structured_content(accumulated_content, response_format)
                        if accumulated_content:
                            if effective_guardrails:
                                try:
                                    accumulated_content = await self._run_guardrails(
                                        accumulated_content, effective_guardrails, "output",
                                        _stream_merged_registry,
                                    )
                                except GuardrailRejectedError as e:
                                    # For streaming, we can't easily retry, so raise blocked error
                                    logger.warning(f"Output guardrails rejected streamed content: {e.reason}")
                                    raise GuardrailBlockedError(
                                        f"Output guardrails rejected streamed content: {e.reason}",
                                        guardrail_name=e.guardrail_name,
                                    ) from e
                            self._conversation.add_message(Role.ASSISTANT, accumulated_content)
                            yield StreamingChunk(
                                content="",
                                done=True,
                                tool_calls_made=tool_calls_made,
                                structured_output=structured_output,
                            )
                        else:
                            yield StreamingChunk(
                                content="",
                                done=True,
                                tool_calls_made=tool_calls_made,
                                structured_output=structured_output,
                            )
                        if _stream_post_iter_hooks:
                            await self._hook_manager.execute_hooks(
                                content=accumulated_content,
                                stage=HookStage.POST_ITERATION,
                                metadata={
                                    "iteration": iteration + 1,
                                    "has_tool_calls": False,
                                    "tool_count": 0,
                                },
                                hooks=_stream_post_iter_hooks,
                            )
                        await record_stream_eval_success(
                            accumulated_content,
                            structured_out=structured_output,
                        )
                        return

                    # With tools: stream so we get token-by-token text and can detect tool_calls from the stream
                    await emit_status(
                        ProcessEvent(
                            kind="stream_start",
                            correlation_id=get_correlation_id(),
                            timestamp=time.time(),
                        ),
                        on_status,
                        sinks=sinks,
                    )
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_start",
                            correlation_id=get_correlation_id(),
                            timestamp=time.time(),
                            iteration=iteration + 1,
                            model=model or self.model,
                            message_count=len(messages),
                        ),
                        on_status,
                        sinks=sinks,
                    )
                    stream_iter = await self._llm_call(
                        _stream_merged_registry,
                        messages=messages,
                        model=model or self.model,
                        tools=active_tools if active_tools else None,
                        response_format=None,
                        stream=True,
                        request_timeout=request_timeout,
                        connect_timeout=connect_timeout,
                        max_tokens=effective_max_tokens,
                        retry_config=effective_retry_config,
                        rate_limit_config=effective_rate_limit_config,
                        **effective_stream_model_kwargs,
                    )
                    accumulated_content = ""
                    assistant_message = None
                    async for is_content, content_or_accumulated, msg in _consume_stream_with_tools(stream_iter):
                        if is_content:
                            await emit_status(
                                ProcessEvent(
                                    kind="stream_chunk",
                                    correlation_id=get_correlation_id(),
                                    timestamp=time.time(),
                                    content=content_or_accumulated,
                                    done=False,
                                ),
                                on_status,
                                sinks=sinks,
                            )
                            yield StreamingChunk(
                                content=content_or_accumulated,
                                done=False,
                                tool_calls_made=tool_calls_made,
                            )
                        else:
                            accumulated_content = content_or_accumulated
                            assistant_message = msg
                            break
                    has_tool_calls = bool(assistant_message and getattr(assistant_message, "tool_calls", None))
                    await emit_status(
                        ProcessEvent(
                            kind="llm_call_end",
                            correlation_id=get_correlation_id(),
                            timestamp=time.time(),
                            iteration=iteration + 1,
                            model=model or self.model,
                            has_tool_calls=has_tool_calls,
                            token_usage=None,
                        ),
                        on_status,
                        sinks=sinks,
                    )
                except LLMError as e:
                    cause_chain = _build_cause_chain(e)
                    logger.error(
                        f"LLM call failed on iteration {iteration + 1}: {e}, "
                        f"error_type={type(e).__name__}, cause_chain={cause_chain}"
                    )
                    error_msg = f"Failed during tool execution loop (iteration {iteration + 1}/{effective_stream_max_tool_iterations})"
                    raise type(e)(f"{error_msg}: {e}") from e

                if _stream_post_iter_hooks:
                    _stream_has_tools = bool(
                        execute_tools and active_tools
                        and assistant_message and getattr(assistant_message, "tool_calls", None)
                    )
                    await self._hook_manager.execute_hooks(
                        content=accumulated_content,
                        stage=HookStage.POST_ITERATION,
                        metadata={
                            "iteration": iteration + 1,
                            "has_tool_calls": _stream_has_tools,
                            "tool_count": len(active_tools),
                        },
                        hooks=_stream_post_iter_hooks,
                    )

                # Check if model wants to call tools (from streamed response)
                if execute_tools and active_tools and assistant_message and getattr(assistant_message, "tool_calls", None):
                    tool_calls = assistant_message.tool_calls
                    logger.info(f"Stream iteration {iteration + 1}: Model requested {len(tool_calls)} tool call(s)")

                    # Handle router call: resolve tools, do NOT append messages, continue
                    if router_tool is not None and is_router_call(tool_calls):
                        query = json.loads(tool_calls[0].function.arguments)["query"]
                        user_context = next(
                            (m.get("content", "") or "" for m in messages if m.get("role") == "user"),
                            query,
                        )
                        if isinstance(user_context, list):
                            user_context = query
                        matched = await self._route_tools(
                            user_context,
                            dynamic_tools,
                            _stream_merged_registry,
                            model=self.tool_route_model,
                            api_key=None,
                            timeout=request_timeout,
                        )
                        active_tools = matched + static_tools
                        logger.debug(
                            f"[{get_correlation_id()}] Dynamic tool routing complete: "
                            f"matched={len(matched)}, static={len(static_tools)}, total_active={len(active_tools)}"
                        )
                        await emit_status(
                            ProcessEvent(
                                kind="tool_route",
                                correlation_id=get_correlation_id(),
                                timestamp=time.time(),
                                route_query=query,
                                matched_tools=[t.__name__ for t in matched],
                            ),
                            on_status,
                            sinks=sinks,
                        )
                        # Update system prompt to reflect matched tools (no longer router)
                        self._apply_system_prompt_after_tool_route(messages, active_tools)
                        continue

                    # Yield a chunk indicating tool execution is happening
                    yield StreamingChunk(
                        content="[Executing tools...]",
                        done=False,
                        tool_calls_made=tool_calls_made,
                    )

                    # Add assistant message with tool calls to history (dict for any_llm validation)
                    msg_dict = _streamed_assistant_message_to_dict(assistant_message)
                    if msg_dict is not None:
                        messages.append(msg_dict)

                    # Execute tool calls (sequential or parallel per effective_tool_execution_order)
                    round_results = await self._execute_tool_calls_round(
                        tool_calls=tool_calls,
                        active_tools=active_tools,
                        parallel=(effective_tool_execution_order == "parallel"),
                        iteration=iteration + 1,
                        correlation_id=get_correlation_id(),
                        on_status=on_status,
                        sinks=sinks,
                    )
                    tool_calls_made += len(round_results)
                    for r in round_results:
                        tool_execution_history.append(r["history"])
                        messages.append(r["message"])

                    if effective_condense:
                        _condense_tool_round(messages, aaak_tool_condensing=self.aaak_tool_condensing)

                    # Continue loop to get next response (stream again)
                    continue

                # No more tool calls: we streamed the final text; run guardrails and finish
                final_content = accumulated_content or ""

                # Output guardrails with retry loop (similar to complete())
                if effective_guardrails and final_content:
                    max_retries = effective_guardrails.max_output_guardrail_retries
                    output_retry_count = 0
                    while output_retry_count <= max_retries:
                        try:
                            # Run output guardrails
                            final_content = await self._run_guardrails(
                                final_content, effective_guardrails, "output",
                                _stream_merged_registry, attempt=output_retry_count,
                            )
                            # Guardrails passed, break out of retry loop
                            break
                        except GuardrailRejectedError as e:
                            output_retry_count += 1
                            if output_retry_count > max_retries:
                                # Max retries exceeded, raise blocked error
                                logger.warning(f"Output guardrails failed after {max_retries} retries: {e.reason}")
                                raise GuardrailBlockedError(
                                    f"Output guardrails failed after {max_retries} retries: {e.reason}",
                                    guardrail_name=e.guardrail_name,
                                ) from e

                            # Add rejected response to conversation (for context)
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": final_content,
                                }
                            )

                            # Append feedback message requesting revised response
                            feedback_message = (
                                f"Your previous response was rejected: {e.reason}. "
                                "Please provide a revised response that addresses this issue."
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": feedback_message,
                                }
                            )
                            logger.info(
                                f"Output guardrail rejected response (attempt {output_retry_count}/{max_retries}): "
                                f"{e.reason}. Requesting revised response."
                            )

                            # Call LLM again for revised response (no tools, just text response)
                            try:
                                response = await self._llm_call(
                                    _stream_merged_registry,
                                    messages=messages,
                                    model=model or self.model,
                                    tools=None,  # No tools on retry
                                    request_timeout=request_timeout,
                                    connect_timeout=connect_timeout,
                                    max_tokens=effective_max_tokens,
                                    retry_config=effective_retry_config,
                                    rate_limit_config=effective_rate_limit_config,
                                    **effective_stream_model_kwargs,
                                )
                            except LLMError as llm_error:
                                # LLM call failed during retry, raise blocked error
                                raise GuardrailBlockedError(
                                    f"Failed to get revised response after guardrail rejection: {llm_error}",
                                    guardrail_name=e.guardrail_name,
                                ) from llm_error

                            if not response.choices:
                                raise InvalidRequestError(
                                    "Empty response from LLM provider during guardrail retry"
                                ) from None

                            # Get the revised response
                            final_content = response.choices[0].message.content or ""
                            logger.debug(
                                f"Received revised response (length={len(final_content)}), re-running output guardrails"
                            )
                            # Continue loop to re-check guardrails

                # Stream the final response character by character (simulated streaming)
                # In a real implementation, you'd stream from the API
                await emit_status(
                    ProcessEvent(
                        kind="stream_end",
                        correlation_id=get_correlation_id(),
                        timestamp=time.time(),
                    ),
                    on_status,
                    sinks=sinks,
                )
                structured_output = None
                if response_format and final_content:
                    structured_output = _parse_structured_content(final_content, response_format)
                if final_content:
                    # Text was already streamed incrementally above; terminal chunk carries metadata only
                    self._conversation.add_message(Role.ASSISTANT, final_content)
                    yield StreamingChunk(
                        content="",
                        done=True,
                        tool_calls_made=tool_calls_made,
                        structured_output=structured_output,
                    )
                else:
                    yield StreamingChunk(
                        content="",
                        done=True,
                        tool_calls_made=tool_calls_made,
                        structured_output=structured_output,
                    )

                await record_stream_eval_success(
                    final_content,
                    structured_out=structured_output,
                )
                return

            # Max iterations reached
            logger.warning(f"Max tool execution iterations ({effective_stream_max_tool_iterations}) reached")
            yield StreamingChunk(
                content="Maximum tool execution iterations reached.",
                done=True,
                tool_calls_made=tool_calls_made,
            )
            await record_stream_eval_success("Maximum tool execution iterations reached.")

        except Exception as e:
            error = e
            raise
        finally:
            if error is not None and not eval_recorded:
                await _record_eval_data(
                    eval_store=effective_eval_store,
                    user_message=user_message,
                    system_prompt=system_prompt_content,
                    model=model or self.model,
                    messages_snapshot=messages_snapshot,
                    start_time=start_time,
                    error=error,
                    tools_available=self.tools,
                    on_eval_record_hooks=_stream_merged_registry.get_hooks(HookStage.PRE_EVAL_RECORD),
                    hook_manager=self._hook_manager,
                )
            clear_correlation_id()

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self._conversation = Conversation()

    async def embed(
        self,
        texts: str | list[str],
        model: str | None = None,
        correlation_id: str | None = None,
        *,
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        api_key: str | None = None,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> "EmbeddingResult":
        """Generate embeddings for the given text(s).

        Args:
            texts: Single text string or list of text strings to embed
            model: Model identifier (defaults to self.embedding_model)
            correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
            connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout); keyword-only.
            request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout); keyword-only.
            api_key: Optional API key override (for key pool usage)
            encoding_format: Optional format to return embeddings in (e.g., "float" or "base64").
                Provider-specific. Note: If using "base64", the embedding format may differ from
                the standard list[float] format.
            dimensions: Optional number of dimensions for the embedding output. Supported by some
                providers (e.g., OpenAI text-embedding-3-* models) to truncate vectors.
            **kwargs: Additional provider-specific arguments passed through to the embedding API.
                Examples: `user` (OpenAI) for end-user identification.

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
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            api_key=api_key,
            encoding_format=encoding_format,
            dimensions=dimensions,
            **kwargs,
        )


# Convenience functions for one-off requests


async def complete(
    user_message: str,
    # -- Core --
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    # -- Model generation --
    logprobs: bool | None = None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    top_logprobs: int | None = None,
    # -- Tools --
    condense_tool_messages: bool | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    parallel_tool_calls: bool | None = None,
    tool_execution_order: ToolExecutionOrder | None = None,
    tool_mode: ToolMode | None = None,
    tool_route_model: str | None = None,
    # -- Timeouts --
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    # -- Resilience --
    rate_limit_config: RateLimitConfig | None = None,
    retry_config: RetryConfig | None = None,
    retry_enabled: bool | None = None,
    # -- Context management --
    summarize_context: SummarizeContextConfig | bool | None = None,
    aaak_compression_enabled: bool | None = None,
    aaak_compression_model: str | None = None,
    aaak_tool_condensing: bool | None = None,
    # -- Observability / behavior --
    correlation_id: str | None = None,
    enable_eval_recording: bool | None = None,
    guardrails: GuardrailsConfig | None = None,
    hook_registry: HookRegistry | None = None,
    on_status: OnStatusCallback = None,
    session_label: str | None = None,
    sinks: list[Sink] | None = None,
    track_costs: bool | None = None,
    **model_kwargs: Any,
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
        request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
        guardrails: Optional guardrails configuration
        on_status: Optional callback for process status events
        max_tokens: Maximum number of tokens to generate. Required for Anthropic models.
        condense_tool_messages: If True, each completed tool-call round is condensed into a single
            assistant message, reducing context size across multi-iteration tool loops. Defaults to settings.default_condense_tool_messages.
        tool_mode: "standard" (all tools in prompt) or "dynamic" (router discovers tools). Defaults to settings.default_tool_mode.
        tool_route_model: Fast model for dynamic tool routing when tool_mode="dynamic" (defaults to settings.tool_route_model).
        retry_enabled: If False, disables retries for this call.
        retry_config: Optional retry configuration (set retry_config.callback for custom retry logic).
        rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
        track_costs: If False, skip cost tracking for this call (defaults to settings.track_costs).
        enable_eval_recording: If False, skip eval recording for this call (defaults to using instance eval_store).
        hook_registry: Optional hook registry for pre/post tool and iteration hooks.
        summarize_context: ``True``/``False``, or ``SummarizeContextConfig`` for summarization defaults.
        aaak_compression_enabled: When context summarization runs, use AAAK compression (defaults to settings).
        aaak_compression_model: Model for AAAK compression when enabled (defaults to settings).
        aaak_tool_condensing: Encode condensed tool rounds as AAAK ``[AT]`` blocks (defaults to settings).
        **model_kwargs: Extra params for acompletion (e.g. temperature, top_p).

    Returns:
        ToolExecutionResult with final response and execution history
    """
    effective_tool_mode = tool_mode if tool_mode is not None else settings.default_tool_mode
    effective_tool_execution_order = tool_execution_order if tool_execution_order is not None else settings.default_tool_execution_order
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        logprobs=logprobs,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs if model_kwargs else None,
        reasoning_effort=reasoning_effort,
        top_logprobs=top_logprobs,
        condense_tool_messages=condense_tool_messages,
        max_tool_iterations=max_tool_iterations,
        parallel_tool_calls=parallel_tool_calls,
        tool_execution_order=effective_tool_execution_order,
        tool_mode=effective_tool_mode,
        tool_route_model=tool_route_model,
        tools=tools,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        summarize_context=summarize_context,
        aaak_compression_enabled=aaak_compression_enabled,
        aaak_compression_model=aaak_compression_model,
        aaak_tool_condensing=aaak_tool_condensing,
        guardrails=guardrails,
        hook_registry=hook_registry,
        session_label=session_label,
    )
    return await client.complete(
        user_message,
        model=model,
        logprobs=logprobs,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        top_logprobs=top_logprobs,
        condense_tool_messages=condense_tool_messages,
        execute_tools=execute_tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_execution_order=tool_execution_order,
        tool_mode=tool_mode,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        retry_enabled=retry_enabled,
        summarize_context=summarize_context,
        correlation_id=correlation_id,
        enable_eval_recording=enable_eval_recording,
        guardrails=guardrails,
        on_status=on_status,
        session_label=session_label,
        sinks=sinks,
        track_costs=track_costs,
        **model_kwargs,
    )


async def structured_complete(
    user_message: str,
    response_format: type[T],
    # -- Core --
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    # -- Model generation --
    logprobs: bool | None = None,
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    top_logprobs: int | None = None,
    # -- Tools --
    condense_tool_messages: bool | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    parallel_tool_calls: bool | None = None,
    tool_execution_order: ToolExecutionOrder | None = None,
    tool_mode: ToolMode | None = None,
    tool_route_model: str | None = None,
    # -- Timeouts --
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    # -- Resilience --
    rate_limit_config: RateLimitConfig | None = None,
    retry_config: RetryConfig | None = None,
    retry_enabled: bool | None = None,
    # -- Context management --
    summarize_context: SummarizeContextConfig | bool | None = None,
    aaak_compression_enabled: bool | None = None,
    aaak_compression_model: str | None = None,
    aaak_tool_condensing: bool | None = None,
    # -- Observability / behavior --
    correlation_id: str | None = None,
    enable_eval_recording: bool | None = None,
    guardrails: GuardrailsConfig | None = None,
    max_validation_retries: int | None = None,
    on_status: OnStatusCallback = None,
    session_label: str | None = None,
    sinks: list[Sink] | None = None,
    hook_registry: HookRegistry | None = None,
    track_costs: bool | None = None,
    **model_kwargs: Any,
) -> ExecutionResult[T]:
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
        request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
        guardrails: Optional guardrails configuration
        on_status: Optional callback for process status events
        max_tokens: Maximum number of tokens to generate. Required for Anthropic models.
        condense_tool_messages: If True, each completed tool-call round is condensed into a single
            assistant message, reducing context size across multi-iteration tool loops. Defaults to settings.default_condense_tool_messages.
        tool_mode: "standard" (all tools in prompt) or "dynamic" (router discovers tools). Defaults to settings.default_tool_mode.
        tool_route_model: Fast model for dynamic tool routing when tool_mode="dynamic" (defaults to settings.tool_route_model).
        retry_enabled: If False, disables retries for this call.
        retry_config: Optional retry configuration (set retry_config.callback for custom retry logic).
        rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
        track_costs: If False, skip cost tracking for this call (defaults to settings.track_costs).
        enable_eval_recording: If False, skip eval recording for this call (defaults to using instance eval_store).
        max_validation_retries: Max retries when Pydantic validation fails (default 3).
        summarize_context: ``True``/``False``, or ``SummarizeContextConfig`` for summarization defaults.
        aaak_compression_enabled: When context summarization runs, use AAAK compression (defaults to settings).
        aaak_compression_model: Model for AAAK compression when enabled (defaults to settings).
        aaak_tool_condensing: Encode condensed tool rounds as AAAK ``[AT]`` blocks (defaults to settings).
        **model_kwargs: Extra params for acompletion (e.g. temperature, top_p).

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
    effective_tool_mode = tool_mode if tool_mode is not None else settings.default_tool_mode
    effective_tool_execution_order = tool_execution_order if tool_execution_order is not None else settings.default_tool_execution_order
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        logprobs=logprobs,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs if model_kwargs else None,
        reasoning_effort=reasoning_effort,
        top_logprobs=top_logprobs,
        condense_tool_messages=condense_tool_messages,
        max_tool_iterations=max_tool_iterations,
        parallel_tool_calls=parallel_tool_calls,
        tool_execution_order=effective_tool_execution_order,
        tool_mode=effective_tool_mode,
        tool_route_model=tool_route_model,
        tools=tools,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        summarize_context=summarize_context,
        aaak_compression_enabled=aaak_compression_enabled,
        aaak_compression_model=aaak_compression_model,
        aaak_tool_condensing=aaak_tool_condensing,
        guardrails=guardrails,
        hook_registry=hook_registry,
        session_label=session_label,
    )
    return await client.structured_complete(
        user_message,
        response_format,
        model=model,
        tools=tools,
        logprobs=logprobs,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        top_logprobs=top_logprobs,
        condense_tool_messages=condense_tool_messages,
        execute_tools=execute_tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_execution_order=tool_execution_order,
        tool_mode=tool_mode,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        retry_enabled=retry_enabled,
        summarize_context=summarize_context,
        correlation_id=correlation_id,
        enable_eval_recording=enable_eval_recording,
        guardrails=guardrails,
        max_validation_retries=max_validation_retries,
        on_status=on_status,
        session_label=session_label,
        sinks=sinks,
        track_costs=track_costs,
        **model_kwargs,
    )


async def structured_response(
    user_input: str | list[dict[str, Any]],
    response_format: type[T],
    # -- Core --
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    # -- Model generation --
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    # -- Tools --
    condense_tool_messages: bool | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    parallel_tool_calls: bool | None = None,
    tool_execution_order: ToolExecutionOrder | None = None,
    tool_mode: ToolMode | None = None,
    tool_route_model: str | None = None,
    # -- Timeouts --
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    # -- Resilience --
    rate_limit_config: RateLimitConfig | None = None,
    retry_config: RetryConfig | None = None,
    retry_enabled: bool | None = None,
    # -- Context management --
    summarize_context: SummarizeContextConfig | bool | None = None,
    aaak_compression_enabled: bool | None = None,
    aaak_compression_model: str | None = None,
    aaak_tool_condensing: bool | None = None,
    # -- Observability / behavior --
    correlation_id: str | None = None,
    enable_eval_recording: bool | None = None,
    guardrails: GuardrailsConfig | None = None,
    max_validation_retries: int | None = None,
    on_status: OnStatusCallback = None,
    sinks: list[Sink] | None = None,
    hook_registry: HookRegistry | None = None,
    track_costs: bool | None = None,
    **model_kwargs: Any,
) -> ExecutionResult[T]:
    """Quick structured completion via the OpenAI Responses API.

    The Responses-API analogue of :func:`structured_complete`. The LLM may
    optionally use tools to gather information before returning the final
    structured output; tools are executed in a loop just like the chat
    completions path. All context-management features (summarisation, AAAK
    condensing, validation retries, guardrails, hooks, eval recording, ...)
    are preserved by translating between the internal chat-completions
    message log and the Responses API ``input``/``instructions`` shape at
    the wire boundary.

    Args:
        user_input: The user's message as a string, or a list of typed
            Responses ``ResponseInputParam`` items (multimodal, prefilled
            tool history, ...).
        response_format: Pydantic model class for structured output.
        model: Model identifier in ``provider:model_name`` form.
        system_prompt: System/developer instructions content.
        tools: Optional list of callables to use as tools.
        execute_tools: Whether to automatically execute tools and loop.
        max_tool_iterations: Max iterations of the tool loop.
        max_tokens: Maximum number of tokens to generate (forwarded as
            ``max_output_tokens``).
        reasoning_effort: Wrapped into ``reasoning={"effort": ...}`` for
            the Responses API.
        max_validation_retries: Max retries when Pydantic validation fails.
        (other args mirror :func:`structured_complete`)
        **model_kwargs: Extra params for ``aresponses`` (e.g. ``temperature``,
            ``top_p``). Chat-completions-only params (``logprobs``,
            ``response_format``, ...) are dropped automatically.

    Returns:
        :class:`ExecutionResult` with ``structured_output`` populated.

    Example:
        >>> import asyncio
        >>> from gluellm.api import structured_response
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
        ...     # Without tools
        ...     result = await structured_response(
        ...         "What is 2+2?",
        ...         response_format=Answer,
        ...     )
        ...     print(result.structured_output.number)
        ...
        ...     # With tools
        ...     result = await structured_response(
        ...         "Use the calculator to compute 2+2 and explain.",
        ...         response_format=Answer,
        ...         tools=[get_calculator_result],
        ...     )
        ...     print(result.tool_calls_made)
        >>>
        >>> asyncio.run(main())
    """
    effective_tool_mode = tool_mode if tool_mode is not None else settings.default_tool_mode
    effective_tool_execution_order = (
        tool_execution_order if tool_execution_order is not None else settings.default_tool_execution_order
    )
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs if model_kwargs else None,
        reasoning_effort=reasoning_effort,
        condense_tool_messages=condense_tool_messages,
        max_tool_iterations=max_tool_iterations,
        parallel_tool_calls=parallel_tool_calls,
        tool_execution_order=effective_tool_execution_order,
        tool_mode=effective_tool_mode,
        tool_route_model=tool_route_model,
        tools=tools,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        summarize_context=summarize_context,
        aaak_compression_enabled=aaak_compression_enabled,
        aaak_compression_model=aaak_compression_model,
        aaak_tool_condensing=aaak_tool_condensing,
        guardrails=guardrails,
        hook_registry=hook_registry,
    )
    return await client.structured_response(
        user_input,
        response_format,
        model=model,
        tools=tools,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        condense_tool_messages=condense_tool_messages,
        execute_tools=execute_tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_execution_order=tool_execution_order,
        tool_mode=tool_mode,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        retry_enabled=retry_enabled,
        summarize_context=summarize_context,
        correlation_id=correlation_id,
        enable_eval_recording=enable_eval_recording,
        guardrails=guardrails,
        max_validation_retries=max_validation_retries,
        on_status=on_status,
        sinks=sinks,
        track_costs=track_costs,
        **model_kwargs,
    )


async def list_models(
    provider: str = "openai",
    api_key: str | None = None,
    timeout: float | None = None,
) -> Sequence[Any]:
    """List available models for a provider.

    Args:
        provider: Provider name (e.g. "openai", "anthropic").
        api_key: Optional API key override.
        timeout: Timeout in seconds for the request (defaults to
            settings.default_request_timeout).

    Returns:
        Sequence of Model objects with id, created, owned_by, etc.

    Example:
        >>> import asyncio
        >>> from gluellm.api import list_models
        >>>
        >>> async def main():
        ...     models = await list_models("openai")
        ...     for m in models[:5]:
        ...         print(m.id)
        >>>
        >>> asyncio.run(main())
    """
    if timeout is None:
        timeout = settings.default_request_timeout
    llm = AnyLLM.create(provider, api_key=api_key)
    return await asyncio.wait_for(asyncio.to_thread(llm.list_models), timeout=timeout)


async def embed(
    texts: str | list[str],
    model: str | None = None,
    correlation_id: str | None = None,
    *,
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    encoding_format: str | None = None,
    dimensions: int | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **kwargs: Any,
) -> "EmbeddingResult":
    """Quick embedding generation.

    Args:
        texts: Single text string or list of text strings to embed
        model: Model identifier (defaults to settings.default_embedding_model)
        correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
        connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout); keyword-only.
        request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout); keyword-only.
        encoding_format: Optional format to return embeddings in (e.g., "float" or "base64").
            Provider-specific. Note: If using "base64", the embedding format may differ from
            the standard list[float] format.
        dimensions: Optional number of dimensions for the embedding output. Supported by some
            providers (e.g., OpenAI text-embedding-3-* models) to truncate vectors.
        rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
        **kwargs: Additional provider-specific arguments passed through to the embedding API.
            Examples: `user` (OpenAI) for end-user identification.

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
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        encoding_format=encoding_format,
        dimensions=dimensions,
        rate_limit_config=rate_limit_config,
        **kwargs,
    )


@overload
async def stream_complete(
    user_message: str,
    # -- Core --
    model: str | None = ...,
    system_prompt: str | None = ...,
    tools: list[Callable] | None = ...,
    guardrails: GuardrailsConfig | None = ...,
    response_format: type[T] = ...,
    # -- Model generation --
    max_tokens: int | None = ...,
    # -- Tools --
    condense_tool_messages: bool | None = ...,
    execute_tools: bool = ...,
    max_tool_iterations: int | None = ...,
    tool_execution_order: ToolExecutionOrder | None = ...,
    tool_mode: ToolMode | None = ...,
    tool_route_model: str | None = ...,
    # -- Timeouts --
    connect_timeout: float | None = ...,
    request_timeout: float | None = ...,
    # -- Resilience --
    rate_limit_config: RateLimitConfig | None = ...,
    retry_config: RetryConfig | None = ...,
    retry_enabled: bool | None = ...,
    # -- Context management --
    summarize_context: SummarizeContextConfig | bool | None = ...,
    # -- Observability / behavior --
    correlation_id: str | None = ...,
    enable_eval_recording: bool | None = ...,
    on_status: OnStatusCallback = ...,
    sinks: list[Sink] | None = ...,
    track_costs: bool | None = ...,
    **model_kwargs: Any,
) -> AsyncIterator[StreamingChunk[T]]: ...


@overload
async def stream_complete(
    user_message: str,
    # -- Core --
    model: str | None = ...,
    system_prompt: str | None = ...,
    tools: list[Callable] | None = ...,
    guardrails: GuardrailsConfig | None = ...,
    response_format: None = ...,
    # -- Model generation --
    max_tokens: int | None = ...,
    # -- Tools --
    condense_tool_messages: bool | None = ...,
    execute_tools: bool = ...,
    max_tool_iterations: int | None = ...,
    tool_execution_order: ToolExecutionOrder | None = ...,
    tool_mode: ToolMode | None = ...,
    tool_route_model: str | None = ...,
    # -- Timeouts --
    connect_timeout: float | None = ...,
    request_timeout: float | None = ...,
    # -- Resilience --
    rate_limit_config: RateLimitConfig | None = ...,
    retry_config: RetryConfig | None = ...,
    retry_enabled: bool | None = ...,
    # -- Context management --
    summarize_context: SummarizeContextConfig | bool | None = ...,
    # -- Observability / behavior --
    correlation_id: str | None = ...,
    enable_eval_recording: bool | None = ...,
    on_status: OnStatusCallback = ...,
    sinks: list[Sink] | None = ...,
    track_costs: bool | None = ...,
    **model_kwargs: Any,
) -> AsyncIterator[StreamingChunk[Any]]: ...


async def stream_complete(
    user_message: str,
    # -- Core --
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    guardrails: GuardrailsConfig | None = None,
    response_format: type[T] | None = None,
    # -- Model generation --
    max_tokens: int | None = None,
    # -- Tools --
    condense_tool_messages: bool | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    tool_execution_order: ToolExecutionOrder | None = None,
    tool_mode: ToolMode | None = None,
    tool_route_model: str | None = None,
    # -- Timeouts --
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
    # -- Resilience --
    rate_limit_config: RateLimitConfig | None = None,
    retry_config: RetryConfig | None = None,
    retry_enabled: bool | None = None,
    # -- Context management --
    summarize_context: SummarizeContextConfig | bool | None = None,
    aaak_compression_enabled: bool | None = None,
    aaak_compression_model: str | None = None,
    aaak_tool_condensing: bool | None = None,
    # -- Observability / behavior --
    correlation_id: str | None = None,
    enable_eval_recording: bool | None = None,
    on_status: OnStatusCallback = None,
    sinks: list[Sink] | None = None,
    track_costs: bool | None = None,
    **model_kwargs: Any,
) -> AsyncIterator[StreamingChunk[Any]]:
    """Stream completion with automatic tool execution.

    Yields chunks of the response as they arrive. Note: tool execution
    interrupts streaming - when tools are called, streaming pauses until
    tool results are processed.

    When response_format is set, the final chunk may include structured_output
    (parsed Pydantic instance).

    Note:
        When tools are enabled, responses are streamed token-by-token; tool calls
        are detected from the stream and executed before the next turn. See
        GlueLLM.stream_complete() for details.

    Args:
        user_message: The user's message/request
        model: Model identifier in format "provider:model_name" (defaults to settings.default_model)
        system_prompt: System prompt content (defaults to settings.default_system_prompt)
        tools: List of callable functions to use as tools
        execute_tools: Whether to automatically execute tools
        max_tool_iterations: Maximum number of tool call iterations (defaults to settings.max_tool_iterations)
        correlation_id: Optional correlation ID for request tracking (auto-generated if not provided)
        guardrails: Optional guardrails configuration
        response_format: Optional Pydantic model; final chunk may include structured_output
        on_status: Optional callback for process status events
        request_timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        connect_timeout: Connection timeout in seconds (defaults to settings.default_connect_timeout)
        condense_tool_messages: If True, each completed tool-call round is condensed into a single
            assistant message, reducing context size across multi-iteration tool loops. Defaults to settings.default_condense_tool_messages.
        tool_mode: "standard" (all tools in prompt) or "dynamic" (router discovers tools). Defaults to settings.default_tool_mode.
        tool_route_model: Fast model for dynamic tool routing when tool_mode="dynamic" (defaults to settings.tool_route_model).
        retry_enabled: If False, disables retries for this call.
        retry_config: Optional retry configuration (set retry_config.callback for custom retry logic).
        rate_limit_config: Per-call rate limit configuration override (use ``algorithm=`` for algorithm).
        track_costs: If False, skip cost tracking for this call (defaults to settings.track_costs).
        max_tokens: Optional maximum completion tokens per LLM call (None = provider default; not all models support this).
        summarize_context: ``True``/``False``, or ``SummarizeContextConfig`` for summarization defaults.
        aaak_compression_enabled: When context summarization runs, use AAAK compression (defaults to settings).
        aaak_compression_model: Model for AAAK compression when enabled (defaults to settings).
        aaak_tool_condensing: Encode condensed tool rounds as AAAK ``[AT]`` blocks (defaults to settings).
        **model_kwargs: Extra params for acompletion (e.g. temperature, top_p), same as ``complete``.

    Yields:
        StreamingChunk objects with content and metadata (and optional structured_output on the final chunk)

    Example:
        >>> async for chunk in stream_complete("Tell me a story"):
        ...     print(chunk.content, end="", flush=True)
        ...     if chunk.done:
        ...         print(f"\\nTool calls: {chunk.tool_calls_made}")
    """
    effective_tool_mode = tool_mode if tool_mode is not None else settings.default_tool_mode
    effective_tool_execution_order = tool_execution_order if tool_execution_order is not None else settings.default_tool_execution_order
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs if model_kwargs else None,
        condense_tool_messages=condense_tool_messages,
        max_tool_iterations=max_tool_iterations,
        tool_execution_order=effective_tool_execution_order,
        tool_mode=effective_tool_mode,
        tool_route_model=tool_route_model,
        tools=tools,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        summarize_context=summarize_context,
        aaak_compression_enabled=aaak_compression_enabled,
        aaak_compression_model=aaak_compression_model,
        aaak_tool_condensing=aaak_tool_condensing,
        guardrails=guardrails,
    )
    async for chunk in client.stream_complete(
        user_message,
        model=model,
        guardrails=guardrails,
        response_format=response_format,
        max_tokens=max_tokens,
        condense_tool_messages=condense_tool_messages,
        execute_tools=execute_tools,
        tool_execution_order=tool_execution_order,
        tool_mode=tool_mode,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        rate_limit_config=rate_limit_config,
        retry_config=retry_config,
        retry_enabled=retry_enabled,
        summarize_context=summarize_context,
        correlation_id=correlation_id,
        enable_eval_recording=enable_eval_recording,
        on_status=on_status,
        sinks=sinks,
        track_costs=track_costs,
        **model_kwargs,
    ):
        yield chunk
