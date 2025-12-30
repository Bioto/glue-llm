"""Embedding generation for GlueLLM.

This module provides embedding generation capabilities with error handling,
rate limiting, cost tracking, and observability, following the same patterns
as the completion API.
"""

import asyncio
import hashlib
import logging
import os
import time
from contextlib import contextmanager

from any_llm import aembedding as any_llm_aembedding
from openai.types.create_embedding_response import CreateEmbeddingResponse
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gluellm.api import (
    APIConnectionError,
    RateLimitError,
    classify_llm_error,
)
from gluellm.config import settings
from gluellm.costing.pricing_data import calculate_embedding_cost
from gluellm.models.embedding import EmbeddingResult
from gluellm.observability.logging_config import get_logger
from gluellm.rate_limiting.rate_limiter import acquire_rate_limit
from gluellm.runtime.context import clear_correlation_id, get_correlation_id, set_correlation_id
from gluellm.runtime.shutdown import ShutdownContext, is_shutting_down
from gluellm.telemetry import (
    is_tracing_enabled,
    log_llm_metrics,
    record_token_usage,
)

# Configure logging
logger = get_logger(__name__)

# Mapping of provider names to their API key environment variables
PROVIDER_ENV_VAR_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}


def _extract_provider_from_embedding_model(model: str) -> str:
    """Extract provider name from embedding model string.

    Embedding models use '/' separator (e.g., "openai/text-embedding-3-small")
    while completion models use ':' separator (e.g., "openai:gpt-4o").

    Args:
        model: Model string in format "provider/model_name" or "provider:model_name"

    Returns:
        Provider name (e.g., "openai", "anthropic", "xai")
    """
    if "/" in model:
        return model.split("/")[0].lower()
    if ":" in model:
        return model.split(":")[0].lower()
    # Default to openai if no provider specified
    return "openai"


def _extract_model_name_from_embedding_model(model: str) -> str:
    """Extract model name from embedding model string.

    Args:
        model: Model string in format "provider/model_name" or "provider:model_name"

    Returns:
        Model name without provider prefix
    """
    if "/" in model:
        return model.split("/", 1)[1]
    if ":" in model:
        return model.split(":", 1)[1]
    return model


@contextmanager
def _temporary_api_key(model: str, api_key: str | None):
    """Context manager for temporarily setting an API key in the environment.

    Temporarily sets the appropriate environment variable for the given provider,
    and restores the original value (or removes it) when the context exits.

    Args:
        model: Model identifier in format "provider/model_name" or "provider:model_name"
        api_key: The API key to set temporarily, or None to skip

    Yields:
        None
    """
    if not api_key:
        yield
        return

    provider = _extract_provider_from_embedding_model(model)
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


def _extract_token_usage(response: CreateEmbeddingResponse) -> dict[str, int] | None:
    """Extract token usage from an embedding response.

    Args:
        response: The CreateEmbeddingResponse object from the embedding API

    Returns:
        Dictionary with 'prompt' (input tokens) and 'total' token counts,
        or None if usage information is not available.
    """
    if not hasattr(response, "usage") or not response.usage:
        return None

    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    return {
        "prompt": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0,
        "completion": 0,  # Embeddings don't have completion tokens
        "total": int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
    }


async def _safe_embedding_call(
    model: str,
    inputs: str | list[str],
    timeout: float | None = None,
    api_key: str | None = None,
) -> CreateEmbeddingResponse:
    """Make an embedding call with error classification and tracing.

    This wraps the any_llm_aembedding call to catch and classify errors,
    and optionally trace the call with OpenTelemetry.
    Raises our custom exception types for better error handling.

    Args:
        model: Model identifier (e.g., "openai/text-embedding-3-small")
        inputs: Single text string or list of text strings to embed
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        api_key: Optional API key override (for key pool usage)

    Returns:
        CreateEmbeddingResponse from the embedding API

    Raises:
        asyncio.TimeoutError: If the request exceeds the timeout
        LLMError: Various LLM-related errors (rate limit, auth, etc.)
    """
    correlation_id = get_correlation_id()
    timeout = timeout or settings.default_request_timeout
    timeout = min(timeout, settings.max_request_timeout)  # Enforce max timeout

    # Apply rate limiting before making the call
    provider = _extract_provider_from_embedding_model(model)
    rate_limit_key = (
        f"global:{provider}" if not api_key else f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"
    )
    await acquire_rate_limit(rate_limit_key)

    start_time = time.time()
    input_count = len(inputs) if isinstance(inputs, list) else 1
    logger.debug(
        f"Making embedding call: model={model}, input_count={input_count}, "
        f"timeout={timeout}s, correlation_id={correlation_id}"
    )

    try:
        # Use tracing context if enabled
        span_name = f"embedding.{provider}"
        span_attributes = {
            "llm.model": model,
            "llm.provider": provider,
            "llm.input_count": input_count,
        }
        if correlation_id:
            span_attributes["correlation_id"] = correlation_id

        if is_tracing_enabled():
            from gluellm.telemetry import _tracer

            if _tracer is not None:
                span = _tracer.start_as_current_span(span_name, attributes=span_attributes)
                span.__enter__()
            else:
                span = None
        else:
            span = None

        try:
            # Use context manager for temporary API key
            with _temporary_api_key(model, api_key):
                # Make embedding call with timeout
                response = await asyncio.wait_for(
                    any_llm_aembedding(
                        model=model,
                        inputs=inputs,
                    ),
                    timeout=timeout,
                )

            elapsed_time = time.time() - start_time

            # Extract token usage
            tokens_used = _extract_token_usage(response)
            if tokens_used:
                # Calculate cost for this embedding call
                provider = _extract_provider_from_embedding_model(model)
                model_name = _extract_model_name_from_embedding_model(model)
                call_cost = calculate_embedding_cost(
                    provider=provider,
                    model_name=model_name,
                    input_tokens=tokens_used.get("prompt", 0),
                )

                # Record tokens and cost to span
                if span:
                    record_token_usage(span, tokens_used, cost_usd=call_cost)

                cost_str = f", cost=${call_cost:.6f}" if call_cost else ""
                logger.info(
                    f"Embedding call completed: model={model}, latency={elapsed_time:.3f}s, "
                    f"tokens={tokens_used['total']}, embeddings={len(response.data)}{cost_str}"
                )

            # Log metrics to MLflow
            log_llm_metrics(
                model=model,
                latency=elapsed_time,
                tokens_used=tokens_used,
                finish_reason=None,
                has_tool_calls=False,
                error=False,
            )

            return response

        finally:
            if span:
                span.__exit__(None, None, None)

    except TimeoutError:
        elapsed_time = time.time() - start_time
        logger.error(
            f"Embedding call timed out after {elapsed_time:.3f}s (timeout={timeout}s): model={model}, "
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
            f"Embedding call failed after {elapsed_time:.3f}s: model={model}, error={classified_error}, "
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
async def _embedding_call_with_retry(
    model: str,
    inputs: str | list[str],
    timeout: float | None = None,
    api_key: str | None = None,
) -> CreateEmbeddingResponse:
    """Make an embedding call with automatic retry on transient errors.

    Retries up to configured attempts with exponential backoff for:
    - Rate limit errors (429)
    - Connection errors (5xx)

    Does NOT retry for:
    - Token limit errors (need to reduce input)
    - Authentication errors (bad credentials)
    - Invalid request errors (bad parameters)
    - Timeout errors

    Args:
        model: Model identifier
        inputs: Single text string or list of text strings to embed
        timeout: Request timeout in seconds (defaults to settings.default_request_timeout)
        api_key: Optional API key override (for key pool usage)
    """
    return await _safe_embedding_call(
        model=model,
        inputs=inputs,
        timeout=timeout,
        api_key=api_key,
    )


async def embed(
    texts: str | list[str],
    model: str | None = None,
    correlation_id: str | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> EmbeddingResult:
    """Generate embeddings for the given text(s).

    This is the main function for generating embeddings. It handles error classification,
    retries, rate limiting, cost tracking, and observability.

    Args:
        texts: Single text string or list of text strings to embed
        model: Model identifier (defaults to settings.default_embedding_model)
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
        >>> from gluellm.embeddings import embed
        >>>
        >>> async def main():
        ...     # Single text
        ...     result = await embed("Hello, world!")
        ...     print(f"Embedding dimension: {result.dimension}")
        ...
        ...     # Multiple texts
        ...     result = await embed(["Hello", "World"])
        ...     print(f"Generated {result.count} embeddings")
        >>>
        >>> asyncio.run(main())
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
    model = model or settings.default_embedding_model

    input_count = len(texts) if isinstance(texts, list) else 1
    logger.info(
        f"Starting embedding request: correlation_id={correlation_id}, model={model}, input_count={input_count}"
    )

    # Use shutdown context to track in-flight requests
    try:
        with ShutdownContext():
            # Make embedding call with retry
            response = await _embedding_call_with_retry(
                model=model,
                inputs=texts,
                timeout=timeout,
                api_key=api_key,
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            # Sort by index to ensure correct order
            embeddings = sorted(
                zip([item.index for item in response.data], embeddings, strict=True), key=lambda x: x[0]
            )
            embeddings = [emb for _, emb in embeddings]

            # Extract token usage
            tokens_used = _extract_token_usage(response)
            total_tokens = tokens_used.get("total", 0) if tokens_used else 0

            # Calculate cost
            provider = _extract_provider_from_embedding_model(model)
            model_name = _extract_model_name_from_embedding_model(model)
            estimated_cost = calculate_embedding_cost(
                provider=provider,
                model_name=model_name,
                input_tokens=total_tokens,
            )

            # Record to session tracker
            # Note: We use the private _session_tracker directly because _calculate_and_record_cost
            # uses calculate_cost() which doesn't support embedding models. We've already
            # calculated the cost using calculate_embedding_cost().
            if tokens_used and estimated_cost is not None:
                from gluellm.api import _session_tracker

                # Convert model format from "provider/model" to "provider:model" for compatibility
                normalized_model = model.replace("/", ":") if "/" in model else model
                _session_tracker.record_usage(
                    model=normalized_model,
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    cost_usd=estimated_cost,
                )

            cost_str = f"${estimated_cost:.6f}" if estimated_cost is not None else "N/A"
            logger.debug(
                f"Embedding generation completed: model={model}, embeddings={len(embeddings)}, "
                f"dimension={len(embeddings[0]) if embeddings else 0}, tokens={total_tokens}, "
                f"cost={cost_str}"
            )

            return EmbeddingResult(
                embeddings=embeddings,
                model=model,
                tokens_used=total_tokens,
                estimated_cost_usd=estimated_cost,
            )
    finally:
        # Clear correlation ID after request completes
        clear_correlation_id()
