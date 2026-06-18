"""Model fallback chain support for GlueLLM LLM calls."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from pydantic import BaseModel, Field

from gluellm.events import ProcessEvent, StatusEmitter

from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Set during call_with_model_fallback to the model that succeeded.
successful_call_model: ContextVar[str | None] = ContextVar("successful_call_model", default=None)


def _default_fallback_on() -> list[type[Exception]]:
    from gluellm.api import (
        APIConnectionError,
        AuthenticationError,
        RateLimitError,
        TokenLimitError,
    )

    return [RateLimitError, TokenLimitError, AuthenticationError, APIConnectionError]


DEFAULT_FALLBACK_ON: tuple[type[Exception], ...] = ()  # populated lazily via _default_fallback_on


class ModelFallbackConfig(BaseModel):
    """Ordered model fallback chain when the primary model fails.

    The primary model is always tried first (from ``GlueLLM.model`` or per-call
    ``model=``). ``models`` lists additional models to try in order when a
    configured error type is raised after retries are exhausted.

    Attributes:
        models: Fallback models in priority order (primary is not included).
        fallback_on: Exception types that trigger advancing to the next model.
        retry_per_model: When True, each model receives the full ``RetryConfig``
            attempt budget before falling back (default).
    """

    models: list[str] = Field(default_factory=list, min_length=0)
    fallback_on: list[type[Exception]] = Field(default_factory=_default_fallback_on)
    retry_per_model: bool = True

    model_config = {"arbitrary_types_allowed": True}


def resolve_fallback_chain(
    primary_model: str,
    fallback_config: ModelFallbackConfig | None = None,
    fallback_models: list[str] | None = None,
) -> list[str] | None:
    """Return ordered model chain ``[primary, *fallbacks]`` or None if no fallbacks."""
    fallbacks: list[str] = []
    if fallback_models is not None:
        fallbacks = list(fallback_models)
    elif fallback_config is not None and fallback_config.models:
        fallbacks = list(fallback_config.models)

    if not fallbacks:
        return None

    chain = [primary_model]
    for model in fallbacks:
        if model not in chain:
            chain.append(model)
    return chain


def _should_fallback(error: BaseException, fallback_on: list[type[Exception]]) -> bool:
    return isinstance(error, tuple(fallback_on))


async def _emit_model_fallback(
    status: StatusEmitter | None,
    *,
    from_model: str,
    to_model: str,
    error_type: str,
    correlation_id: str | None = None,
) -> None:
    if status is None:
        return
    import time

    await status.emit(
        ProcessEvent(
            kind="model_fallback",
            model=to_model,
            from_model=from_model,
            to_model=to_model,
            error_type=error_type,
            correlation_id=correlation_id,
            timestamp=time.time(),
        )
    )


async def call_with_model_fallback(
    call_fn: Callable[..., Awaitable[Any]],
    *,
    primary_model: str,
    fallback_config: ModelFallbackConfig | None = None,
    fallback_models: list[str] | None = None,
    status: StatusEmitter | None = None,
    correlation_id: str | None = None,
    **call_kwargs: Any,
) -> Any:
    """Try ``call_fn`` with primary model, then fallbacks on configured errors.

    Args:
        call_fn: Async callable (e.g. ``_llm_call_with_retry``) accepting ``model=``.
        primary_model: First model to attempt.
        fallback_config: Instance-level fallback configuration.
        fallback_models: Per-call fallback override (takes precedence over config).
        status: Optional status emitter for ``model_fallback`` events.
        correlation_id: Correlation ID for status events.
        **call_kwargs: Forwarded to ``call_fn`` (except ``model``).

    Returns:
        Result from the first successful ``call_fn`` invocation.

    Raises:
        The last exception if all models in the chain fail.
    """
    chain = resolve_fallback_chain(primary_model, fallback_config, fallback_models)
    if chain is None:
        invoke_kwargs = dict(call_kwargs)
        invoke_kwargs.setdefault("model", primary_model)
        result = await call_fn(**invoke_kwargs)
        successful_call_model.set(invoke_kwargs["model"])
        return result

    fallback_on = (
        fallback_config.fallback_on
        if fallback_config is not None
        else _default_fallback_on()
    )

    last_error: BaseException | None = None
    for index, model in enumerate(chain):
        try:
            invoke_kwargs = dict(call_kwargs)
            invoke_kwargs["model"] = model
            result = await call_fn(**invoke_kwargs)
            successful_call_model.set(model)
            return result
        except Exception as e:
            last_error = e
            is_last = index + 1 >= len(chain)
            if is_last or not _should_fallback(e, fallback_on):
                raise
            next_model = chain[index + 1]
            logger.warning(
                "Model fallback: %s failed (%s), trying %s",
                model,
                type(e).__name__,
                next_model,
            )
            await _emit_model_fallback(
                status,
                from_model=model,
                to_model=next_model,
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )

    assert last_error is not None
    raise last_error
