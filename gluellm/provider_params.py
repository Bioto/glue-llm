"""Provider-specific parameter normalization.

Translates the unified gluellm interface into provider-specific params before
the API call — zero wasted round-trips for known param quirks.
"""

import re
from typing import Any

from gluellm.config import settings
from gluellm.rate_limiting.api_key_pool import extract_provider_from_model

ANTHROPIC_DEFAULT_MAX_TOKENS = 8192

# OpenAI models that use max_completion_tokens instead of max_tokens (o-series, gpt-5, gpt-4.1, etc.)
_MAX_COMPLETION_TOKENS_MODELS_RE = re.compile(
    r"^(o\d|gpt-5|gpt-4\.1)",
    re.IGNORECASE,
)

_REASONING_EFFORT_ORDER: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

# gpt-5.1+ (and gpt-5.4, etc.) support the full effort range including xhigh.
_GPT_51_PLUS_RE = re.compile(r"^gpt-5\.(?:[1-9]|[1-9][0-9])", re.IGNORECASE)


def _openai_supported_reasoning_efforts(model_name: str) -> tuple[str, ...]:
    """Return supported reasoning effort values for an OpenAI model name."""
    name = model_name.lower()
    if re.match(r"^o\d", name):
        return ("low", "medium", "high")
    if _GPT_51_PLUS_RE.match(name):
        return _REASONING_EFFORT_ORDER
    if name.startswith("gpt-5"):
        return ("none", "minimal", "low", "medium", "high")
    return _REASONING_EFFORT_ORDER


def _normalize_openai_reasoning_effort(model_name: str, effort: str) -> str:
    """Downgrade unsupported reasoning effort to the nearest lower supported value."""
    supported = _openai_supported_reasoning_efforts(model_name)
    if effort in supported:
        return effort
    if effort not in _REASONING_EFFORT_ORDER:
        return supported[0]
    effort_idx = _REASONING_EFFORT_ORDER.index(effort)
    for candidate in reversed(_REASONING_EFFORT_ORDER[: effort_idx + 1]):
        if candidate in supported:
            return candidate
    return supported[0]


def _update_kwargs_for_provider_reasoning_effort(
    provider: str,
    model: str,
    effort: str | None,
    kwargs: dict[str, Any],
    *,
    use_responses_api: bool = False,
    reasoning_summary: str | None = None,
) -> dict[str, Any]:
    """Normalize and inject provider-specific reasoning effort/summary kwargs.

    For the Responses API, builds ``reasoning={"effort": ..., "summary": ...}``.
    ``reasoning_summary`` is only meaningful on the Responses path; chat
    completions ignore it. An existing ``reasoning`` dict in kwargs (with a
    caller-provided ``summary``) is preserved when merging.
    """
    if effort is None and reasoning_summary is None:
        return kwargs

    model_name = model.split(":", 1)[1] if ":" in model else model.split("/", 1)[-1]
    updated = dict(kwargs)
    existing_reasoning = updated.get("reasoning")
    existing_summary: str | None = None
    if isinstance(existing_reasoning, dict):
        summary_val = existing_reasoning.get("summary")
        if isinstance(summary_val, str):
            existing_summary = summary_val

    updated.pop("reasoning_effort", None)
    updated.pop("reasoning_summary", None)

    if use_responses_api:
        reasoning: dict[str, Any] = {}
        if isinstance(existing_reasoning, dict):
            reasoning.update(existing_reasoning)
        if effort is not None:
            normalized = effort
            if provider == "openai":
                normalized = _normalize_openai_reasoning_effort(model_name, effort)
            reasoning["effort"] = normalized
        # Explicit reasoning_summary wins; else keep caller-provided summary on reasoning dict
        summary = reasoning_summary if reasoning_summary is not None else existing_summary
        if summary is not None:
            reasoning["summary"] = summary
        if reasoning:
            updated["reasoning"] = reasoning
        elif "reasoning" in updated and not isinstance(updated["reasoning"], dict):
            updated.pop("reasoning", None)
    else:
        updated.pop("reasoning", None)
        if effort is not None:
            normalized = effort
            if provider == "openai":
                normalized = _normalize_openai_reasoning_effort(model_name, effort)
            updated["reasoning_effort"] = normalized
    return updated


def normalize_model_params(
    model: str,
    max_tokens: int | None,
    extra_kwargs: dict[str, Any],
) -> tuple[int | None, dict[str, Any]]:
    """Normalize model params for the target provider.

    - Anthropic: max_tokens is required — inject default if caller omitted it
    - OpenAI o-series (o1, o3, o4-mini, …): use max_completion_tokens, not max_tokens
    - All other providers: pass through unchanged

    Returns:
        (final_max_tokens, kwargs) — kwargs may contain max_completion_tokens for o-series
    """
    provider = extract_provider_from_model(model)
    model_name = model.split(":", 1)[1] if ":" in model else model
    kwargs = dict(extra_kwargs)

    if provider == "anthropic":
        if max_tokens is None:
            max_tokens = settings.default_max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS
    elif provider == "openai" and _MAX_COMPLETION_TOKENS_MODELS_RE.match(model_name):
        if max_tokens is not None:
            kwargs.setdefault("max_completion_tokens", max_tokens)
            max_tokens = None
    elif provider == "gemini":
        kwargs["timeout"] = settings.default_request_timeout  # Gemini SDK doesn't support httpx.Timeout, so we set it here for consistency

    return max_tokens, kwargs
