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

# Models that support reasoning_effort (o1, o3, o4-mini; not o1-mini or gpt-4o-mini)
_REASONING_EFFORT_MODELS_RE = re.compile(
    r"^o1(?!-mini)(?:-|$)|^o3|^o4",
    re.IGNORECASE,
)


def _model_supports_reasoning_effort(model: str) -> bool:
    """Return True if the model supports the reasoning_effort parameter."""
    model_name = model.split(":", 1)[1] if ":" in model else model
    return bool(_REASONING_EFFORT_MODELS_RE.search(model_name))


def normalize_model_params(
    model: str,
    max_tokens: int | None,
    extra_kwargs: dict[str, Any],
) -> tuple[int | None, dict[str, Any]]:
    """Normalize model params for the target provider.

    - Anthropic: max_tokens is required — inject default if caller omitted it
    - OpenAI o-series (o1, o3, o4-mini, …): use max_completion_tokens, not max_tokens
    - OpenAI: strip reasoning_effort for models that don't support it (e.g. gpt-4o-mini)
    - All other providers: pass through unchanged

    Returns:
        (final_max_tokens, kwargs) — kwargs may contain max_completion_tokens for o-series
    """
    provider = extract_provider_from_model(model)
    model_name = model.split(":", 1)[1] if ":" in model else model
    kwargs = dict(extra_kwargs)

    # reasoning_effort is only supported by o1, o3, o4-mini (not gpt-4o-mini, o1-mini, etc.)
    if "reasoning_effort" in kwargs and not _model_supports_reasoning_effort(model):
        kwargs.pop("reasoning_effort", None)

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
