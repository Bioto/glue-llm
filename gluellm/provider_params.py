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

    return max_tokens, kwargs
