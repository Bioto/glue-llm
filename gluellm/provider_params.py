"""Provider-specific parameter normalization.

Translates the unified gluellm interface into provider-specific params before
the API call — zero wasted round-trips for known param quirks.
"""

import re
from collections.abc import Sequence
from logging import getLogger
from typing import Any, cast

from gluellm.config import settings
from gluellm.models.agent import REASONING_EFFORTS
from gluellm.rate_limiting.api_key_pool import extract_provider_from_model

ANTHROPIC_DEFAULT_MAX_TOKENS = 8192
logger = getLogger(__name__)

# OpenAI models that use max_completion_tokens instead of max_tokens (o-series, gpt-5, gpt-4.1, etc.)
_MAX_COMPLETION_TOKENS_MODELS_RE = re.compile(
    r"^(o\d|gpt-5|gpt-4\.1)",
    re.IGNORECASE,
)
_OPENAI_REASONING_MODEL_RE = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)
_OPENAI_GPT5_MINOR_RE = re.compile(r"^gpt-5(?:\.(\d+))?(?:-|$)", re.IGNORECASE)
_OPENAI_GPT51_REASONING_EFFORTS = ["none", "low", "medium", "high"]
_OPENAI_PRE_GPT51_REASONING_EFFORTS = ["minimal", "low", "medium", "high"]


def _model_name(model: str) -> str:
    if ":" in model:
        return model.split(":", 1)[1]
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _openai_supported_reasoning_efforts(model_name: str) -> Sequence[str]:
    normalized_model = model_name.lower()
    if normalized_model.startswith("gpt-5-pro"):
        return ["high"]

    version_match = _OPENAI_GPT5_MINOR_RE.match(normalized_model)
    if version_match:
        minor = version_match.group(1)
        if minor is not None and int(minor) == 1:
            return _OPENAI_GPT51_REASONING_EFFORTS
        if minor is not None and int(minor) > 1:
            return cast("Sequence[str]", REASONING_EFFORTS)
        return _OPENAI_PRE_GPT51_REASONING_EFFORTS

    if _OPENAI_REASONING_MODEL_RE.match(normalized_model):
        return _OPENAI_PRE_GPT51_REASONING_EFFORTS

    return []


def _closest_lower_supported_effort(
    effort: str,
    supported_efforts: Sequence[str],
) -> str | None:
    """Walk back GlueLLM effort order to find the closest lower supported value."""
    gluellm_reasoning_efforts = cast("Sequence[str]", REASONING_EFFORTS)
    if effort not in gluellm_reasoning_efforts:
        return None

    requested_index = gluellm_reasoning_efforts.index(effort)
    supported = set(supported_efforts)
    for candidate in reversed(gluellm_reasoning_efforts[:requested_index]):
        if candidate in supported:
            return candidate
    return None


def normalize_reasoning_effort_for_provider(model: str, effort: str) -> str | None:
    """Normalize reasoning effort to values supported by the target provider/model."""
    provider = extract_provider_from_model(model)
    if provider != "openai":
        return effort

    model_name = _model_name(model)
    supported_efforts = _openai_supported_reasoning_efforts(model_name)
    if not supported_efforts:
        logger.warning(
            "OpenAI model %s does not support reasoning_effort; omitting unsupported effort %s",
            model_name,
            effort,
        )
        return None

    if effort in supported_efforts:
        return effort

    mapped_effort = _closest_lower_supported_effort(effort, supported_efforts)
    if mapped_effort is None and effort not in cast("Sequence[str]", REASONING_EFFORTS):
        return effort
    if mapped_effort is None:
        logger.warning(
            "OpenAI model %s does not support reasoning effort %s and no lower supported GlueLLM effort exists; omitting it",
            model_name,
            effort,
        )
        return None

    logger.warning(
        "OpenAI model %s does not support reasoning effort %s; using %s instead",
        model_name,
        effort,
        mapped_effort,
    )
    return mapped_effort


def _normalize_openai_reasoning_effort(model_name: str, effort: str) -> str:
    """Normalize effort for a bare OpenAI model name (used in unit tests)."""
    normalized = normalize_reasoning_effort_for_provider(f"openai:{model_name}", effort)
    if normalized is None:
        supported = _openai_supported_reasoning_efforts(model_name)
        return supported[0] if supported else effort
    return normalized


def _update_kwargs_for_provider_reasoning_effort(
    provider: str,
    model: str,
    effort: str | None,
    kwargs: dict[str, Any],
    *,
    use_responses_api: bool = False,
    reasoning_summary: str | None = None,
) -> dict[str, Any]:
    """Normalize and inject provider-specific reasoning effort/summary kwargs."""
    if effort is None and reasoning_summary is None:
        return kwargs

    model_name = _model_name(model)
    updated = dict(kwargs)
    existing_reasoning = updated.get("reasoning")
    existing_summary: str | None = None
    if isinstance(existing_reasoning, dict):
        summary_val = existing_reasoning.get("summary")
        if isinstance(summary_val, str):
            existing_summary = summary_val

    updated.pop("reasoning_effort", None)
    updated.pop("reasoning_summary", None)

    def _normalize_effort(raw_effort: str) -> str | None:
        if provider == "openai":
            return normalize_reasoning_effort_for_provider(model, raw_effort)
        return raw_effort

    if use_responses_api:
        reasoning: dict[str, Any] = {}
        if isinstance(existing_reasoning, dict):
            reasoning.update(existing_reasoning)
        if effort is not None:
            normalized = _normalize_effort(effort)
            if normalized is not None:
                reasoning["effort"] = normalized
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
            normalized = _normalize_effort(effort)
            if normalized is not None:
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
    model_name = _model_name(model)
    kwargs = dict(extra_kwargs)

    reasoning_effort = kwargs.get("reasoning_effort")
    if isinstance(reasoning_effort, str):
        normalized_reasoning_effort = normalize_reasoning_effort_for_provider(model, reasoning_effort)
        if normalized_reasoning_effort is None:
            kwargs.pop("reasoning_effort", None)
        else:
            kwargs["reasoning_effort"] = normalized_reasoning_effort

    if provider == "anthropic":
        if max_tokens is None:
            max_tokens = settings.default_max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS
    elif provider == "openai" and _MAX_COMPLETION_TOKENS_MODELS_RE.match(model_name):
        if max_tokens is not None:
            kwargs.setdefault("max_completion_tokens", max_tokens)
            max_tokens = None
    elif provider == "gemini":
        kwargs["timeout"] = settings.default_request_timeout

    return max_tokens, kwargs
