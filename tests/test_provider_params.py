"""Tests for provider-specific parameter normalization."""

import pytest

from gluellm.provider_params import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    normalize_model_params,
)

pytestmark = pytest.mark.asyncio


class TestNormalizeModelParams:
    """Tests for normalize_model_params."""

    @pytest.mark.parametrize(
        "model,max_tokens,extra_kwargs,expected_max_tokens,expected_max_completion_tokens",
        [
            # Anthropic: max_tokens None → injected
            (
                "anthropic:claude-3-5-sonnet-20241022",
                None,
                {},
                8192,
                None,
            ),
            # Anthropic: max_tokens set → unchanged
            (
                "anthropic:claude-3-5-sonnet-20241022",
                4096,
                {},
                4096,
                None,
            ),
            # OpenAI o-series: max_tokens → swapped to max_completion_tokens
            (
                "openai:o3-mini",
                4096,
                {},
                None,
                4096,
            ),
            (
                "openai:o1-mini",
                8192,
                {},
                None,
                8192,
            ),
            (
                "openai:o1-preview",
                2048,
                {"temperature": 0.5},
                None,
                2048,
            ),
            # OpenAI gpt-5: same as o-series, uses max_completion_tokens
            (
                "openai:gpt-5-mini-2025-08-07",
                4096,
                {},
                None,
                4096,
            ),
            # OpenAI o-series: max_tokens None → no max_completion_tokens added
            (
                "openai:o3-mini",
                None,
                {},
                None,
                None,
            ),
            # OpenAI standard: unchanged
            (
                "openai:gpt-4o",
                4096,
                {},
                4096,
                None,
            ),
            (
                "openai:gpt-5.4-2026-03-05",
                None,
                {},
                None,
                None,
            ),
        ],
    )
    async def test_normalize_model_params(
        self,
        model: str,
        max_tokens: int | None,
        extra_kwargs: dict,
        expected_max_tokens: int | None,
        expected_max_completion_tokens: int | None,
    ):
        """Parametrized cases for provider param normalization."""
        result_max_tokens, result_kwargs = normalize_model_params(
            model, max_tokens, extra_kwargs
        )
        assert result_max_tokens == expected_max_tokens
        if expected_max_completion_tokens is not None:
            assert result_kwargs.get("max_completion_tokens") == expected_max_completion_tokens
        else:
            assert "max_completion_tokens" not in result_kwargs
        # Extra kwargs preserved (except any we add)
        for k, v in extra_kwargs.items():
            assert result_kwargs.get(k) == v

    async def test_anthropic_uses_settings_default_max_tokens_when_available(
        self, monkeypatch
    ):
        """Anthropic with max_tokens None uses settings.default_max_tokens if set."""
        monkeypatch.setattr(
            "gluellm.provider_params.settings.default_max_tokens", 1024
        )
        result_max, _ = normalize_model_params("anthropic:claude-3-5-sonnet", None, {})
        assert result_max == 1024

    async def test_anthropic_fallback_when_settings_default_none(self, monkeypatch):
        """Anthropic with max_tokens None and settings.default_max_tokens None uses ANTHROPIC_DEFAULT_MAX_TOKENS."""
        monkeypatch.setattr(
            "gluellm.provider_params.settings.default_max_tokens", None
        )
        result_max, _ = normalize_model_params("anthropic:claude-3-5-sonnet", None, {})
        assert result_max == ANTHROPIC_DEFAULT_MAX_TOKENS

    async def test_reasoning_effort_stripped_for_unsupported_models(self):
        """reasoning_effort is removed when model does not support it (e.g. gpt-5.4-2026-03-05)."""
        _, kwargs = normalize_model_params(
            "openai:gpt-5.4-2026-03-05", None, {"reasoning_effort": "high"}
        )
        assert "reasoning_effort" not in kwargs

    async def test_reasoning_effort_preserved_for_supported_models(self):
        """reasoning_effort is preserved for o1, o3, o4-mini."""
        for model in ("openai:o1", "openai:o3-mini", "openai:o4-mini", "openai:o4"):
            _, kwargs = normalize_model_params(model, None, {"reasoning_effort": "high"})
            assert kwargs.get("reasoning_effort") == "high", f"Failed for {model}"

    async def test_reasoning_effort_stripped_for_o1_mini(self):
        """reasoning_effort is not supported by o1-mini."""
        _, kwargs = normalize_model_params(
            "openai:o1-mini", None, {"reasoning_effort": "high"}
        )
        assert "reasoning_effort" not in kwargs
