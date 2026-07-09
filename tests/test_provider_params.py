"""Tests for provider-specific parameter normalization."""

import pytest

from gluellm.provider_params import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    _normalize_openai_reasoning_effort,
    _update_kwargs_for_provider_reasoning_effort,
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


class TestReasoningEffortNormalization:
    @pytest.mark.parametrize(
        "model,effort,expected",
        [
            ("o3-mini", "xhigh", "high"),
            ("o3-mini", "high", "high"),
            ("o3-mini", "minimal", "low"),
            ("gpt-5-mini-2025-08-07", "xhigh", "high"),
            ("gpt-5.4-2026-03-05", "xhigh", "xhigh"),
            ("gpt-5.1-2025-11-13", "xhigh", "xhigh"),
        ],
    )
    async def test_openai_reasoning_effort_downgraded_to_supported(self, model, effort, expected):
        assert _normalize_openai_reasoning_effort(model, effort) == expected

    async def test_responses_api_wraps_reasoning_effort(self):
        kwargs = _update_kwargs_for_provider_reasoning_effort(
            "openai",
            "openai:gpt-5.4-2026-03-05",
            "high",
            {},
            use_responses_api=True,
        )
        assert kwargs == {"reasoning": {"effort": "high"}}
        assert "reasoning_effort" not in kwargs

    async def test_chat_api_sets_flat_reasoning_effort(self):
        kwargs = _update_kwargs_for_provider_reasoning_effort(
            "openai",
            "openai:o3-mini",
            "xhigh",
            {},
            use_responses_api=False,
        )
        assert kwargs == {"reasoning_effort": "high"}

    async def test_responses_kwargs_include_reasoning_summary_when_set(self):
        """reasoning_summary merges into Responses reasoning dict alongside effort."""
        kwargs = _update_kwargs_for_provider_reasoning_effort(
            "openai",
            "openai:o4-mini",
            "high",
            {},
            use_responses_api=True,
            reasoning_summary="auto",
        )
        assert kwargs == {"reasoning": {"effort": "high", "summary": "auto"}}
        assert "reasoning_summary" not in kwargs
        assert "reasoning_effort" not in kwargs

    async def test_responses_reasoning_summary_preserves_caller_reasoning_dict(self):
        """Caller-provided reasoning.summary is kept when reasoning_summary is unset."""
        kwargs = _update_kwargs_for_provider_reasoning_effort(
            "openai",
            "openai:o4-mini",
            "medium",
            {"reasoning": {"summary": "detailed", "effort": "low"}},
            use_responses_api=True,
            reasoning_summary=None,
        )
        assert kwargs["reasoning"]["effort"] == "medium"
        assert kwargs["reasoning"]["summary"] == "detailed"

    async def test_responses_reasoning_summary_only_without_effort(self):
        """Summary can be set without effort on the Responses path."""
        kwargs = _update_kwargs_for_provider_reasoning_effort(
            "openai",
            "openai:o4-mini",
            None,
            {},
            use_responses_api=True,
            reasoning_summary="concise",
        )
        assert kwargs == {"reasoning": {"summary": "concise"}}
