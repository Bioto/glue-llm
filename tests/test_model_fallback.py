"""Tests for model fallback chains."""

from unittest.mock import AsyncMock, patch

import pytest

from gluellm.api import GlueLLM, RateLimitError, RetryConfig
from gluellm.events import ProcessEvent
from gluellm.resilience.fallback import ModelFallbackConfig, resolve_fallback_chain


class TestResolveFallbackChain:
    def test_returns_none_when_no_fallbacks(self):
        assert resolve_fallback_chain("openai:gpt-4", None, None) is None

    def test_per_call_override(self):
        chain = resolve_fallback_chain("openai:gpt-4", None, ["anthropic:claude-3"])
        assert chain == ["openai:gpt-4", "anthropic:claude-3"]

    def test_deduplicates_models(self):
        cfg = ModelFallbackConfig(models=["openai:gpt-4", "anthropic:claude-3"])
        chain = resolve_fallback_chain("openai:gpt-4", cfg, None)
        assert chain == ["openai:gpt-4", "anthropic:claude-3"]


@pytest.mark.asyncio
async def test_fallback_advances_on_rate_limit_after_retries_exhausted():
    """Primary model exhausts retries; fallback model succeeds."""
    call_models: list[str] = []

    async def fake_llm_call(**kwargs):
        call_models.append(kwargs["model"])
        if kwargs["model"] == "openai:primary":
            raise RateLimitError("rate limited")
        return _fake_chat_response("ok from fallback")

    client = GlueLLM(
        model="openai:primary",
        retry_config=RetryConfig(retry_enabled=False, max_attempts=1),
    )

    with patch.object(client, "_llm_call", new=AsyncMock(side_effect=fake_llm_call)):
        # Bypass _llm_call wrapper - test call_with_model_fallback via complete internals
        pass

    from gluellm.resilience.fallback import call_with_model_fallback

    await call_with_model_fallback(
        fake_llm_call,
        primary_model="openai:primary",
        fallback_models=["openai:fallback"],
        retry_config=RetryConfig(retry_enabled=False),
    )
    assert call_models == ["openai:primary", "openai:fallback"]


@pytest.mark.asyncio
async def test_fallback_does_not_advance_on_invalid_request():
    from gluellm.api import InvalidRequestError
    from gluellm.resilience.fallback import call_with_model_fallback

    async def fake_llm_call(**kwargs):
        raise InvalidRequestError("bad request")

    with pytest.raises(InvalidRequestError):
        await call_with_model_fallback(
            fake_llm_call,
            primary_model="openai:primary",
            fallback_models=["openai:fallback"],
        )


@pytest.mark.asyncio
async def test_fallback_records_successful_model_in_execution_result():
    client = GlueLLM(
        model="openai:primary",
        retry_config=RetryConfig(retry_enabled=False, max_attempts=1),
    )

    async def fake_safe_llm_call(*, model, **kwargs):
        if model == "openai:primary":
            raise RateLimitError("rate limited")
        return _fake_chat_response("hello")

    with patch("gluellm.api._llm_call_with_retry", new=AsyncMock(side_effect=fake_safe_llm_call)):
        result = await client.complete(
            "hi",
            fallback_models=["openai:fallback"],
            execute_tools=False,
        )

    assert result.model == "openai:fallback"
    assert result.final_response == "hello"


@pytest.mark.asyncio
async def test_fallback_emits_model_fallback_status_event():
    events: list[ProcessEvent] = []

    async def on_status(event: ProcessEvent) -> None:
        events.append(event)

    client = GlueLLM(
        model="openai:primary",
        retry_config=RetryConfig(retry_enabled=False, max_attempts=1),
    )

    async def fake_safe_llm_call(*, model, **kwargs):
        if model == "openai:primary":
            raise RateLimitError("rate limited")
        return _fake_chat_response("ok")

    with patch("gluellm.api._llm_call_with_retry", new=AsyncMock(side_effect=fake_safe_llm_call)):
        await client.complete(
            "hi",
            fallback_models=["openai:fallback"],
            execute_tools=False,
            on_status=on_status,
        )

    fallback_events = [e for e in events if e.kind == "model_fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0].from_model == "openai:primary"
    assert fallback_events[0].to_model == "openai:fallback"


@pytest.mark.asyncio
async def test_responses_path_uses_same_fallback_wrapper():
    client = GlueLLM(
        model="openai:primary",
        retry_config=RetryConfig(retry_enabled=False, max_attempts=1),
    )

    async def fake_responses_call(*, model, **kwargs):
        if model == "openai:primary":
            raise RateLimitError("rate limited")
        return _fake_responses_response("responses ok")

    with patch("gluellm.api._responses_call_with_retry", new=AsyncMock(side_effect=fake_responses_call)):
        result = await client.response(
            "hi",
            fallback_models=["openai:fallback"],
            execute_tools=False,
        )

    assert result.model == "openai:fallback"
    assert "responses ok" in result.final_response


def _fake_chat_response(content: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="test",
    )


def _fake_responses_response(text: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        id="resp_test123",
        output=[],
        output_text=text,
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        model="test",
    )
