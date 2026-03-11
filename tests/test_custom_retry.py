"""Tests for custom retry logic: RetryConfig, RetryCallback, retry_enabled, model_kwargs."""

import asyncio
import os
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from gluellm.api import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    GlueLLM,
    InvalidRequestError,
    RateLimitError,
    RetryConfig,
    TokenLimitError,
    complete,
    structured_complete,
)

pytestmark = pytest.mark.asyncio

# Skip integration tests if no API key
requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; run with real API for integration tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str = "OK") -> Mock:
    """Build a minimal mock ChatCompletion."""
    msg = Mock(content=content, tool_calls=None)
    choice = Mock(message=msg)
    return Mock(choices=[choice], usage=None)


# ===========================================================================
# Unit tests – all mocked, no network
# ===========================================================================


class TestRetryConfigDefaults:
    """RetryConfig field defaults and validation."""

    async def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.retry_enabled is True
        assert cfg.max_attempts == 3
        assert cfg.min_wait == 2.0
        assert cfg.max_wait == 30.0
        assert cfg.multiplier == 1.0
        assert cfg.callback is None

    async def test_retry_disabled(self):
        cfg = RetryConfig(retry_enabled=False)
        assert cfg.retry_enabled is False

    async def test_custom_values(self):
        cfg = RetryConfig(max_attempts=5, min_wait=0.5, max_wait=10.0, multiplier=2.0)
        assert cfg.max_attempts == 5
        assert cfg.min_wait == 0.5
        assert cfg.max_wait == 10.0
        assert cfg.multiplier == 2.0

    async def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=0)

    async def test_callback_field_accepts_callable(self):
        def cb(exc, attempt):
            return True, None
        cfg = RetryConfig(callback=cb)
        assert cfg.callback is cb

    async def test_callback_field_accepts_async_callable(self):
        async def cb(exc, attempt):
            return True, None
        cfg = RetryConfig(callback=cb)
        assert cfg.callback is cb


# ---------------------------------------------------------------------------


class TestRetryDisabled:
    """retry_enabled=False enforces a single attempt."""

    @patch("gluellm.api._safe_llm_call")
    async def test_instance_retry_disabled_no_retries_on_rate_limit(self, mock):
        mock.side_effect = RateLimitError("429")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(retry_enabled=False))
        with pytest.raises(RateLimitError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_instance_retry_disabled_no_retries_on_connection_error(self, mock):
        mock.side_effect = APIConnectionError("network fail")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(retry_enabled=False))
        with pytest.raises(APIConnectionError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_per_call_retry_enabled_false_overrides_enabled_instance(self, mock):
        """Per-call retry_enabled=False beats instance config that has retries on."""
        mock.side_effect = RateLimitError("429")
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(retry_enabled=True, max_attempts=5),
        )
        with pytest.raises(RateLimitError):
            await client.complete("Test", retry_enabled=False)
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_standalone_complete_retry_disabled(self, mock):
        mock.side_effect = RateLimitError("429")
        with pytest.raises(RateLimitError):
            await complete("Test", retry_enabled=False)
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_standalone_structured_complete_retry_disabled(self, mock):
        mock.side_effect = RateLimitError("429")

        class Out(BaseModel):
            v: int

        with pytest.raises(RateLimitError):
            await structured_complete("Test", response_format=Out, retry_enabled=False)
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_retry_disabled_preserves_callback_from_config(self, mock):
        """When retry_enabled=False is applied, the callback on the config is preserved
        but never called (since no retry happens)."""
        invocations = []

        def cb(exc, attempt):
            invocations.append(attempt)
            return True, None

        mock.side_effect = RateLimitError("429")
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(callback=cb),
        )
        with pytest.raises(RateLimitError):
            await client.complete("Test", retry_enabled=False)
        assert mock.call_count == 1
        assert len(invocations) == 0


# ---------------------------------------------------------------------------


class TestRetryOn:
    """retry_on: list[type[Exception]] — only retry on listed error types."""

    @patch("gluellm.api._safe_llm_call")
    async def test_retries_on_listed_error(self, mock):
        """Retries when error matches a type in retry_on."""
        mock.side_effect = [RateLimitError("429"), _make_response()]
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, retry_on=[RateLimitError]),
        )
        result = await client.complete("Test")
        assert mock.call_count == 2
        assert result.final_response == "OK"

    @patch("gluellm.api._safe_llm_call")
    async def test_does_not_retry_unlisted_error(self, mock):
        """Does not retry when error is not in retry_on, even if it would be retried by default."""
        mock.side_effect = APIConnectionError("network fail")
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, retry_on=[RateLimitError]),  # connection errors excluded
        )
        with pytest.raises(APIConnectionError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_multiple_types_in_retry_on(self, mock):
        """Retries on any listed type."""
        mock.side_effect = [APIConnectionError("net"), RateLimitError("429"), _make_response()]
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, max_attempts=3, retry_on=[RateLimitError, APIConnectionError]),
        )
        result = await client.complete("Test")
        assert mock.call_count == 3
        assert result.final_response == "OK"

    @patch("gluellm.api._safe_llm_call")
    async def test_retry_on_does_not_affect_non_matching_errors(self, mock):
        """TokenLimitError is not retried even if retry_on=[RateLimitError]."""
        mock.side_effect = TokenLimitError("too long")
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, retry_on=[RateLimitError]),
        )
        with pytest.raises(TokenLimitError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_takes_precedence_over_retry_on(self, mock):
        """When both callback and retry_on are set, callback wins."""
        callback_called = []
        mock.side_effect = [RateLimitError("429"), _make_response()]

        def cb(exc, attempt):
            callback_called.append(attempt)
            return True, None

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(
                min_wait=0,
                retry_on=[TokenLimitError],  # would block RateLimitError
                callback=cb,                  # callback overrides and allows it
            ),
        )
        result = await client.complete("Test")
        assert result.final_response == "OK"
        assert len(callback_called) == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_retry_on_subclass_matches(self, mock):
        """retry_on works with isinstance, so subclasses match parent types."""
        mock.side_effect = [APITimeoutError("timeout"), _make_response()]
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, retry_on=[APIConnectionError]),  # APITimeoutError is a subclass
        )
        result = await client.complete("Test")
        assert mock.call_count == 2
        assert result.final_response == "OK"


class TestDefaultRetryBehavior:
    """When no callback is set, default logic retries RateLimit and Connection errors only."""

    @patch("gluellm.api._safe_llm_call")
    async def test_retries_on_rate_limit(self, mock):
        mock.side_effect = [RateLimitError("429"), RateLimitError("429"), _make_response()]
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, max_attempts=3))
        result = await client.complete("Test")
        assert mock.call_count == 3
        assert result.final_response == "OK"

    @patch("gluellm.api._safe_llm_call")
    async def test_retries_on_connection_error(self, mock):
        mock.side_effect = [APIConnectionError("net"), _make_response()]
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, max_attempts=3))
        result = await client.complete("Test")
        assert mock.call_count == 2
        assert result.final_response == "OK"

    @patch("gluellm.api._safe_llm_call")
    async def test_does_not_retry_token_limit(self, mock):
        mock.side_effect = TokenLimitError("too long")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0))
        with pytest.raises(TokenLimitError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_does_not_retry_auth_error(self, mock):
        mock.side_effect = AuthenticationError("bad key")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0))
        with pytest.raises(AuthenticationError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_does_not_retry_invalid_request(self, mock):
        mock.side_effect = InvalidRequestError("bad params")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0))
        with pytest.raises(InvalidRequestError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_max_attempts_respected(self, mock):
        mock.side_effect = RateLimitError("429")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, max_attempts=2))
        with pytest.raises(RateLimitError):
            await client.complete("Test")
        assert mock.call_count == 2

    @patch("gluellm.api._safe_llm_call")
    async def test_raises_after_all_attempts_exhausted(self, mock):
        mock.side_effect = APIConnectionError("network")
        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, max_attempts=4))
        with pytest.raises(APIConnectionError):
            await client.complete("Test")
        assert mock.call_count == 4


# ---------------------------------------------------------------------------


class TestRetryCallback:
    """Callback controls retry decisions and can mutate model kwargs."""

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_retries_on_matching_error(self, mock):
        mock.side_effect = [RateLimitError("429"), _make_response()]

        def cb(exc, attempt):
            return isinstance(exc, RateLimitError), None

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        result = await client.complete("Test")
        assert mock.call_count == 2
        assert result.final_response == "OK"

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_aborts_on_non_matching_error(self, mock):
        """Callback returns False for a non-rate-limit error → single attempt."""
        mock.side_effect = TokenLimitError("too long")

        def cb(exc, attempt):
            return isinstance(exc, RateLimitError), None  # only retry rate limits

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        with pytest.raises(TokenLimitError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_abort_false_stops_retries(self, mock):
        """Callback returning (False, ...) stops retrying even when attempts remain."""
        mock.side_effect = RateLimitError("429")

        def cb(exc, attempt):
            return False, None

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, max_attempts=5, callback=cb))
        with pytest.raises(RateLimitError):
            await client.complete("Test")
        assert mock.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_injects_temperature_on_retry(self, mock):
        mock.side_effect = [RateLimitError("429"), _make_response()]

        def cb(exc, attempt):
            return True, {"temperature": 0.0}

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        await client.complete("Test")

        assert mock.call_count == 2
        assert mock.call_args_list[1][1].get("temperature") == 0.0

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_params_accumulate_across_retries(self, mock):
        """next_params from successive callback invocations accumulate correctly."""
        temps = [0.8, 0.4]
        mock.side_effect = [RateLimitError("429"), RateLimitError("429"), _make_response()]

        def cb(exc, attempt):
            return True, {"temperature": temps[attempt - 1]}

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, max_attempts=3, callback=cb),
        )
        await client.complete("Test")
        assert mock.call_count == 3
        assert mock.call_args_list[1][1].get("temperature") == 0.8
        assert mock.call_args_list[2][1].get("temperature") == 0.4

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_receives_correct_attempt_numbers(self, mock):
        """Callback receives attempt=1 on first retry, attempt=2 on second, etc."""
        invocations: list[int] = []
        mock.side_effect = [RateLimitError("429"), RateLimitError("429"), _make_response()]

        def cb(exc, attempt):
            invocations.append(attempt)
            return True, None

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, max_attempts=3, callback=cb),
        )
        await client.complete("Test")
        assert invocations == [1, 2]

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_receives_original_exception(self, mock):
        """Callback receives the actual exception, not a wrapper."""
        received_exc: list[Exception] = []
        mock.side_effect = [RateLimitError("specific message"), _make_response()]

        def cb(exc, attempt):
            received_exc.append(exc)
            return True, None

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        await client.complete("Test")
        assert len(received_exc) == 1
        assert isinstance(received_exc[0], RateLimitError)
        assert "specific message" in str(received_exc[0])

    @patch("gluellm.api._safe_llm_call")
    async def test_async_callback_is_supported(self, mock):
        """Async callbacks work the same as sync ones."""
        mock.side_effect = [RateLimitError("429"), _make_response()]

        async def cb(exc, attempt):
            await asyncio.sleep(0)  # yield control
            return True, {"temperature": 0.5}

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        result = await client.complete("Test")
        assert result.final_response == "OK"
        assert mock.call_args_list[1][1].get("temperature") == 0.5

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_not_called_on_success(self, mock):
        """Callback is never invoked when first attempt succeeds."""
        invocations: list[int] = []
        mock.return_value = _make_response()

        def cb(exc, attempt):
            invocations.append(attempt)
            return True, None

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        await client.complete("Test")
        assert mock.call_count == 1
        assert len(invocations) == 0

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_on_per_call_retry_config(self, mock):
        """Per-call retry_config with callback overrides instance config."""
        instance_cb_called = []
        per_call_cb_called = []

        def instance_cb(exc, attempt):
            instance_cb_called.append(attempt)
            return True, None

        def per_call_cb(exc, attempt):
            per_call_cb_called.append(attempt)
            return True, None

        mock.side_effect = [RateLimitError("429"), _make_response()]
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            retry_config=RetryConfig(min_wait=0, callback=instance_cb),
        )
        await client.complete("Test", retry_config=RetryConfig(min_wait=0, callback=per_call_cb))

        assert len(per_call_cb_called) == 1
        assert len(instance_cb_called) == 0


# ---------------------------------------------------------------------------


class TestModelKwargs:
    """model_kwargs pass-through to provider.acompletion."""

    @patch("gluellm.api._safe_llm_call")
    async def test_instance_model_kwargs_forwarded(self, mock):
        mock.return_value = _make_response()
        client = GlueLLM(model="openai:gpt-4o-mini", model_kwargs={"temperature": 0.5, "top_p": 0.9})
        await client.complete("Test")
        kw = mock.call_args[1]
        assert kw["temperature"] == 0.5
        assert kw["top_p"] == 0.9

    @patch("gluellm.api._safe_llm_call")
    async def test_per_call_kwargs_override_instance_kwargs(self, mock):
        mock.return_value = _make_response()
        client = GlueLLM(model="openai:gpt-4o-mini", model_kwargs={"temperature": 0.3})
        await client.complete("Test", temperature=0.9)
        assert mock.call_args[1]["temperature"] == 0.9

    @patch("gluellm.api._safe_llm_call")
    async def test_instance_kwargs_merged_with_per_call(self, mock):
        """Instance kwargs that are not overridden remain in the final call."""
        mock.return_value = _make_response()
        client = GlueLLM(model="openai:gpt-4o-mini", model_kwargs={"temperature": 0.3, "top_p": 0.8})
        await client.complete("Test", temperature=0.9)
        kw = mock.call_args[1]
        assert kw["temperature"] == 0.9  # overridden
        assert kw["top_p"] == 0.8       # carried from instance

    @patch("gluellm.api._safe_llm_call")
    async def test_no_model_kwargs_by_default(self, mock):
        mock.return_value = _make_response()
        client = GlueLLM(model="openai:gpt-4o-mini")
        await client.complete("Test")
        kw = mock.call_args[1]
        assert "temperature" not in kw
        assert "top_p" not in kw

    @patch("gluellm.api._safe_llm_call")
    async def test_standalone_complete_forwards_kwargs(self, mock):
        mock.return_value = _make_response()
        await complete("Test", temperature=0.1, top_p=0.95)
        kw = mock.call_args[1]
        assert kw["temperature"] == 0.1
        assert kw["top_p"] == 0.95

    @patch("gluellm.api._safe_llm_call")
    async def test_callback_injected_kwargs_used_on_retry_attempt(self, mock):
        """model kwargs set by the callback are present on retry calls."""
        mock.side_effect = [RateLimitError("429"), _make_response()]

        def cb(exc, attempt):
            return True, {"temperature": 0.0, "top_p": 0.5}

        client = GlueLLM(model="openai:gpt-4o-mini", retry_config=RetryConfig(min_wait=0, callback=cb))
        await client.complete("Test")

        retry_kw = mock.call_args_list[1][1]
        assert retry_kw["temperature"] == 0.0
        assert retry_kw["top_p"] == 0.5


# ===========================================================================
# Integration tests – skipped unless OPENAI_API_KEY is set
# ===========================================================================


@requires_api_key
class TestCustomRetryIntegration:
    """End-to-end tests against real OpenAI API.

    Run all:
        OPENAI_API_KEY=sk-... uv run pytest tests/test_custom_retry.py -v -k integration

    See callback working:
        OPENAI_API_KEY=sk-... uv run pytest tests/test_custom_retry.py -v -k integration -s
    """

    # ------------------------------------------------------------------
    # retry_enabled / retry_config
    # ------------------------------------------------------------------

    async def test_complete_with_retry_disabled(self):
        """retry_enabled=False succeeds on a normal call without retrying."""
        result = await complete(
            "Reply with the single word: PONG",
            system_prompt="You are a bot. Reply with only the exact word asked, no punctuation.",
            retry_enabled=False,
        )
        assert "PONG" in result.final_response.upper()
        assert result.tool_calls_made == 0

    async def test_complete_with_retry_config_instance(self):
        """RetryConfig on GlueLLM instance works normally on successful calls."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a bot. Reply with only the exact word asked, no punctuation.",
            retry_config=RetryConfig(retry_enabled=True, max_attempts=3, min_wait=0.1, max_wait=1.0),
        )
        result = await client.complete("Reply with the single word: PONG")
        assert "PONG" in result.final_response.upper()

    async def test_retry_config_disabled_via_per_call_flag(self):
        """Per-call retry_enabled=False overrides instance RetryConfig."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a bot. Reply with only the exact word asked, no punctuation.",
            retry_config=RetryConfig(retry_enabled=True, max_attempts=5),
        )
        result = await client.complete("Reply with the single word: PONG", retry_enabled=False)
        assert "PONG" in result.final_response.upper()

    # ------------------------------------------------------------------
    # model_kwargs (temperature, max_tokens, etc.)
    # ------------------------------------------------------------------

    async def test_temperature_zero_is_deterministic(self):
        result = await complete(
            "Reply with only the number 42, nothing else.",
            system_prompt="You are a precise assistant. Reply with only the value asked for, no other text.",
            temperature=0.0,
        )
        assert "42" in result.final_response

    async def test_max_tokens_truncates_response(self):
        result = await complete("Write a 500-word essay about the ocean.", max_tokens=10)
        assert len(result.final_response.split()) <= 20

    async def test_model_kwargs_on_instance(self):
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a precise assistant. Reply with only the value asked for, no other text.",
            model_kwargs={"temperature": 0.0},
        )
        result = await client.complete("Reply with only the number 99, nothing else.")
        assert "99" in result.final_response

    async def test_per_call_model_kwargs_override_instance(self):
        """Per-call temperature=0.0 overrides instance temperature=1.5."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a precise assistant. Reply with only the value asked for, no other text.",
            model_kwargs={"temperature": 1.5},
        )
        result = await client.complete(
            "Reply with only the number 77, nothing else.",
            temperature=0.0,
        )
        assert "77" in result.final_response

    # ------------------------------------------------------------------
    # callback — invocation and param mutation
    # ------------------------------------------------------------------

    async def test_callback_not_invoked_on_success(self):
        """Callback is never called when the first attempt succeeds."""
        invocations: list[int] = []

        def cb(exc, attempt):
            invocations.append(attempt)
            return False, None

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a precise assistant. Reply with only the value asked for, no other text.",
            retry_config=RetryConfig(callback=cb),
        )
        await client.complete("Reply with only the number 1, nothing else.")
        assert len(invocations) == 0

    async def test_callback_changes_temperature_on_retry(self):
        """Callback injects temperature=0.0 on retry; confirms kwargs per attempt."""
        import gluellm.api as api_module

        attempts_kwargs: list[dict] = []
        original = api_module._safe_llm_call

        async def capturing(*args, **kwargs):
            attempts_kwargs.append(dict(kwargs))
            if len(attempts_kwargs) == 1:
                raise RateLimitError("simulated 429")
            return await original(*args, **kwargs)

        def cb(exc, attempt):
            print(f"  [cb] attempt={attempt} → temperature=0.0")
            return True, {"temperature": 0.0}

        api_module._safe_llm_call = capturing
        try:
            client = GlueLLM(
                model="openai:gpt-4o-mini",
                system_prompt="You are a precise assistant. Reply with only the value asked for.",
                model_kwargs={"temperature": 0.9},
                retry_config=RetryConfig(callback=cb),
            )
            result = await client.complete("Reply with only the number 55, nothing else.")
        finally:
            api_module._safe_llm_call = original

        assert "55" in result.final_response
        assert len(attempts_kwargs) == 2
        assert attempts_kwargs[0]["temperature"] == 0.9
        assert attempts_kwargs[1]["temperature"] == 0.0

    async def test_callback_changes_max_tokens_on_retry(self):
        """Callback injects max_tokens on retry."""
        import gluellm.api as api_module

        attempts_kwargs: list[dict] = []
        original = api_module._safe_llm_call

        async def capturing(*args, **kwargs):
            attempts_kwargs.append(dict(kwargs))
            if len(attempts_kwargs) == 1:
                raise RateLimitError("simulated 429")
            return await original(*args, **kwargs)

        def cb(exc, attempt):
            print(f"  [cb] attempt={attempt} → max_tokens=50")
            return True, {"max_tokens": 50}

        api_module._safe_llm_call = capturing
        try:
            client = GlueLLM(
                model="openai:gpt-4o-mini",
                retry_config=RetryConfig(callback=cb),
            )
            result = await client.complete("Say hello briefly.")
        finally:
            api_module._safe_llm_call = original

        assert len(result.final_response) > 0
        assert attempts_kwargs[0].get("max_tokens") is None
        assert attempts_kwargs[1]["max_tokens"] == 50

    async def test_callback_temperature_steps_down_across_multiple_retries(self):
        """Temperature steps 0.9 → 0.5 → 0.0 across three attempts."""
        import gluellm.api as api_module

        attempts_kwargs: list[dict] = []
        original = api_module._safe_llm_call
        temps = [0.9, 0.5, 0.0]

        async def capturing(*args, **kwargs):
            attempts_kwargs.append(dict(kwargs))
            if len(attempts_kwargs) < 3:
                raise RateLimitError(f"simulated 429 #{len(attempts_kwargs)}")
            return await original(*args, **kwargs)

        def cb(exc, attempt):
            t = temps[attempt] if attempt < len(temps) else 0.0
            print(f"  [cb] attempt={attempt} → temperature={t}")
            return True, {"temperature": t}

        api_module._safe_llm_call = capturing
        try:
            client = GlueLLM(
                model="openai:gpt-4o-mini",
                system_prompt="You are a precise assistant. Reply with only the value asked for.",
                model_kwargs={"temperature": 0.9},
                retry_config=RetryConfig(max_attempts=4, callback=cb),
            )
            result = await client.complete("Reply with only the number 33, nothing else.")
        finally:
            api_module._safe_llm_call = original

        assert "33" in result.final_response
        assert len(attempts_kwargs) == 3
        assert attempts_kwargs[0]["temperature"] == 0.9
        assert attempts_kwargs[1]["temperature"] == 0.5
        assert attempts_kwargs[2]["temperature"] == 0.0

    async def test_callback_abort_stops_retries(self):
        """Callback returning False stops retrying after the first failure."""
        import gluellm.api as api_module

        call_count = 0
        original = api_module._safe_llm_call

        async def always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RateLimitError("simulated 429")

        def cb(exc, attempt):
            print(f"  [cb] attempt={attempt} → aborting")
            return False, None

        api_module._safe_llm_call = always_fail
        try:
            client = GlueLLM(
                model="openai:gpt-4o-mini",
                retry_config=RetryConfig(max_attempts=5, callback=cb),
            )
            with pytest.raises(RateLimitError):
                await client.complete("Say hello.")
        finally:
            api_module._safe_llm_call = original

        assert call_count == 1

    # ------------------------------------------------------------------
    # structured_complete
    # ------------------------------------------------------------------

    async def test_structured_complete_with_retry_disabled(self):
        class Answer(BaseModel):
            value: int = Field(description="The result of 3+4")

        result = await structured_complete(
            "What is 3+4? Put the integer result in the value field.",
            response_format=Answer,
            retry_enabled=False,
        )
        assert result.structured_output is not None
        assert result.structured_output.value == 7

    async def test_structured_complete_with_temperature(self):
        class Flag(BaseModel):
            yes: bool = Field(description="Whether 2+2 equals 4")

        result = await structured_complete(
            "Does 2+2 equal 4? Set yes=true if yes, yes=false if no.",
            response_format=Flag,
            temperature=0.0,
        )
        assert result.structured_output is not None
        assert result.structured_output.yes is True

    async def test_structured_complete_with_retry_callback(self):
        """structured_complete passes retry_config.callback through correctly."""
        import gluellm.api as api_module

        attempts: list[dict] = []
        original = api_module._safe_llm_call

        async def capturing(*args, **kwargs):
            attempts.append(dict(kwargs))
            if len(attempts) == 1:
                raise RateLimitError("simulated 429")
            return await original(*args, **kwargs)

        def cb(exc, attempt):
            return True, {"temperature": 0.0}

        class Out(BaseModel):
            value: int = Field(description="Result of 5+5")

        api_module._safe_llm_call = capturing
        try:
            result = await structured_complete(
                "What is 5+5? Put the integer result in the value field.",
                response_format=Out,
                retry_config=RetryConfig(callback=cb),
            )
        finally:
            api_module._safe_llm_call = original

        assert result.structured_output is not None
        assert result.structured_output.value == 10
        assert len(attempts) == 2
        assert attempts[1].get("temperature") == 0.0
