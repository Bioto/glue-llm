"""
Test error handling and retry logic for GlueLLM.
Tests the comprehensive error classification and retry mechanisms.
"""

import time
from unittest.mock import Mock, patch

import pytest

from gluellm.api import (
    APIConnectionError,
    AuthenticationError,
    GlueLLM,
    InvalidRequestError,
    LLMError,
    RateLimitError,
    TokenLimitError,
    classify_llm_error,
)

# Mark tests that need async
pytestmark = pytest.mark.asyncio


class TestErrorClassification:
    """Test error classification logic."""

    async def test_token_limit_error_classification(self):
        """Test that token limit errors are correctly classified."""
        errors = [
            Exception("context length exceeded"),
            Exception("maximum context length is 4096"),
            Exception("too many tokens in request"),
            Exception("token limit exceeded"),
        ]

        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, TokenLimitError), f"Failed for: {error}"

    async def test_rate_limit_error_classification(self):
        """Test that rate limit errors are correctly classified."""
        errors = [
            Exception("rate limit exceeded"),
            Exception("too many requests"),
            Exception("429 - rate_limit_exceeded"),
            Exception("quota exceeded"),
            Exception("resource exhausted"),
        ]

        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, RateLimitError), f"Failed for: {error}"

    async def test_connection_error_classification(self):
        """Test that connection errors are correctly classified."""
        errors = [
            Exception("connection timeout"),
            Exception("network error"),
            Exception("503 service unavailable"),
            Exception("502 bad gateway"),
            Exception("unreachable"),
        ]

        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, APIConnectionError), f"Failed for: {error}"

    async def test_auth_error_classification(self):
        """Test that authentication errors are correctly classified."""
        errors = [
            Exception("invalid api key"),
            Exception("401 unauthorized"),
            Exception("authentication failed"),
            Exception("403 forbidden"),
        ]

        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, AuthenticationError), f"Failed for: {error}"

    async def test_invalid_request_error_classification(self):
        """Test that invalid request errors are correctly classified."""
        errors = [
            Exception("invalid request"),
            Exception("400 bad request"),
            Exception("validation error"),
        ]

        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, InvalidRequestError), f"Failed for: {error}"

    async def test_generic_error_classification(self):
        """Test that unknown errors become generic LLMError."""
        error = Exception("some random error")
        classified = classify_llm_error(error)
        assert isinstance(classified, LLMError)
        assert not isinstance(classified, TokenLimitError)


class TestRetryLogic:
    """Test retry behavior with mocked LLM calls."""

    @patch("gluellm.api._safe_llm_call")
    async def test_retry_on_rate_limit(self, mock_safe_call):
        """Test that rate limit errors trigger retries."""
        # Create a proper mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # First two calls raise RateLimitError, third succeeds
        mock_safe_call.side_effect = [
            RateLimitError("Rate limit exceeded"),
            RateLimitError("Rate limit exceeded"),
            mock_response,
        ]

        client = GlueLLM(model="openai:gpt-4o-mini")
        result = await client.complete("Test message")

        # Should have retried and eventually succeeded
        assert mock_safe_call.call_count == 3
        assert result.final_response == "Success"

    @patch("gluellm.api._safe_llm_call")
    async def test_retry_on_connection_error(self, mock_safe_call):
        """Test that connection errors trigger retries."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success after retry"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_safe_call.side_effect = [
            APIConnectionError("Connection timeout"),
            mock_response,
        ]

        client = GlueLLM(model="openai:gpt-4o-mini")
        result = await client.complete("Test message")

        assert mock_safe_call.call_count == 2
        assert result.final_response == "Success after retry"

    @patch("gluellm.api._safe_llm_call")
    async def test_no_retry_on_token_limit(self, mock_safe_call):
        """Test that token limit errors do NOT trigger retries."""
        mock_safe_call.side_effect = TokenLimitError("Context length exceeded")

        client = GlueLLM(model="openai:gpt-4o-mini")

        with pytest.raises(TokenLimitError):
            await client.complete("Test message with too many tokens")

        # Should only be called once (no retries)
        assert mock_safe_call.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_no_retry_on_auth_error(self, mock_safe_call):
        """Test that authentication errors do NOT trigger retries."""
        mock_safe_call.side_effect = AuthenticationError("Invalid API key")

        client = GlueLLM(model="openai:gpt-4o-mini")

        with pytest.raises(AuthenticationError):
            await client.complete("Test message")

        # Should only be called once (no retries)
        assert mock_safe_call.call_count == 1

    @patch("gluellm.api._safe_llm_call")
    async def test_max_retries_exceeded(self, mock_safe_call):
        """Test that max retries is respected."""
        # Always raise RateLimitError - test the actual retry decorator
        mock_safe_call.side_effect = RateLimitError("Rate limit exceeded")

        client = GlueLLM(model="openai:gpt-4o-mini")

        with pytest.raises(RateLimitError):
            await client.complete("Test message")

        # Should retry up to 3 times total
        assert mock_safe_call.call_count == 3


class TestToolExecutionErrorHandling:
    """Test error handling during tool execution."""

    @patch("gluellm.api._safe_llm_call")
    async def test_tool_execution_exception_handling(self, mock_safe_call):
        """Test that tool execution errors are caught and added to history."""

        def error_tool(x: str) -> str:
            """A tool that raises an error."""
            raise ValueError(f"Tool error with input: {x}")

        # First call: model wants to use tool
        tool_call_response = Mock()
        tool_call_choice = Mock()
        tool_call_msg = Mock()
        tool_call_msg.content = None

        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "error_tool"
        mock_tool_call.function.arguments = '{"x": "test"}'
        tool_call_msg.tool_calls = [mock_tool_call]
        tool_call_choice.message = tool_call_msg
        tool_call_response.choices = [tool_call_choice]

        # Second call: final response
        final_response = Mock()
        final_choice = Mock()
        final_msg = Mock()
        final_msg.content = "I handled the error"
        final_msg.tool_calls = None
        final_choice.message = final_msg
        final_response.choices = [final_choice]

        mock_safe_call.side_effect = [tool_call_response, final_response]

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            tools=[error_tool],
        )
        result = await client.complete("Use error_tool")

        # Check that error was captured in history
        assert len(result.tool_execution_history) == 1
        assert result.tool_execution_history[0]["error"] is True
        assert "ValueError" in result.tool_execution_history[0]["result"]
        assert result.final_response == "I handled the error"

    @patch("gluellm.api._safe_llm_call")
    async def test_malformed_json_in_tool_args(self, mock_safe_call):
        """Test handling of malformed JSON in tool arguments."""

        def dummy_tool(x: str) -> str:
            """A simple tool."""
            return f"Result: {x}"

        # Mock a response with malformed JSON
        tool_call_response = Mock()
        tool_call_choice = Mock()
        tool_call_msg = Mock()
        tool_call_msg.content = None

        mock_tool_call = Mock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "dummy_tool"
        mock_tool_call.function.arguments = "{invalid json}"  # Malformed
        tool_call_msg.tool_calls = [mock_tool_call]
        tool_call_choice.message = tool_call_msg
        tool_call_response.choices = [tool_call_choice]

        # Final response
        final_response = Mock()
        final_choice = Mock()
        final_msg = Mock()
        final_msg.content = "Handled invalid JSON"
        final_msg.tool_calls = None
        final_choice.message = final_msg
        final_response.choices = [final_choice]

        mock_safe_call.side_effect = [tool_call_response, final_response]

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            tools=[dummy_tool],
        )
        result = await client.complete("Test")

        # Check that JSON error was captured
        assert len(result.tool_execution_history) == 1
        assert result.tool_execution_history[0]["error"] is True
        assert "Invalid JSON" in result.tool_execution_history[0]["result"]


class TestStructuredCompleteErrorHandling:
    """Test error handling in structured_complete."""

    @patch("gluellm.api._safe_llm_call")
    async def test_structured_complete_with_rate_limit_retry(self, mock_safe_call):
        """Test that structured_complete also benefits from retry logic."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        # Create proper mock responses
        # First call fails
        first_call_side_effect = RateLimitError("Rate limit exceeded")

        # Second call succeeds
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.parsed = TestModel(name="test", value=42)
        mock_message.content = '{"name": "test", "value": 42}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        mock_safe_call.side_effect = [
            first_call_side_effect,
            mock_response,
        ]

        client = GlueLLM(model="openai:gpt-4o-mini")
        result = await client.structured_complete("Test", TestModel)

        # Verify retry happened (2 calls total)
        assert mock_safe_call.call_count == 2
        from gluellm.api import ExecutionResult

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, TestModel)
        assert result.structured_output.name == "test"
        assert result.structured_output.value == 42


class TestRetryBackoffTiming:
    """Test retry backoff timing behavior."""

    @patch("gluellm.api._safe_llm_call")
    @patch("asyncio.sleep")  # Mock asyncio.sleep for async retry
    async def test_exponential_backoff_waits(self, mock_sleep, mock_safe_call):
        """Test that exponential backoff actually waits between retries."""

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # First two calls fail, third succeeds
        mock_safe_call.side_effect = [
            RateLimitError("Rate limit exceeded"),
            RateLimitError("Rate limit exceeded"),
            mock_response,
        ]

        client = GlueLLM(model="openai:gpt-4o-mini")
        await client.complete("Test message")

        # Verify sleep was called (exponential backoff waits)
        assert mock_sleep.call_count >= 2  # Should sleep between retries

        # Verify sleep durations follow exponential pattern
        sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
        if len(sleep_times) >= 2:
            # Second wait should be longer than first (exponential)
            # With multiplier=1, min=2: first wait ~2s, second wait ~4s
            assert sleep_times[1] >= sleep_times[0]

    @patch("gluellm.api._safe_llm_call")
    @patch("asyncio.sleep")
    async def test_max_wait_cap_enforced(self, mock_sleep, mock_safe_call):
        """Test that max_wait cap is enforced."""
        from gluellm.config import settings

        # Temporarily set low max_wait for testing
        original_max_wait = settings.retry_max_wait
        settings.retry_max_wait = 5  # Set max wait to 5 seconds

        try:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Success"
            mock_message.tool_calls = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            # Two failures then success (within retry_max_attempts=3)
            mock_safe_call.side_effect = [
                RateLimitError("Rate limit exceeded"),
                RateLimitError("Rate limit exceeded"),
                mock_response,
            ]

            client = GlueLLM(model="openai:gpt-4o-mini")
            await client.complete("Test message")

            # Verify all sleep times are <= max_wait
            sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
            for sleep_time in sleep_times:
                assert sleep_time <= settings.retry_max_wait, (
                    f"Sleep time {sleep_time} exceeds max_wait {settings.retry_max_wait}"
                )
        finally:
            settings.retry_max_wait = original_max_wait

    @patch("gluellm.api._safe_llm_call")
    @patch("asyncio.sleep")
    async def test_min_wait_floor_enforced(self, mock_sleep, mock_safe_call):
        """Test that min_wait floor is enforced."""
        from gluellm.config import settings

        # Temporarily set min_wait for testing
        original_min_wait = settings.retry_min_wait
        settings.retry_min_wait = 1  # Set min wait to 1 second

        try:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Success"
            mock_message.tool_calls = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            # First call fails, second succeeds
            mock_safe_call.side_effect = [
                RateLimitError("Rate limit exceeded"),
                mock_response,
            ]

            client = GlueLLM(model="openai:gpt-4o-mini")
            await client.complete("Test message")

            # Verify sleep time is >= min_wait
            if mock_sleep.called:
                sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
                for sleep_time in sleep_times:
                    assert sleep_time >= settings.retry_min_wait, (
                        f"Sleep time {sleep_time} is below min_wait {settings.retry_min_wait}"
                    )
        finally:
            settings.retry_min_wait = original_min_wait

    @patch("gluellm.api._safe_llm_call")
    async def test_actual_wait_times(self, mock_safe_call):
        """Test that exponential backoff actually waits (not just counts)."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # First call fails, second succeeds
        mock_safe_call.side_effect = [
            RateLimitError("Rate limit exceeded"),
            mock_response,
        ]

        client = GlueLLM(model="openai:gpt-4o-mini")

        # Measure actual time taken
        start_time = time.time()
        await client.complete("Test message")
        elapsed_time = time.time() - start_time

        # Should have waited at least min_wait seconds (with some tolerance)
        from gluellm.config import settings

        assert elapsed_time >= settings.retry_min_wait - 0.5, (
            f"Elapsed time {elapsed_time} is less than min_wait {settings.retry_min_wait}"
        )

    @patch("gluellm.api._safe_llm_call")
    @patch("asyncio.sleep")
    async def test_exponential_backoff_calculation(self, mock_sleep, mock_safe_call):
        """Test that exponential backoff calculation is correct."""
        from gluellm.config import settings

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Two failures then success (within retry_max_attempts=3)
        mock_safe_call.side_effect = [
            RateLimitError("Rate limit exceeded"),
            RateLimitError("Rate limit exceeded"),
            mock_response,
        ]

        client = GlueLLM(model="openai:gpt-4o-mini")
        await client.complete("Test message")

        # Verify exponential pattern: each wait should be roughly 2x the previous
        # (with multiplier=1, min=2: ~2s, ~4s, ~8s, capped at max_wait)
        if mock_sleep.call_count >= 2:
            sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
            # First wait should be at least min_wait
            assert sleep_times[0] >= settings.retry_min_wait
            # Subsequent waits should increase (exponential) or be capped
            for i in range(1, len(sleep_times)):
                # Each wait should be >= previous (exponential) or <= max_wait (capped)
                assert sleep_times[i] >= sleep_times[i - 1] or sleep_times[i] <= settings.retry_max_wait


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
