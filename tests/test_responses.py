"""Tests for the OpenResponses API wrapper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gluellm.responses import (
    CODE_INTERPRETER,
    FILE_SEARCH,
    WEB_SEARCH,
    ResponseResult,
    responses,
)

pytestmark = pytest.mark.asyncio


class TestResponsesBasic:
    """Basic tests for responses() API."""

    async def test_responses_basic_completion(self):
        """Test basic completion returns ResponseResult."""
        mock_resp = SimpleNamespace(
            output_text="4",
            output=[],
            model="gpt-5.4-2026-03-05",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        with patch("any_llm.aresponses", new=AsyncMock(return_value=mock_resp)):
            result = await responses("What is 2+2?")

        assert isinstance(result, ResponseResult)
        assert result.output == "4"
        assert result.model == "gpt-5.4-2026-03-05"
        assert result.usage is not None
        assert result.raw_response == mock_resp

    async def test_responses_with_web_search(self):
        """Test responses with WEB_SEARCH tool."""
        mock_resp = SimpleNamespace(
            output_text="Latest news: ...",
            output=[],
            model="gpt-5.4-2026-03-05",
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        )

        with patch("any_llm.aresponses", new=AsyncMock(return_value=mock_resp)) as mock:
            result = await responses("What's the news?", tools=[WEB_SEARCH])

        assert result.output == "Latest news: ..."
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("tools") == [WEB_SEARCH]

    async def test_responses_streaming_returns_iterator(self):
        """Test streaming returns raw iterator, not ResponseResult."""
        async def fake_stream():
            yield SimpleNamespace(type="response_text_delta", delta=SimpleNamespace(text="Hi"))

        with patch("any_llm.aresponses", new=AsyncMock(return_value=fake_stream())):
            result = await responses("Hi", stream=True)

        assert result is not None
        # When stream=True, we return the raw async iterator
        async for _ in result:
            break


class TestResponseResultExtraction:
    """Tests for output extraction from various response formats."""

    async def test_extract_output_text_from_output_text_attr(self):
        """Response with output_text attribute is extracted."""
        mock_resp = SimpleNamespace(
            output_text="Hello world",
            output=[],
            model="gpt-5.4-2026-03-05",
            usage=None,
        )

        with patch("any_llm.aresponses", new=AsyncMock(return_value=mock_resp)):
            result = await responses("Hi")

        assert result.output == "Hello world"

    async def test_extract_output_text_from_empty_output(self):
        """Response with empty output returns empty string when no output_text."""
        mock_resp = SimpleNamespace(
            output_text=None,
            output=[],
            model="gpt-5.4-2026-03-05",
            usage=None,
        )

        with patch("any_llm.aresponses", new=AsyncMock(return_value=mock_resp)):
            result = await responses("Hi")

        assert result.output == ""


class TestBuiltinTools:
    """Tests for built-in tool constants."""

    async def test_web_search_constant(self):
        """WEB_SEARCH has correct structure."""
        assert WEB_SEARCH == {"type": "web_search_preview"}

    async def test_code_interpreter_constant(self):
        """CODE_INTERPRETER has correct structure."""
        assert CODE_INTERPRETER == {"type": "code_interpreter"}

    async def test_file_search_constant(self):
        """FILE_SEARCH has correct structure."""
        assert FILE_SEARCH == {"type": "file_search"}


@pytest.mark.integration
class TestResponsesIntegration:
    """Integration tests (require API key)."""

    async def test_responses_integration(self):
        """Test responses with real API (skipped if no key)."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        result = await responses("Say 'test' and nothing else.")
        assert isinstance(result, ResponseResult)
        assert len(result.output) > 0
        assert result.model
