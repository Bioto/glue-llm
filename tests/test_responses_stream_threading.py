"""Tests for Responses API streaming and threading."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gluellm import GlueLLM, StreamingChunk


@pytest.mark.asyncio
async def test_stream_response_yields_chunks_and_final_done():
    client = GlueLLM(model="openai:gpt-test")

    async def fake_stream():
        yield SimpleNamespace(type="response_text_delta", delta=SimpleNamespace(text="Hel"))
        yield SimpleNamespace(type="response_text_delta", delta=SimpleNamespace(text="lo"))

    with patch.object(
        client,
        "_llm_responses_call",
        new=AsyncMock(return_value=fake_stream()),
    ):
        chunks: list[StreamingChunk] = []
        async for chunk in client.stream_response("Hi", execute_tools=False):
            chunks.append(chunk)

    assert any(c.content == "Hel" for c in chunks)
    assert chunks[-1].done is True


@pytest.mark.asyncio
async def test_response_threading_tool_loop_chains_previous_response_id():
    client = GlueLLM(model="openai:gpt-test", response_threading=True)
    seen_previous: list[str | None] = []
    call_count = 0

    async def fake_responses_call(_registry, **kwargs):
        nonlocal call_count
        call_count += 1
        seen_previous.append(kwargs.get("previous_response_id"))
        if call_count == 1:
            tc = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="echo", arguments='{"text":"hi"}'),
            )
            return SimpleNamespace(
                id="resp_1",
                output=[],
                output_text="",
                usage=None,
                choices=None,
            )
        return SimpleNamespace(
            id="resp_2",
            output=[],
            output_text="done",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    def echo(text: str) -> str:
        return text

    with patch("gluellm.api._extract_response_function_calls") as mock_extract:
        mock_extract.side_effect = [
            [SimpleNamespace(id="call_1", function=SimpleNamespace(name="echo", arguments='{"text":"hi"}'))],
            [],
        ]
        with patch.object(client, "_llm_responses_call", new=AsyncMock(side_effect=fake_responses_call)):
            result = await client.response("hello", tools=[echo], execute_tools=True, max_tool_iterations=2)

    assert call_count >= 2
    assert seen_previous[0] is None
    assert seen_previous[1] == "resp_1"
    assert result.response_id == "resp_2"


@pytest.mark.asyncio
async def test_execution_result_includes_response_id():
    client = GlueLLM(model="openai:gpt-test")

    fake = SimpleNamespace(
        id="resp_abc",
        output=[],
        output_text="hello",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )

    with patch.object(client, "_llm_responses_call", new=AsyncMock(return_value=fake)):
        result = await client.response("hi", execute_tools=False)

    assert result.response_id == "resp_abc"


@pytest.mark.asyncio
async def test_response_threading_rebuild_mode_unchanged_by_default():
    client = GlueLLM(model="openai:gpt-test", response_threading=False)
    captured: dict = {}

    async def fake_responses_call(_registry, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="resp_x",
            output=[],
            output_text="ok",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    with patch.object(client, "_llm_responses_call", new=AsyncMock(side_effect=fake_responses_call)):
        await client.response("hi", execute_tools=False)

    assert "previous_response_id" not in captured
