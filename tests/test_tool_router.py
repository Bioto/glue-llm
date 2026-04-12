"""Tests for dynamic tool routing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gluellm.api import complete
from gluellm.tool_router import (
    ROUTER_TOOL_NAME,
    build_router_tool,
    is_router_call,
    resolve_tool_route,
)


def dummy_tool(value: str) -> str:
    """A dummy tool for testing."""
    return f"Tool received: {value}"


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: sunny"


def calculate(expr: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expr)


# Unit tests for build_router_tool
class TestBuildRouterTool:
    """Unit tests for build_router_tool()."""

    def test_router_tool_has_correct_name(self):
        tools = [dummy_tool, get_weather]
        router = build_router_tool(tools)
        assert router.__name__ == ROUTER_TOOL_NAME

    def test_router_docstring_contains_all_tool_names(self):
        tools = [dummy_tool, get_weather, calculate]
        router = build_router_tool(tools)
        doc = router.__doc__ or ""
        assert "dummy_tool" in doc
        assert "get_weather" in doc
        assert "calculate" in doc

    def test_router_tool_is_callable(self):
        tools = [dummy_tool]
        router = build_router_tool(tools)
        # The router returns "" when called (placeholder - never normally invoked)
        result = router(query="test")
        assert result == ""


# Unit tests for is_router_call
class TestIsRouterCall:
    """Unit tests for is_router_call()."""

    def test_detects_router_call(self):
        tc = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name=ROUTER_TOOL_NAME, arguments='{"query": "weather"}'),
        )
        assert is_router_call([tc]) is True

    def test_rejects_non_router_call(self):
        tc = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="dummy_tool", arguments='{"value": "x"}'),
        )
        assert is_router_call([tc]) is False

    def test_empty_list_returns_false(self):
        assert is_router_call([]) is False

    def test_mixed_calls_detects_router(self):
        tc1 = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="dummy_tool", arguments="{}"),
        )
        tc2 = SimpleNamespace(
            id="call_2",
            type="function",
            function=SimpleNamespace(name=ROUTER_TOOL_NAME, arguments='{"query": "x"}'),
        )
        assert is_router_call([tc1, tc2]) is True


# Unit tests for resolve_tool_route
class TestResolveToolRoute:
    """Unit tests for resolve_tool_route() with mocked LLM."""

    @pytest.mark.asyncio
    async def test_returns_matched_tools_from_llm_response(self):
        async def fake_llm(*, messages, model, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='["get_weather", "dummy_tool"]',
                        )
                    )
                ],
            )

        with patch("gluellm.tool_router.any_llm_acompletion", side_effect=fake_llm):
            tools = [dummy_tool, get_weather, calculate]
            matched = await resolve_tool_route("weather and dummy", tools, model="openai:gpt-5.4-2026-03-05")
            assert len(matched) == 2
            names = [t.__name__ for t in matched]
            assert "get_weather" in names
            assert "dummy_tool" in names

    @pytest.mark.asyncio
    async def test_fallback_to_all_tools_on_llm_error(self):
        async def failing_llm(*args, **kwargs):
            raise RuntimeError("API error")

        with patch("gluellm.tool_router.any_llm_acompletion", side_effect=failing_llm):
            tools = [dummy_tool, get_weather]
            matched = await resolve_tool_route("weather", tools, model="openai:gpt-5.4-2026-03-05")
            assert matched == tools

    @pytest.mark.asyncio
    async def test_fallback_to_all_tools_on_invalid_json(self):
        async def bad_json_llm(*, messages, model, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="I think you need get_weather",
                        )
                    )
                ],
            )

        with patch("gluellm.tool_router.any_llm_acompletion", side_effect=bad_json_llm):
            tools = [dummy_tool, get_weather]
            matched = await resolve_tool_route("weather", tools, model="openai:gpt-5.4-2026-03-05")
            assert matched == tools

    @pytest.mark.asyncio
    async def test_empty_tools_returns_empty(self):
        matched = await resolve_tool_route("anything", [], model="openai:gpt-5.4-2026-03-05")
        assert matched == []


# Shared scenario for integration tests (same input for both modes)
SCENARIO_USER_MESSAGE = "Get weather in Paris and use the dummy tool with value 'hello'"
SCENARIO_SYSTEM_PROMPT = "Use tools when needed."
SCENARIO_TOOLS = [dummy_tool, get_weather, calculate]
SCENARIO_FINAL_RESPONSE = "All done"


def make_tool_call_response(tool_name: str, arguments: str, call_id: str) -> SimpleNamespace:
    """Build a fake LLM response that requests a tool call."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(name=tool_name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="openai:gpt-5.4-2026-03-05",
    )


def make_text_response(content: str) -> SimpleNamespace:
    """Build a fake LLM response with plain text."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="openai:gpt-5.4-2026-03-05",
    )


# Integration tests (mocked LLM) - same scenario, different tool_mode
class TestDynamicToolModeIntegration:
    """Integration tests: identical scenario in standard vs dynamic mode."""

    @pytest.mark.asyncio
    async def test_standard_mode_same_scenario(self):
        """Same scenario in tool_mode='standard': LLM sees all tools, calls dummy_tool, then done."""
        call_count = 0

        async def fake_llm(*, messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_tool_call_response("dummy_tool", '{"value": "hello"}', "call_1")
            return make_text_response(SCENARIO_FINAL_RESPONSE)

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            result = await complete(
                user_message=SCENARIO_USER_MESSAGE,
                system_prompt=SCENARIO_SYSTEM_PROMPT,
                tools=SCENARIO_TOOLS,
                tool_mode="standard",
            )

        assert result.tool_calls_made == 1
        assert SCENARIO_FINAL_RESPONSE in result.final_response
        assert "hello" in str(result.tool_execution_history)

    @pytest.mark.asyncio
    async def test_dynamic_mode_same_scenario(self):
        """Same scenario in tool_mode='dynamic': router first, then dummy_tool, then done."""
        main_llm_calls = 0
        routing_llm_calls = 0

        async def fake_main_llm(*, messages, tools, **kwargs):
            nonlocal main_llm_calls
            main_llm_calls += 1
            tool_names = [t.__name__ for t in (tools or [])]
            if main_llm_calls == 1:
                assert ROUTER_TOOL_NAME in tool_names
                return make_tool_call_response(
                    ROUTER_TOOL_NAME, '{"query": "weather and dummy"}', "call_1"
                )
            if main_llm_calls == 2:
                assert ROUTER_TOOL_NAME not in tool_names
                assert "get_weather" in tool_names
                assert "dummy_tool" in tool_names
                return make_tool_call_response("dummy_tool", '{"value": "hello"}', "call_2")
            return make_text_response(SCENARIO_FINAL_RESPONSE)

        async def fake_routing_llm(*, messages, model, **kwargs):
            nonlocal routing_llm_calls
            routing_llm_calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='["get_weather", "dummy_tool"]'),
                    )
                ],
            )

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_main_llm), patch(
            "gluellm.tool_router.any_llm_acompletion", side_effect=fake_routing_llm
        ):
            result = await complete(
                user_message=SCENARIO_USER_MESSAGE,
                system_prompt=SCENARIO_SYSTEM_PROMPT,
                tools=SCENARIO_TOOLS,
                tool_mode="dynamic",
            )

        assert routing_llm_calls >= 1
        assert main_llm_calls >= 3
        assert result.tool_calls_made == 1
        assert SCENARIO_FINAL_RESPONSE in result.final_response
        assert "hello" in str(result.tool_execution_history)

    @pytest.mark.asyncio
    async def test_routing_uses_user_message_not_llm_query(self):
        """Router receives the full user message, not the LLM's narrow query arg."""
        routing_prompt_received: list[str] = []

        async def fake_main_llm(*, messages, tools, **kwargs):
            if ROUTER_TOOL_NAME in [t.__name__ for t in (tools or [])]:
                return make_tool_call_response(
                    ROUTER_TOOL_NAME, '{"query": "just weather"}', "call_1"
                )
            return make_tool_call_response("dummy_tool", '{"value": "ok"}', "call_2")

        async def capture_routing(*, messages, model, **kwargs):
            last_content = (messages[-1].get("content", "") or "") if messages else ""
            routing_prompt_received.append(last_content)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='["dummy_tool", "get_weather"]'),
                    )
                ],
            )

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_main_llm), patch(
            "gluellm.tool_router.any_llm_acompletion", side_effect=capture_routing
        ):
            await complete(
                user_message=SCENARIO_USER_MESSAGE,
                system_prompt=SCENARIO_SYSTEM_PROMPT,
                tools=SCENARIO_TOOLS,
                tool_mode="dynamic",
            )

        assert len(routing_prompt_received) >= 1
        # Router prompt should contain the full user message (weather + dummy), not "just weather"
        combined = " ".join(routing_prompt_received).lower()
        assert "weather" in combined
        assert "dummy" in combined
        assert "hello" in combined

    @pytest.mark.asyncio
    async def test_router_messages_scrubbed_from_history(self):
        """Same scenario: router call is NOT in messages for the second LLM call."""
        second_call_messages: list = []
        call_count = 0

        async def capture_main_llm(*, messages, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if tools and ROUTER_TOOL_NAME not in [t.__name__ for t in tools]:
                second_call_messages.extend(messages)
            if call_count == 1:
                return make_tool_call_response(
                    ROUTER_TOOL_NAME, '{"query": "weather and dummy"}', "call_1"
                )
            if call_count == 2:
                return make_tool_call_response("dummy_tool", '{"value": "hello"}', "call_2")
            return make_text_response(SCENARIO_FINAL_RESPONSE)

        async def fake_routing(*, messages, model, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='["dummy_tool", "get_weather"]'),
                    )
                ],
            )

        with patch("gluellm.api._llm_call_with_retry", side_effect=capture_main_llm), patch(
            "gluellm.tool_router.any_llm_acompletion", side_effect=fake_routing
        ):
            await complete(
                user_message=SCENARIO_USER_MESSAGE,
                system_prompt=SCENARIO_SYSTEM_PROMPT,
                tools=SCENARIO_TOOLS,
                tool_mode="dynamic",
            )

        assert ROUTER_TOOL_NAME not in str(second_call_messages)
