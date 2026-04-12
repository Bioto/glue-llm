"""Tests for executor implementations — SimpleExecutor, AgentExecutor, AgentStructuredExecutor."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from gluellm.api import ExecutionResult
from gluellm.executors import AgentExecutor, AgentStructuredExecutor, Executor, SimpleExecutor
from gluellm.models.agent import Agent
from gluellm.models.hook import HookConfig, HookErrorStrategy, HookRegistry, HookStage
from gluellm.models.prompt import SystemPrompt


def _fake_llm_response(content: str = "Hello"):
    """Build a minimal fake LLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.usage = None
    resp.model = "openai:gpt-5.4-2026-03-05"
    return resp


@pytest.mark.asyncio
class TestSimpleExecutor:
    async def test_returns_execution_result(self):
        with patch("gluellm.api._llm_call_with_retry", return_value=_fake_llm_response("Hi")):
            executor = SimpleExecutor(system_prompt="Be helpful.")
            result = await executor.execute("Hello")
        assert isinstance(result, ExecutionResult)
        assert result.final_response == "Hi"

    async def test_uses_configured_model(self):
        executor = SimpleExecutor(model="openai:gpt-4o")
        assert executor.model == "openai:gpt-4o"

    async def test_defaults_to_settings_model(self):
        executor = SimpleExecutor()
        assert executor.model is not None

    async def test_with_tools(self):
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        with patch("gluellm.api._llm_call_with_retry", return_value=_fake_llm_response("done")):
            executor = SimpleExecutor(tools=[dummy_tool])
            result = await executor.execute("Use the tool")
        assert isinstance(result, ExecutionResult)
        assert result.final_response == "done"


@pytest.mark.asyncio
class TestAgentExecutor:
    def _make_agent(self, **overrides):
        defaults = {
            "name": "Test Agent",
            "description": "A test agent",
            "system_prompt": SystemPrompt(content="You are a test agent."),
            "tools": [],
            "max_tool_iterations": 5,
        }
        defaults.update(overrides)
        return Agent(**defaults)

    async def test_returns_execution_result(self):
        agent = self._make_agent()
        with patch("gluellm.api._llm_call_with_retry", return_value=_fake_llm_response("Agent says hi")):
            executor = AgentExecutor(agent)
            result = await executor.execute("Hello agent")
        assert isinstance(result, ExecutionResult)
        assert result.final_response == "Agent says hi"

    async def test_uses_agent_model(self):
        agent = self._make_agent(model="anthropic:claude-3-5-sonnet-20241022")
        executor = AgentExecutor(agent)
        assert executor.agent.model == "anthropic:claude-3-5-sonnet-20241022"

    async def test_agent_without_system_prompt(self):
        agent = self._make_agent(system_prompt=None)
        with patch("gluellm.api._llm_call_with_retry", return_value=_fake_llm_response("ok")):
            executor = AgentExecutor(agent)
            result = await executor.execute("test")
        assert isinstance(result, ExecutionResult)
        assert result.final_response == "ok"


@pytest.mark.asyncio
class TestAgentStructuredExecutor:
    class Answer(BaseModel):
        value: int

    def _make_agent(self):
        return Agent(
            name="Structured Agent",
            description="Structured output agent",
            system_prompt=SystemPrompt(content="Return structured data."),
            tools=[],
            max_tool_iterations=3,
        )

    async def test_returns_execution_result_with_structured_output(self):
        agent = self._make_agent()
        resp = _fake_llm_response('{"value": 42}')
        resp.choices[0].message.parsed = self.Answer(value=42)
        with patch("gluellm.api._llm_call_with_retry", return_value=resp):
            executor = AgentStructuredExecutor(agent, response_format=self.Answer)
            result = await executor.execute("What is 6*7?")
        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert result.structured_output.value == 42


class TestExecutorIsAbstract:
    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError):
            Executor()


@pytest.mark.asyncio
class TestExecutorHookIntegration:
    async def test_pre_hook_modifies_query(self):
        def uppercase_hook(context):
            context.content = context.content.upper()
            return context

        registry = HookRegistry()
        registry.add_hook(
            HookStage.PRE_EXECUTOR,
            HookConfig(handler=uppercase_hook, name="upper", error_strategy=HookErrorStrategy.ABORT),
        )

        with patch("gluellm.api._llm_call_with_retry", return_value=_fake_llm_response("done")):
            executor = SimpleExecutor(hook_registry=registry)
            result = await executor.execute("hello")
        assert isinstance(result, ExecutionResult)
        assert result.final_response == "done"

    async def test_post_hook_modifies_result(self):
        def suffix_hook(context):
            context.content = context.content + " [reviewed]"
            return context

        registry = HookRegistry()
        registry.add_hook(
            HookStage.POST_EXECUTOR,
            HookConfig(handler=suffix_hook, name="suffix", error_strategy=HookErrorStrategy.ABORT),
        )

        with patch("gluellm.api._llm_call_with_retry", return_value=_fake_llm_response("response")):
            executor = SimpleExecutor(hook_registry=registry)
            result = await executor.execute("query")
        assert isinstance(result, ExecutionResult)
        assert result.final_response == "response [reviewed]"
