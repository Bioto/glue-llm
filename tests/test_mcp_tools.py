"""Tests for MCP tool integration (mocked — no live MCP server)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gluellm.api import GlueLLM
from gluellm.tools.mcp import MCPToolSource, _require_mcp
from gluellm.tools.registry import ToolRegistry
from gluellm.tools.schema import mcp_tool_to_openai_schema


class _FakeMCPSource:
    """Minimal ToolSource stand-in for registry tests."""

    def __init__(self) -> None:
        self._schemas = {
            "remote_add": mcp_tool_to_openai_schema(
                "remote_add",
                description="Add two numbers",
                input_schema={
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            )
        }

    async def list_tools(self) -> list[str]:
        return list(self._schemas)

    def get_schema(self, name: str) -> dict[str, Any]:
        return dict(self._schemas[name])

    def exported_schemas(self) -> dict[str, dict[str, Any]]:
        return dict(self._schemas)

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        if name == "remote_add":
            return str(args["a"] + args["b"])
        raise KeyError(name)


class TestSchemaConversion:
    def test_mcp_tool_to_openai_schema_includes_parameters(self):
        schema = mcp_tool_to_openai_schema(
            "echo",
            description="Echo input",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "text" in schema["function"]["parameters"]["properties"]


class TestToolRegistry:
    def test_registry_merges_local_and_remote_tools(self):
        def local_tool() -> str:
            """Local tool."""
            return "local"

        registry = ToolRegistry(callables=[local_tool], sources=[_FakeMCPSource()])
        merged = registry.merge_for_llm([local_tool])
        names = [
            (t.__name__ if callable(t) else t["function"]["name"])
            for t in merged
        ]
        assert "local_tool" in names
        assert "remote_add" in names

    @pytest.mark.asyncio
    async def test_registry_call_dispatches_to_remote_source(self):
        registry = ToolRegistry(sources=[_FakeMCPSource()])
        result = await registry.call("remote_add", {"a": 2, "b": 3})
        assert result == "5"

    def test_routing_stubs_expand_to_schemas(self):
        registry = ToolRegistry(sources=[_FakeMCPSource()])
        stubs = registry.routing_stubs()
        assert stubs[0].__name__ == "remote_add"
        expanded = registry.expand_routed(stubs)
        assert expanded[0]["function"]["name"] == "remote_add"


class TestGlueLLMRegistryDispatch:
    @pytest.mark.asyncio
    async def test_execute_tool_calls_round_dispatches_registry_tools(self):
        registry = ToolRegistry(sources=[_FakeMCPSource()])
        client = GlueLLM(tool_registry=registry)

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "remote_add"
        tool_call.function.arguments = '{"a": 4, "b": 5}'

        active_tools = registry.openai_tools()
        results = await client._execute_tool_calls_round(
            tool_calls=[tool_call],
            active_tools=active_tools,
            parallel=False,
            iteration=1,
            correlation_id="test-corr",
            on_status=None,
        )

        assert len(results) == 1
        assert results[0]["history"]["result"] == "9"
        assert results[0]["history"]["error"] is False

    def test_find_tool_returns_none_for_registry_only_tools(self):
        registry = ToolRegistry(sources=[_FakeMCPSource()])
        client = GlueLLM(tool_registry=registry)
        assert client._find_tool("remote_add", registry=registry) is None
        assert registry.has_tool("remote_add")


class TestMCPImportError:
    def test_mcp_missing_extra_raises_import_error_with_message(self):
        with patch.dict("sys.modules", {"mcp": None}), pytest.raises(ImportError, match="gluellm\\[mcp\\]"):
            _require_mcp()


class TestMCPToolSourceMocked:
    @pytest.mark.asyncio
    async def test_mcp_tool_schema_registered_and_executable(self):
        source = MCPToolSource(transport="stdio", command="echo", args=["hello"])

        fake_tool = MagicMock()
        fake_tool.name = "greet"
        fake_tool.description = "Say hello"
        fake_tool.inputSchema = {"type": "object", "properties": {"name": {"type": "string"}}}

        fake_session = AsyncMock()
        fake_session.initialize = AsyncMock()
        fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[fake_tool]))
        fake_session.call_tool = AsyncMock(
            return_value=MagicMock(content=[MagicMock(type="text", text="Hello, Ada")], isError=False)
        )

        source._session = fake_session
        await source._load_tools()

        assert await source.list_tools() == ["greet"]
        schema = source.get_schema("greet")
        assert schema["function"]["name"] == "greet"
        result = await source.call("greet", {"name": "Ada"})
        assert result == "Hello, Ada"
