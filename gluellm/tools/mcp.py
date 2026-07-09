"""MCP (Model Context Protocol) tool source backed by the official Python SDK."""

from __future__ import annotations

import logging
from typing import Any, Literal

from gluellm.tools.schema import mcp_tool_to_openai_schema

logger = logging.getLogger(__name__)

MCPTransport = Literal["stdio", "sse", "streamable-http"]

_MCP_INSTALL_MESSAGE = (
    "MCP support requires the optional 'mcp' package. Install with: pip install 'gluellm[mcp]'"
)


def _require_mcp() -> Any:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MCP_INSTALL_MESSAGE) from exc
    return mcp


def _format_call_tool_result(result: Any) -> str:
    """Convert an MCP CallToolResult into a string for the LLM tool message."""
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", "") or "")
        else:
            parts.append(str(block))
    if getattr(result, "isError", False):
        body = "\n".join(parts) if parts else "Tool returned an error"
        return f"Error: {body}"
    return "\n".join(parts) if parts else ""


class MCPToolSource:
    """Expose tools from a remote MCP server via the official MCP Python SDK."""

    def __init__(
        self,
        *,
        transport: MCPTransport = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        prefix: str | None = None,
    ) -> None:
        self.transport = transport
        self._command = command
        self._args = args or []
        self._env = env
        self._url = url
        self._headers = headers or {}
        self._prefix = prefix or ""

        self._transport_ctx: Any = None
        self._session_ctx: Any = None
        self._session: Any = None
        self._tool_names: list[str] = []
        self._schemas: dict[str, dict[str, Any]] = {}

    def _prefixed(self, name: str) -> str:
        return f"{self._prefix}{name}" if self._prefix else name

    def _unprefixed(self, name: str) -> str:
        if self._prefix and name.startswith(self._prefix):
            return name[len(self._prefix) :]
        return name

    async def __aenter__(self) -> MCPToolSource:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the MCP transport and load available tools."""
        if self._session is not None:
            return

        _require_mcp()

        if self.transport == "stdio":
            if not self._command:
                raise ValueError("MCPToolSource stdio transport requires command=...")
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
            self._transport_ctx = stdio_client(params)
            read, write = await self._transport_ctx.__aenter__()
        elif self.transport == "sse":
            if not self._url:
                raise ValueError("MCPToolSource sse transport requires url=...")
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            self._transport_ctx = sse_client(self._url, headers=self._headers)
            read, write = await self._transport_ctx.__aenter__()
        elif self.transport == "streamable-http":
            if not self._url:
                raise ValueError("MCPToolSource streamable-http transport requires url=...")
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            self._transport_ctx = streamable_http_client(self._url, headers=self._headers)
            read, write, _ = await self._transport_ctx.__aenter__()
        else:
            raise ValueError(f"Unsupported MCP transport: {self.transport}")

        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        await self._load_tools()

    async def close(self) -> None:
        """Close the MCP session and transport."""
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None
        self._tool_names = []
        self._schemas = {}

    async def _load_tools(self) -> None:
        assert self._session is not None
        result = await self._session.list_tools()
        self._tool_names = []
        self._schemas = {}
        for tool in result.tools:
            exposed_name = self._prefixed(tool.name)
            input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
            self._tool_names.append(exposed_name)
            self._schemas[exposed_name] = mcp_tool_to_openai_schema(
                exposed_name,
                description=getattr(tool, "description", None),
                input_schema=input_schema,
            )
        logger.debug("Loaded %d MCP tools via %s transport", len(self._tool_names), self.transport)

    def exported_schemas(self) -> dict[str, dict[str, Any]]:
        """Return cached OpenAI tool schemas after ``connect()``."""
        return dict(self._schemas)

    async def list_tools(self) -> list[str]:
        if self._session is None:
            await self.connect()
        return list(self._tool_names)

    def get_schema(self, name: str) -> dict[str, Any]:
        if name not in self._schemas:
            raise KeyError(f"MCP tool '{name}' is not registered on this source")
        return dict(self._schemas[name])

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        if self._session is None:
            await self.connect()
        remote_name = self._unprefixed(name)
        result = await self._session.call_tool(remote_name, arguments=args or {})
        return _format_call_tool_result(result)


async def load_mcp_tools(
    *,
    transport: MCPTransport = "stdio",
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    prefix: str | None = None,
) -> MCPToolSource:
    """Connect to an MCP server and return a ready ``MCPToolSource``."""
    source = MCPToolSource(
        transport=transport,
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
        prefix=prefix,
    )
    await source.connect()
    return source
