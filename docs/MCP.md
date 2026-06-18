# MCP Tool Integration

GlueLLM can expose tools from [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers alongside local Python callables. MCP tools are advertised to the LLM as OpenAI function schemas and executed remotely through the official MCP Python SDK.

## Installation

MCP support is optional:

```bash
pip install 'gluellm[mcp]'
```

Without the extra, importing MCP helpers raises a clear error pointing to this install command.

## Quick start

```python
import asyncio
from gluellm import GlueLLM, load_mcp_tools

async def main():
    source = await load_mcp_tools(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
    )
    client = GlueLLM(mcp_sources=[source])

    result = await client.complete("Use an MCP tool to help with my request.")
    print(result.final_response)

    await source.close()

asyncio.run(main())
```

See [`examples/mcp_tools_example.py`](../examples/mcp_tools_example.py) for a runnable script.

## Transports

`MCPToolSource` and `load_mcp_tools()` support three transports:

| Transport | Parameters | Notes |
|-----------|------------|-------|
| `stdio` (default) | `command`, `args`, `env` | Spawn a subprocess MCP server |
| `sse` | `url`, `headers` | Legacy SSE transport |
| `streamable-http` | `url`, `headers` | Preferred HTTP transport for new servers |

```python
# Streamable HTTP
source = await load_mcp_tools(
    transport="streamable-http",
    url="http://localhost:8000/mcp",
    headers={"Authorization": "Bearer token"},
)
```

Use `prefix="myserver_"` when connecting multiple MCP sources to avoid tool name collisions.

## ToolRegistry

For finer control, build a `ToolRegistry` that merges local callables and remote sources:

```python
from gluellm import GlueLLM, ToolRegistry, load_mcp_tools

def local_lookup(key: str) -> str:
    """Look up a value from the local cache."""
    return cache.get(key, "")

async def setup():
    mcp = await load_mcp_tools(transport="stdio", command="my-mcp-server")
    registry = ToolRegistry(callables=[local_lookup], sources=[mcp])
    await registry.refresh()  # optional if sources were pre-connected via load_mcp_tools

    client = GlueLLM(tool_registry=registry)
    return client, mcp
```

## How it works

1. **Schema registration** — MCP `inputSchema` JSON is converted to OpenAI function tool dicts via `gluellm.tools.schema`.
2. **LLM calls** — Registry schemas are merged into the `tools=` list passed to the provider (alongside local callables).
3. **Execution** — `_execute_tool_calls_round` dispatches registry-only tools through `ToolRegistry.call()`, which forwards to the MCP session.
4. **Dynamic routing** — In `tool_mode="dynamic"`, MCP tools participate in routing via lightweight stubs; matched tools are expanded back to schemas before the main LLM call.

Hooks (`PRE_TOOL`, `POST_TOOL`) apply to MCP tools the same way as local callables.

## Lifecycle

- `await load_mcp_tools(...)` connects and caches tool schemas.
- `async with MCPToolSource(...) as src:` or `await source.close()` closes the MCP session.
- Call `await registry.close()` to shut down all registered MCP sources.

## See also

- [TOOL_EXECUTION.md](TOOL_EXECUTION.md) — local tool loops and dynamic routing
- [EXTENDING.md](EXTENDING.md) — hooks and custom integrations
