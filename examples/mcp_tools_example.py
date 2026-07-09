"""Example: use MCP tools with GlueLLM.

Requires the optional MCP extra::

    pip install 'gluellm[mcp]'

This example assumes you have an MCP server available. For a quick local
stdio server, many MCP tutorials use ``npx @modelcontextprotocol/server-everything``
or a project-specific server binary.

The pattern is:

1. Connect to the MCP server with ``load_mcp_tools()``
2. Pass the source to ``GlueLLM(mcp_sources=[...])``
3. Call ``complete()`` — MCP tools are advertised to the model and executed remotely
"""

import asyncio

from gluellm import GlueLLM, load_mcp_tools


async def main() -> None:
    # Connect to an MCP server (stdio transport example)
    mcp_source = await load_mcp_tools(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
    )

    client = GlueLLM(
        mcp_sources=[mcp_source],
        system_prompt="You are a helpful assistant with access to MCP tools.",
    )

    try:
        result = await client.complete(
            "Use an MCP tool if one can help answer: what tools do you have?"
        )
        print(result.final_response)
        print(f"Tool calls made: {result.tool_calls_made}")
    finally:
        await mcp_source.close()


if __name__ == "__main__":
    asyncio.run(main())
