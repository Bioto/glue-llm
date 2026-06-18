"""Convert MCP JSON Schema tool definitions to OpenAI function schemas."""

from typing import Any


def mcp_input_schema_to_parameters(input_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize an MCP tool input schema for OpenAI ``parameters``."""
    if not input_schema:
        return {"type": "object", "properties": {}}

    params = dict(input_schema)
    if params.get("type") is None:
        params["type"] = "object"
    if params["type"] == "object" and "properties" not in params:
        params["properties"] = {}
    return params


def mcp_tool_to_openai_schema(
    name: str,
    *,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a single MCP tool definition to OpenAI chat-completions tool format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (description or name).strip(),
            "parameters": mcp_input_schema_to_parameters(input_schema),
        },
    }
