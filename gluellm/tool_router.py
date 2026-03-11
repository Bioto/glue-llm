"""Dynamic tool routing for GlueLLM.

When tool_mode="dynamic", the primary LLM sees only a single search_tools
router tool. When it calls that tool, a fast LLM resolves which real tools
match the query, and those are injected for the next iteration. The router
call is scrubbed from message history to keep context lean.
"""

import asyncio
import json
import re
from collections.abc import Callable
from typing import Literal

from any_llm import acompletion as any_llm_acompletion

from gluellm.config import settings

ToolMode = Literal["standard", "dynamic"]
ROUTER_TOOL_NAME = "search_tools"


def _tool_catalog(tools: list[Callable]) -> str:
    """Build a compact catalog of tool names and first-line descriptions."""
    lines = []
    for tool in tools:
        desc = (tool.__doc__ or "").strip().split("\n")[0].strip()
        lines.append(f"- {tool.__name__}: {desc or '(no description)'}")
    return "\n".join(lines)


def build_router_tool(tools: list[Callable]) -> Callable[..., str]:
    """Build the synthetic search_tools router function.

    The docstring lists all available tools so the primary LLM knows
    what to ask for. The function itself is a placeholder; routing
    is handled by resolve_tool_route() when the tool is intercepted.
    """

    catalog = _tool_catalog(tools)

    def search_tools(query: str) -> str:
        """Search for tools you need to complete your task.

        Available tools:
        {catalog}

        Args:
            query: Describe what you need to accomplish so the right tools can be found.
        """
        # This is never actually called - api.py intercepts the router call
        # and invokes resolve_tool_route() instead. The return value is unused.
        return ""

    search_tools.__name__ = ROUTER_TOOL_NAME
    search_tools.__doc__ = f"""Search for tools you need to complete your task.

Available tools:
{catalog}

Args:
    query: Describe what you need to accomplish so the right tools can be found.
"""
    return search_tools


def is_router_call(tool_calls: list) -> bool:
    """Check if the tool calls contain the search_tools router."""
    if not tool_calls:
        return False
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", None)
        if name == ROUTER_TOOL_NAME:
            return True
    return False


async def resolve_tool_route(
    query: str,
    tools: list[Callable],
    *,
    model: str,
    api_key: str | None = None,
    timeout: float | None = None,
) -> list[Callable]:
    """Use a fast LLM to select which tools match the user's query.

    Sends the tool catalog and query to a cheap model, which returns a
    JSON array of tool names. Falls back to all tools if the call fails
    or the response is unparseable.
    """
    if not tools:
        return []

    catalog = _tool_catalog(tools)
    tool_names = [t.__name__ for t in tools]

    prompt = f"""Given a user's query and a catalog of available tools, return the names of the tools that are relevant for the task.

User query: {query}

Tool catalog:
{catalog}

Return ONLY a JSON array of tool names, e.g. ["get_weather", "calculate"]. Do not include any other text."""

    messages = [
        {"role": "system", "content": "You return only JSON arrays of tool names. No explanation."},
        {"role": "user", "content": prompt},
    ]

    effective_timeout = timeout or settings.default_request_timeout

    try:
        response = await asyncio.wait_for(
            any_llm_acompletion(
                messages=messages,
                model=model,
                tools=None,
                stream=False,
            ),
            timeout=effective_timeout,
        )
    except Exception:
        return list(tools)

    if not response.choices:
        return list(tools)

    content = getattr(response.choices[0].message, "content", None) or ""
    content = content.strip()

    # Try to parse JSON array from the response (may be wrapped in markdown)
    parsed: list[str] = []
    json_match = re.search(r"\[[\s\S]*?\]", content)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    if not isinstance(parsed, list):
        return list(tools)

    # Filter to valid tool names and preserve order of tools list
    valid_names = {n for n in parsed if isinstance(n, str) and n in tool_names}
    if not valid_names:
        return list(tools)

    name_to_tool = {t.__name__: t for t in tools}
    # Preserve order from parsed list, falling back to tools order for unspecified
    result = []
    seen = set()
    for name in parsed:
        if isinstance(name, str) and name in name_to_tool and name not in seen:
            result.append(name_to_tool[name])
            seen.add(name)

    return result if result else list(tools)
