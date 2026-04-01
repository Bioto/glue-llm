"""Dynamic tool routing and static tool pinning for GlueLLM.

Tool modes
----------
standard (default)
    All tool schemas are injected into every LLM call. Simple and predictable;
    best for small toolsets or when most tools are used on every request.

dynamic
    Only a lightweight router tool is exposed on the first turn. A fast model
    decides which tools are actually needed, then only those schemas are
    injected. Reduces token usage and latency for large toolsets where each
    request uses only a small subset.

Static tools (@static_tool)
---------------------------
In dynamic mode some tools should always be in scope regardless of routing —
e.g. fetching the current time, reading user context, or any tool that is
nearly always needed. Decorate them with @static_tool:

    @static_tool
    def get_current_time() -> str:
        \"\"\"Get the current UTC time.\"\"\"
        return datetime.utcnow().isoformat()

Static tools are included in the initial active_tools list alongside the
router tool, and are always merged back after resolve_tool_route runs.
In standard mode the decorator has no effect.
"""

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any, Literal

from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)

ROUTER_TOOL_NAME = "request_tools"

ToolMode = Literal["standard", "dynamic"]

# Provider env vars for resolve_tool_route (mirrors api.PROVIDER_ENV_VAR_MAP)
_PROVIDER_ENV_VAR_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}


def static_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a tool as always available, bypassing dynamic routing.

    When tool_mode="dynamic", the dynamic tool router selects a subset of tools
    per request. Applying @static_tool to a function opts it out of that
    selection process — it is always included in the active tool list.

    In tool_mode="standard" this decorator has no effect; all tools are already
    always available.

    Example::

        from gluellm import GlueLLM, static_tool

        @static_tool
        def get_current_time() -> str:
            \"\"\"Get the current UTC time.\"\"\"
            return datetime.utcnow().isoformat()

        def search_products(query: str) -> list[str]:
            \"\"\"Search the product catalog.\"\"\"
            ...

        client = GlueLLM(
            tools=[get_current_time, search_products],
            tool_mode="dynamic",
        )
        # get_current_time is always injected; search_products goes through routing

    Args:
        fn: The tool callable to mark as static.

    Returns:
        The same callable, with ``_gluellm_static = True`` set on it.
    """
    fn._gluellm_static = True  # type: ignore[attr-defined]
    return fn


def is_static_tool(fn: Callable[..., Any]) -> bool:
    """Return True if the callable is marked as a static (always-available) tool."""
    return getattr(fn, "_gluellm_static", False)


def build_router_tool(tools: list[Callable[..., Any]]) -> Callable[..., str]:
    """Build the meta router tool that the LLM calls to discover available tools."""
    tool_names = [getattr(fn, "__name__", str(fn)) for fn in tools]
    names_str = ", ".join(tool_names)

    def request_tools(query: str) -> str:
        """Internal use only — never normally invoked at runtime."""
        return ""

    request_tools.__name__ = ROUTER_TOOL_NAME
    request_tools.__doc__ = (
        "Call this first to discover which tools are needed for the user's request. "
        f"Available tools: {names_str}. "
        "Pass the user's query (or a short summary) as the 'query' argument. "
        "After calling, you will receive the relevant tool schemas."
    )
    return request_tools


def is_router_call(tool_calls: Any) -> bool:
    """Return True if any of the tool calls is the router tool."""
    if not tool_calls or len(tool_calls) == 0:
        return False
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", None)
        if name == ROUTER_TOOL_NAME:
            return True
    return False


async def any_llm_acompletion(
    *,
    messages: list,
    model: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call the LLM provider via the shared provider cache.

    Defined at module level so tests can patch ``gluellm.tool_router.any_llm_acompletion``
    without touching the provider cache or network.
    """
    # Late import to avoid circular dependency (api imports us)
    from gluellm.api import _provider_cache

    if ":" in model:
        provider_name, _ = model.split(":", 1)
    elif "/" in model:
        provider_name, _ = model.split("/", 1)
    else:
        provider_name = model
    provider_name = provider_name.lower()

    resolved_key = api_key
    if resolved_key is None:
        env_var = _PROVIDER_ENV_VAR_MAP.get(provider_name)
        if env_var:
            resolved_key = os.environ.get(env_var)

    provider, model_id = _provider_cache.get_provider(model, resolved_key)
    return await provider.acompletion(model=model_id, messages=messages, **kwargs)


async def resolve_tool_route(
    user_context: str,
    tools: list[Callable[..., Any]],
    *,
    model: str,
    api_key: str | None = None,
    timeout: float | None = None,
) -> list[Callable[..., Any]]:
    """Use an LLM to select the subset of tools relevant to the user context.

    Returns all tools on any error or unparseable response, so callers always
    get a usable toolset even when routing fails.
    """
    if not tools:
        logger.debug("resolve_tool_route called with empty tools list")
        return []

    tool_names_available = [getattr(fn, "__name__", str(fn)) for fn in tools]
    logger.debug(
        f"Starting dynamic tool routing: model={model}, available_tools={tool_names_available}, "
        f"context_preview={user_context[:100]}..."
    )

    tool_descriptions = []
    name_to_fn: dict[str, Callable[..., Any]] = {}
    for fn in tools:
        name = getattr(fn, "__name__", str(fn))
        desc = (getattr(fn, "__doc__") or "").strip().split("\n")[0] or name
        tool_descriptions.append(f"- {name}: {desc}")
        name_to_fn[name] = fn

    system = (
        "You are a tool router. Given a user request, select which tools are needed. "
        'Respond with a JSON array of tool names: ["name1", "name2", ...] '
        "Use only the exact tool names from the list. If no tools are needed, use []."
    )
    tools_list = "\n".join(tool_descriptions)
    user_msg = (
        f"User request:\n{user_context}\n\n"
        f"Available tools:\n{tools_list}\n\n"
        "Which tools are needed? Respond with a JSON array only."
    )

    request_timeout = timeout or 30.0
    try:
        response = await asyncio.wait_for(
            any_llm_acompletion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                model=model,
                max_tokens=256,
                api_key=api_key,
            ),
            timeout=request_timeout,
        )
    except Exception as e:
        logger.warning(
            f"Tool routing failed ({type(e).__name__}: {e}), falling back to all {len(tools)} tools"
        )
        return list(tools)

    text = ""
    if response.choices:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        text = content or ""

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            logger.warning(
                f"Tool routing returned non-list response: {type(parsed).__name__}, "
                f"falling back to all {len(tools)} tools"
            )
            return list(tools)
        names = parsed
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(
            f"Tool routing returned invalid JSON ({type(e).__name__}): {text[:200]}, "
            f"falling back to all {len(tools)} tools"
        )
        return list(tools)

    result = [name_to_fn[n] for n in names if n in name_to_fn]
    unrecognized = [n for n in names if n not in name_to_fn]
    if unrecognized:
        logger.warning(f"Tool routing returned unrecognized tool names: {unrecognized}")

    logger.info(
        f"Tool routing selected {len(result)}/{len(tools)} tools: {[fn.__name__ for fn in result]}"
    )
    return result
