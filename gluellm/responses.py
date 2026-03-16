"""OpenResponses API wrapper for agentic AI workflows.

This module provides a clean interface to the OpenResponses specification
(https://github.com/openresponsesspec/openresponses), supporting built-in
tools like web search, code interpreter, and file search.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gluellm.config import settings

# Built-in tools for OpenAI Responses API
WEB_SEARCH = {"type": "web_search_preview"}
CODE_INTERPRETER = {"type": "code_interpreter"}
FILE_SEARCH = {"type": "file_search"}


@dataclass
class ResponseResult:
    """Parsed result from an OpenResponses API call.

    Attributes:
        output: Final text output from the model.
        output_items: Raw response output items (messages, tool calls, etc.).
        tool_calls: Any tool calls made during the response.
        usage: Token usage (prompt_tokens, completion_tokens, total_tokens) if available.
        model: Model identifier used.
        raw_response: The raw ResponseResource or Response from the provider.
    """

    output: str
    output_items: list[Any]
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any] | None
    model: str
    raw_response: Any


def _extract_output_text(resp: Any) -> str:
    """Extract final text from response object."""
    if hasattr(resp, "output_text") and resp.output_text is not None:
        return str(resp.output_text)
    output = getattr(resp, "output", None) or []
    texts = []
    for item in output:
        if isinstance(item, dict):
            if item.get("type") == "message" and "content" in item:
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
            elif item.get("type") == "output_text":
                texts.append(item.get("text", ""))
        elif hasattr(item, "type") and getattr(item, "type", None) == "output_text":
            texts.append(getattr(item, "text", ""))
        elif hasattr(item, "content"):
            # Message/object with nested content (OpenResponses ResponseResource)
            for part in getattr(item, "content", []) or []:
                if hasattr(part, "type") and getattr(part, "type", None) == "output_text":
                    texts.append(getattr(part, "text", ""))
                elif isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    return "".join(texts)


def _extract_tool_calls(resp: Any) -> list[dict[str, Any]]:
    """Extract tool calls from response output."""
    output = getattr(resp, "output", None) or []
    tool_calls = []
    for item in output:
        if isinstance(item, dict):
            if item.get("type") in ("function_call", "tool_call", "web_search_call", "code_interpreter_call"):
                tool_calls.append(item)
        elif hasattr(item, "type") and "call" in str(getattr(item, "type", "")):
            tool_calls.append(item if isinstance(item, dict) else {"raw": str(item)})
    return tool_calls


def _extract_usage(resp: Any) -> dict[str, Any] | None:
    """Extract usage from response."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return {"prompt_tokens": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens"),
            "completion_tokens": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens"),
            "total_tokens": getattr(usage, "total_tokens", None)}


async def responses(
    prompt: str,
    model: str | None = None,
    *,
    provider: str | None = None,
    tools: list[Callable[..., Any] | dict[str, Any]] | None = None,
    instructions: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    reasoning: dict[str, Any] | None = None,
    stream: bool = False,
    api_key: str | None = None,
    **kwargs: Any,
) -> ResponseResult | Any:
    """Execute an OpenResponses API call with optional tool handling.

    Wraps any_llm.aresponses for the OpenResponses specification, providing
    built-in tools support (web search, code interpreter, file search).

    Args:
        prompt: User prompt or input text.
        model: Model identifier (e.g. "gpt-4.1-mini"). Defaults to settings.default_model.
        provider: Provider name (e.g. "openai"). Extracted from model if not set.
        tools: Optional tools (callables or dicts). Use WEB_SEARCH, CODE_INTERPRETER,
            FILE_SEARCH for built-in tools.
        instructions: System/developer instructions.
        max_output_tokens: Max tokens to generate.
        temperature: Sampling temperature (0.0-2.0).
        reasoning: Config for reasoning models.
        stream: If True, returns async iterator of events instead of ResponseResult.
        api_key: Optional API key override.
        **kwargs: Additional parameters passed to any_llm.aresponses.

    Returns:
        ResponseResult with output, output_items, tool_calls, usage, model, raw_response.
        If stream=True, returns AsyncIterator[ResponseStreamEvent].

    Example:
        >>> from gluellm.responses import responses, WEB_SEARCH
        >>>
        >>> result = await responses(
        ...     "What's the latest news about AI?",
        ...     tools=[WEB_SEARCH],
        ... )
        >>> print(result.output)
    """
    from gluellm.rate_limiting.api_key_pool import extract_provider_from_model

    model_str = model or settings.default_model
    prov = provider or extract_provider_from_model(model_str)
    if ":" in model_str and provider is None:
        prov, model_str = model_str.split(":", 1)

    call_kwargs: dict[str, Any] = {
        "provider": prov,
        "input_data": prompt,
        "model": model_str,
        "tools": tools,
        "instructions": instructions,
        "max_output_tokens": max_output_tokens or settings.default_max_tokens,
        "temperature": temperature,
        "reasoning": reasoning,
        "stream": stream,
        "api_key": api_key,
        **kwargs,
    }
    call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

    from any_llm import aresponses

    raw = await aresponses(**call_kwargs)

    if stream:
        return raw

    output_items = list(getattr(raw, "output", []) or [])
    return ResponseResult(
        output=_extract_output_text(raw),
        output_items=output_items,
        tool_calls=_extract_tool_calls(raw),
        usage=_extract_usage(raw),
        model=getattr(raw, "model", model_str),
        raw_response=raw,
    )
