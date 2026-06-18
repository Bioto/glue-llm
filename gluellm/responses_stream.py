"""Responses API streaming utilities for GlueLLM."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any


async def consume_responses_stream_with_tools(
    stream_iter: AsyncIterator[Any],
) -> AsyncIterator[tuple[bool, str, SimpleNamespace | None]]:
    """Consume a streaming Responses API response with optional function calls."""
    from gluellm.api import _adapt_response_function_call, _build_message_from_stream

    accumulated_content = ""
    tool_calls_accumulator: dict[int, dict[str, Any]] = {}

    async for event in stream_iter:
        event_type = _event_type(event)
        if event_type in ("response.output_text.delta", "response_text_delta", "output_text.delta"):
            delta_text = _extract_text_delta(event)
            if delta_text:
                accumulated_content += delta_text
                yield (True, delta_text, None)
            continue

        if event_type in (
            "response.function_call_arguments.delta",
            "function_call_arguments.delta",
        ):
            idx, call_id, name, args_delta = _extract_function_call_delta(event)
            if idx is None:
                continue
            if idx not in tool_calls_accumulator:
                tool_calls_accumulator[idx] = {
                    "id": call_id,
                    "function": {"name": name or "", "arguments": ""},
                }
            acc = tool_calls_accumulator[idx]
            if call_id:
                acc["id"] = call_id
            if name:
                acc["function"]["name"] = name
            if args_delta:
                acc["function"]["arguments"] += args_delta
            continue

        if event_type in ("response.output_item.done", "response.function_call_arguments.done"):
            _merge_completed_function_call(event, tool_calls_accumulator)

    message = _build_message_from_stream(accumulated_content, tool_calls_accumulator)
    yield (False, accumulated_content, message)


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type", ""))
    return str(getattr(event, "type", ""))


def _extract_text_delta(event: Any) -> str:
    if isinstance(event, dict):
        delta = event.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("text", "") or "")
        return str(event.get("text", "") or "")
    delta = getattr(event, "delta", None)
    if delta is not None:
        return str(getattr(delta, "text", "") or "")
    return str(getattr(event, "text", "") or "")


def _extract_function_call_delta(
    event: Any,
) -> tuple[int | None, str | None, str | None, str]:
    if isinstance(event, dict):
        idx = event.get("output_index", event.get("index"))
        call_id = event.get("call_id") or event.get("id")
        name = event.get("name")
        delta = event.get("delta", "")
        if isinstance(delta, dict):
            args_delta = str(delta.get("arguments", "") or "")
            name = name or delta.get("name")
        else:
            args_delta = str(delta or "")
        return (int(idx) if idx is not None else 0, call_id, name, args_delta)

    idx = getattr(event, "output_index", getattr(event, "index", 0))
    call_id = getattr(event, "call_id", getattr(event, "id", None))
    name = getattr(event, "name", None)
    delta = getattr(event, "delta", "")
    if hasattr(delta, "arguments"):
        args_delta = str(getattr(delta, "arguments", "") or "")
        name = name or getattr(delta, "name", None)
    else:
        args_delta = str(delta or "")
    return (int(idx) if idx is not None else 0, call_id, name, args_delta)


def _merge_completed_function_call(
    event: Any,
    tool_calls_accumulator: dict[int, dict[str, Any]],
) -> None:
    item = event.get("item") if isinstance(event, dict) else getattr(event, "item", None)
    if item is None:
        return
    adapted = _adapt_response_function_call(item if isinstance(item, dict) else item)
    if adapted is None:
        return
    idx = len(tool_calls_accumulator)
    tool_calls_accumulator[idx] = {
        "id": adapted.id,
        "function": {
            "name": adapted.function.name,
            "arguments": adapted.function.arguments,
        },
    }
