"""Responses API streaming utilities for GlueLLM."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, Literal

StreamYieldKind = Literal["content", "reasoning", "done"]

_REASONING_DELTA_SUFFIXES = (
    "reasoning_summary_text.delta",
    "reasoning_summary.delta",
    "reasoning.delta",
)
_REASONING_DONE_SUFFIXES = (
    "reasoning_summary_text.done",
    "reasoning_summary.done",
    "reasoning.done",
)


async def consume_responses_stream_with_tools(
    stream_iter: AsyncIterator[Any],
) -> AsyncIterator[tuple[StreamYieldKind, str, SimpleNamespace | None]]:
    """Consume a streaming Responses API response with optional function calls.

    Yields ``(kind, text, message)`` triples:
    - ``("content", delta, None)`` for answer text deltas
    - ``("reasoning", delta, None)`` for reasoning summary text deltas
    - ``("done", accumulated_content, message)`` as the final sentinel
    """
    from gluellm.api import _adapt_response_function_call, _build_message_from_stream

    accumulated_content = ""
    accumulated_reasoning = ""
    tool_calls_accumulator: dict[int, dict[str, Any]] = {}

    async for event in stream_iter:
        event_type = _event_type(event)

        if _matches_suffix(event_type, ("output_text.delta", "response_text_delta")):
            delta_text = _extract_text_delta(event)
            if delta_text:
                accumulated_content += delta_text
                yield ("content", delta_text, None)
            continue

        if _matches_suffix(event_type, _REASONING_DELTA_SUFFIXES):
            delta_text = _extract_reasoning_summary_delta(event)
            if delta_text:
                accumulated_reasoning += delta_text
                yield ("reasoning", delta_text, None)
            continue

        if _matches_suffix(event_type, _REASONING_DONE_SUFFIXES):
            # Full text on done events — emit only the suffix we have not already
            # streamed via deltas (providers may send both).
            full_text = _extract_reasoning_done_text(event)
            if full_text and full_text.startswith(accumulated_reasoning):
                remainder = full_text[len(accumulated_reasoning) :]
                if remainder:
                    accumulated_reasoning = full_text
                    yield ("reasoning", remainder, None)
            elif full_text and not accumulated_reasoning:
                accumulated_reasoning = full_text
                yield ("reasoning", full_text, None)
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
            # Reasoning items may only appear as completed output items (no deltas).
            reasoning_text = _extract_reasoning_from_output_item_done(event)
            if reasoning_text and not accumulated_reasoning:
                accumulated_reasoning = reasoning_text
                yield ("reasoning", reasoning_text, None)
            _merge_completed_function_call(event, tool_calls_accumulator)

    # Final response object (when the SDK attaches it) can still hold a summary
    # even if no reasoning stream events were emitted.
    final_resp = getattr(stream_iter, "response", None)
    if final_resp is not None and not accumulated_reasoning:
        from gluellm.api import _extract_response_reasoning

        leftover = _extract_response_reasoning(final_resp)
        if leftover:
            accumulated_reasoning = leftover
            yield ("reasoning", leftover, None)

    message = _build_message_from_stream(accumulated_content, tool_calls_accumulator)
    yield ("done", accumulated_content, message)


def _matches_suffix(event_type: str, suffixes: tuple[str, ...]) -> bool:
    return any(event_type == s or event_type.endswith(s) for s in suffixes)


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        raw = event.get("type", "")
    else:
        raw = getattr(event, "type", "")
    if hasattr(raw, "value"):
        return str(raw.value)
    return str(raw or "")


def _extract_text_delta(event: Any) -> str:
    if isinstance(event, dict):
        delta = event.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("text", "") or "")
        return str(event.get("text", "") or delta or "")
    delta = getattr(event, "delta", None)
    if delta is not None and not isinstance(delta, str):
        return str(getattr(delta, "text", "") or "")
    if isinstance(delta, str):
        return delta
    return str(getattr(event, "text", "") or "")


def _extract_reasoning_summary_delta(event: Any) -> str:
    """Extract text from a reasoning summary delta event.

    OpenAI emits ``response.reasoning_summary_text.delta`` with a string
    ``delta`` field (not nested under ``delta.text``).
    """
    if isinstance(event, dict):
        delta = event.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("text", "") or "")
        return str(delta or "")
    delta = getattr(event, "delta", None)
    if isinstance(delta, str):
        return delta
    if delta is not None:
        return str(getattr(delta, "text", "") or "")
    return str(getattr(event, "text", "") or "")


def _extract_reasoning_done_text(event: Any) -> str:
    """Extract full text from a reasoning summary/text done event."""
    if isinstance(event, dict):
        text = event.get("text")
        if text:
            return str(text)
        delta = event.get("delta")
        return str(delta or "")
    text = getattr(event, "text", None)
    if text:
        return str(text)
    delta = getattr(event, "delta", None)
    return str(delta or "") if delta else ""


def _extract_reasoning_from_output_item_done(event: Any) -> str | None:
    """Pull summary text from a completed reasoning output item, if present."""
    item = event.get("item") if isinstance(event, dict) else getattr(event, "item", None)
    if item is None:
        return None
    item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    if hasattr(item_type, "value"):
        item_type = item_type.value
    if str(item_type) != "reasoning":
        return None

    from gluellm.api import _extract_response_reasoning

    return _extract_response_reasoning(SimpleNamespace(output=[item]))


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
    """Merge a completed function_call output item into the accumulator.

    ``response.output_item.done`` fires for every output item type (including
    ``reasoning`` and ``message``). Only ``function_call`` items are adapted.
    """
    from gluellm.api import _adapt_response_function_call

    item = event.get("item") if isinstance(event, dict) else getattr(event, "item", None)
    if item is None:
        return
    item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    if hasattr(item_type, "value"):
        item_type = item_type.value
    if str(item_type) != "function_call":
        return
    adapted = _adapt_response_function_call(item if isinstance(item, dict) else item)
    idx = len(tool_calls_accumulator)
    tool_calls_accumulator[idx] = {
        "id": adapted.id,
        "function": {
            "name": adapted.function.name,
            "arguments": adapted.function.arguments,
        },
    }
