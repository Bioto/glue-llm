"""Process status events for GlueLLM LLM execution.

This module defines event types emitted during complete(), stream_complete(),
structured_complete(), response(), and structured_response() so callers can
observe progress (LLM calls, tool execution, streaming chunks) via an optional
on_status callback, typed sinks, or a StatusEmitter.
"""

import inspect
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Event kind literals for type-safe payloads
ProcessEventKind = Literal[
    "llm_call_start",
    "llm_call_end",
    "llm_call_error",
    "tool_call_start",
    "tool_call_end",
    "tool_route",
    "stream_start",
    "stream_chunk",
    "stream_end",
    "model_fallback",
    "complete",
]


class ProcessEvent(BaseModel):
    """A status event emitted during LLM process execution.

    Use the `kind` field to discriminate; optional payload fields are
    populated depending on the event kind.
    """

    kind: ProcessEventKind = Field(description="Event type discriminator")
    correlation_id: str | None = Field(default=None, description="Request correlation ID")
    timestamp: float | None = Field(default=None, description="Event time (e.g. time.time())")

    # llm_call_start / llm_call_end / llm_call_error
    iteration: int | None = Field(default=None, description="Tool loop iteration (1-based)")
    model: str | None = Field(default=None, description="Model identifier")
    message_count: int | None = Field(default=None, description="Number of messages in the request")
    tool_call_count: int | None = Field(default=None, description="Number of tool calls in the response")
    token_usage: dict[str, int] | None = Field(default=None, description="Token usage dict")
    estimated_cost_usd: float | None = Field(default=None, description="Estimated USD cost for this LLM call")
    error_type: str | None = Field(default=None, description="Exception class name for llm_call_error events")

    # model_fallback
    from_model: str | None = Field(default=None, description="Model that failed before fallback")
    to_model: str | None = Field(default=None, description="Next model in fallback chain")

    # tool_call_start / tool_call_end
    tool_name: str | None = Field(default=None, description="Name of the tool")

    # tool_route
    route_query: str | None = Field(default=None, description="Query used for tool routing")
    matched_tools: list[str] | None = Field(default=None, description="Tool names matched by router")
    call_index: int | None = Field(default=None, description="1-based index of this tool call in the round")
    success: bool | None = Field(default=None, description="Whether the tool call succeeded")
    duration_seconds: float | None = Field(default=None, description="Tool execution duration")
    error: str | None = Field(default=None, description="Error message if tool failed")

    # stream_chunk
    content: str | None = Field(default=None, description="Chunk content (for stream_chunk)")
    done: bool | None = Field(default=None, description="Whether stream is done (for stream_chunk)")

    # complete
    tool_calls_made: int | None = Field(default=None, description="Total tool calls made")
    response_length: int | None = Field(default=None, description="Final response length")

    model_config = {"extra": "allow"}


@runtime_checkable
class Sink(Protocol):
    """Protocol for ProcessEvent sinks. Implement handle() to receive events."""

    async def handle(self, event: ProcessEvent) -> None:
        """Process the event. Sync or async implementations supported."""
        ...


class ConsoleSink:
    """Prints events to stderr."""

    async def handle(self, event: ProcessEvent) -> None:
        print(f"[{event.kind}] {event.model_dump_json()}", file=sys.stderr)


class JsonFileSink:
    """Appends events as JSON lines to a file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def handle(self, event: ProcessEvent) -> None:
        import aiofiles

        async with aiofiles.open(self._path, "a") as f:
            await f.write(event.model_dump_json() + "\n")


OnStatusCallback = Callable[[ProcessEvent], None] | Callable[[ProcessEvent], Awaitable[None]] | None


class StatusEmitter:
    """Fan-out dispatcher for process status events."""

    def __init__(
        self,
        on_status: OnStatusCallback = None,
        sinks: list[Sink] | None = None,
    ) -> None:
        self.on_status = on_status
        self.sinks = list(sinks) if sinks else []

    def bind(
        self,
        on_status: OnStatusCallback = None,
        sinks: list[Sink] | None = None,
    ) -> "StatusEmitter":
        """Merge instance-level observers with per-call overrides."""
        merged_on_status = on_status if on_status is not None else self.on_status
        merged_sinks = list(self.sinks)
        if sinks:
            merged_sinks.extend(sinks)
        return StatusEmitter(on_status=merged_on_status, sinks=merged_sinks or None)

    async def emit(self, event: ProcessEvent) -> None:
        """Dispatch an event to the bound callback and sinks."""
        await emit_status(event, self.on_status, sinks=self.sinks or None)


async def emit_status(
    event: ProcessEvent,
    on_status: OnStatusCallback = None,
    sinks: list[Sink] | None = None,
) -> None:
    """Invoke on_status and sinks with the event.

    Call from api.py with: await emit_status(event, on_status, sinks=sinks).
    Both on_status and sinks are optional; no-op when neither is provided.
    Supports sync and async callbacks for on_status. Exceptions are not caught.
    """
    if on_status is not None:
        result = on_status(event)
        if inspect.iscoroutine(result):
            await result
    if sinks:
        for sink in sinks:
            await sink.handle(event)
