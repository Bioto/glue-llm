"""Process status events for GlueLLM LLM execution.

This module defines event types emitted during complete(), stream_complete(),
and structured_complete() so callers can observe progress (LLM calls, tool
execution, streaming chunks) via an optional on_status callback or typed sinks.
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
    "tool_call_start",
    "tool_call_end",
    "tool_route",
    "stream_start",
    "stream_chunk",
    "stream_end",
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

    # llm_call_start / llm_call_end
    iteration: int | None = Field(default=None, description="Tool loop iteration (1-based)")
    model: str | None = Field(default=None, description="Model identifier")
    message_count: int | None = Field(default=None, description="Number of messages in the request")
    has_tool_calls: bool | None = Field(default=None, description="Whether response requested tool calls")
    token_usage: dict[str, int] | None = Field(default=None, description="Token usage dict")

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


async def emit_status(
    event: ProcessEvent,
    on_status: Callable[[ProcessEvent], None] | Callable[[ProcessEvent], Awaitable[None]] | None = None,
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
