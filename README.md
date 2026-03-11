# GlueLLM

> **TL;DR:** A high-level Python SDK for LLMs that handles the annoying stuff (tools, retries, structured output, batching) so you can ship features instead of glue code.

GlueLLM is opinionated in the “I’ve been burned by this in production” way. If you like sensible defaults, clear APIs, and fewer bespoke wrappers, you’ll feel at home.

## What is this?

GlueLLM is a high-level SDK that makes working with LLMs actually pleasant:

- You call `complete()` or `structured_complete()` and get results.
- Tools are plain Python functions.
- Retries and error classification are built-in.
- Batching and rate limiting are first-class.
- Providers are unified via `any-llm-sdk`.

## Why you might like it

- **Zero ceremony**: minimal code to get real results
- **Tool execution loop**: automatic tool calling orchestration
- **Structured output**: Pydantic models, validated (including streaming: parse on final chunk)
- **Streaming**: `stream_complete()` with optional structured output on the last chunk
- **Process status events**: optional `on_status` callback for LLM/tool/stream progress
- **Provider-agnostic**: one API for OpenAI, Anthropic, XAI, and others
- **Embeddings**: same ergonomics + error handling
- **Batch processing**: concurrency control, retry strategies, key pools
- **Observability hooks**: logging + optional tracing
- **Context condensing** *(opt-in)*: compress completed tool rounds to reduce prompt tokens across long tool chains
- **Dynamic tool routing** *(opt-in)*: route to relevant tools on demand instead of sending all schemas every call

## Why you might not

- If you want a thin client that exposes every raw provider knob, GlueLLM isn’t trying to be that.
- If you hate opinions, you’ll hate opinions (mine included).

## Installation

```bash
# Using uv (recommended)
uv pip install gluellm

# From source (dev)
uv pip install -e ".[dev]"
```

## Quick start

### Simple completion

```python
import asyncio
from gluellm.api import complete

async def main():
    result = await complete(
        user_message="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
    )
    print(result.final_response)

asyncio.run(main())
```

### Tool calling (tools are just functions)

```python
import asyncio
from gluellm.api import complete

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 22°{unit[0].upper()}, sunny"

async def main():
    result = await complete(
        user_message="What's the weather in Tokyo and Paris?",
        system_prompt="Use get_weather for weather queries.",
        tools=[get_weather],
    )
    print(result.final_response)

asyncio.run(main())
```

### Structured output

```python
import asyncio
from pydantic import BaseModel, Field
from typing import Annotated

from gluellm.api import structured_complete

class PersonInfo(BaseModel):
    name: Annotated[str, Field(description="Full name")]
    age: Annotated[int, Field(description="Age in years")]
    city: Annotated[str, Field(description="City of residence")]

async def main():
    person = await structured_complete(
        user_message="Extract info: John Smith, 35, lives in Seattle",
        response_format=PersonInfo,
    )
    print(person.model_dump())

asyncio.run(main())
```

### Streaming

Stream token-by-token with `stream_complete()`. When tools are enabled, the final response after tool runs is returned as one chunk (streaming resumes between tool rounds).

```python
import asyncio
from gluellm import stream_complete

async def main():
    async for chunk in stream_complete("Tell me a short joke."):
        print(chunk.content, end="", flush=True)
        if chunk.done:
            print(f"\nTool calls: {chunk.tool_calls_made}")

asyncio.run(main())
```

**Streaming + structured output:** Pass `response_format` to get a parsed Pydantic instance on the final chunk (the stream is plain text; we parse when the stream ends).

```python
from pydantic import BaseModel, Field
from gluellm import stream_complete

class Answer(BaseModel):
    word: str

async for chunk in stream_complete(
    "Reply with JSON: {\"word\": \"hello\"}",
    response_format=Answer,
    tools=[],
):
    if chunk.done and chunk.structured_output:
        print(chunk.structured_output.word)  # hello
```

### Context condensing (opt-in)

When a tool round completes, `condense_tool_messages=True` replaces the raw `assistant(tool_calls) + tool(results)` messages with a single compact summary. This keeps the prompt from growing linearly with every tool call — useful for long multi-step chains.

**Off by default.** Enable per-call or on the client:

```python
# Per-call
result = await complete(
    "Do ten things with tools...",
    tools=[...],
    condense_tool_messages=True,  # opt-in
)

# On the client (applies to all calls)
client = GlueLLM(tools=[...], condense_tool_messages=True)
result = await client.complete("Do ten things with tools...")
```

Without condensing, context grows by 2 messages per tool round (assistant + tool). With condensing, each completed round collapses to 1 message regardless of how many tools ran in parallel.

### Dynamic tool routing (opt-in)

In standard mode every LLM call sees the full list of tool schemas in the system prompt. With a large toolset this wastes tokens and increases latency. `tool_mode="dynamic"` replaces the upfront schema dump with a lightweight router call: the LLM is first asked *which* tools it needs, then only those schemas are injected for the actual tool execution.

**Off by default (`tool_mode="standard"`).** Enable per-call or on the client:

```python
# Per-call
result = await complete(
    "Check the weather and search flights...",
    tools=[get_weather, search_flights, book_hotel, calculate, ...],
    tool_mode="dynamic",  # opt-in
)

# On the client
client = GlueLLM(
    tools=[...],
    tool_mode="dynamic",
    tool_route_model="openai:gpt-4o-mini",  # fast cheap model for routing
)
result = await client.complete("Check the weather and search flights...")
```

Dynamic routing is most effective when you have 6+ tools and the task only uses a few of them per call. For small toolsets or when every call uses most tools, standard mode is simpler and equally efficient.

Both `condense_tool_messages` and `tool_mode` can be combined:

```python
result = await complete(
    "Plan a trip with 9 sequential steps...",
    tools=[...],
    condense_tool_messages=True,
    tool_mode="dynamic",
)
```

### Process status events

Use the optional `on_status` callback to observe what’s happening (LLM call start/end, tool execution, stream start/chunk/end, completion). Handy for progress UIs or logging.

```python
from gluellm import complete, ProcessEvent

def on_status(e: ProcessEvent) -> None:
    print(f"{e.kind}: {e.tool_name or e.iteration or ''}")

result = await complete(
    "What is 2+2?",
    on_status=on_status,
)
# llm_call_start, llm_call_end, complete (and tool_call_* if tools run)
```

`on_status` is supported on `complete()`, `stream_complete()`, and `structured_complete()` (and the `GlueLLM` client methods).

### Embeddings

```python
import asyncio
from gluellm import embed

async def main():
    result = await embed("Hello, world!")
    print(result.dimension, result.tokens_used)

asyncio.run(main())
```

## Configuration (the boring part)

Providers are configured via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export XAI_API_KEY=xai-...
```

Models use `provider:model` strings:

- `openai:gpt-4o-mini`
- `anthropic:claude-3-5-sonnet-20241022`

## Docs (when you want the details)

GlueLLM keeps deeper docs in `docs/` so the README stays readable:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/BATCH_PROCESSING.md`](docs/BATCH_PROCESSING.md)
- [`docs/CONNECTION_POOLING.md`](docs/CONNECTION_POOLING.md)
- [`docs/WORKFLOW_PATTERNS.md`](docs/WORKFLOW_PATTERNS.md)
- [`docs/CONTEXT_OPTIMIZATION.md`](docs/CONTEXT_OPTIMIZATION.md) — condensing + dynamic routing deep-dive

More runnable examples live in [`examples/`](examples/).

## Contributing

PRs welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
