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

### Parallel tool execution (opt-in)

By default, when the model returns multiple tool calls in a single round, they are executed sequentially. Use `tool_execution_order="parallel"` to run them concurrently (via `asyncio.gather`), which can reduce latency when tools are I/O-bound.

**Off by default (`tool_execution_order="sequential"`).** Enable globally, per client, or per call:

```python
# Per-call
result = await complete("Get weather in Tokyo and Paris", tools=[get_weather], tool_execution_order="parallel")

# On the client
client = GlueLLM(tools=[...], tool_execution_order="parallel")

# Global default (env: GLUELLM_DEFAULT_TOOL_EXECUTION_ORDER)
import gluellm
gluellm.configure(default_tool_execution_order="parallel")
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

### Timeouts

Two independent timeouts control how long GlueLLM waits for the network:

| Parameter | What it governs | Default |
|-----------|-----------------|---------|
| `connect_timeout` | Time to establish the TCP connection | 10s |
| `request_timeout` | Total time for the full LLM response | 60s |

Both can be set per-call or left at their defaults (configurable via environment variables):

```python
from gluellm import complete

# Set both per-call
result = await complete(
    "Write a short story.",
    request_timeout=120.0,   # allow 2 minutes for a long generation
    connect_timeout=5.0,     # fail fast if we can't reach the API
)

# Or just one — the other uses its default
result = await complete("Hello", request_timeout=30.0)
```

On the `GlueLLM` client, set them per method call:

```python
from gluellm import GlueLLM

client = GlueLLM()
result = await client.complete(
    "Summarise this document...",
    request_timeout=180.0,
    connect_timeout=10.0,
)
```

A connection timeout raises `APIConnectionError`; a request timeout raises `APITimeoutError` (subclass of `APIConnectionError`). Both are retried by default.

For full details and environment variable configuration, see [`docs/TIMEOUTS.md`](docs/TIMEOUTS.md).

### Retry configuration

Retries are enabled by default (exponential backoff for rate limits and connection errors). You can customise or disable them per client or per call with `retry_config` and `retry_enabled`:

```python
from gluellm import complete, GlueLLM, RetryConfig

# Disable retries for a single call
result = await complete("What is 2+2?", retry_enabled=False)

# Or pass a RetryConfig
result = await complete(
    "What is 2+2?",
    retry_config=RetryConfig(retry_enabled=False),
)

# Per-client: disable retries for all calls
client = GlueLLM(retry_config=RetryConfig(retry_enabled=False))

# Filter by exception type (only retry RateLimitError)
from gluellm import RateLimitError
result = await complete(
    "...",
    retry_config=RetryConfig(retry_on=[RateLimitError]),
)

# Custom callback: decide per error and inject params for next attempt
def on_retry(err: Exception, attempt: int) -> tuple[bool, dict | None]:
    if attempt >= 2:
        return False, None  # stop retrying
    return True, {"temperature": 0.0}  # lower temp on retry

result = await complete(
    "...",
    retry_config=RetryConfig(callback=on_retry),
)
```

When `callback` is set, it takes precedence over `retry_on`; the callback receives `(error, attempt)` and returns `(should_retry, next_params | None)`.

For full details (precedence rules, backoff formula, exception hierarchy), see [`docs/RETRY.md`](docs/RETRY.md).

### Embeddings

```python
import asyncio
from gluellm import embed

async def main():
    result = await embed("Hello, world!")
    print(result.dimension, result.tokens_used)

    # Request a specific output dimension (OpenAI text-embedding-3-* only)
    result = await embed("Hello, world!", dimensions=512)
    print(result.dimension)  # 512

asyncio.run(main())
```

The `dimensions` parameter truncates the output vector — useful for reducing storage costs while preserving most of the semantic signal. You can also set a global default via `GLUELLM_DEFAULT_EMBEDDING_DIMENSIONS` or `gluellm.configure(default_embedding_dimensions=512)` so every call uses it without repeating the argument.

## Configuration

Providers are configured via environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export XAI_API_KEY=xai-...
```

Models use `provider:model` strings:

- `openai:gpt-4o-mini`
- `anthropic:claude-3-5-sonnet-20241022`

Key GlueLLM-specific env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `GLUELLM_DEFAULT_MODEL` | `openai:gpt-4o-mini` | Default model |
| `GLUELLM_DEFAULT_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Default embedding model |
| `GLUELLM_DEFAULT_EMBEDDING_DIMENSIONS` | _(unset)_ | Default output dimensions for embeddings (e.g. `512`) |
| `GLUELLM_DEFAULT_REQUEST_TIMEOUT` | `60.0` | Request timeout (seconds) |
| `GLUELLM_MAX_REQUEST_TIMEOUT` | `300.0` | Maximum allowed request timeout |
| `GLUELLM_DEFAULT_CONNECT_TIMEOUT` | `10.0` | Connection timeout (seconds) |
| `GLUELLM_MAX_CONNECT_TIMEOUT` | `60.0` | Maximum allowed connection timeout |
| `GLUELLM_RETRY_MAX_ATTEMPTS` | `3` | Max retry attempts |
| `GLUELLM_RETRY_MIN_WAIT` | `2` | Min backoff seconds |
| `GLUELLM_RETRY_MAX_WAIT` | `30` | Max backoff seconds |
| `GLUELLM_LOG_LEVEL` | `INFO` | Console log level |
| `GLUELLM_LOG_CONSOLE_OUTPUT` | `false` | Enable console output (off by default for library usage) |
| `GLUELLM_DISABLE_LOGGING` | `false` | Disable GlueLLM logging setup (use your app's config) |

## Docs (when you want the details)

GlueLLM keeps deeper docs in `docs/` so the README stays readable:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/BATCH_PROCESSING.md`](docs/BATCH_PROCESSING.md)
- [`docs/RETRY.md`](docs/RETRY.md) — retry configuration, `RetryConfig`, callbacks
- [`docs/ERROR_HANDLING.md`](docs/ERROR_HANDLING.md) — exception hierarchy, classification, handling patterns
- [`docs/TIMEOUTS.md`](docs/TIMEOUTS.md) — `connect_timeout`, `request_timeout`, defaults, env vars
- [`docs/CONNECTION_POOLING.md`](docs/CONNECTION_POOLING.md)
- [`docs/WORKFLOW_PATTERNS.md`](docs/WORKFLOW_PATTERNS.md)
- [`docs/CONTEXT_OPTIMIZATION.md`](docs/CONTEXT_OPTIMIZATION.md) — condensing + dynamic routing deep-dive

More runnable examples live in [`examples/`](examples/).

## Contributing

PRs welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
