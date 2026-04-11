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
- **Context condensing** *(opt-in)*: compress completed tool rounds; optionally with **AAAK lossless encoding** that preserves technical facts exactly across long conversations
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
    result = await structured_complete(
        user_message="Extract info: John Smith, 35, lives in Seattle",
        response_format=PersonInfo,
    )
    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")
    person = result.structured_output  # typed as PersonInfo
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

### AAAK lossless compression (opt-in)

AAAK is a structured shorthand encoding that makes context compression **lossless for technical facts** — exact numbers, config values, algorithm names, security attributes, and ordered multi-step flows survive compression intact. It replaces the default lossy prose summarization when you can't afford to drop details.

Two levels, composable:

| Level | Trigger | LLM call? | Output |
|---|---|---|---|
| Tool-round encoding | `condense_tool_messages=True` + `aaak_tool_condensing=True` | ❌ none | `[AT]` block |
| History compression | `summarize_context=True` + `aaak_compression_enabled=True` | ✅ one extra call | `[AAAK CTX]` block |

```python
from gluellm import GlueLLM, SummarizeContextConfig

client = GlueLLM(
    model="openai:gpt-4o",
    condense_tool_messages=True,       # collapses each tool round into an [AT] block (free)
    summarize_context=SummarizeContextConfig(
        enabled=True,                  # compresses old history when it grows long
        threshold=20,                # compress after 20 non-system messages
        keep_recent=6,               # always keep last 6 messages verbatim
    ),
    aaak_compression_enabled=True,     # opt-in; enable AAAK explicitly (not implied by summarize_context alone)
    aaak_compression_model="openai:gpt-4-turbo",  # stronger compressor → smaller context
)
```

**What gets preserved exactly:** numbers with units (`8500ms`, `15min`), config key=value pairs (`pool_size=10`, `JWT_SECRET_KEY`), security cookie attributes (`HttpOnly,Secure,SameSite=Strict`), algorithm names (`HS256`), HTTP verbs and paths, ordered multi-step flows (`1→DELETE /auth/session;2→hash;3→revoke`).

**When it pays off:** long technical conversations (API design, infra, auth, SRE incidents, DB schemas) where context will span many turns. The compression call is paid once and the smaller context is reused for every subsequent turn.

**When it doesn't:** short conversations, mathematical reasoning, general prose, single one-shot queries. Below ~400 tokens of context the overhead exceeds the savings — GlueLLM won't compress if there isn't enough to compress.

**Out-of-domain benchmark results** (NarrativeQA, long narrative prose — not AAAK's intended domain):

| Compressor | Compression ratio | Accuracy preserved |
|---|---|---|
| `llama-3.1-8b-instant` | 4.3× | 78.9% |
| `gpt-4-turbo` | 15.6× | 89.9% |

On its home turf (agent conversation history) AAAK targets lossless recall — see `benchmarks/aaak_live_benchmark.py`.

→ **[Full reference: docs/AAAK_COMPRESSION.md](docs/AAAK_COMPRESSION.md)**

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
    tool_route_model="openai:gpt-5.4-nano",  # fast cheap model for routing
)
result = await client.complete("Check the weather and search flights...")
```

Dynamic routing is most effective when you have 6+ tools and the task only uses a few of them per call. For small toolsets or when every call uses most tools, standard mode is simpler and equally efficient.

#### Always-available tools with `@static_tool`

Some tools should never go through routing — utility functions like getting the current time, fetching the user's profile, or anything that should always be in scope. Decorate them with `@static_tool` to pin them to every LLM call, bypassing the router entirely.

```python
from gluellm import GlueLLM, static_tool

@static_tool
def get_current_time() -> str:
    """Get the current UTC time."""
    return datetime.utcnow().isoformat()

def search_products(query: str) -> list[str]:
    """Search the product catalog."""
    ...

def check_inventory(sku: str) -> int:
    """Check stock level for a product."""
    ...

client = GlueLLM(
    tools=[get_current_time, search_products, check_inventory],
    tool_mode="dynamic",
)
# get_current_time is always injected; search_products and check_inventory go through routing
result = await client.complete("Find a widget and tell me if it's in stock, also what time is it?")
```

In `tool_mode="standard"` the decorator has no effect — all tools are always present anyway.

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

### Reasoning effort and logprobs

For reasoning models (o3, o4-mini, Claude 3.7 thinking), use `reasoning_effort` to control thinking depth:

```python
result = await complete(
    "Solve this logic puzzle...",
    model="openai:o4-mini",
    reasoning_effort="high",  # "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"auto"
)
```

Set a default via `GLUELLM_DEFAULT_REASONING_EFFORT` or on the client.

For eval and confidence scoring, enable `logprobs`:

```python
result = await complete(
    "Is this claim true?",
    logprobs=True,
    top_logprobs=5,
)
# Inspect token-level probabilities in the raw response
```

### OpenResponses API (agentic tools)

The `responses()` API supports OpenAI's agentic interface with built-in tools like web search, code interpreter, and file search:

```python
from gluellm import responses, WEB_SEARCH

result = await responses(
    "What's the latest news about AI?",
    tools=[WEB_SEARCH],
)
print(result.output)
# Also: result.tool_calls, result.usage
```

Built-in tool constants: `WEB_SEARCH`, `CODE_INTERPRETER`, `FILE_SEARCH`.

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

### Listing models

Enumerate available models for a provider:

```python
from gluellm import list_models

models = await list_models(provider="openai")
for m in models[:5]:
    print(m.id)
```

CLI: `gluellm list-models -p openai`

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
| `GLUELLM_DEFAULT_MODEL` | `openai:gpt-5.4-nano` | Default model |
| `GLUELLM_DEFAULT_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Default embedding model |
| `GLUELLM_DEFAULT_EMBEDDING_DIMENSIONS` | _(unset)_ | Default output dimensions for embeddings (e.g. `512`) |
| `GLUELLM_DEFAULT_REQUEST_TIMEOUT` | `300.0` | Request timeout (seconds) |
| `GLUELLM_MAX_REQUEST_TIMEOUT` | `1800.0` | Maximum allowed request timeout |
| `GLUELLM_DEFAULT_CONNECT_TIMEOUT` | `10.0` | Connection timeout (seconds) |
| `GLUELLM_MAX_CONNECT_TIMEOUT` | `60.0` | Maximum allowed connection timeout |
| `GLUELLM_DEFAULT_REASONING_EFFORT` | _(unset)_ | Default reasoning effort for thinking models |
| `GLUELLM_DEFAULT_PARALLEL_TOOL_CALLS` | _(unset)_ | Default for parallel tool calls |
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

More runnable examples live in [`examples/`](examples/). Every example is also a test case (see `tests/test_examples.py`), so if you follow an example, it should work as intended.

## Contributing

PRs welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
