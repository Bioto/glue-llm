# GlueLLM API Reference

This document provides comprehensive API reference for GlueLLM's main completion interface.

## Quick Start

```python
import asyncio
from gluellm import complete, structured_complete, GlueLLM
from pydantic import BaseModel

async def main():
    # Simple completion
    result = await complete("What is the capital of France?")
    print(result.final_response)

    # With tools
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    result = await complete("What's the weather in Paris?", tools=[get_weather])
    print(result.final_response)

    # Structured output
    class City(BaseModel):
        name: str
        country: str

    city = await structured_complete("Extract: Paris, France", response_format=City)
    print(f"{city.name}, {city.country}")

asyncio.run(main())
```

## High-Level Functions

### complete()

Quick completion function with optional tool execution. Creates a fresh conversation per call.

```python
async def complete(
    user_message: str,
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    correlation_id: str | None = None,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    guardrails: GuardrailsConfig | None = None,
    on_status: OnStatusCallback = None,
    max_tokens: int | None = None,
    condense_tool_messages: bool | None = None,
    tool_mode: ToolMode | None = None,
    tool_execution_order: ToolExecutionOrder | None = None,
    tool_route_model: str | None = None,
    retry_enabled: bool | None = None,
    retry_config: RetryConfig | None = None,
    rate_limit_algorithm: RateLimitAlgorithm | str | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    track_costs: bool | None = None,
    enable_eval_recording: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    session_label: str | None = None,
    parallel_tool_calls: bool | None = None,
    sinks: list[Sink] | None = None,
    **model_kwargs: Any,
) -> ExecutionResult
```

**Parameters:**
- `user_message`: The user's message/request
- `model`: Model identifier in "provider:model_name" format (defaults to `GLUELLM_DEFAULT_MODEL`)
- `system_prompt`: System prompt content
- `tools`: List of callable functions to use as tools
- `execute_tools`: Whether to automatically execute tool calls and loop until done
- `max_tool_iterations`: Maximum tool call rounds (default: 10)
- `correlation_id`: Optional ID for request tracing (auto-generated if not provided)
- `request_timeout`: Request timeout in seconds
- `connect_timeout`: Connection timeout in seconds
- `guardrails`: Optional guardrails configuration
- `on_status`: Callback for process status events
- `max_tokens`: Maximum completion tokens
- `condense_tool_messages`: Condense tool rounds to reduce context size
- `tool_mode`: `"standard"` (all tools) or `"dynamic"` (router discovers tools)
- `tool_execution_order`: `"sequential"` or `"parallel"`
- `retry_config`: Custom retry configuration
- `rate_limit_config`: Custom rate limit configuration

**Returns:** `ExecutionResult` with `final_response`, `tool_calls_made`, `tool_execution_history`, etc.

### structured_complete()

Quick structured output completion. Returns a Pydantic model instance directly (not wrapped in `ExecutionResult`).

```python
async def structured_complete(
    user_message: str,
    response_format: type[T],
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    execute_tools: bool = True,
    max_tool_iterations: int | None = None,
    # ... same params as complete() ...
    max_validation_retries: int | None = None,
) -> T
```

**Parameters:** Same as `complete()`, plus:
- `response_format`: Pydantic model class for structured output
- `max_validation_retries`: Retries when LLM output fails to parse into the schema

**Returns:** Instance of `response_format` (e.g., your Pydantic model), not `ExecutionResult`.

### stream_complete()

Stream completion with automatic tool execution. Yields chunks as they arrive.

```python
async def stream_complete(
    user_message: str,
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    execute_tools: bool = True,
    # ... similar params as complete() ...
    response_format: type[BaseModel] | None = None,
) -> AsyncIterator[StreamingChunk]
```

**Returns:** Async iterator of `StreamingChunk` with `content`, `done`, `tool_calls_made`, and optional `structured_output` on final chunk.

### embed()

Quick embedding generation.

```python
async def embed(
    texts: str | list[str],
    model: str | None = None,
    correlation_id: str | None = None,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    encoding_format: str | None = None,
    dimensions: int | None = None,
    rate_limit_algorithm: RateLimitAlgorithm | str | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **kwargs: Any,
) -> EmbeddingResult
```

**Parameters:**
- `texts`: Single text or list of texts to embed
- `model`: Model identifier (defaults to `GLUELLM_DEFAULT_EMBEDDING_MODEL`)
- `dimensions`: Optional dimension truncation (e.g., OpenAI text-embedding-3-*)

**Returns:** `EmbeddingResult` with `embeddings`, `model`, `usage`, etc.

## GlueLLM Class

Main client for multi-turn conversations with persistent state.

### Constructor

```python
client = GlueLLM(
    model: str | None = None,
    embedding_model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    max_tool_iterations: int | None = None,
    eval_store: EvalStore | None = None,
    guardrails: GuardrailsConfig | None = None,
    condense_tool_messages: bool | None = None,
    tool_mode: ToolMode = "standard",
    tool_execution_order: ToolExecutionOrder | None = None,
    tool_route_model: str | None = None,
    max_tokens: int | None = None,
    retry_config: RetryConfig | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    model_kwargs: dict[str, Any] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    session_label: str | None = None,
    parallel_tool_calls: bool | None = None,
)
```

### Methods

#### complete()

Same as module-level `complete()` but uses instance configuration and maintains conversation history.

#### reset_conversation()

```python
client.reset_conversation()
```

Clears the conversation history. Use for starting a new thread.

### Multi-Turn Example

```python
client = GlueLLM(system_prompt="You are a helpful assistant.")
await client.complete("My name is Alice")
response = await client.complete("What's my name?")
print(response.final_response)  # "Your name is Alice."
```

## Result Types

### ExecutionResult

| Field | Type | Description |
|------|------|--------------|
| `final_response` | `str` | Final text response |
| `tool_calls_made` | `int` | Number of tool calls |
| `tool_execution_history` | `list[dict]` | History of tool calls and results |
| `raw_response` | `ChatCompletion \| None` | Raw LLM response |
| `tokens_used` | `dict \| None` | `{prompt, completion, total}` |
| `estimated_cost_usd` | `float \| None` | Estimated cost in USD |
| `model` | `str \| None` | Model used |
| `structured_output` | `Any \| None` | Parsed Pydantic model (for `structured_complete`) |

### StreamingChunk

| Field | Type | Description |
|------|------|--------------|
| `content` | `str` | Chunk text |
| `done` | `bool` | True if final chunk |
| `tool_calls_made` | `int` | Tool calls so far |
| `structured_output` | `Any \| None` | Parsed output (final chunk only, when `response_format` used) |

## Configuration Types

### RetryConfig

```python
from gluellm import RetryConfig

RetryConfig(
    retry_enabled: bool = True,
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 30.0,
    multiplier: float = 1.0,
    retry_on: list[type[Exception]] | None = None,
    callback: RetryCallback = None,
)
```

- `retry_on`: Override which exception types trigger retry (default: `RateLimitError`, `APIConnectionError`)
- `callback`: `(error, attempt) -> (should_retry, next_params | None)` for custom logic

### RateLimitConfig

```python
from gluellm import RateLimitConfig, RateLimitAlgorithm

RateLimitConfig(algorithm: RateLimitAlgorithm | str | None = None)
```

## Utility Functions

### close_providers()

```python
await close_providers()
```

Closes all cached HTTP clients. Call during application shutdown to prevent "Event loop is closed" warnings.

### list_models()

Returns available models from the configured provider.

### get_session_summary()

Returns token and cost summary for the current session.

### reset_session_tracker()

Resets session tracking.

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Configuration system
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Exceptions and retry logic
- [TOOL_EXECUTION.md](TOOL_EXECUTION.md) - Tool modes and execution
