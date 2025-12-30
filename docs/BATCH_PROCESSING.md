Batch Processing
================

GlueLLM provides powerful batch processing capabilities to efficiently handle multiple LLM requests in parallel.

## Overview

Batch processing allows you to:
- **Process multiple requests concurrently** with configurable concurrency limits
- **Handle errors gracefully** with different error strategies
- **Track metadata** for each request
- **Aggregate results** and token usage across all requests
- **Retry failed requests** automatically

## Quick Start

### Simple Batch Processing

The easiest way to batch process messages:

```python
import asyncio
from gluellm import batch_complete_simple

async def main():
    messages = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain quantum computing briefly.",
    ]

    responses = await batch_complete_simple(messages)

    for msg, resp in zip(messages, responses):
        print(f"Q: {msg}")
        print(f"A: {resp}\n")

asyncio.run(main())
```

### Advanced Batch Processing

For more control, use `BatchRequest` objects:

```python
import asyncio
from gluellm import BatchRequest, batch_complete, BatchConfig

async def main():
    requests = [
        BatchRequest(
            id="req-1",
            user_message="What is 2+2?",
            metadata={"category": "math"},
        ),
        BatchRequest(
            id="req-2",
            user_message="What is the capital of France?",
            metadata={"category": "geography"},
        ),
    ]

    response = await batch_complete(
        requests,
        config=BatchConfig(max_concurrent=5),
    )

    print(f"Processed {response.successful_requests}/{response.total_requests} requests")
    print(f"Total time: {response.total_elapsed_time:.2f}s")

    for result in response.results:
        if result.success:
            print(f"{result.id}: {result.response}")
        else:
            print(f"{result.id}: ERROR - {result.error}")

asyncio.run(main())
```

## Configuration

### BatchConfig

Control batch processing behavior with `BatchConfig`:

```python
from gluellm import BatchConfig, BatchErrorStrategy

config = BatchConfig(
    max_concurrent=10,           # Process up to 10 requests at once
    error_strategy=BatchErrorStrategy.CONTINUE,  # Continue on errors
    show_progress=False,          # Don't show progress bar
    retry_failed=True,            # Retry failed requests once
)
```

#### Parameters

- **max_concurrent** (int, default=5): Maximum number of concurrent requests
- **error_strategy** (BatchErrorStrategy): How to handle errors
  - `FAIL_FAST`: Stop on first error
  - `CONTINUE`: Process all requests, collect errors
  - `SKIP`: Skip failed requests, return only successful ones
- **show_progress** (bool, default=False): Show progress during processing
- **retry_failed** (bool, default=False): Automatically retry failed requests once

## Error Handling

### Error Strategies

**FAIL_FAST**: Stop immediately on first error

```python
config = BatchConfig(error_strategy=BatchErrorStrategy.FAIL_FAST)
response = await batch_complete(requests, config=config)
# Raises exception on first error
```

**CONTINUE**: Process all requests, include failed results

```python
config = BatchConfig(error_strategy=BatchErrorStrategy.CONTINUE)
response = await batch_complete(requests, config=config)

for result in response.results:
    if result.success:
        print(f"✓ {result.id}: {result.response}")
    else:
        print(f"✗ {result.id}: {result.error}")
```

**SKIP**: Only return successful results

```python
config = BatchConfig(error_strategy=BatchErrorStrategy.SKIP)
response = await batch_complete(requests, config=config)

# response.results contains only successful results
assert all(r.success for r in response.results)
```

### Retry Failed Requests

Enable automatic retry for transient failures:

```python
config = BatchConfig(
    retry_failed=True,
    error_strategy=BatchErrorStrategy.CONTINUE,
)
response = await batch_complete(requests, config=config)

# Check which requests were retried
for result in response.results:
    if result.metadata.get("_retried"):
        print(f"{result.id} was retried")
```

## Batch Processing with Tools

You can use tools in batch requests:

```python
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return f"Result: {eval(expression)}"

requests = [
    BatchRequest(
        user_message="What's the weather in Tokyo?",
        tools=[get_weather],
    ),
    BatchRequest(
        user_message="Calculate 123 * 456",
        tools=[calculate],
    ),
    BatchRequest(
        user_message="What's the weather in Paris and what's 50 + 50?",
        tools=[get_weather, calculate],
    ),
]

response = await batch_complete(requests)

for result in response.results:
    print(f"Tool calls: {result.tool_calls_made}")
    print(f"Response: {result.response}")
```

## Metadata and Custom IDs

Attach metadata to requests for tracking:

```python
requests = [
    BatchRequest(
        id="user-123-req-1",
        user_message="Hello",
        metadata={
            "user_id": "123",
            "session_id": "abc",
            "priority": "high",
        },
    ),
]

response = await batch_complete(requests)

for result in response.results:
    user_id = result.metadata.get("user_id")
    print(f"Response for user {user_id}: {result.response}")
```

## Performance Optimization

### Concurrency Tuning

Adjust `max_concurrent` based on your rate limits and resources:

```python
# Conservative (good for rate-limited APIs)
config = BatchConfig(max_concurrent=3)

# Moderate (balanced)
config = BatchConfig(max_concurrent=10)

# Aggressive (maximize throughput)
config = BatchConfig(max_concurrent=20)
```

### Measuring Performance

```python
import time

start = time.time()
response = await batch_complete(
    requests,
    config=BatchConfig(max_concurrent=10),
)
wall_time = time.time() - start

print(f"Wall time: {wall_time:.2f}s")
print(f"Total processing time: {response.total_elapsed_time:.2f}s")
print(f"Concurrency benefit: {response.total_elapsed_time / wall_time:.1f}x")
```

## Using BatchProcessor Class

For advanced use cases, use `BatchProcessor` directly:

```python
from gluellm import BatchProcessor, BatchConfig

processor = BatchProcessor(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],
    max_tool_iterations=5,
    config=BatchConfig(max_concurrent=10),
)

# Process multiple batches with the same configuration
batch1_response = await processor.process(batch1_requests)
batch2_response = await processor.process(batch2_requests)
```

## Response Structure

### BatchResponse

```python
response = await batch_complete(requests)

# Overall statistics
print(f"Total requests: {response.total_requests}")
print(f"Successful: {response.successful_requests}")
print(f"Failed: {response.failed_requests}")
print(f"Total time: {response.total_elapsed_time:.2f}s")

# Token usage (if available)
if response.total_tokens_used:
    print(f"Total tokens: {response.total_tokens_used['total']}")
    print(f"Prompt tokens: {response.total_tokens_used['prompt']}")
    print(f"Completion tokens: {response.total_tokens_used['completion']}")

# Individual results
for result in response.results:
    print(f"{result.id}: {result.success}")
```

### BatchResult

Each result contains:

```python
result = response.results[0]

# Core fields
result.id                    # Request ID
result.success              # bool: Whether request succeeded
result.response             # str: Response text (if successful)
result.elapsed_time         # float: Time taken in seconds

# Tool execution
result.tool_calls_made      # int: Number of tool calls
result.tool_execution_history  # list: Tool execution details

# Error information (if failed)
result.error                # str: Error message
result.error_type           # str: Exception type

# Token usage (if available)
result.tokens_used          # dict: {"prompt": int, "completion": int, "total": int}

# Custom data
result.metadata             # dict: Metadata from request
```

## Best Practices

1. **Set appropriate concurrency limits** based on your API rate limits
2. **Use CONTINUE strategy** for most use cases to maximize throughput
3. **Add metadata** to track requests and correlate results
4. **Monitor token usage** to stay within budget
5. **Handle errors gracefully** - check `result.success` before using results
6. **Use retry for transient failures** but not for validation errors
7. **Batch similar requests** together for better efficiency

## Examples

See `examples/batch_processing.py` for comprehensive examples including:
- Simple batch processing
- Metadata and custom IDs
- Different error handling strategies
- Tools in batch requests
- High concurrency scenarios
- Automatic retry

## API Reference

### Functions

- `batch_complete(requests, model, system_prompt, tools, max_tool_iterations, config)`
  → `BatchResponse`
- `batch_complete_simple(messages, model, system_prompt, tools, config)`
  → `list[str]`

### Classes

- `BatchProcessor(model, system_prompt, tools, max_tool_iterations, config)`
- `BatchRequest(id, user_message, system_prompt, tools, execute_tools, max_tool_iterations, timeout, metadata)`
- `BatchResult(id, success, response, tool_calls_made, tool_execution_history, tokens_used, error, error_type, metadata, elapsed_time)`
- `BatchResponse(results, total_requests, successful_requests, failed_requests, total_elapsed_time, total_tokens_used)`
- `BatchConfig(max_concurrent, error_strategy, show_progress, retry_failed)`

### Enums

- `BatchErrorStrategy.FAIL_FAST`
- `BatchErrorStrategy.CONTINUE`
- `BatchErrorStrategy.SKIP`
