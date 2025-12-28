# Error Handling and Retry Logic

GlueLLM includes comprehensive error handling and automatic retry logic to make your applications more robust when dealing with LLM APIs.

## Features

### Automatic Retry with Exponential Backoff

GlueLLM automatically retries failed requests with exponential backoff for transient errors:

- **Rate Limit Errors (429)**: Automatically retries up to 3 times with exponential backoff (2s, 4s, 8s)
- **Connection Errors (5xx)**: Automatically retries network/timeout issues
- **No retry for permanent failures**: Token limit, authentication, and invalid request errors fail immediately

### Error Classification

All errors from any LLM provider are automatically classified into specific exception types:

```python
from source import (
    GlueLLM,
    # Exception types
    LLMError,          # Base exception
    TokenLimitError,   # Context length exceeded
    RateLimitError,    # Rate limit hit
    APIConnectionError,# Network/connection issues
    AuthenticationError,# Invalid API key
    InvalidRequestError,# Bad request parameters
)
```

### Smart Error Detection

The library intelligently detects errors from various providers by analyzing error messages:

- **Token limit**: "context length", "token limit", "maximum context", "too many tokens"
- **Rate limiting**: "rate limit", "too many requests", "quota exceeded", "429"
- **Connection**: "timeout", "network", "unreachable", "503", "502", "504"
- **Authentication**: "unauthorized", "invalid api key", "401", "403"
- **Invalid request**: "invalid", "bad request", "400"

## Usage Examples

### Basic Usage with Error Handling

```python
from source import complete, RateLimitError, TokenLimitError

try:
    result = complete(
        user_message="Analyze this text...",
        model="openai:gpt-4o",
    )
    print(result.final_response)
    
except TokenLimitError as e:
    print(f"Input too long: {e}")
    # Reduce input size and retry
    
except RateLimitError as e:
    print(f"Rate limit exceeded after retries: {e}")
    # Wait longer or reduce request frequency
    
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
    # Check your API credentials
```

### Using the Client Class

```python
from source import GlueLLM, LLMError

client = GlueLLM(
    model="openai:gpt-4o-mini",
    tools=[my_tool],
    max_tool_iterations=10,
)

try:
    result = client.complete("Process this request")
    print(f"Success! Made {result.tool_calls_made} tool calls")
    
except LLMError as e:
    print(f"LLM error occurred: {e}")
    # Handle any LLM-related error
```

### Tool Execution Errors

Tool execution errors are automatically caught and reported back to the model:

```python
def my_tool(param: str) -> str:
    """A tool that might fail."""
    if not param:
        raise ValueError("Parameter required")
    return f"Processed: {param}"

client = GlueLLM(tools=[my_tool])
result = client.complete("Use my_tool")

# Check if any tools had errors
for execution in result.tool_execution_history:
    if execution.get('error'):
        print(f"Tool error: {execution['result']}")
```

### Handling Malformed Tool Arguments

If the LLM provides invalid JSON in tool arguments, it's automatically caught:

```python
# This is handled automatically - no action needed!
# The error is reported back to the model and logged
result = client.complete("Use tools...")

for execution in result.tool_execution_history:
    if "Invalid JSON" in execution.get('result', ''):
        print(f"Model provided bad JSON: {execution['arguments']}")
```

## Error Handling in Tool Execution Loop

The tool execution loop includes comprehensive error handling:

1. **LLM API Errors**: Caught and classified with automatic retry
2. **JSON Parsing Errors**: Malformed tool arguments are caught and reported
3. **Tool Execution Errors**: Exceptions in tool functions are caught and passed back to the model
4. **Missing Tools**: Requests for non-existent tools are gracefully handled

Example execution history with errors:

```python
result = client.complete("Use some tools")

# Execution history tracks all tool calls and errors
for i, execution in enumerate(result.tool_execution_history):
    print(f"Tool {i+1}: {execution['tool_name']}")
    print(f"  Arguments: {execution['arguments']}")
    print(f"  Error: {execution.get('error', False)}")
    print(f"  Result: {execution['result']}")
```

## Logging

Error handling activities are logged at appropriate levels:

```python
import logging

# Enable logging to see retry attempts and errors
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('source.api')

# You'll see messages like:
# WARNING - Retrying _llm_call_with_retry in 2.0 seconds as it raised RateLimitError
# WARNING - Tool my_tool execution failed: ValueError: invalid input
# ERROR - LLM call failed: Rate limit exceeded
```

## Configuration

You can customize retry behavior by modifying the retry decorator:

```python
# In source/api.py - default configuration:

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(3),  # Max 3 attempts
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 2s, 4s, 8s, ...
    reraise=True,
)
```

## Best Practices

1. **Catch specific exceptions**: Handle `TokenLimitError`, `RateLimitError`, etc. specifically
2. **Use the base `LLMError` for catch-all**: Catches any LLM-related error
3. **Check tool execution history**: Review `tool_execution_history` for tool-specific errors
4. **Enable logging**: Use Python logging to debug retry behavior
5. **Handle token limits proactively**: Truncate input before hitting token limits
6. **Design resilient tools**: Make tools fail gracefully with helpful error messages

## Error Hierarchy

```
Exception
└── LLMError (base for all LLM errors)
    ├── TokenLimitError (context length exceeded)
    ├── RateLimitError (rate limit hit) [RETRYABLE]
    ├── APIConnectionError (network issues) [RETRYABLE]
    ├── AuthenticationError (invalid credentials)
    └── InvalidRequestError (bad parameters)
```

## Testing

The error handling system includes comprehensive tests in `tests/test_error_handling.py`:

```bash
# Run error handling tests
uv run pytest tests/test_error_handling.py -v

# Run all tests
uv run pytest tests/ -v
```

## Additional Notes

- The raw LLM response is available in `result.raw_response` for debugging
- All errors preserve the original exception in the chain (use `raise ... from e`)
- Retry delays use exponential backoff to avoid overwhelming APIs
- Connection errors include 502, 503, 504 HTTP status codes

