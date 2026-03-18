# GlueLLM Troubleshooting

Common issues and solutions.

## Authentication

### "Authentication failed" / 401 / 403

- **Check API keys**: Ensure `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `XAI_API_KEY` is set
- **Verify key format**: Keys are usually prefixed (e.g., `sk-...`)
- **Provider**: Ensure the key matches the provider in the model string (e.g., `openai:gpt-4o` needs OpenAI key)

### Key not found

```python
# Explicit key override
result = await complete("Hi", api_key="sk-...")
# or
gluellm.configure(openai_api_key="sk-...")
```

## Rate Limiting

### "Rate limit hit" / 429

- **Wait and retry**: GlueLLM retries automatically with backoff
- **Reduce concurrency**: Use `BatchConfig(max_concurrent=3)` for batch processing
- **API Key Pool**: Add multiple keys via `APIKeyPool` or `BatchConfig.api_keys`
- **Adjust limits**: `gluellm.configure(rate_limit_requests_per_minute=30)` if you are under provider limits

## Timeouts

### "API request timed out"

- **Increase timeout**: `result = await complete("Hi", request_timeout=120)`
- **Check network**: Verify connectivity to the provider
- **Large context**: Reduce input size or use summarization
- **Max values**: Timeouts are capped by `max_request_timeout` and `max_connect_timeout` in settings

## Token Limits

### "Token limit exceeded" / "context length exceeded"

- **Reduce input**: Shorten messages or conversation history
- **Use condense_tool_messages**: `condense_tool_messages=True` to shrink tool rounds
- **Use dynamic tool routing**: `tool_mode="dynamic"` to inject fewer tool schemas
- **Split content**: Use MapReduce or similar workflow for long documents

## Shutdown

### "Cannot process request: shutdown in progress"

Shutdown has been initiated (e.g., SIGTERM). New requests are rejected. Ensure:

- `setup_signal_handlers()` is called if you want graceful shutdown
- In-flight requests complete before process exit
- Call `await close_providers()` in shutdown callbacks

### "RuntimeError: Event loop is closed"

HTTP clients may be closed after the event loop. Call `await close_providers()` before the event loop exits:

```python
import atexit
import asyncio
from gluellm import close_providers

async def _shutdown():
    await close_providers()

def shutdown():
    asyncio.run(_shutdown())

atexit.register(shutdown)
```

## Structured Output

### ValidationError when parsing structured output

- **Schema mismatch**: Ensure your Pydantic model matches what the LLM can produce
- **Retries**: `structured_complete` retries on parse failure; increase `max_validation_retries` if needed
- **Simpler schema**: Use fewer fields or more permissive types

## Hooks

### Hook not running

- **Stage**: Verify the hook is registered for the correct `HookStage`
- **Registry**: Ensure the workflow uses the registry (global or instance) where the hook was added
- **Enabled**: Check `HookConfig.enabled=True`

## Evaluation Recording

### Records not written

- **Store**: Call `enable_file_recording()` or `enable_callback_recording()` before any LLM calls
- **Per-call**: Ensure `enable_eval_recording` is not `False`
- **aiofiles**: JSONLFileStore requires `pip install aiofiles`

## Logging

### No logs or wrong level

- **Disable flag**: Check `GLUELLM_DISABLE_LOGGING` is not set
- **Console**: Set `GLUELLM_LOG_CONSOLE_OUTPUT=true` for console output
- **Level**: Set `GLUELLM_LOG_LEVEL=DEBUG` for verbose logs
- **Directory**: Ensure `log_dir` is writable

## See Also

- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Exception types and retry
- [CONFIGURATION.md](CONFIGURATION.md) - Settings reference
- [RUNTIME.md](RUNTIME.md) - Shutdown and context
