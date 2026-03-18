# GlueLLM Runtime Management

Runtime utilities for shutdown handling, correlation IDs, and request metadata.

## Graceful Shutdown

### Flow

1. Signal (SIGTERM/SIGINT) sets shutdown event
2. New requests raise `RuntimeError: Cannot process request: shutdown in progress`
3. In-flight requests complete
4. `wait_for_shutdown()` waits (with timeout)
5. Shutdown callbacks run
6. Shutdown complete

### Setup Signal Handlers

```python
from gluellm.runtime.shutdown import setup_signal_handlers

setup_signal_handlers()  # Registers SIGTERM, SIGINT
```

### Shutdown Callbacks

```python
from gluellm.runtime.shutdown import register_shutdown_callback, unregister_shutdown_callback

async def cleanup():
    await close_providers()
    # Close MLflow runs, flush buffers, etc.

register_shutdown_callback(cleanup)
# Later: unregister_shutdown_callback(cleanup)
```

### Manual Graceful Shutdown

```python
from gluellm.runtime.shutdown import graceful_shutdown

await graceful_shutdown(max_wait_time=30.0)
```

### ShutdownContext

Used internally. New requests check `is_shutting_down()` and raise if True.

### Functions

| Function | Description |
|----------|-------------|
| `is_shutting_down()` | Whether shutdown is in progress |
| `register_shutdown_callback(cb)` | Register callback for shutdown |
| `unregister_shutdown_callback(cb)` | Remove callback |
| `get_in_flight_count()` | Current in-flight request count |
| `wait_for_shutdown(max_wait_time)` | Wait for in-flight to complete |
| `execute_shutdown_callbacks()` | Run all callbacks |
| `graceful_shutdown(max_wait_time)` | Full shutdown sequence |

## Correlation IDs

For request tracing across async operations.

```python
from gluellm.runtime.context import (
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    with_correlation_id,
)

# Get/set
cid = set_correlation_id("req-123")  # Auto-generates UUID if not provided
print(get_correlation_id())
clear_correlation_id()

# Context manager
with with_correlation_id("req-456"):
    result = await complete("Hello")
```

Pass to API:

```python
result = await complete("Hello", correlation_id="req-789")
```

Log records include `correlation_id` when set.

## Request Metadata

Arbitrary key-value metadata for the current request context:

```python
from gluellm.runtime.context import get_request_metadata, set_request_metadata, clear_request_metadata

set_request_metadata(user_id="u-1", session_id="s-1")
meta = get_request_metadata()  # {"user_id": "u-1", "session_id": "s-1"}
clear_request_metadata()
```

## get_context_dict

Returns both correlation ID and metadata:

```python
from gluellm.runtime.context import get_context_dict

ctx = get_context_dict()  # {"correlation_id": "...", "metadata": {...}}
```

## See Also

- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Shutdown-related errors
- [OBSERVABILITY.md](OBSERVABILITY.md) - Logging with correlation IDs
- [API.md](API.md) - close_providers
