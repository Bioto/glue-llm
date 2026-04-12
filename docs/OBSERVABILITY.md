# GlueLLM Observability

Logging, tracing, and metrics for GlueLLM operations.

## Logging

### Setup

GlueLLM initializes logging on package import. By default:
- **File logging**: `logs/gluellm.log`, rotating (10MB, 5 backups), DEBUG level
- **Console logging**: Off (to avoid conflicts when used as a library)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GLUELLM_DISABLE_LOGGING` | Set to `true` to disable GlueLLM logging entirely |
| `GLUELLM_LOG_LEVEL` | Console level (INFO, DEBUG, etc.) |
| `GLUELLM_LOG_FILE_LEVEL` | File level |
| `GLUELLM_LOG_DIR` | Log directory |
| `GLUELLM_LOG_FILE_NAME` | Log file name |
| `GLUELLM_LOG_JSON_FORMAT` | Set to `true` for JSON structured logging |
| `GLUELLM_LOG_CONSOLE_OUTPUT` | Set to `true` to enable console output |
| `GLUELLM_LOG_MAX_BYTES` | Max file size before rotation |
| `GLUELLM_LOG_BACKUP_COUNT` | Number of backup files |

### Programmatic Setup

```python
from gluellm.observability.logging_config import setup_logging

setup_logging(
    log_level="INFO",
    log_file_level="DEBUG",
    log_dir="logs",
    console_output=True,
    log_json_format=True,
)
```

### Correlation IDs

Log records include `correlation_id` for request tracing. Set via:
- `complete(..., correlation_id="req-123")`
- `set_correlation_id("req-123")` or `with_correlation_id("req-123")` from `gluellm.runtime.context`

### Get Logger

```python
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)
```

## OpenTelemetry Tracing

### Enable Tracing

```python
gluellm.configure(
    enable_tracing=True,
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="gluellm",
    otel_exporter_endpoint="http://localhost:4318/v1/traces",
)
```

Or via env: `GLUELLM_ENABLE_TRACING=true`, `GLUELLM_MLFLOW_TRACKING_URI`, etc.

### trace_llm_call

Context manager for manual spans:

```python
from gluellm.telemetry import trace_llm_call

with trace_llm_call("my_llm_call", model="openai:gpt-5.4-2026-03-05"):
    result = await complete("Hello")
```

### MLflow Integration

When `mlflow_tracking_uri` is set:
- Experiments and runs are created
- Token usage and model info are logged
- Metrics can be queried via MLflow UI

## Process Events (Sinks)

Status events (LLM start/end, tool start/end) can be emitted to sinks:

```python
from gluellm.events import ConsoleSink, JsonFileSink, ProcessEvent

# Built-in sinks
sinks = [ConsoleSink(), JsonFileSink("status.jsonl")]
result = await complete("Hello", sinks=sinks)
```

Custom sink: implement `Sink` with `emit(event: ProcessEvent)`.

## on_status Callback

Per-call callback for status events:

```python
async def on_status(event: ProcessEvent):
    print(f"{event.stage}: {event}")

result = await complete("Hello", on_status=on_status)
```

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Logging and tracing settings
- [RUNTIME.md](RUNTIME.md) - Correlation IDs and context
