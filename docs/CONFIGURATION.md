# GlueLLM Configuration

Configuration is managed via `GlueLLMSettings` (Pydantic settings) with support for environment variables, `.env` files, and programmatic overrides.

## Configuration Sources (Priority Order)

1. **Programmatic overrides** - `gluellm.configure(**kwargs)` or constructor params
2. **Environment variables** - `GLUELLM_*` prefixed
3. **`.env` file** - In project root

## Programmatic Configuration

Call `configure()` once at application startup before any LLM calls:

```python
import gluellm
from gluellm import RateLimitAlgorithm

gluellm.configure(
    default_model="anthropic:claude-3-5-sonnet-20241022",
    rate_limit_backend="redis",
    rate_limit_redis_url="redis://localhost:6379",
    rate_limit_algorithm=RateLimitAlgorithm.LEAKING_BUCKET,
    openai_api_key=my_settings.openai_key,
)
```

`configure()` mutates the global `settings` singleton in place. All internal modules pick up new values immediately.

## GlueLLMSettings Reference

### Model Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `default_model` | `GLUELLM_DEFAULT_MODEL` | `openai:gpt-5.4-2026-03-05` | Default completion model |
| `default_embedding_model` | `GLUELLM_DEFAULT_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Default embedding model |
| `default_embedding_dimensions` | `GLUELLM_DEFAULT_EMBEDDING_DIMENSIONS` | `None` | Embedding dimensions (e.g., 512) |
| `default_system_prompt` | `GLUELLM_DEFAULT_SYSTEM_PROMPT` | `You are a helpful assistant.` | Default system prompt |
| `default_max_tokens` | `GLUELLM_DEFAULT_MAX_TOKENS` | `None` | Max completion tokens |
| `default_reasoning_effort` | `GLUELLM_DEFAULT_REASONING_EFFORT` | `None` | For o3, o4-mini, Claude thinking models |
| `default_parallel_tool_calls` | `GLUELLM_DEFAULT_PARALLEL_TOOL_CALLS` | `None` | Allow parallel tool calls |

### Tool Execution

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `max_tool_iterations` | `GLUELLM_MAX_TOOL_ITERATIONS` | `10` | Max tool call rounds |
| `default_tool_mode` | `GLUELLM_DEFAULT_TOOL_MODE` | `standard` | `standard` or `dynamic` |
| `default_tool_execution_order` | `GLUELLM_DEFAULT_TOOL_EXECUTION_ORDER` | `sequential` | `sequential` or `parallel` |
| `tool_route_model` | `GLUELLM_TOOL_ROUTE_MODEL` | `openai:gpt-5.4-2026-03-05` | Model for dynamic tool routing |
| `default_condense_tool_messages` | `GLUELLM_DEFAULT_CONDENSE_TOOL_MESSAGES` | `False` | Condense tool rounds in context |

### Retry Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `retry_max_attempts` | `GLUELLM_RETRY_MAX_ATTEMPTS` | `3` | Max retry attempts |
| `retry_min_wait` | `GLUELLM_RETRY_MIN_WAIT` | `2` | Min wait between retries (seconds) |
| `retry_max_wait` | `GLUELLM_RETRY_MAX_WAIT` | `30` | Max wait between retries (seconds) |
| `retry_multiplier` | `GLUELLM_RETRY_MULTIPLIER` | `1` | Exponential backoff multiplier |

### Timeout Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `default_request_timeout` | `GLUELLM_DEFAULT_REQUEST_TIMEOUT` | `60.0` | Request timeout (seconds) |
| `max_request_timeout` | `GLUELLM_MAX_REQUEST_TIMEOUT` | `300.0` | Max allowed request timeout |
| `default_connect_timeout` | `GLUELLM_DEFAULT_CONNECT_TIMEOUT` | `10.0` | Connection timeout (seconds) |
| `max_connect_timeout` | `GLUELLM_MAX_CONNECT_TIMEOUT` | `60.0` | Max allowed connect timeout |

### API Keys

| Setting | Env Var | Fallback |
|---------|---------|----------|
| `openai_api_key` | `GLUELLM_OPENAI_API_KEY` | `OPENAI_API_KEY` |
| `anthropic_api_key` | `GLUELLM_ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` |
| `xai_api_key` | `GLUELLM_XAI_API_KEY` | `XAI_API_KEY` |

### Logging

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `log_level` | `GLUELLM_LOG_LEVEL` | `INFO` | Root log level |
| `log_file_level` | `GLUELLM_LOG_FILE_LEVEL` | `DEBUG` | File log level |
| `log_dir` | `GLUELLM_LOG_DIR` | `logs` | Log directory |
| `log_file_name` | `GLUELLM_LOG_FILE_NAME` | `gluellm.log` | Log file name |
| `log_json_format` | `GLUELLM_LOG_JSON_FORMAT` | `False` | Structured JSON logging |
| `log_max_bytes` | `GLUELLM_LOG_MAX_BYTES` | `10485760` (10MB) | Rotating file max size |
| `log_backup_count` | `GLUELLM_LOG_BACKUP_COUNT` | `5` | Backup count |
| `log_console_output` | `GLUELLM_LOG_CONSOLE_OUTPUT` | `False` | Enable console logging |

### Rate Limiting

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `rate_limit_enabled` | `GLUELLM_RATE_LIMIT_ENABLED` | `True` | Enable rate limiting |
| `rate_limit_requests_per_minute` | `GLUELLM_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Global RPM cap |
| `rate_limit_burst` | `GLUELLM_RATE_LIMIT_BURST` | `10` | Burst allowance |
| `rate_limit_backend` | `GLUELLM_RATE_LIMIT_BACKEND` | `memory` | `memory` or `redis` |
| `rate_limit_redis_url` | `GLUELLM_RATE_LIMIT_REDIS_URL` | `None` | Redis URL (for redis backend) |
| `rate_limit_algorithm` | `GLUELLM_RATE_LIMIT_ALGORITHM` | `sliding_window` | See [RATE_LIMITING.md](RATE_LIMITING.md) |

### Telemetry

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `enable_tracing` | `GLUELLM_ENABLE_TRACING` | `False` | OpenTelemetry tracing |
| `mlflow_tracking_uri` | `GLUELLM_MLFLOW_TRACKING_URI` | `None` | MLflow tracking server |
| `mlflow_experiment_name` | `GLUELLM_MLFLOW_EXPERIMENT_NAME` | `gluellm` | MLflow experiment |
| `otel_exporter_endpoint` | `GLUELLM_OTEL_EXPORTER_ENDPOINT` | `None` | OTLP export endpoint |

### Cost & Evaluation

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `track_costs` | `GLUELLM_TRACK_COSTS` | `True` | Include cost in responses |
| `print_session_summary_on_exit` | `GLUELLM_PRINT_SESSION_SUMMARY_ON_EXIT` | `True` | Print summary on exit |
| `eval_recording_enabled` | `GLUELLM_EVAL_RECORDING_ENABLED` | `False` | Enable eval recording |
| `eval_recording_path` | `GLUELLM_EVAL_RECORDING_PATH` | `None` | JSONL path for eval records |

## Functions

### get_settings()

```python
from gluellm import get_settings

settings = get_settings()
print(settings.default_model)
```

Returns the global settings instance.

### reload_settings()

```python
from gluellm import reload_settings

settings = reload_settings()
```

Reloads settings from environment and `.env`. Returns new instance and updates global `settings`.

## Model Format

Use `provider:model_name` or `provider/model_name`:

- `openai:gpt-5.4-2026-03-05`
- `anthropic:claude-3-5-sonnet-20241022`
- `xai:grok-2`

See [ARCHITECTURE.md](ARCHITECTURE.md) for full configuration details.
