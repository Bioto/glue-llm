# GlueLLM Evaluation Recording

Evaluation recording captures request/response lifecycle data for debugging, analysis, and model evaluation.

## EvalRecord

Each record includes:
- Request: `user_message`, `system_prompt`, `model`, `messages_snapshot`
- Response: `final_response`, `structured_output`, `raw_response`
- Tools: `tool_calls_made`, `tool_execution_history`, `tools_available`
- Metrics: `latency_ms`, `tokens_used`, `estimated_cost_usd`
- Outcome: `success`, `error_type`, `error_message`
- Agent info (when using AgentExecutor)

## Enable Recording

### File Recording (JSONL)

```python
from gluellm.eval import enable_file_recording

# Default path: logs/eval_records.jsonl
store = enable_file_recording()

# Custom path
store = enable_file_recording("./my_eval/records.jsonl")

# Now all GlueLLM instances record automatically
```

Requires `aiofiles`: `pip install aiofiles`

### Callback Recording

```python
from gluellm.eval import enable_callback_recording
from gluellm.models.eval import EvalRecord

async def save_to_db(record: EvalRecord):
    await db.insert("eval_records", record.model_dump_dict())

store = enable_callback_recording(save_to_db)
```

### Per-Instance Store

```python
from gluellm import GlueLLM
from gluellm.eval import JSONLFileStore

store = JSONLFileStore("./my_records.jsonl")
client = GlueLLM(eval_store=store)
```

### Disable Per-Call

```python
result = await complete("Hello", enable_eval_recording=False)
```

## Store Backends

### JSONLFileStore

Writes newline-delimited JSON. One record per line.

```python
from gluellm.eval import JSONLFileStore

store = JSONLFileStore("./records.jsonl")
await store.record(eval_record)
await store.close()
```

### CallbackStore

Wraps a callable that receives `EvalRecord`:

```python
from gluellm.eval import CallbackStore

def my_callback(record: EvalRecord):
    print(record.final_response)

store = CallbackStore(my_callback)
```

### MultiStore

Forwards to multiple stores:

```python
from gluellm.eval import MultiStore, JSONLFileStore, CallbackStore

store = MultiStore([
    JSONLFileStore("./a.jsonl"),
    CallbackStore(my_callback),
])
```

### Custom Store

Implement `EvalStore` protocol:

```python
from gluellm.eval.store import EvalStore

class MyStore:
    async def record(self, record: EvalRecord) -> None:
        ...

    async def close(self) -> None:
        ...
```

## Global Store

```python
from gluellm.eval import set_global_eval_store, get_global_eval_store

set_global_eval_store(my_store)
store = get_global_eval_store()
```

## Configuration

| Setting | Env Var | Default |
|---------|---------|---------|
| `eval_recording_enabled` | `GLUELLM_EVAL_RECORDING_ENABLED` | `False` |
| `eval_recording_path` | `GLUELLM_EVAL_RECORDING_PATH` | `None` |

`enable_file_recording()` and `enable_callback_recording()` set the global store regardless of these settings.

## See Also

- [MODELS.md](MODELS.md) - EvalRecord schema
- [API.md](API.md) - enable_eval_recording parameter
