# GlueLLM Hooks

Hooks intercept and transform content at defined points in the execution pipeline.

## Hook Stages

| Stage | When Executed | `content` |
|-------|---------------|-----------|
| `PRE_WORKFLOW` | Before workflow starts | Workflow input string |
| `POST_WORKFLOW` | After workflow completes | Workflow output string |
| `PRE_EXECUTOR` | Before each executor (LLM) call | Executor query string |
| `POST_EXECUTOR` | After each executor call | Final response string |
| `PRE_TOOL` | Before each tool function is invoked | JSON-serialised tool arguments (modifiable) |
| `POST_TOOL` | After each tool function returns | Tool result string (modifiable) |
| `PRE_ITERATION` | Before each LLM API call in `complete`, `structured_complete`, and `stream_complete` | Last user message string |
| `POST_ITERATION` | After each LLM response, before tool dispatch (all three methods) | LLM response text |
| `PRE_GUARDRAIL` | Before the guardrail chain runs (input or output) | Text entering the chain (modifiable) |
| `POST_GUARDRAIL` | After the guardrail chain completes | Text exiting the chain (modifiable) |
| `ON_LLM_RETRY` | Before each LLM retry sleep | Stringified exception |
| `PRE_TOOL_ROUTE` | Before the tool-routing LLM call | User context string |
| `POST_TOOL_ROUTE` | After tool routing resolves | JSON array of matched tool names (modifiable) |
| `ON_VALIDATION_RETRY` | Before each structured-output validation retry | Raw bad JSON string |
| `PRE_BATCH_ITEM` | Before each `BatchProcessor` item | User message |
| `POST_BATCH_ITEM` | After each `BatchProcessor` item (success or failure) | Result string / error string |
| `PRE_EVAL_RECORD` | Before writing to the eval store | `user_message` field (modifiable — for PII scrubbing) |

### PRE_TOOL / POST_TOOL detail

`PRE_TOOL` hooks receive the tool's arguments as a JSON string in `content`. A hook
may return modified JSON to change what arguments the tool receives — the framework
will `json.loads` the returned string before calling the function.

`POST_TOOL` hooks receive the raw tool result string in `content`. The returned
content replaces what the LLM sees as the tool's output.

Both stages **only fire for valid tool calls** (parse errors and unknown-tool
errors bypass them, consistent with how those paths skip invocation entirely).

In **parallel** tool mode, PRE_TOOL hooks are awaited before `asyncio.gather`
collects the parallel tasks, preserving the same per-tool semantics.

### PRE_ITERATION / POST_ITERATION detail

`PRE_ITERATION` fires once per loop iteration **before** the LLM API call.
Use it for observability, per-call message injection (via metadata inspection),
or circuit-breaker logic.

`POST_ITERATION` fires after the LLM responds but **before** tool calls are
dispatched. Use it for response monitoring or early-termination patterns.

The return value of PRE_ITERATION and POST_ITERATION hooks is **not** applied
back to the conversation — these stages are observation-only by design.

### Metadata keys by stage

| Stage | Metadata keys |
|-------|---------------|
| `PRE_EXECUTOR` | `executor_type` |
| `POST_EXECUTOR` | `executor_type`, `original_query`, `processed_query` |
| `PRE_WORKFLOW` | `workflow_type`, `context` |
| `POST_WORKFLOW` | `workflow_type`, `context`, `original_input`, `processed_input`, `iterations` |
| `PRE_TOOL` | `tool_name`, `call_index`, `iteration` |
| `POST_TOOL` | `tool_name`, `call_index`, `iteration`, `duration_seconds`, `error` (bool) |
| `PRE_ITERATION` | `iteration`, `tool_count` |
| `POST_ITERATION` | `iteration`, `tool_count`, `has_tool_calls` |
| `PRE_GUARDRAIL` | `direction` (`"input"` or `"output"`), `attempt` |
| `POST_GUARDRAIL` | `direction`, `attempt` |
| `ON_LLM_RETRY` | `attempt`, `max_attempts`, `wait_seconds`, `exception_type` |
| `PRE_TOOL_ROUTE` | `route_query`, `available_tool_count` |
| `POST_TOOL_ROUTE` | `route_query`, `available_tool_count`, `matched_tool_names`, `fallback_to_all` |
| `ON_VALIDATION_RETRY` | `validation_attempt`, `max_validation_retries`, `error`, `response_format_name` |
| `PRE_BATCH_ITEM` | `batch_request_id`, `index`, `attempt` |
| `POST_BATCH_ITEM` | `batch_request_id`, `index`, `attempt`, `success` |
| `PRE_EVAL_RECORD` | `correlation_id`, `success`, `model` |

### PRE_GUARDRAIL / POST_GUARDRAIL detail

`PRE_GUARDRAIL` fires before `run_input_guardrails` or `run_output_guardrails` is called. The hook
may return modified content to change what enters the guardrail chain.

`POST_GUARDRAIL` fires with the text that **passed** guardrails. Return a modified string to
transform the output before it continues through the pipeline.

The `direction` metadata key is `"input"` for input guardrails and `"output"` for output
guardrails. `attempt` is the current output-guardrail retry count (0 on the first pass).

### ON_LLM_RETRY detail

Fires before each sleep between LLM API retries. The hook receives the stringified exception in
`content` and `attempt` / `max_attempts` / `wait_seconds` / `exception_type` in metadata.
Return value is ignored — this stage is observation-only.

### PRE_TOOL_ROUTE / POST_TOOL_ROUTE detail

`PRE_TOOL_ROUTE` fires before the routing LLM is asked which tools are relevant.
Use it for logging or to short-circuit routing via metadata inspection.
Return value is ignored.

`POST_TOOL_ROUTE` fires after routing resolves. The hook receives a JSON array of matched
tool names (e.g. `'["search", "calculator"]'`). Returning a modified JSON array overrides
which tools become active for the current iteration — useful for allowlisting or forcing
specific tools.

### ON_VALIDATION_RETRY detail

Fires before each retry in the `structured_complete` Pydantic validation loop.
`content` is the raw bad JSON string, and metadata includes the error message and
`response_format_name`. Return value is ignored — use this for observability.

### PRE_BATCH_ITEM / POST_BATCH_ITEM detail

`PRE_BATCH_ITEM` fires once per `BatchRequest`, before the retry loop, with
`content = request.user_message`. The inner `GlueLLM` instance automatically
inherits the `BatchProcessor`'s `hook_registry`.

`POST_BATCH_ITEM` fires after the item completes (whether it succeeded or
all retry attempts were exhausted). `success` in metadata is `True` on success.

### PRE_EVAL_RECORD detail

Fires inside `_record_eval_data` just before `eval_store.record(record)` is called.
`content` is `record.user_message`; the returned string replaces the `user_message`
field in the stored record. Use this to scrub PII from evaluation logs without
affecting the runtime conversation.

## HookContext

Passed to every hook:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Text being processed (semantics depend on stage) |
| `stage` | `HookStage` | Current stage |
| `metadata` | `dict` | Stage-specific context (see table above) |
| `original_content` | `str \| None` | Unmodified content at the start of the hook chain |

## HookConfig

| Field | Type | Default |
|-------|------|---------|
| `handler` | `Callable[[HookContext], HookContext \| str]` | Required |
| `name` | `str` | Required |
| `error_strategy` | `HookErrorStrategy` | `SKIP` |
| `fallback_value` | `str \| None` | `None` |
| `enabled` | `bool` | `True` |
| `timeout` | `float \| None` | `None` |

## HookErrorStrategy

| Strategy | Behavior on Error |
|----------|-------------------|
| `ABORT` | Re-raise the exception |
| `SKIP` | Use current content (no change) |
| `FALLBACK` | Use `fallback_value` if set |

## Basic Usage

### Sync Hook

```python
from gluellm.models.hook import HookConfig, HookContext, HookStage

def log_content(context: HookContext) -> HookContext:
    print(f"[{context.stage.value}] content length: {len(context.content)}")
    return context

config = HookConfig(
    handler=log_content,
    name="log_content",
)
registry.add_hook(HookStage.PRE_EXECUTOR, config)
```

### Async Hook

```python
async def redact_pii(context: HookContext) -> str:
    content = context.content
    # Redact emails, phones, etc.
    return redacted_content

config = HookConfig(
    handler=redact_pii,
    name="redact_pii",
    error_strategy=HookErrorStrategy.FALLBACK,
    fallback_value="[content unavailable]",
)
```

### Return Types

Handlers may return:
- `HookContext` - Use `result.content` as new content
- `str` - Use string directly as new content

## HookRegistry

Container for hooks by stage.

```python
from gluellm.models.hook import HookRegistry, HookStage, HookConfig

registry = HookRegistry()

# Add
registry.add_hook(HookStage.PRE_EXECUTOR, config)

# Get
hooks = registry.get_hooks(HookStage.PRE_EXECUTOR)

# Remove
registry.remove_hook(HookStage.PRE_EXECUTOR, "my_hook")

# Clear
registry.clear_stage(HookStage.PRE_EXECUTOR)

# Merge registries
merged = registry_a.merge(registry_b)
```

## Global Hooks

Register hooks that apply to all workflows:

```python
from gluellm.hooks.manager import register_global_hook, unregister_global_hook, clear_global_hooks

register_global_hook(HookStage.PRE_EXECUTOR, config)
unregister_global_hook(HookStage.PRE_EXECUTOR, "my_hook")
clear_global_hooks()  # All stages
clear_global_hooks(HookStage.PRE_EXECUTOR)  # Single stage
```

## GlueLLM Hooks

Pass a registry directly to `GlueLLM` (or the convenience functions) to use
per-tool and per-iteration hooks:

```python
from gluellm import GlueLLM, complete, HookRegistry, HookStage, HookConfig

registry = HookRegistry()
registry.add_hook(HookStage.PRE_TOOL, HookConfig(handler=log_tool_call, name="log_tool"))
registry.add_hook(HookStage.POST_TOOL, HookConfig(handler=sanitize_result, name="sanitize"))

# Instance-level
client = GlueLLM(tools=[my_tool], hook_registry=registry)
result = await client.complete("Do something")

# Convenience function
result = await complete("Do something", tools=[my_tool], hook_registry=registry)
```

### Modifying tool arguments (PRE_TOOL)

```python
import json
from gluellm.models.hook import HookContext

def inject_user_id(context: HookContext) -> str:
    args = json.loads(context.content)
    args["user_id"] = "u_42"          # Inject a field before the tool runs
    return json.dumps(args)

registry.add_hook(HookStage.PRE_TOOL, HookConfig(handler=inject_user_id, name="inject_user_id"))
```

### Transforming tool results (POST_TOOL)

```python
def truncate_long_result(context: HookContext) -> str:
    return context.content[:2000] if len(context.content) > 2000 else context.content

registry.add_hook(HookStage.POST_TOOL, HookConfig(handler=truncate_long_result, name="truncate"))
```

### Observing each LLM iteration (PRE_ITERATION / POST_ITERATION)

```python
def log_iteration(context: HookContext) -> HookContext:
    print(f"Iteration {context.metadata['iteration']}: {len(context.content)} chars")
    return context   # return unchanged — no side-effect on the conversation

registry.add_hook(HookStage.PRE_ITERATION, HookConfig(handler=log_iteration, name="log_iter"))
```

### Observing guardrail runs (PRE_GUARDRAIL / POST_GUARDRAIL)

```python
def log_guardrail(context: HookContext) -> HookContext:
    direction = context.metadata["direction"]   # "input" or "output"
    print(f"[guardrail:{direction}] content length = {len(context.content)}")
    return context

registry.add_hook(HookStage.PRE_GUARDRAIL, HookConfig(handler=log_guardrail, name="log_gr"))
registry.add_hook(HookStage.POST_GUARDRAIL, HookConfig(handler=log_guardrail, name="log_gr_out"))
```

### Monitoring LLM retries (ON_LLM_RETRY)

```python
def on_retry(context: HookContext) -> HookContext:
    m = context.metadata
    print(f"LLM retry {m['attempt']}/{m['max_attempts']}: {m['exception_type']} — "
          f"waiting {m['wait_seconds']:.1f}s")
    return context

registry.add_hook(HookStage.ON_LLM_RETRY, HookConfig(handler=on_retry, name="retry_log"))
```

### Overriding matched tools (POST_TOOL_ROUTE)

```python
import json

ALLOWED_TOOLS = {"search", "calculator"}

def allowlist_tools(context: HookContext) -> str:
    names = json.loads(context.content)
    filtered = [n for n in names if n in ALLOWED_TOOLS]
    return json.dumps(filtered or names)  # fallback to original if all filtered

registry.add_hook(HookStage.POST_TOOL_ROUTE,
                  HookConfig(handler=allowlist_tools, name="allowlist"))
```

### Observing validation retries (ON_VALIDATION_RETRY)

```python
def log_val_retry(context: HookContext) -> HookContext:
    m = context.metadata
    print(f"Validation retry {m['validation_attempt']}/{m['max_validation_retries']} "
          f"for {m['response_format_name']}: {m['error']}")
    return context

registry.add_hook(HookStage.ON_VALIDATION_RETRY,
                  HookConfig(handler=log_val_retry, name="val_retry_log"))
```

### Scrubbing PII before eval recording (PRE_EVAL_RECORD)

```python
import re

def scrub_pii(context: HookContext) -> str:
    # Replace email addresses with a placeholder
    return re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                  "[EMAIL]", context.content)

registry.add_hook(HookStage.PRE_EVAL_RECORD,
                  HookConfig(handler=scrub_pii, name="pii_scrub"))

client = GlueLLM(hook_registry=registry)
result = await client.complete("My email is alice@example.com, please help.")
# The eval store records "[EMAIL]" instead of the actual address.
```

### Batch processing hooks (PRE_BATCH_ITEM / POST_BATCH_ITEM)

```python
from gluellm.batch import BatchProcessor
from gluellm.models.batch import BatchConfig, BatchRequest

def log_batch_item(context: HookContext) -> HookContext:
    stage = context.stage.value
    rid = context.metadata["batch_request_id"]
    print(f"[{stage}] item {rid}: {context.content[:60]}")
    return context

registry = HookRegistry()
registry.add_hook(HookStage.PRE_BATCH_ITEM,
                  HookConfig(handler=log_batch_item, name="pre_item"))
registry.add_hook(HookStage.POST_BATCH_ITEM,
                  HookConfig(handler=log_batch_item, name="post_item"))

processor = BatchProcessor(
    model="openai:gpt-5.4-2026-03-05",
    hook_registry=registry,
    config=BatchConfig(max_concurrent=3),
)
response = await processor.process([
    BatchRequest(user_message="What is 2+2?"),
    BatchRequest(user_message="Capital of France?"),
])
```

## Workflow Hooks

Pass a registry when creating a workflow:

```python
from gluellm.workflows.pipeline import PipelineWorkflow

registry = HookRegistry()
registry.add_hook(HookStage.PRE_WORKFLOW, my_hook_config)

workflow = PipelineWorkflow(stages=[...], hook_registry=registry)
```

Global and instance registries are merged; both run.

## Built-in Hook Utilities

`gluellm.hooks.utils` provides ready-made handlers:

- `remove_emails(context)` - Redact emails
- `remove_phone_numbers(context)` - Redact phone numbers
- `remove_ssn(context)` - Redact SSN-like patterns
- `remove_credit_cards(context)` - Redact credit card numbers
- `remove_pii(context)` - Redact all PII types
- `validate_length_factory(min_len, max_len)` - Length validation
- `validate_no_profanity(context)` - Profanity check
- `require_citations(context)` - Ensure citations
- `normalize_whitespace(context)` - Normalize spacing
- `escape_html(context)` - Escape HTML
- `truncate_output_factory(max_chars, max_tokens)` - Truncation

Example:

```python
from gluellm.hooks.utils import remove_pii
from gluellm.models.hook import HookConfig, HookStage

config = HookConfig(handler=remove_pii, name="pii_redaction")
registry.add_hook(HookStage.PRE_EXECUTOR, config)
```

## Timeout

Hooks can have a per-hook timeout:

```python
config = HookConfig(
    handler=slow_hook,
    name="slow_hook",
    timeout=5.0,  # Seconds
)
```

On timeout, `TimeoutError` is caught and handled by `error_strategy`.

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Hook system in architecture
- [WORKFLOWS_API.md](WORKFLOWS_API.md) - Workflow hook usage
