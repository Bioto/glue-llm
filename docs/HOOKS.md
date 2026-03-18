# GlueLLM Hooks

Hooks intercept and transform content before and after workflow/executor execution.

## Hook Stages

| Stage | When Executed |
|-------|---------------|
| `PRE_WORKFLOW` | Before workflow starts |
| `POST_WORKFLOW` | After workflow completes |
| `PRE_EXECUTOR` | Before each executor (LLM) call |
| `POST_EXECUTOR` | After each executor call |

## HookContext

Passed to every hook:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Text being processed |
| `stage` | `HookStage` | Current stage |
| `metadata` | `dict` | Workflow/executor info, iteration counts |
| `original_content` | `str \| None` | Unmodified original content |

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
