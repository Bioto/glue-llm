# GlueLLM Guardrails

Guardrails validate and transform LLM inputs and outputs: blocklists, PII redaction, length limits, and custom checks.

## Overview

- **Input guardrails**: Run before the LLM receives the request
- **Output guardrails**: Run after the LLM responds; failures trigger retries with feedback (up to `max_output_guardrail_retries`)

## GuardrailsConfig

```python
from gluellm import GuardrailsConfig, GuardrailBlockedError, GuardrailRejectedError
from gluellm.guardrails.config import BlocklistConfig, PIIConfig, MaxLengthConfig, PromptGuidedConfig

guardrails = GuardrailsConfig(
    enabled=True,
    blocklist=BlocklistConfig(patterns=["secret"], on_input="block", on_output="redact"),
    pii=PIIConfig(redact_emails=True, redact_phones=True, replacement="[REDACTED]"),
    max_length=MaxLengthConfig(max_input_length=10000, max_output_length=5000, strategy="block"),
    max_output_guardrail_retries=3,
)
```

## Configuration Options

### BlocklistConfig

| Field | Type | Default |
|-------|------|---------|
| `patterns` | `list[str]` | Required |
| `case_sensitive` | `bool` | `False` |
| `on_input` | `"block" \| "redact"` | `"block"` |
| `on_output` | `"block" \| "redact"` | `"redact"` |

- `block`: Raise exception
- `redact`: Replace matches with placeholder

### PIIConfig

| Field | Type | Default |
|-------|------|---------|
| `redact_emails` | `bool` | `True` |
| `redact_phones` | `bool` | `True` |
| `redact_ssn` | `bool` | `True` |
| `redact_credit_cards` | `bool` | `True` |
| `custom_patterns` | `list[str] \| None` | `None` |
| `replacement` | `str` | `"[REDACTED]"` |

### MaxLengthConfig

| Field | Type | Default |
|-------|------|---------|
| `max_input_length` | `int \| None` | `None` |
| `max_output_length` | `int \| None` | `None` |
| `strategy` | `"truncate" \| "block"` | `"block"` |

### PromptGuidedConfig

Uses an evaluator (e.g., LLM) to check content against a prompt/criteria.

| Field | Type |
|-------|------|
| `prompt` | `str` - Instruction/criteria |
| `evaluator` | `Callable[[str, str], tuple[bool, str \| None]]` |

Evaluator: `(content, prompt) -> (passed, failure_reason_if_not_passed)`

```python
def my_evaluator(content: str, prompt: str) -> tuple[bool, str | None]:
    # E.g., call another LLM to verify content
    passed = check_content(content, prompt)
    return (passed, None if passed else "Content violated policy")

config = GuardrailsConfig(
    prompt_guided=PromptGuidedConfig(prompt="Be professional", evaluator=my_evaluator),
)
```

### Custom Guardrails

| Field | Type |
|-------|------|
| `custom_input` | `list[Callable[[str], str]] \| None` |
| `custom_output` | `list[Callable[[str], str]] \| None` |

Each callable receives content and returns transformed content.

## Exceptions

| Exception | When |
|-----------|------|
| `GuardrailBlockedError` | Input guardrails block the request; no retry |
| `GuardrailRejectedError` | Output guardrails reject content; triggers retry with feedback |

## Usage

### Per-Call

```python
from gluellm import complete, GuardrailsConfig

result = await complete(
    "User message",
    guardrails=GuardrailsConfig(enabled=True, blocklist=BlocklistConfig(patterns=["secret"])),
)
```

### GlueLLM Client

```python
from gluellm import GlueLLM, GuardrailsConfig

client = GlueLLM(guardrails=GuardrailsConfig(enabled=True, pii=PIIConfig()))
result = await client.complete("Message with email user@example.com")
```

Per-call `guardrails` overrides client-level when provided.

## Output Guardrail Retries

When output guardrails fail:
1. Feedback is sent to the LLM (e.g., "Previous response was rejected: ...")
2. LLM generates a new response
3. Output guardrails run again
4. Repeat up to `max_output_guardrail_retries` (default 3)
5. If still failing, `GuardrailRejectedError` is raised

## See Also

- [API.md](API.md) - complete(), GlueLLM
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Guardrail exceptions
