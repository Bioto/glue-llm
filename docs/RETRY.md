# Retry Configuration

GlueLLM retries transient LLM failures (rate limits, connection errors) with exponential backoff by default. You can customise or disable retries per client and per call using `RetryConfig`, `retry_enabled`, and `retry_config`.

## Overview

- **Default behaviour**: Retries up to 3 attempts (configurable via `GLUELLM_RETRY_MAX_ATTEMPTS`) with exponential backoff on `RateLimitError` and `APIConnectionError`.
- **Customisation**: Use `RetryConfig` to change attempts, backoff, filter by exception type (`retry_on`), or add a callback for per-error decisions and parameter injection.
- **Disable**: Set `retry_enabled=False` or `RetryConfig(retry_enabled=False)` for a single attempt.

## Where Retry Applies

Retry configuration is supported on:

| API | `retry_config` | `retry_enabled` |
|-----|----------------|-----------------|
| `complete()` (module-level) | Yes | Yes |
| `structured_complete()` (module-level) | Yes | Yes |
| `GlueLLM.__init__()` | Yes (client default) | — |
| `GlueLLM.complete()` | Yes (per-call override) | Yes |
| `GlueLLM.structured_complete()` | Yes (per-call override) | Yes |
| `GlueLLM.stream_complete()` | Not supported (streaming path uses direct LLM calls without retry) | No |
| `stream_complete()` (module-level) | Not supported | No |

`stream_complete` does not use retry logic; its LLM calls are made without retry wrapping.

---

## RetryConfig

```python
from gluellm import RetryConfig

# All fields optional; defaults shown
cfg = RetryConfig(
    retry_enabled=True,      # If False, single attempt only
    max_attempts=3,         # Total attempts including first
    min_wait=2.0,           # Min seconds between retries
    max_wait=30.0,          # Max seconds (cap on backoff)
    multiplier=1.0,          # Exponential backoff: min_wait * multiplier^attempt
    retry_on=None,          # None = default (RateLimit + APIConnection), or list[type[Exception]]
    callback=None,          # Optional (error, attempt) -> (should_retry, next_params | None)
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `retry_enabled` | `bool` | `True` | If `False`, no retries (single attempt). |
| `max_attempts` | `int` | `3` | Max attempts including the first call. Must be > 0. |
| `min_wait` | `float` | `2.0` | Min wait in seconds between retries. |
| `max_wait` | `float` | `30.0` | Max wait in seconds (caps exponential backoff). |
| `multiplier` | `float` | `1.0` | Backoff multiplier: wait = min(min_wait * multiplier^attempt, max_wait). |
| `retry_on` | `list[type[Exception]] \| None` | `None` | Only retry if error is an instance of one of these types. Ignored when `callback` is set. |
| `callback` | `RetryCallback \| None` | `None` | Optional callback for per-error retry decisions and param injection. When set, `retry_on` is ignored. |

---

## Default Retryable Errors

When neither `retry_on` nor `callback` is set, only these errors trigger a retry:

- **RateLimitError** — 429, quota exceeded
- **APIConnectionError** — Network failures, timeouts (includes `APITimeoutError` subclass)

Errors that are **not** retried by default:

- **TokenLimitError** — Context length exceeded
- **AuthenticationError** — 401, 403
- **InvalidRequestError** — 400, validation errors

---

## Disabling Retries

### Per-call shorthand

```python
result = await complete("What is 2+2?", retry_enabled=False)
```

### Per-call via RetryConfig

```python
result = await complete(
    "What is 2+2?",
    retry_config=RetryConfig(retry_enabled=False),
)
```

### Per-client (all calls)

```python
client = GlueLLM(retry_config=RetryConfig(retry_enabled=False))
result = await client.complete("What is 2+2?")  # no retries
```

### Per-call override of client config

```python
client = GlueLLM(retry_config=RetryConfig(retry_enabled=True, max_attempts=5))
result = await client.complete("Quick call, no retries.", retry_enabled=False)
# This call uses a single attempt despite client default
```

---

## retry_on: Filter by Exception Type

Use `retry_on` to restrict retries to specific exception types. When set, only errors that are instances of one of the listed types trigger a retry; all others are re-raised immediately.

**Subclass matching**: `isinstance()` is used, so subclasses match (e.g. `APITimeoutError` matches `retry_on=[APIConnectionError]`).

```python
from gluellm import RetryConfig, RateLimitError, APIConnectionError

# Only retry rate limits
result = await complete(
    "...",
    retry_config=RetryConfig(retry_on=[RateLimitError]),
)

# Retry both rate limits and connection errors (explicit, same as default)
result = await complete(
    "...",
    retry_config=RetryConfig(retry_on=[RateLimitError, APIConnectionError]),
)

# APITimeoutError is subclass of APIConnectionError — it matches
result = await complete(
    "...",
    retry_config=RetryConfig(retry_on=[APIConnectionError]),
)
```

If the error is not in `retry_on`, it is **not** retried:

```python
# TokenLimitError not in list — single attempt, then raise
client = GlueLLM(retry_config=RetryConfig(retry_on=[RateLimitError]))
await client.complete("...")  # TokenLimitError → immediate raise, no retry
```

---

## Callback: Custom Retry Logic and Parameter Injection

When `callback` is set, it **takes precedence over `retry_on`**; `retry_on` is ignored.

### Signature

```python
def callback(error: Exception, attempt: int) -> tuple[bool, dict[str, Any] | None]:
    # Return (should_retry, next_params)
    # - should_retry: if False, stop retrying and re-raise
    # - next_params: dict merged into model kwargs for next attempt, or None to keep current
    return True, {"temperature": 0.0}
```

Async callbacks are supported:

```python
async def callback(error: Exception, attempt: int) -> tuple[bool, dict | None]:
    await log_to_external(error)
    return True, {"temperature": 0.5}
```

### attempt numbering

- First retry (after initial failure): `attempt=1`
- Second retry: `attempt=2`
- etc.

### Examples

**Retry only on rate limit**

```python
def on_retry(err: Exception, attempt: int) -> tuple[bool, dict | None]:
    if isinstance(err, RateLimitError):
        return True, None
    return False, None

result = await complete("...", retry_config=RetryConfig(callback=on_retry))
```

**Lower temperature on retry**

```python
def on_retry(err: Exception, attempt: int) -> tuple[bool, dict | None]:
    return True, {"temperature": 0.0}

result = await complete("...", retry_config=RetryConfig(callback=on_retry))
```

**Stop after N retries**

```python
def on_retry(err: Exception, attempt: int) -> tuple[bool, dict | None]:
    if attempt >= 2:
        return False, None
    return True, None

result = await complete(
    "...",
    retry_config=RetryConfig(max_attempts=5, callback=on_retry),
)
```

**Accumulate params across retries**

```python
def on_retry(err: Exception, attempt: int) -> tuple[bool, dict | None]:
    # First retry: temp 0.8, second retry: temp 0.4
    temps = [0.8, 0.4]
    return True, {"temperature": temps[attempt - 1]}

result = await complete(
    "...",
    retry_config=RetryConfig(max_attempts=3, callback=on_retry),
)
```

**Callback never called on success**

The callback is invoked only when an error occurs and a retry is about to happen. If the first attempt succeeds, the callback is never called.

---

## Precedence Rules

1. **Per-call vs client**
   - Per-call `retry_config` overrides client `retry_config`.
   - Per-call `retry_enabled=False` overrides client config (creates a `RetryConfig` with `retry_enabled=False`, preserving callback if present).

2. **callback vs retry_on**
   - When `callback` is set, `retry_on` is ignored. The callback decides whether to retry.

3. **retry_on vs default**
   - When `retry_on` is set (and `callback` is not), it replaces the default retryable set (`RateLimitError`, `APIConnectionError`). Only listed types are retried.

---

## Global Settings (Defaults)

When no `RetryConfig` is provided, defaults come from `gluellm.config.settings`:

| Setting | Env var | Default |
|---------|---------|---------|
| `retry_max_attempts` | `GLUELLM_RETRY_MAX_ATTEMPTS` | `3` |
| `retry_min_wait` | `GLUELLM_RETRY_MIN_WAIT` | `2` |
| `retry_max_wait` | `GLUELLM_RETRY_MAX_WAIT` | `30` |
| `retry_multiplier` | `GLUELLM_RETRY_MULTIPLIER` | `1` |

`RetryConfig` takes precedence over these when explicitly passed.

---

## Backoff Formula

Wait time before retry `attempt` (0-based):

```
wait = min(max_wait, min_wait * (multiplier ** attempt))
```

Example with `min_wait=2`, `max_wait=30`, `multiplier=2`:
- Attempt 0 (first retry): 2s
- Attempt 1: 4s
- Attempt 2: 8s
- Attempt 3: 16s
- Attempt 4: 30s (capped)

---

## Exception Hierarchy

```
LLMError (base)
├── TokenLimitError      # Not retried by default
├── RateLimitError      # Retried by default
├── APIConnectionError  # Retried by default
│   └── APITimeoutError # Subclass, matches APIConnectionError
├── InvalidRequestError # Not retried by default
└── AuthenticationError # Not retried by default
```

---

## Relation to Batch Retry

Batch processing uses a separate mechanism: `BatchConfig.retry_failed` retries failed *requests* in a batch once. That is independent of `RetryConfig`, which controls retries of individual LLM *calls* within a single request. See [`docs/BATCH_PROCESSING.md`](BATCH_PROCESSING.md) for batch retry details.
