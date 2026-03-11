# Error Handling

GlueLLM uses a unified exception hierarchy for LLM-related errors and separate exceptions for guardrails. This document covers all exception types, how they are classified from provider errors, and practical patterns for handling them.

## Exception Hierarchy

### LLM Errors

All LLM errors inherit from `LLMError`. These are raised from `complete()`, `structured_complete()`, `stream_complete()`, and `embed()` when the underlying provider (e.g. OpenAI, Anthropic) fails.

```
LLMError (base)
├── TokenLimitError      # Context/token limit exceeded
├── RateLimitError       # Rate limit or quota hit (429)
├── APIConnectionError   # Network/connection failures
│   └── APITimeoutError  # Request timed out (subclass of APIConnectionError)
├── InvalidRequestError  # Bad parameters, validation (400)
└── AuthenticationError  # API key invalid, unauthorized (401, 403)
```

| Exception | When raised |
|-----------|-------------|
| `LLMError` | Base class; also used as fallback when provider error cannot be classified |
| `TokenLimitError` | Context length exceeded, too many tokens |
| `RateLimitError` | Rate limit (429), quota exceeded, throttled |
| `APIConnectionError` | Connection failed, network unreachable, 502, 503, 504 |
| `APITimeoutError` | Request timed out (subclass of `APIConnectionError`) |
| `InvalidRequestError` | Invalid params, bad request, validation (400) |
| `AuthenticationError` | Invalid API key, unauthorized (401, 403) |

### Guardrail Errors

Guardrail exceptions do **not** subclass `LLMError`. They are raised when using guardrails (e.g. `GuardrailsConfig`).

| Exception | When raised |
|-----------|-------------|
| `GuardrailBlockedError` | Input guardrail blocks user content, or output guardrail blocks after max retries exhausted. No retry. |
| `GuardrailRejectedError` | Output guardrail rejects response. Caught internally to retry the LLM call with feedback. |

Both guardrail errors have `reason` and `guardrail_name` attributes.

## Error Classification

GlueLLM maps raw provider errors to its exception types via `classify_llm_error`. The classifier inspects the error message (case-insensitive) and type name. Keywords are checked in order; the first match wins.

| Exception | Keywords (message or type) |
|-----------|---------------------------|
| `TokenLimitError` | `context length`, `token limit`, `maximum context`, `too many tokens`, `context_length_exceeded`, `max_tokens` |
| `RateLimitError` | `rate limit`, `rate_limit`, `too many requests`, `quota exceeded`, `resource exhausted`, `throttled`, `429` |
| `APITimeoutError` | `timeout`, `timed out`, or `error_type == "APITimeoutError"` |
| `APIConnectionError` | `connection`, `network`, `unreachable`, `503`, `502`, `504` |
| `AuthenticationError` | `unauthorized`, `invalid api key`, `authentication`, `auth`, `401`, `403` |
| `InvalidRequestError` | `invalid`, `bad request`, `400`, `validation` |
| `LLMError` | Fallback when no other match |

Timeouts are checked before connection errors, so timeout messages map to `APITimeoutError` rather than `APIConnectionError`.

## Handling Errors

### Basic pattern

```python
from gluellm import complete, LLMError, TokenLimitError, RateLimitError, AuthenticationError, APIConnectionError

try:
    result = await complete("What is 2+2?")
    print(result.final_response)
except TokenLimitError as e:
    print(f"Input too long: {e}")
except RateLimitError as e:
    print(f"Rate limit hit: {e}")
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except APIConnectionError as e:
    print(f"Connection failed: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
```

### Checking subtypes

When you catch a parent type, use `isinstance()` to distinguish subtypes:

```python
from gluellm import complete, APIConnectionError, APITimeoutError

try:
    result = await complete("...")
except APIConnectionError as e:
    if isinstance(e, APITimeoutError):
        print("Request timed out")
    else:
        print("Other connection error (network, 502, 503, etc.)")
```

You can also inspect the exact type:

```python
type(e).__name__  # "APITimeoutError" or "APIConnectionError"
```

### Catch-all for LLM errors

`LLMError` is the base for all LLM exceptions, so catching it handles every LLM error:

```python
except LLMError as e:
    print(f"LLM failed: {type(e).__name__}: {e}")
```

### Guardrail errors

When using guardrails, also handle guardrail exceptions:

```python
from gluellm import complete, GuardrailBlockedError, GuardrailRejectedError

try:
    result = await complete("...", guardrails_config=config)
except GuardrailBlockedError as e:
    print(f"Blocked: {e.reason} (guardrail: {e.guardrail_name})")
except GuardrailRejectedError:
    # Usually caught internally for retry; re-raised if retries exhausted
    print("Output rejected by guardrail")
```

## Retryable vs Non-Retryable

By default, GlueLLM retries on:

- `RateLimitError`
- `APIConnectionError` (including `APITimeoutError`)

It does **not** retry on:

- `TokenLimitError`
- `InvalidRequestError`
- `AuthenticationError`
- `GuardrailBlockedError`

`GuardrailRejectedError` is caught internally to retry the LLM call with feedback; it is not surfaced to the user unless retries are exhausted (then converted to `GuardrailBlockedError`).

To customise retries (per-call, per-client, callback, backoff), see [RETRY.md](RETRY.md).

## See also

- [RETRY.md](RETRY.md) — Retry configuration, `RetryConfig`, callbacks, backoff
- [`examples/error_handling_example.py`](../examples/error_handling_example.py) — Runnable error-handling examples
