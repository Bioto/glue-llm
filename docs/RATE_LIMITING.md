# GlueLLM Rate Limiting

Rate limiting prevents excessive API calls and helps stay within provider quotas. GlueLLM uses [throttled-py](https://pypi.org/project/throttled-py/) with multiple algorithms and backends.

## Configuration

| Setting | Env Var | Default |
|---------|---------|---------|
| `rate_limit_enabled` | `GLUELLM_RATE_LIMIT_ENABLED` | `True` |
| `rate_limit_requests_per_minute` | `GLUELLM_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` |
| `rate_limit_burst` | `GLUELLM_RATE_LIMIT_BURST` | `10` |
| `rate_limit_backend` | `GLUELLM_RATE_LIMIT_BACKEND` | `memory` |
| `rate_limit_redis_url` | `GLUELLM_RATE_LIMIT_REDIS_URL` | `None` |
| `rate_limit_algorithm` | `GLUELLM_RATE_LIMIT_ALGORITHM` | `sliding_window` |

## Algorithms (RateLimitAlgorithm)

| Enum | String | Description |
|------|--------|-------------|
| `FIXED_WINDOW` | `fixed_window` | Counts in fixed time windows |
| `SLIDING_WINDOW` | `sliding_window` | Rolling window (default) |
| `LEAKING_BUCKET` | `leaking_bucket` | Constant outflow, absorbs bursts |
| `TOKEN_BUCKET` | `token_bucket` | Tokens refill at fixed rate |
| `GCRA` | `gcra` | Generic Cell Rate Algorithm, low jitter |

## Backends

### Memory

In-process store. Use for single-process apps.

```python
gluellm.configure(rate_limit_backend="memory")
```

### Redis

Distributed store for multi-process/multi-host deployments.

```python
gluellm.configure(
    rate_limit_backend="redis",
    rate_limit_redis_url="redis://localhost:6379",
)
```

## Per-Call Overrides

```python
from gluellm import complete, RateLimitConfig, RateLimitAlgorithm

# Algorithm only
result = await complete("Hello", rate_limit_algorithm=RateLimitAlgorithm.LEAKING_BUCKET)

# Full config
result = await complete(
    "Hello",
    rate_limit_config=RateLimitConfig(algorithm=RateLimitAlgorithm.TOKEN_BUCKET),
)
```

## Client-Level Override

```python
from gluellm import GlueLLM, RateLimitConfig, RateLimitAlgorithm

client = GlueLLM(
    rate_limit_config=RateLimitConfig(algorithm=RateLimitAlgorithm.LEAKING_BUCKET),
)
result = await client.complete("Hello")
```

## Priority Order

(highest → lowest): per-call `rate_limit_config` → per-call `rate_limit_algorithm` → client `rate_limit_config` → `configure()` / env vars.

## API Key Pool

For higher throughput, use multiple API keys with per-key rate limiting:

```python
from gluellm import APIKeyPool, GlueLLM
from gluellm.models.batch import APIKeyConfig

pool = APIKeyPool(
    keys=[
        APIKeyConfig(key="sk-...", provider="openai", requests_per_minute=60),
        APIKeyConfig(key="sk-...", provider="openai", requests_per_minute=60),
    ],
)

# Use with batch processing
from gluellm import batch_complete, BatchConfig

config = BatchConfig(api_keys=pool.keys)
result = await batch_complete(requests, config=config)
```

See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for BatchConfig.api_keys.

## Clear Cache

For testing or config changes:

```python
from gluellm.rate_limiting.rate_limiter import clear_rate_limiter_cache

clear_rate_limiter_cache()
```

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Full config reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Rate limiting in data flow
