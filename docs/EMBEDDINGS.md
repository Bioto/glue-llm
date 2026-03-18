# GlueLLM Embeddings

Embedding generation via `embed()` or `GlueLLM` client, with rate limiting, retries, and cost tracking.

## Quick Usage

```python
from gluellm import embed

# Single text
result = await embed("Hello world")

# Multiple texts
result = await embed(["Hello", "World"])

print(result.embeddings)  # list[list[float]]
print(result.model)
print(result.tokens_used)
print(result.estimated_cost_usd)
```

## API

```python
async def embed(
    texts: str | list[str],
    model: str | None = None,
    correlation_id: str | None = None,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    encoding_format: str | None = None,
    dimensions: int | None = None,
    rate_limit_algorithm: RateLimitAlgorithm | str | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    **kwargs: Any,
) -> EmbeddingResult
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `texts` | Single string or list of strings |
| `model` | Model identifier (default: `GLUELLM_DEFAULT_EMBEDDING_MODEL`) |
| `dimensions` | Dimension truncation (e.g., OpenAI text-embedding-3-*) |
| `encoding_format` | `"float"` or `"base64"` (provider-specific) |

## EmbeddingResult

| Field | Type |
|-------|------|
| `embeddings` | `list[list[float]]` |
| `model` | `str` |
| `tokens_used` | `int` |
| `estimated_cost_usd` | `float \| None` |

**Methods:**
- `get_embedding(index=0)` - Single vector by index
- `dimension` - Vector dimension
- `count` - Number of embeddings

## Model Format

Use `provider/model_name` or `provider:model_name`:

- `openai/text-embedding-3-small`
- `openai/text-embedding-3-large`

## Configuration

| Setting | Env Var | Default |
|---------|---------|---------|
| `default_embedding_model` | `GLUELLM_DEFAULT_EMBEDDING_MODEL` | `openai/text-embedding-3-small` |
| `default_embedding_dimensions` | `GLUELLM_DEFAULT_EMBEDDING_DIMENSIONS` | `None` |

## GlueLLM Client

```python
client = GlueLLM(embedding_model="openai/text-embedding-3-small")
result = await embed("Hello", model=client.embedding_model)
```

## See Also

- [API.md](API.md) - embed() reference
- [MODELS.md](MODELS.md) - EmbeddingResult
- [COST_TRACKING.md](COST_TRACKING.md) - estimated_cost_usd
