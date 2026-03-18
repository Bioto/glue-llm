# GlueLLM Cost Tracking

Cost tracking estimates USD cost from token usage using built-in pricing data.

## Configuration

| Setting | Env Var | Default |
|---------|---------|---------|
| `track_costs` | `GLUELLM_TRACK_COSTS` | `True` |
| `print_session_summary_on_exit` | `GLUELLM_PRINT_SESSION_SUMMARY_ON_EXIT` | `True` |

## In ExecutionResult

When `track_costs` is True (default):

```python
result = await complete("Hello")
print(result.estimated_cost_usd)  # e.g., 0.00012
print(result.tokens_used)  # {"prompt": 10, "completion": 20, "total": 30}
```

## Disable Per-Call

```python
result = await complete("Hello", track_costs=False)
# result.estimated_cost_usd will be None
```

## Pricing Data

Pricing is maintained in `gluellm.costing.pricing_data`:

- `OPENAI_PRICING`, `ANTHROPIC_PRICING`, `XAI_PRICING`
- `get_model_pricing(model)` - Lookup by model string
- `calculate_cost(model, prompt_tokens, completion_tokens)` - Compute USD
- `list_available_models()` - Models with pricing

## CostTracker

For aggregated tracking:

```python
from gluellm.costing import CostTracker, get_global_tracker, reset_global_tracker

tracker = get_global_tracker()  # Or CostTracker()
# GlueLLM records usage when track_costs=True
summary = tracker.get_summary()
print(summary.total_cost_usd, summary.total_prompt_tokens)
reset_global_tracker()
```

## Embeddings

Embedding costs use `calculate_embedding_cost()` in `gluellm.costing.pricing_data`. `EmbeddingResult.estimated_cost_usd` is populated when available.

## Session Summary

With `print_session_summary_on_exit=True`, a token/cost summary is printed when the program exits (when using the default at-exit handler).

## See Also

- [API.md](API.md) - track_costs parameter
- [CONFIGURATION.md](CONFIGURATION.md) - Settings reference
