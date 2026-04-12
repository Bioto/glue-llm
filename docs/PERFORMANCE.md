# GlueLLM Performance

Guidance for optimizing GlueLLM performance.

## Rate Limiting

- **Match provider limits**: Set `rate_limit_requests_per_minute` to your provider's RPM
- **Burst**: Use `LEAKING_BUCKET` or `TOKEN_BUCKET` for smoother handling of bursts
- **Redis backend**: Use Redis for distributed rate limiting in multi-process deployments
- **API Key Pool**: Add multiple keys to increase effective throughput

## Batch Processing

- **Concurrency**: Tune `BatchConfig(max_concurrent=...)` to balance throughput and rate limits
- **Error strategy**: Use `CONTINUE` to maximize throughput when some failures are acceptable
- **Retries**: Enable `retry_failed=True` for transient errors

## Tool Execution

- **condense_tool_messages**: Enable to reduce context size in long tool loops
- **tool_mode="dynamic"**: Use for large toolsets to reduce prompt tokens per call
- **@static_tool**: Pin only essential tools; let routing handle the rest
- **tool_execution_order="parallel"**: Use when the model often requests multiple tools per round

## Connection Pooling

GlueLLM reuses HTTP clients via the provider cache. Call `close_providers()` only at shutdown. See [CONNECTION_POOLING.md](CONNECTION_POOLING.md).

## Timeouts

- **Request timeout**: Set appropriately for model size and context length
- **Connect timeout**: Usually 10–30 seconds
- **Avoid excessive values**: Keeps resources from hanging

## Model Selection

- **Lighter models**: Use `gpt-5.4-2026-03-05` or similar for routing, summarization, simple tasks
- **tool_route_model**: In dynamic mode, use a fast model for routing to reduce latency
- **Embeddings**: Use smaller embedding models when dimensions allow

## Workflows

- **Limit iterations**: Set `max_iterations`, `max_rounds` etc. to avoid runaway loops
- **Parallel executors**: Use `parallel_workers=True` in Hierarchical/MapReduce when applicable
- **Reduce agent count**: Fewer agents means fewer API calls
- **Cache**: Cache retrieval results in RAG workflows when possible

## Cost Tracking

- **track_costs**: Keep enabled to monitor usage
- **Session summary**: Use `print_session_summary_on_exit` to review costs

## See Also

- [BATCH_PROCESSING.md](BATCH_PROCESSING.md) - Batch optimization
- [TOOL_EXECUTION.md](TOOL_EXECUTION.md) - Tool optimization
- [RATE_LIMITING.md](RATE_LIMITING.md) - Rate limit configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) - Performance considerations
