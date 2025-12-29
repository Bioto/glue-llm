# Connection Pooling

GlueLLM uses `any-llm-sdk` for LLM provider communication, which handles HTTP connection pooling automatically. This document explains how connection pooling works and how to configure it if needed.

## How Connection Pooling Works

The `any-llm-sdk` library manages HTTP connections to LLM providers (OpenAI, Anthropic, xAI, etc.) using connection pools. This provides several benefits:

- **Performance**: Reuses existing connections instead of creating new ones for each request
- **Efficiency**: Reduces overhead from TCP handshakes and TLS negotiations
- **Resource Management**: Limits the number of concurrent connections

## Default Behavior

By default, `any-llm-sdk` uses the underlying HTTP client's default connection pool settings:

- **httpx** (used by most providers): Default pool size of 100 connections
- **aiohttp** (if used): Default pool size of 100 connections

These defaults are typically sufficient for most use cases.

## Configuration

If you need to customize connection pooling, you can configure the underlying HTTP client used by `any-llm-sdk`. However, since GlueLLM doesn't directly expose HTTP client configuration, you may need to:

1. **Use any-llm-sdk's configuration options** (if available)
2. **Set environment variables** that the HTTP client respects
3. **Configure at the provider SDK level** (e.g., OpenAI SDK, Anthropic SDK)

### Example: Configuring httpx Connection Limits

If you're using httpx (which most providers use), you can set environment variables:

```bash
# Set connection pool size (if supported by the provider SDK)
export HTTPX_MAX_CONNECTIONS=100
export HTTPX_MAX_KEEPALIVE_CONNECTIONS=20
```

### Example: Using Provider-Specific Configuration

For provider-specific configuration, refer to the provider's SDK documentation:

- **OpenAI**: Uses `httpx` with default pooling
- **Anthropic**: Uses `httpx` with default pooling
- **xAI**: Uses `httpx` with default pooling

## Best Practices

1. **Default Settings**: The default connection pool settings are usually sufficient. Only customize if you have specific requirements.

2. **High Concurrency**: If you're making many concurrent requests, ensure your connection pool size is adequate. The default of 100 connections should handle most scenarios.

3. **Connection Timeouts**: Connection timeouts are handled by GlueLLM's request timeout configuration (`GLUELLM_DEFAULT_REQUEST_TIMEOUT`).

4. **Monitoring**: Monitor connection pool usage if you experience connection-related issues. Most HTTP clients provide metrics or logging for pool usage.

## Troubleshooting

If you experience connection-related issues:

1. **Connection Pool Exhaustion**: If you see errors about connection pool exhaustion, you may need to increase the pool size or reduce concurrent requests.

2. **Connection Timeouts**: Adjust `GLUELLM_DEFAULT_REQUEST_TIMEOUT` or use per-request timeouts.

3. **Keep-Alive Issues**: Some providers may close idle connections. The HTTP client should handle reconnection automatically.

## Summary

- Connection pooling is handled automatically by `any-llm-sdk`
- Default settings are sufficient for most use cases
- Customization may be needed for high-concurrency scenarios
- Connection timeouts are managed by GlueLLM's timeout configuration

For more details, refer to:
- [any-llm-sdk documentation](https://github.com/BerkYeni/any_llm_client)
- Provider-specific SDK documentation (OpenAI, Anthropic, etc.)
