"""Infrastructure-related CLI commands.

Commands for testing error handling, hooks, telemetry, rate limiting, etc.
"""

import click

from gluellm.cli.utils import (
    console,
    print_error,
    print_header,
    print_info,
    print_result,
    print_step,
    print_success,
    print_warning,
    run_async,
)


@click.command("test-error-handling")
def test_error_handling() -> None:
    """Test error handling and classification."""
    from gluellm.api import LLMError, RateLimitError, TokenLimitError, classify_llm_error

    print_header("Test Error Handling")

    # Test error classification
    test_cases = [
        ("context length exceeded", TokenLimitError),
        ("rate limit exceeded", RateLimitError),
        ("connection timeout", LLMError),
    ]

    for msg, expected_type in test_cases:
        error = classify_llm_error(Exception(msg))
        status = "✓" if isinstance(error, expected_type) else "✗"
        console.print(f"  {status} {msg} -> {type(error).__name__}")

    print_success("Error handling test passed")


@click.command("test-hooks")
def test_hooks() -> None:
    """Test hooks system for lifecycle events."""
    from gluellm.executors import SimpleExecutor
    from gluellm.hooks.utils import normalize_whitespace, remove_pii
    from gluellm.models.hook import HookConfig, HookErrorStrategy, HookRegistry, HookStage

    print_header("Test Hooks System")

    registry = HookRegistry()
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(handler=remove_pii, name="pii_filter", error_strategy=HookErrorStrategy.SKIP),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(handler=normalize_whitespace, name="normalize", error_strategy=HookErrorStrategy.SKIP),
    )

    async def run_test():
        executor = SimpleExecutor(hook_registry=registry)
        return await executor.execute("My email is test@example.com. What is Python?")

    try:
        result = run_async(run_test())
        print_result("Response", result[:200])
        print_success("Hooks test passed")
    except Exception as e:
        print_error(f"Hooks test failed: {e}")


@click.command("test-correlation-ids")
def test_correlation_ids() -> None:
    """Test correlation ID tracking."""
    import uuid

    from gluellm.api import complete

    print_header("Test Correlation IDs")

    async def run_test():
        custom_id = f"test-{uuid.uuid4().hex[:8]}"
        result = await complete("Say hello", correlation_id=custom_id)
        return custom_id, result

    try:
        custom_id, result = run_async(run_test())
        console.print(f"  Correlation ID: {custom_id}")
        print_success("Correlation ID test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-telemetry")
def test_telemetry() -> None:
    """Test OpenTelemetry tracing integration."""
    from gluellm.api import complete
    from gluellm.telemetry import is_mlflow_enabled, is_tracing_enabled

    print_header("Test Telemetry")

    print_info(f"Tracing enabled: {is_tracing_enabled()}")
    print_info(f"MLflow enabled: {is_mlflow_enabled()}")

    if not is_tracing_enabled():
        print_warning("Tracing is disabled. Enable with GLUELLM_ENABLE_TRACING=true")
        return

    async def run_test():
        return await complete("Test telemetry")

    try:
        run_async(run_test())
        print_success("Telemetry test passed")
    except Exception as e:
        print_error(f"Telemetry test failed: {e}")


@click.command("test-rate-limiting")
@click.option("--requests", "-r", default=5, type=int, help="Number of rapid requests")
def test_rate_limiting(requests: int) -> None:
    """Test rate limiting functionality."""
    import time

    from gluellm.rate_limiting.rate_limiter import acquire_rate_limit

    print_header("Test Rate Limiting", f"Requests: {requests}")

    async def run_test():
        start = time.time()
        for i in range(requests):
            await acquire_rate_limit("test_key")
            print_step(i + 1, requests, f"Request {i + 1} allowed")
        return time.time() - start

    try:
        elapsed = run_async(run_test())
        console.print(f"  Total time: {elapsed:.2f}s")
        print_success("Rate limiting test passed")
    except Exception as e:
        print_error(f"Rate limiting test failed: {e}")


@click.command("test-timeout")
@click.option("--timeout", "-t", default=5.0, type=float, help="Timeout in seconds")
def test_timeout(timeout: float) -> None:
    """Test timeout handling."""
    from gluellm.api import complete

    print_header("Test Timeout Handling", f"Timeout: {timeout}s")

    async def run_test():
        return await complete(
            "Write a very short poem about time.",
            timeout=timeout,
        )

    try:
        result = run_async(run_test())
        print_result("Response", result.final_response[:200])
        print_success("Timeout test passed")
    except TimeoutError:
        print_warning("Request timed out as expected")
    except Exception as e:
        print_error(f"Timeout test failed: {e}")


@click.command("test-api-key-pool")
@click.option("--keys", "-k", default=2, type=int, help="Number of API keys to simulate")
def test_api_key_pool(keys: int) -> None:
    """Test API key pooling functionality."""
    from gluellm.rate_limiting.api_key_pool import APIKeyPool

    print_header("Test API Key Pool", f"Keys: {keys}")

    pool = APIKeyPool()
    test_keys = [f"sk-test-key-{i}" for i in range(keys)]

    for key in test_keys:
        pool.add_key("openai", key)

    for i in range(keys * 2):
        key = pool.get_key("openai")
        console.print(f"  Request {i + 1}: Using key {key[-8:]}")

    print_success("API key pool test passed")


@click.command("test-different-models")
@click.option("--models", "-m", multiple=True, default=["openai:gpt-4o-mini", "openai:gpt-3.5-turbo"])
def test_different_models(models: tuple) -> None:
    """Test switching between different models."""
    from gluellm.api import complete

    print_header("Test Different Models", f"Models: {len(models)}")

    async def run_test():
        results = {}
        for model in models:
            try:
                result = await complete("Say 'hello' in one word", model=model)
                results[model] = result.final_response[:50]
            except Exception as e:
                results[model] = f"Error: {e}"
        return results

    try:
        results = run_async(run_test())
        for model, response in results.items():
            console.print(f"  {model}: {response}")
        print_success("Model switching test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


# Export all commands
infrastructure_commands = [
    test_error_handling,
    test_hooks,
    test_correlation_ids,
    test_telemetry,
    test_rate_limiting,
    test_timeout,
    test_api_key_pool,
    test_different_models,
]
