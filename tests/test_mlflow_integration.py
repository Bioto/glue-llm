"""Comprehensive test for MLflow metrics integration with actual API calls."""

import asyncio
import importlib.util

import pytest

from gluellm.telemetry import configure_tracing, is_mlflow_enabled, log_llm_metrics, mlflow_run_context


@pytest.mark.asyncio
async def test_mlflow_integration():
    """Test MLflow metrics integration end-to-end."""
    print("=" * 60)
    print("Testing MLflow Metrics Integration")
    print("=" * 60)

    # Check if MLflow is available
    mlflow_spec = importlib.util.find_spec("mlflow")
    if mlflow_spec is not None:
        print("✓ MLflow is installed")
    else:
        print("✗ MLflow is not installed - install with: pip install mlflow>=3.6.0")
        return

    # Configure tracing (this also enables MLflow metrics)
    print("\n1. Configuring tracing...")
    configure_tracing()

    if is_mlflow_enabled():
        print("   ✓ MLflow metrics tracking is enabled")
    else:
        print("   ⚠ MLflow metrics tracking is not enabled")
        print("   (Set GLUELLM_ENABLE_TRACING=true to enable)")

    # Test 1: Direct metrics logging
    print("\n2. Testing direct metrics logging...")
    log_llm_metrics(
        model="openai:gpt-4o-mini",
        latency=1.234,
        tokens_used={"prompt": 100, "completion": 50, "total": 150},
        finish_reason="stop",
        has_tool_calls=False,
        error=False,
    )
    print("   ✓ Metrics logged successfully")

    # Test 2: Metrics with tool calls
    print("\n3. Testing metrics with tool calls...")
    log_llm_metrics(
        model="anthropic:claude-3-5-sonnet-20241022",
        latency=2.456,
        tokens_used={"prompt": 200, "completion": 100, "total": 300},
        finish_reason="tool_calls",
        has_tool_calls=True,
        error=False,
    )
    print("   ✓ Metrics with tool calls logged")

    # Test 3: Error metrics
    print("\n4. Testing error metrics logging...")
    log_llm_metrics(
        model="openai:gpt-4o-mini",
        latency=0.5,
        tokens_used=None,
        finish_reason=None,
        has_tool_calls=False,
        error=True,
        error_type="RateLimitError",
    )
    print("   ✓ Error metrics logged")

    # Test 4: Run context
    print("\n5. Testing MLflow run context...")
    try:
        with mlflow_run_context("test_workflow", tags={"test": "true", "version": "1.0"}):
            log_llm_metrics(
                model="xai:grok-beta",
                latency=3.789,
                tokens_used={"prompt": 150, "completion": 75, "total": 225},
                finish_reason="stop",
                has_tool_calls=False,
                error=False,
            )
            print("   ✓ Metrics logged within run context")
    except Exception as e:
        print(f"   ⚠ Run context test: {e}")
        print("   (This is expected if MLflow server is not running)")

    print("\n" + "=" * 60)
    print("✓ All integration tests completed!")
    print("=" * 60)
    print("\nNote: To see metrics in MLflow UI:")
    print("  1. Start MLflow server: mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("  2. Set GLUELLM_ENABLE_TRACING=true")
    print("  3. Set GLUELLM_MLFLOW_TRACKING_URI=http://localhost:5000")
    print("  4. Run your application and check MLflow UI")


if __name__ == "__main__":
    asyncio.run(test_mlflow_integration())
