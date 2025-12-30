"""OpenTelemetry and MLflow tracing configuration for GlueLLM.

This module provides OpenTelemetry tracing integration using MLflow for observability
of LLM interactions. It supports both automatic instrumentation via MLflow's autolog
and manual tracing with custom spans.

Features:
    - Automatic tracing of LLM calls through MLflow
    - OpenTelemetry span creation and management
    - Configurable trace export to MLflow tracking server
    - Token usage and cost tracking via span attributes
    - MLflow metrics logging for LLM calls

Configuration:
    Set environment variables or use settings:
    - GLUELLM_ENABLE_TRACING: Enable/disable tracing (default: False)
    - GLUELLM_MLFLOW_TRACKING_URI: MLflow tracking server URI
    - GLUELLM_MLFLOW_EXPERIMENT_NAME: MLflow experiment name
    - OTEL_EXPORTER_OTLP_ENDPOINT: OpenTelemetry OTLP endpoint

Example:
    >>> from gluellm.telemetry import configure_tracing, trace_llm_call
    >>>
    >>> # Configure tracing on startup
    >>> configure_tracing()
    >>>
    >>> # LLM calls will be automatically traced
    >>> result = await complete("Hello, world!")
"""

from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from gluellm.config import settings
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)

# Global tracer instance
_tracer = None
_tracing_enabled = False
_mlflow_enabled = False
_mlflow_client = None
_default_mlflow_run = None


def configure_tracing() -> None:
    """Configure OpenTelemetry tracing with MLflow integration.

    This function sets up the OpenTelemetry SDK to export traces to MLflow.
    It should be called once at application startup.

    The function will:
    1. Check if tracing is enabled via settings
    2. Configure the TracerProvider with appropriate resource attributes
    3. Set up the OTLP exporter to send traces to MLflow
    4. Initialize MLflow experiment if tracking URI is configured
    5. Enable MLflow metrics tracking

    Note:
        This function is idempotent - calling it multiple times is safe.
    """
    global _tracer, _tracing_enabled, _mlflow_enabled, _mlflow_client, _default_mlflow_run

    if not settings.enable_tracing:
        logger.info("OpenTelemetry tracing is disabled")
        return

    try:
        # Import mlflow here to make it optional
        import mlflow

        # Configure MLflow if tracking URI is set
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {settings.mlflow_tracking_uri}")

        # Set or create experiment
        if settings.mlflow_experiment_name:
            mlflow.set_experiment(settings.mlflow_experiment_name)
            logger.info(f"MLflow experiment set to: {settings.mlflow_experiment_name}")

        # Initialize MLflow client for metrics tracking
        try:
            _mlflow_client = mlflow.tracking.MlflowClient()
            _mlflow_enabled = True
            logger.info("MLflow metrics tracking enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow client: {e}")
            _mlflow_enabled = False

        # Create resource with service information
        resource = Resource.create(
            {
                "service.name": "gluellm",
                "service.version": "0.1.0",
            }
        )

        # Set up tracer provider
        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        if settings.otel_exporter_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, headers={})
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured with endpoint: {settings.otel_exporter_endpoint}")

        # Set the tracer provider
        trace.set_tracer_provider(provider)

        # Get a tracer instance
        _tracer = trace.get_tracer("gluellm.api")
        _tracing_enabled = True

        # Register shutdown callback for cleanup
        from gluellm.runtime.shutdown import register_shutdown_callback

        register_shutdown_callback(shutdown_telemetry)

        logger.info("OpenTelemetry tracing configured successfully")

    except ImportError:
        logger.warning("MLflow not installed. Install with: pip install mlflow>=3.6.0")
        _tracing_enabled = False
        _mlflow_enabled = False
    except Exception as e:
        logger.error(f"Failed to configure tracing: {e}")
        _tracing_enabled = False
        _mlflow_enabled = False


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled.

    Returns:
        bool: True if tracing is enabled, False otherwise
    """
    return _tracing_enabled


@contextmanager
def trace_llm_call(
    model: str,
    messages: list[dict],
    tools: list[Any] | None = None,
    correlation_id: str | None = None,
    **attributes: Any,
):
    """Context manager for tracing LLM calls with OpenTelemetry.

    Creates a span for an LLM call with relevant attributes and metrics.
    The span captures:
    - Model name and provider
    - Input messages and token count
    - Tool usage information
    - Response metadata
    - Errors and exceptions
    - Correlation ID (if provided)

    Args:
        model: Model identifier (e.g., "openai:gpt-4o-mini")
        messages: List of message dictionaries
        tools: Optional list of tools available for the call
        correlation_id: Optional correlation ID for request tracking
        **attributes: Additional span attributes to include

    Yields:
        Span: The active span object for adding custom attributes

    Example:
        >>> with trace_llm_call("openai:gpt-4o-mini", messages, correlation_id="req-123") as span:
        ...     response = await llm_call(messages)
        ...     span.set_attribute("response.tokens", response.usage.total_tokens)
    """
    if not _tracing_enabled or _tracer is None:
        # Tracing disabled, yield a no-op context
        class NoOpSpan:
            def set_attribute(self, *args, **kwargs):
                pass

            def set_status(self, *args, **kwargs):
                pass

            def record_exception(self, *args, **kwargs):
                pass

        yield NoOpSpan()
        return

    # Parse model into provider and name
    provider, model_name = model.split(":", 1) if ":" in model else ("unknown", model)

    # Start a new span
    span_attributes = {
        "llm.provider": provider,
        "llm.model": model_name,
        "llm.messages_count": len(messages),
        "llm.tools_available": len(tools) if tools else 0,
        **attributes,
    }
    if correlation_id:
        span_attributes["correlation_id"] = correlation_id

    with _tracer.start_as_current_span("llm.completion", attributes=span_attributes) as span:
        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def set_span_attributes(span: Any, **attributes: Any) -> None:
    """Set multiple attributes on a span safely.

    Args:
        span: OpenTelemetry span object
        **attributes: Key-value pairs to set as span attributes
    """
    if not _tracing_enabled:
        return

    for key, value in attributes.items():
        try:
            # Convert complex types to strings
            if isinstance(value, (dict, list)):
                value = str(value)
            span.set_attribute(key, value)
        except Exception as e:
            logger.debug(f"Failed to set span attribute {key}: {e}")


def record_token_usage(span: Any, usage: dict[str, int], cost_usd: float | None = None) -> None:
    """Record token usage and cost information on a span.

    Args:
        span: OpenTelemetry span object
        usage: Dictionary with token counts (prompt, completion, total)
        cost_usd: Optional estimated cost in USD
    """
    if not _tracing_enabled:
        return

    attributes = {
        "llm.tokens.prompt": usage.get("prompt", 0),
        "llm.tokens.completion": usage.get("completion", 0),
        "llm.tokens.total": usage.get("total", 0),
    }

    if cost_usd is not None:
        attributes["llm.cost.usd"] = cost_usd

    set_span_attributes(span, **attributes)


def record_cost(span: Any, cost_usd: float, model: str | None = None) -> None:
    """Record cost information on a span.

    Args:
        span: OpenTelemetry span object
        cost_usd: Estimated cost in USD
        model: Optional model identifier for context
    """
    if not _tracing_enabled:
        return

    attributes = {"llm.cost.usd": cost_usd}
    if model:
        attributes["llm.cost.model"] = model

    set_span_attributes(span, **attributes)


def record_tool_execution(span: Any, tool_name: str, arguments: dict, result: str, error: bool = False) -> None:
    """Record tool execution details on a span.

    Args:
        span: OpenTelemetry span object
        tool_name: Name of the tool that was executed
        arguments: Tool call arguments
        result: Tool execution result
        error: Whether the tool execution failed
    """
    if not _tracing_enabled:
        return

    set_span_attributes(
        span,
        **{
            f"tool.{tool_name}.called": True,
            f"tool.{tool_name}.error": error,
            # Avoid logging sensitive data
            f"tool.{tool_name}.arg_count": len(arguments),
        },
    )


def log_llm_metrics(
    model: str,
    latency: float,
    tokens_used: dict[str, int] | None = None,
    finish_reason: str | None = None,
    has_tool_calls: bool = False,
    error: bool = False,
    error_type: str | None = None,
) -> None:
    """Log LLM call metrics to MLflow.

    This function logs metrics from any-llm client calls to MLflow for tracking
    and analysis. Metrics include latency, token usage, and call metadata.

    Args:
        model: Model identifier (e.g., "openai:gpt-4o-mini")
        latency: Call latency in seconds
        tokens_used: Dictionary with token counts (prompt, completion, total)
        finish_reason: Reason the completion finished
        has_tool_calls: Whether the response included tool calls
        error: Whether the call failed
        error_type: Type of error if call failed
    """
    if not _mlflow_enabled or _mlflow_client is None:
        return

    try:
        import mlflow

        # Parse model into provider and name
        provider, model_name = model.split(":", 1) if ":" in model else ("unknown", model)

        # Prepare metrics to log
        metrics = {
            "llm.latency_seconds": latency,
        }

        # Add token usage metrics if available
        if tokens_used:
            metrics["llm.tokens.prompt"] = tokens_used.get("prompt", 0)
            metrics["llm.tokens.completion"] = tokens_used.get("completion", 0)
            metrics["llm.tokens.total"] = tokens_used.get("total", 0)

        # Add error metric if call failed
        if error:
            metrics["llm.error"] = 1.0
        else:
            metrics["llm.success"] = 1.0

        # Log metrics to current run or create a default run if none exists
        try:
            current_run = mlflow.active_run()
            if current_run is None:
                # No active run - create a default one for automatic metrics tracking
                # This ensures metrics are always logged without requiring manual run management
                global _default_mlflow_run
                if _default_mlflow_run is None:
                    _default_mlflow_run = mlflow.start_run(run_name="gluellm_auto_metrics")
                    logger.debug("Created default MLflow run for automatic metrics tracking")

            # Log metrics to the active run
            # MLflow accumulates metrics over time, allowing aggregation and analysis
            mlflow.log_metrics(metrics)

            # Log parameters for this call
            # Note: Parameters are overwritten on each call, which is fine for tracking
            # the most recent call's metadata. For historical tracking, use tags or metrics.
            params = {
                "llm.provider": provider,
                "llm.model": model_name,
            }
            if finish_reason:
                params["llm.finish_reason"] = finish_reason
            if error_type:
                params["llm.error_type"] = error_type
            params["llm.has_tool_calls"] = str(has_tool_calls)
            mlflow.log_params(params)
        except Exception as e:
            logger.debug(f"Failed to log metrics to MLflow: {e}")

    except ImportError:
        logger.debug("MLflow not available for metrics logging")
    except Exception as e:
        logger.debug(f"Error logging metrics to MLflow: {e}")


@contextmanager
def mlflow_run_context(run_name: str | None = None, tags: dict[str, str] | None = None):
    """Context manager for MLflow run tracking.

    Creates an MLflow run for tracking a group of LLM calls or operations.
    All metrics logged within this context will be associated with this run.

    Args:
        run_name: Optional name for the run
        tags: Optional dictionary of tags to add to the run

    Yields:
        The MLflow run object

    Example:
        >>> with mlflow_run_context("my_llm_workflow"):
        ...     result = await complete("Hello!")
        ...     log_llm_metrics(...)
    """
    if not _mlflow_enabled:
        # Return a no-op context if MLflow is not enabled
        class NoOpRun:
            pass

        yield NoOpRun()
        return

    try:
        import mlflow

        with mlflow.start_run(run_name=run_name, tags=tags):
            yield mlflow.active_run()
    except ImportError:
        logger.debug("MLflow not available for run context")
        yield None
    except Exception as e:
        logger.debug(f"Error creating MLflow run: {e}")
        yield None


def is_mlflow_enabled() -> bool:
    """Check if MLflow metrics tracking is enabled.

    Returns:
        bool: True if MLflow is enabled and configured, False otherwise
    """
    return _mlflow_enabled


def shutdown_telemetry() -> None:
    """Shutdown telemetry and cleanup resources.

    This function should be called when the application is shutting down to:
    1. Close any active MLflow runs (including the default auto-metrics run)
    2. Flush any pending trace exports
    3. Reset global state

    Call this during application cleanup or use with atexit:

    Example:
        >>> import atexit
        >>> from gluellm.telemetry import configure_tracing, shutdown_telemetry
        >>>
        >>> configure_tracing()
        >>> atexit.register(shutdown_telemetry)
        >>>
        >>> # Or manually during cleanup:
        >>> shutdown_telemetry()
    """
    global _tracer, _tracing_enabled, _mlflow_enabled, _mlflow_client, _default_mlflow_run

    logger.debug("Shutting down telemetry...")

    # Close default MLflow run if one was created
    if _default_mlflow_run is not None:
        try:
            import mlflow

            # End the default run if it's still active
            if mlflow.active_run() is not None and mlflow.active_run().info.run_id == _default_mlflow_run.info.run_id:
                mlflow.end_run()
                logger.debug("Closed default MLflow run")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error closing default MLflow run: {e}")
        finally:
            _default_mlflow_run = None

    # Flush OpenTelemetry trace provider
    if _tracing_enabled:
        try:
            provider = trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush()
                logger.debug("Flushed OpenTelemetry traces")
            if hasattr(provider, "shutdown"):
                provider.shutdown()
                logger.debug("Shutdown OpenTelemetry tracer provider")
        except Exception as e:
            logger.warning(f"Error shutting down OpenTelemetry: {e}")

    # Reset global state
    _tracer = None
    _tracing_enabled = False
    _mlflow_enabled = False
    _mlflow_client = None

    logger.info("Telemetry shutdown complete")


def reset_default_mlflow_run() -> None:
    """Close the current default MLflow run and allow a new one to be created.

    Useful when you want to start fresh metrics tracking without fully
    shutting down telemetry.

    Example:
        >>> # After a batch job completes, reset for the next batch
        >>> reset_default_mlflow_run()
    """
    global _default_mlflow_run

    if _default_mlflow_run is None:
        return

    try:
        import mlflow

        if mlflow.active_run() is not None and mlflow.active_run().info.run_id == _default_mlflow_run.info.run_id:
            mlflow.end_run()
            logger.debug("Closed and reset default MLflow run")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error resetting MLflow run: {e}")
    finally:
        _default_mlflow_run = None


def get_default_mlflow_run():
    """Get the current default MLflow run if one exists.

    Returns:
        The MLflow Run object if a default run is active, None otherwise.

    Example:
        >>> run = get_default_mlflow_run()
        >>> if run:
        ...     print(f"Current run ID: {run.info.run_id}")
    """
    return _default_mlflow_run
