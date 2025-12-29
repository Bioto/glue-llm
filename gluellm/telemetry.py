"""OpenTelemetry and MLflow tracing configuration for GlueLLM.

This module provides OpenTelemetry tracing integration using MLflow for observability
of LLM interactions. It supports both automatic instrumentation via MLflow's autolog
and manual tracing with custom spans.

Features:
    - Automatic tracing of LLM calls through MLflow
    - OpenTelemetry span creation and management
    - Configurable trace export to MLflow tracking server
    - Token usage and cost tracking via span attributes

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

import logging
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from gluellm.config import settings

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer = None
_tracing_enabled = False


def configure_tracing() -> None:
    """Configure OpenTelemetry tracing with MLflow integration.

    This function sets up the OpenTelemetry SDK to export traces to MLflow.
    It should be called once at application startup.

    The function will:
    1. Check if tracing is enabled via settings
    2. Configure the TracerProvider with appropriate resource attributes
    3. Set up the OTLP exporter to send traces to MLflow
    4. Initialize MLflow experiment if tracking URI is configured

    Note:
        This function is idempotent - calling it multiple times is safe.
    """
    global _tracer, _tracing_enabled

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

        logger.info("OpenTelemetry tracing configured successfully")

    except ImportError:
        logger.warning("MLflow not installed. Install with: pip install mlflow>=3.6.0")
        _tracing_enabled = False
    except Exception as e:
        logger.error(f"Failed to configure tracing: {e}")
        _tracing_enabled = False


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled.

    Returns:
        bool: True if tracing is enabled, False otherwise
    """
    return _tracing_enabled


@contextmanager
def trace_llm_call(model: str, messages: list[dict], tools: list[Any] | None = None, **attributes: Any):
    """Context manager for tracing LLM calls with OpenTelemetry.

    Creates a span for an LLM call with relevant attributes and metrics.
    The span captures:
    - Model name and provider
    - Input messages and token count
    - Tool usage information
    - Response metadata
    - Errors and exceptions

    Args:
        model: Model identifier (e.g., "openai:gpt-4o-mini")
        messages: List of message dictionaries
        tools: Optional list of tools available for the call
        **attributes: Additional span attributes to include

    Yields:
        Span: The active span object for adding custom attributes

    Example:
        >>> with trace_llm_call("openai:gpt-4o-mini", messages) as span:
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
    with _tracer.start_as_current_span(
        "llm.completion",
        attributes={
            "llm.provider": provider,
            "llm.model": model_name,
            "llm.messages_count": len(messages),
            "llm.tools_available": len(tools) if tools else 0,
            **attributes,
        },
    ) as span:
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


def record_token_usage(span: Any, usage: dict[str, int]) -> None:
    """Record token usage information on a span.

    Args:
        span: OpenTelemetry span object
        usage: Dictionary with token counts (prompt, completion, total)
    """
    if not _tracing_enabled:
        return

    set_span_attributes(
        span,
        **{
            "llm.tokens.prompt": usage.get("prompt", 0),
            "llm.tokens.completion": usage.get("completion", 0),
            "llm.tokens.total": usage.get("total", 0),
        },
    )


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
