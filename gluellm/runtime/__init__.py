"""Runtime module for context and lifecycle management.

This module provides runtime utilities for request context propagation
and graceful shutdown handling.

Components:
    - context: Correlation ID and request metadata management
    - shutdown: Signal handlers and graceful shutdown support

Note: Imports are done lazily to avoid circular dependencies with observability.
"""

# Context imports (no dependencies on observability)
from gluellm.runtime.context import (
    clear_correlation_id,
    clear_request_metadata,
    get_context_dict,
    get_correlation_id,
    get_request_metadata,
    set_correlation_id,
    set_request_metadata,
    with_correlation_id,
)


def __getattr__(name: str):
    """Lazy import for shutdown module to avoid circular dependencies."""
    shutdown_exports = {
        "ShutdownContext",
        "execute_shutdown_callbacks",
        "get_in_flight_count",
        "graceful_shutdown",
        "is_shutting_down",
        "register_shutdown_callback",
        "setup_signal_handlers",
        "unregister_shutdown_callback",
        "wait_for_shutdown",
    }
    if name in shutdown_exports:
        from gluellm.runtime import shutdown

        return getattr(shutdown, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Context
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "with_correlation_id",
    "get_request_metadata",
    "set_request_metadata",
    "clear_request_metadata",
    "get_context_dict",
    # Shutdown (lazy loaded)
    "ShutdownContext",
    "is_shutting_down",
    "setup_signal_handlers",
    "graceful_shutdown",
    "register_shutdown_callback",
    "unregister_shutdown_callback",
    "get_in_flight_count",
    "wait_for_shutdown",
    "execute_shutdown_callbacks",
]
