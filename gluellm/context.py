"""Request context management for GlueLLM.

This module provides request correlation IDs and context propagation for
distributed tracing and logging across async operations.
"""

import contextvars
import uuid
from contextlib import contextmanager
from typing import Any

# Context variable for storing request correlation ID
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)

# Context variable for storing additional request metadata
# Note: Using None as default to avoid mutable default value issue
_request_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "request_metadata", default=None
)


def get_correlation_id() -> str | None:
    """Get the current request correlation ID.

    Returns:
        The correlation ID string if set, None otherwise
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set the correlation ID for the current context.

    If no correlation_id is provided, generates a new UUID.

    Args:
        correlation_id: Optional correlation ID to use (defaults to generating a new UUID)

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id.set(None)


def get_request_metadata() -> dict[str, Any]:
    """Get the current request metadata.

    Returns:
        Dictionary of request metadata (empty dict if not set)
    """
    metadata = _request_metadata.get()
    if metadata is None:
        return {}
    return metadata.copy()


def set_request_metadata(**metadata: Any) -> None:
    """Set request metadata for the current context.

    Args:
        **metadata: Key-value pairs to add to request metadata
    """
    current = _request_metadata.get()
    if current is None:
        current = {}
    updated = {**current, **metadata}
    _request_metadata.set(updated)


def clear_request_metadata() -> None:
    """Clear request metadata from the current context."""
    _request_metadata.set(None)


@contextmanager
def with_correlation_id(correlation_id: str | None = None):
    """Context manager for setting correlation ID in a scope.

    Args:
        correlation_id: Optional correlation ID (defaults to generating a new UUID)

    Example:
        >>> with with_correlation_id("req-123"):
        ...     result = some_operation()
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    token = _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.reset(token)


def get_context_dict() -> dict[str, Any]:
    """Get a dictionary with all context information.

    Returns:
        Dictionary containing correlation_id and metadata
    """
    return {
        "correlation_id": get_correlation_id(),
        "metadata": get_request_metadata(),
    }
