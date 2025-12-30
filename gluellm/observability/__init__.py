"""Observability module for logging and utilities.

This module provides logging configuration and helper utilities
for debugging and monitoring GlueLLM applications.

Components:
    - logging_config: Production-grade logging setup with rotation
    - logging_utils: Decorators and utilities for function logging
"""

from gluellm.observability.logging_config import (
    CorrelationIDFilter,
    get_logger,
    setup_logging,
)
from gluellm.observability.logging_utils import (
    log_async_function_call,
    log_function_call,
    log_operation,
    log_timing,
    timed_operation,
)

__all__ = [
    # Logging config
    "setup_logging",
    "get_logger",
    "CorrelationIDFilter",
    # Logging utils
    "log_function_call",
    "log_async_function_call",
    "timed_operation",
]
