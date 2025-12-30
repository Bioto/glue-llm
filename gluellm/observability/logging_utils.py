"""Logging utilities and decorators for GlueLLM.

This module provides convenient utilities for logging, including decorators
for automatic function call logging and context managers for timing operations.
"""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from gluellm.observability.logging_config import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def log_function_call(
    logger: logging.Logger | None = None, log_args: bool = True, log_result: bool = False
) -> Callable[[F], F]:
    """Decorator to automatically log function calls.

    Args:
        logger: Logger instance (defaults to module logger)
        log_args: Whether to log function arguments
        log_result: Whether to log function return value

    Returns:
        Decorated function

    Example:
        >>> @log_function_call(log_args=True, log_result=True)
        ... def my_function(x: int, y: int) -> int:
        ...     return x + y
    """

    def decorator(func: F) -> F:
        func_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            if log_args:
                func_logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                func_logger.debug(f"Calling {func_name}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if log_result:
                    func_logger.debug(f"{func_name} completed in {elapsed:.3f}s, result={result}")
                else:
                    func_logger.debug(f"{func_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                func_logger.error(f"{func_name} failed after {elapsed:.3f}s: {e}", exc_info=True)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_async_function_call(
    logger: logging.Logger | None = None, log_args: bool = True, log_result: bool = False
) -> Callable[[F], F]:
    """Decorator to automatically log async function calls.

    Args:
        logger: Logger instance (defaults to module logger)
        log_args: Whether to log function arguments
        log_result: Whether to log function return value

    Returns:
        Decorated async function

    Example:
        >>> @log_async_function_call(log_args=True, log_result=True)
        ... async def my_async_function(x: int, y: int) -> int:
        ...     return x + y
    """

    def decorator(func: F) -> F:
        func_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            if log_args:
                func_logger.debug(f"Calling async {func_name} with args={args}, kwargs={kwargs}")
            else:
                func_logger.debug(f"Calling async {func_name}")

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                if log_result:
                    func_logger.debug(f"Async {func_name} completed in {elapsed:.3f}s, result={result}")
                else:
                    func_logger.debug(f"Async {func_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                func_logger.error(f"Async {func_name} failed after {elapsed:.3f}s: {e}", exc_info=True)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def timed_operation(operation_name: str, logger: logging.Logger | None = None, log_level: int = logging.DEBUG):
    """Context manager for timing operations with logging.

    Alias: log_timing (for backwards compatibility)

    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance (defaults to module logger)
        log_level: Log level to use (default: DEBUG)

    Example:
        >>> with timed_operation("database_query"):
        ...     result = db.query(...)
    """
    func_logger = logger or get_logger(__name__)
    start_time = time.time()
    func_logger.log(log_level, f"Starting {operation_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        func_logger.log(log_level, f"Completed {operation_name} in {elapsed:.3f}s")


# Backwards compatibility alias
log_timing = timed_operation


@contextmanager
def log_operation(operation_name: str, logger: logging.Logger | None = None, log_level: int = logging.INFO):
    """Context manager for logging operation start/end.

    Args:
        operation_name: Name of the operation
        logger: Logger instance (defaults to module logger)
        log_level: Log level to use (default: INFO)

    Example:
        >>> with log_operation("processing_file", log_level=logging.INFO):
        ...     process_file(...)
    """
    func_logger = logger or get_logger(__name__)
    func_logger.log(log_level, f"Starting {operation_name}")
    try:
        yield
        func_logger.log(log_level, f"Completed {operation_name}")
    except Exception as e:
        func_logger.error(f"Failed {operation_name}: {e}", exc_info=True)
        raise
