"""Graceful shutdown support for GlueLLM.

This module provides signal handlers and shutdown management for
zero-downtime deployments and clean resource cleanup.
"""

import asyncio
import signal
import threading
from collections.abc import Awaitable, Callable

from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)

# Global shutdown state
_shutdown_event = threading.Event()
_shutdown_callbacks: list[Callable[[], None] | Callable[[], Awaitable[None]]] = []
_in_flight_requests = 0
_shutdown_lock = threading.Lock()


def is_shutting_down() -> bool:
    """Check if shutdown has been initiated.

    Returns:
        True if shutdown is in progress, False otherwise
    """
    return _shutdown_event.is_set()


def register_shutdown_callback(callback: Callable[[], None] | Callable[[], Awaitable[None]]) -> None:
    """Register a callback to be called during graceful shutdown.

    Callbacks are executed in the order they were registered.

    Args:
        callback: Function or coroutine to call during shutdown
    """
    with _shutdown_lock:
        _shutdown_callbacks.append(callback)
    logger.debug(f"Registered shutdown callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")


def unregister_shutdown_callback(callback: Callable[[], None] | Callable[[], Awaitable[None]]) -> None:
    """Unregister a shutdown callback.

    Args:
        callback: Callback to remove
    """
    with _shutdown_lock:
        if callback in _shutdown_callbacks:
            _shutdown_callbacks.remove(callback)
            logger.debug(
                f"Unregistered shutdown callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}"
            )


def increment_in_flight() -> None:
    """Increment the count of in-flight requests."""
    global _in_flight_requests
    with _shutdown_lock:
        _in_flight_requests += 1


def decrement_in_flight() -> None:
    """Decrement the count of in-flight requests."""
    global _in_flight_requests
    with _shutdown_lock:
        _in_flight_requests = max(0, _in_flight_requests - 1)


def get_in_flight_count() -> int:
    """Get the current count of in-flight requests.

    Returns:
        Number of requests currently being processed
    """
    with _shutdown_lock:
        return _in_flight_requests


async def wait_for_shutdown(max_wait_time: float = 30.0) -> None:
    """Wait for all in-flight requests to complete or timeout.

    Args:
        max_wait_time: Maximum time to wait in seconds (default: 30)
    """
    import time

    start_time = time.time()
    while get_in_flight_count() > 0 and (time.time() - start_time) < max_wait_time:
        await asyncio.sleep(0.1)
        logger.debug(f"Waiting for {get_in_flight_count()} in-flight requests to complete...")

    remaining = get_in_flight_count()
    if remaining > 0:
        logger.warning(f"Shutdown timeout: {remaining} requests still in flight")
    else:
        logger.info("All in-flight requests completed")


async def execute_shutdown_callbacks() -> None:
    """Execute all registered shutdown callbacks."""
    with _shutdown_lock:
        callbacks = _shutdown_callbacks.copy()

    logger.info(f"Executing {len(callbacks)} shutdown callbacks...")
    for callback in callbacks:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
            logger.debug(
                f"Shutdown callback executed: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}"
            )
        except Exception as e:
            logger.error(f"Error executing shutdown callback: {e}", exc_info=True)


def _signal_handler(signum: int, frame) -> None:  # type: ignore
    """Handle shutdown signals (SIGTERM, SIGINT).

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
    _shutdown_event.set()


async def graceful_shutdown(max_wait_time: float = 30.0) -> None:
    """Perform graceful shutdown.

    This function:
    1. Sets the shutdown event
    2. Waits for in-flight requests to complete
    3. Executes shutdown callbacks
    4. Logs completion

    Args:
        max_wait_time: Maximum time to wait for requests to complete (default: 30 seconds)
    """
    logger.info("Starting graceful shutdown...")
    _shutdown_event.set()

    # Wait for in-flight requests
    await wait_for_shutdown(max_wait_time)

    # Execute shutdown callbacks
    await execute_shutdown_callbacks()

    logger.info("Graceful shutdown completed")


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown.

    Registers handlers for SIGTERM and SIGINT signals.
    """
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    logger.info("Signal handlers registered for graceful shutdown")


class ShutdownContext:
    """Context manager for tracking request lifecycle during shutdown."""

    def __init__(self):
        self._entered = False

    def __enter__(self) -> "ShutdownContext":
        """Enter the context and increment in-flight counter."""
        if is_shutting_down():
            raise RuntimeError("Cannot start new request: shutdown in progress")
        increment_in_flight()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and decrement in-flight counter."""
        if self._entered:
            decrement_in_flight()

    async def __aenter__(self) -> "ShutdownContext":
        """Async enter the context."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async exit the context."""
        self.__exit__(exc_type, exc_val, exc_tb)
