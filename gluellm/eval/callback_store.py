"""Callback-based storage backend for evaluation records.

This module provides a storage backend that delegates to a user-provided
callback function, enabling custom storage logic (e.g., database writes).
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from gluellm.models.eval import EvalRecord
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)


class CallbackStore:
    """Storage backend that delegates to a user-provided callback.

    Supports both sync and async callbacks. Async callbacks are awaited,
    sync callbacks are run in an executor to avoid blocking.

    Attributes:
        callback: The callback function to invoke for each record
        _closed: Whether the store has been closed

    Example:
        >>> async def save_to_db(record: EvalRecord):
        ...     await db.insert("eval_records", record.model_dump_dict())
        >>>
        >>> from gluellm.eval import CallbackStore
        >>> store = CallbackStore(save_to_db)
        >>> await store.record(eval_record)
    """

    def __init__(self, callback: Callable[[EvalRecord], Awaitable[None] | None]):
        """Initialize callback store.

        Args:
            callback: Callable that accepts an EvalRecord. Can be sync or async.
        """
        self.callback = callback
        self._closed = False
        self._is_async = inspect.iscoroutinefunction(callback)

    async def record(self, record: EvalRecord) -> None:
        """Invoke the callback with the record.

        Args:
            record: The evaluation record to store
        """
        if self._closed:
            logger.warning("Attempted to record to closed CallbackStore")
            return

        try:
            if self._is_async:
                await self.callback(record)  # type: ignore
            else:
                # Run sync callback in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.callback, record)

        except Exception as e:
            # Log but don't raise - recording failures shouldn't break completions
            logger.error(f"Callback store failed to record evaluation: {e}", exc_info=True)

    async def close(self) -> None:
        """Mark the store as closed.

        Note: This doesn't call the callback - it just prevents further records.
        If your callback needs cleanup, handle that separately.
        """
        self._closed = True
