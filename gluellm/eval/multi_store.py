"""Multi-store backend for fan-out to multiple storage backends.

This module provides a storage backend that writes to multiple stores
simultaneously, enabling scenarios like file backup + database storage.
"""

import asyncio

from gluellm.eval.store import EvalStore
from gluellm.models.eval import EvalRecord
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)


class MultiStore:
    """Storage backend that fans out to multiple stores.

    Writes each record to all configured stores in parallel. If any store
    fails, the others continue to receive the record.

    Attributes:
        stores: List of storage backends to write to

    Example:
        >>> from gluellm.eval import MultiStore, JSONLFileStore, CallbackStore
        >>>
        >>> store = MultiStore([
        ...     JSONLFileStore("./backup.jsonl"),
        ...     CallbackStore(send_to_analytics),
        ... ])
        >>> await store.record(eval_record)  # Writes to both stores
    """

    def __init__(self, stores: list[EvalStore]):
        """Initialize multi-store backend.

        Args:
            stores: List of storage backends to write to
        """
        self.stores = stores

    async def record(self, record: EvalRecord) -> None:
        """Write record to all configured stores in parallel.

        Args:
            record: The evaluation record to store
        """
        if not self.stores:
            return

        # Write to all stores in parallel
        async def write_to_store(store: EvalStore) -> None:
            try:
                await store.record(record)
            except Exception as e:
                # Log but don't raise - individual store failures shouldn't break others
                logger.error(f"MultiStore: Failed to write to store {store}: {e}", exc_info=True)

        await asyncio.gather(*[write_to_store(store) for store in self.stores], return_exceptions=True)

    async def close(self) -> None:
        """Close all configured stores."""

        async def close_store(store: EvalStore) -> None:
            try:
                await store.close()
            except Exception as e:
                logger.error(f"MultiStore: Failed to close store {store}: {e}", exc_info=True)

        await asyncio.gather(*[close_store(store) for store in self.stores], return_exceptions=True)
