"""Protocol definition for evaluation data storage backends."""

from typing import Protocol

from gluellm.models.eval import EvalRecord


class EvalStore(Protocol):
    """Protocol for evaluation data storage backends.

    Any class implementing this protocol can be used as an evaluation store.
    The protocol uses duck-typing, so no explicit inheritance is required.

    Example:
        >>> class MyCustomStore:
        ...     async def record(self, record: EvalRecord) -> None:
        ...         # Store the record
        ...         pass
        ...
        ...     async def close(self) -> None:
        ...         # Clean up resources
        ...         pass
        >>>
        >>> store = MyCustomStore()
        >>> client = GlueLLM(eval_store=store)
    """

    async def record(self, record: EvalRecord) -> None:
        """Store a single evaluation record.

        This method should be non-blocking and handle errors gracefully.
        Failures should be logged but not raise exceptions that would
        interrupt the main LLM request flow.

        Args:
            record: The evaluation record to store
        """
        ...

    async def close(self) -> None:
        """Clean up resources.

        Called when the store is no longer needed. Implementations should
        flush any buffered writes, close file handles, etc.
        """
        ...
