"""Evaluation data storage for GlueLLM.

This module provides pluggable storage backends for capturing evaluation data
from LLM interactions. Supports both built-in file storage and custom handlers.

Example:
    >>> from gluellm import GlueLLM
    >>> from gluellm.eval import enable_file_recording
    >>>
    >>> # Enable global file recording
    >>> enable_file_recording("./eval_data/records.jsonl")
    >>>
    >>> client = GlueLLM()
    >>> result = await client.complete("Hello!")  # Automatically recorded
"""

from collections.abc import Callable
from typing import Any

from gluellm.eval.callback_store import CallbackStore
from gluellm.eval.jsonl_store import JSONLFileStore
from gluellm.eval.multi_store import MultiStore
from gluellm.eval.store import EvalStore
from gluellm.models.eval import EvalRecord

# Global store instance
_global_eval_store: EvalStore | None = None


def set_global_eval_store(store: EvalStore) -> None:
    """Set the global evaluation store.

    This store will be used by all GlueLLM instances that don't have
    an explicit eval_store parameter.

    Args:
        store: The evaluation store to use globally
    """
    global _global_eval_store
    _global_eval_store = store


def get_global_eval_store() -> EvalStore | None:
    """Get the global evaluation store.

    Returns:
        The global evaluation store, or None if not set
    """
    return _global_eval_store


def enable_file_recording(path: str | None = None) -> JSONLFileStore:
    """Enable file-based evaluation recording globally.

    Creates a JSONLFileStore and sets it as the global store.

    Args:
        path: Path to the JSONL file (defaults to logs/eval_records.jsonl)

    Returns:
        The created JSONLFileStore instance

    Example:
        >>> from gluellm.eval import enable_file_recording
        >>> store = enable_file_recording("./my_records.jsonl")
        >>> # Now all GlueLLM instances will record to this file
    """
    from pathlib import Path

    from gluellm.config import settings

    if path is None:
        # Default to logs/eval_records.jsonl
        log_dir = settings.log_dir or Path("logs")
        path = str(log_dir / "eval_records.jsonl")

    store = JSONLFileStore(path)
    set_global_eval_store(store)
    return store


def enable_callback_recording(
    callback: Callable[[EvalRecord], Any],  # type: ignore
) -> CallbackStore:
    """Enable callback-based evaluation recording globally.

    Creates a CallbackStore wrapping the provided callback and sets it as
    the global store.

    Args:
        callback: Async or sync callable that accepts an EvalRecord

    Returns:
        The created CallbackStore instance

    Example:
        >>> async def save_to_db(record: EvalRecord):
        ...     await db.insert("eval_records", record.model_dump_dict())
        >>>
        >>> from gluellm.eval import enable_callback_recording
        >>> store = enable_callback_recording(save_to_db)
    """
    store = CallbackStore(callback)
    set_global_eval_store(store)
    return store


__all__ = [
    "EvalStore",
    "EvalRecord",
    "JSONLFileStore",
    "CallbackStore",
    "MultiStore",
    "set_global_eval_store",
    "get_global_eval_store",
    "enable_file_recording",
    "enable_callback_recording",
]
