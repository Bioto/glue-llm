"""JSONL file storage backend for evaluation records.

This module provides a file-based storage backend that writes evaluation
records as newline-delimited JSON (JSONL) files.
"""

import asyncio
from pathlib import Path
from typing import Any

try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore

from gluellm.models.eval import EvalRecord
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)


class JSONLFileStore:
    """File-based storage backend using JSONL format.

    Writes evaluation records as newline-delimited JSON, one record per line.
    This format is efficient for streaming writes and easy to parse.

    Attributes:
        file_path: Path to the JSONL file
        _file: Async file handle (opened on first write)
        _write_lock: Lock for thread-safe writes
        _closed: Whether the store has been closed

    Example:
        >>> from gluellm.eval import JSONLFileStore
        >>> store = JSONLFileStore("./eval_records.jsonl")
        >>> await store.record(eval_record)
        >>> await store.close()
    """

    def __init__(self, file_path: str | Path):
        """Initialize JSONL file store.

        Args:
            file_path: Path to the JSONL file (will be created if it doesn't exist)

        Raises:
            ImportError: If aiofiles is not installed
        """
        if aiofiles is None:
            raise ImportError("aiofiles is required for JSONLFileStore. Install it with: pip install aiofiles")

        self.file_path = Path(file_path)
        self._file: Any = None
        self._write_lock = asyncio.Lock()
        self._closed = False

        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, record: EvalRecord) -> None:
        """Write a record to the JSONL file.

        Args:
            record: The evaluation record to store
        """
        if self._closed:
            logger.warning("Attempted to write to closed JSONLFileStore")
            return

        try:
            async with self._write_lock:
                # Open file on first write (lazy initialization)
                if self._file is None:
                    self._file = await aiofiles.open(self.file_path, mode="a", encoding="utf-8")

                # Serialize record to JSON
                json_line = record.model_dump_json()
                await self._file.write(json_line + "\n")
                await self._file.flush()  # Ensure data is written

        except Exception as e:
            # Log but don't raise - recording failures shouldn't break completions
            logger.error(f"Failed to write evaluation record to {self.file_path}: {e}", exc_info=True)

    async def close(self) -> None:
        """Close the file handle and flush any remaining writes."""
        if self._closed:
            return

        async with self._write_lock:
            if self._file is not None:
                try:
                    await self._file.flush()
                    await self._file.close()
                except Exception as e:
                    logger.error(f"Error closing JSONLFileStore file: {e}", exc_info=True)
                finally:
                    self._file = None
                    self._closed = True
