"""Tool source protocol for remote and local tool providers."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolSource(Protocol):
    """A provider of tools that may live outside the local Python process."""

    async def list_tools(self) -> list[str]:
        """Return the names of tools this source exposes."""
        ...

    def get_schema(self, name: str) -> dict[str, Any]:
        """Return the OpenAI function tool schema for ``name``."""
        ...

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke ``name`` with ``args`` and return the raw result."""
        ...
