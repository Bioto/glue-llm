"""Unified registry for local callables and remote tool sources."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from gluellm.tools.protocol import ToolSource

logger = logging.getLogger(__name__)


def _routing_stub(name: str, description: str) -> Callable[..., str]:
    """Lightweight callable used only for dynamic tool routing metadata."""

    def _stub() -> str:
        return ""

    _stub.__name__ = name
    _stub.__doc__ = description
    _stub._gluellm_registry_stub = True  # type: ignore[attr-defined]
    return _stub


class ToolRegistry:
    """Merge local Python callables with remote ``ToolSource`` providers."""

    def __init__(
        self,
        callables: list[Callable[..., Any]] | None = None,
        sources: list[ToolSource] | None = None,
    ) -> None:
        self._callables = list(callables or [])
        self._sources = list(sources or [])
        self._schemas: dict[str, dict[str, Any]] = {}
        self._source_for_name: dict[str, ToolSource] = {}
        self._refresh_from_callables()
        for source in self._sources:
            self._ingest_source_schemas(source)

    def _ingest_source_schemas(self, source: ToolSource) -> None:
        exported = getattr(source, "exported_schemas", None)
        if exported is None:
            return
        for name, schema in exported().items():
            self._schemas[name] = schema
            self._source_for_name[name] = source

    def add_callable(self, fn: Callable[..., Any]) -> None:
        self._callables.append(fn)
        self._refresh_from_callables()

    def add_source(self, source: ToolSource) -> None:
        self._sources.append(source)

    async def ensure_loaded(self) -> None:
        """Load remote tool schemas when sources exist but none are cached yet."""
        if self._schemas or not self._sources:
            return
        await self.refresh()

    async def refresh(self) -> None:
        """Reload tool schemas from all remote sources."""
        for source in self._sources:
            for name in await source.list_tools():
                schema = source.get_schema(name)
                self._schemas[name] = schema
                self._source_for_name[name] = source
        self._refresh_from_callables()

    def _refresh_from_callables(self) -> None:
        for fn in self._callables:
            name = getattr(fn, "__name__", None)
            if name:
                self._source_for_name.pop(name, None)

    def has_tool(self, name: str) -> bool:
        if any(getattr(fn, "__name__", None) == name for fn in self._callables):
            return True
        return name in self._schemas or name in self._source_for_name

    def find_callable(self, name: str) -> Callable[..., Any] | None:
        for fn in self._callables:
            if getattr(fn, "__name__", None) == name:
                return fn
        return None

    def get_openai_tool(self, name: str) -> dict[str, Any] | None:
        return self._schemas.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [dict(schema) for schema in self._schemas.values()]

    def routing_stubs(self) -> list[Callable[..., Any]]:
        stubs: list[Callable[..., Any]] = []
        for schema in self._schemas.values():
            fn = schema.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            description = fn.get("description") or name
            stubs.append(_routing_stub(name, description))
        return stubs

    def merge_for_llm(self, callables: list[Callable[..., Any]]) -> list[Callable[..., Any] | dict[str, Any]]:
        """Return callables plus remote OpenAI tool dicts for provider APIs."""
        return list(callables) + self.openai_tools()

    def expand_routed(self, matched: list[Callable[..., Any]]) -> list[Callable[..., Any] | dict[str, Any]]:
        """Replace routing stubs with OpenAI schemas after dynamic routing."""
        expanded: list[Callable[..., Any] | dict[str, Any]] = []
        for item in matched:
            if getattr(item, "_gluellm_registry_stub", False):
                schema = self.get_openai_tool(item.__name__)
                if schema is not None:
                    expanded.append(schema)
            else:
                expanded.append(item)
        return expanded

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        for fn in self._callables:
            if getattr(fn, "__name__", None) == name:
                return fn(**args)

        source = self._source_for_name.get(name)
        if source is None:
            for src in self._sources:
                if name in await src.list_tools():
                    source = src
                    self._source_for_name[name] = src
                    break
        if source is None:
            raise KeyError(f"Tool '{name}' is not registered")

        return await source.call(name, args)

    async def close(self) -> None:
        for source in self._sources:
            close = getattr(source, "close", None)
            if close is not None:
                await close()
