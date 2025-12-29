"""Base executor interface for GlueLLM.

This module defines the abstract base class for all executors,
which are responsible for executing queries using LLM agents.
"""

from abc import ABC, abstractmethod
from typing import Any

from gluellm.hooks import manager as hooks_manager
from gluellm.hooks.manager import HookManager
from gluellm.models.hook import HookRegistry, HookStage


class Executor(ABC):
    """Abstract base class for query executors.

    Executors are responsible for processing queries and returning responses.
    Subclasses must implement the _execute_internal method to define their specific
    execution strategy (e.g., simple execution, agent-based execution, etc.).

    Example:
        >>> class MyExecutor(Executor):
        ...     async def _execute_internal(self, query: str) -> str:
        ...         # Custom execution logic
        ...         return "Response"
    """

    def __init__(self, hook_registry: HookRegistry | None = None):
        """Initialize an Executor.

        Args:
            hook_registry: Optional hook registry for this executor instance
        """
        self.hook_registry = hook_registry
        self._hook_manager = HookManager()

    async def execute(self, query: str) -> str:
        """Execute a query with webhook support and return the response.

        This method wraps the internal execution with pre/post-executor webhooks.

        Args:
            query: The query string to execute

        Returns:
            str: The response from the LLM after webhook processing
        """
        # Merge global and instance hooks
        merged_registry = self._get_merged_registry()

        # Execute pre-executor hooks
        metadata: dict[str, Any] = {"executor_type": self.__class__.__name__}
        processed_query = await self._hook_manager.execute_hooks(
            query, HookStage.PRE_EXECUTOR, metadata, merged_registry.get_hooks(HookStage.PRE_EXECUTOR)
        )

        # Execute the actual query
        result = await self._execute_internal(processed_query)

        # Execute post-executor hooks
        metadata["original_query"] = query
        metadata["processed_query"] = processed_query
        return await self._hook_manager.execute_hooks(
            result, HookStage.POST_EXECUTOR, metadata, merged_registry.get_hooks(HookStage.POST_EXECUTOR)
        )

    @abstractmethod
    async def _execute_internal(self, query: str) -> str:
        """Execute a query and return the response.

        This is the internal implementation that subclasses must provide.
        This method is called by execute() after pre-executor hooks have run.

        Args:
            query: The query string to execute (may have been modified by webhooks)

        Returns:
            str: The response from the LLM

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def _get_merged_registry(self) -> HookRegistry:
        """Get merged hook registry (global + instance).

        Returns:
            Merged HookRegistry
        """
        from gluellm.models.hook import HookRegistry

        global_registry = hooks_manager.GLOBAL_HOOK_REGISTRY or HookRegistry()
        instance_registry = self.hook_registry or HookRegistry()
        return global_registry.merge(instance_registry)
