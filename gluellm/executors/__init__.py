"""Executor implementations for query processing.

This module provides concrete implementations of the Executor interface,
including simple and agent-based execution strategies.
"""

from collections.abc import Callable
from typing import Optional, TypeVar

from pydantic import BaseModel

from gluellm.api import ExecutionResult, GlueLLM
from gluellm.config import settings
from gluellm.models.agent import Agent

from ._base import Executor

T = TypeVar("T", bound=Executor)


class SimpleExecutor(Executor):
    """Simple executor for direct LLM query processing.

    This executor provides a straightforward way to execute queries
    using the GlueLLM client with customizable configuration.

    Attributes:
        model: LLM model identifier (provider:model_name format)
        system_prompt: Optional system prompt for the LLM
        tools: Optional list of callable tools
        max_tool_iterations: Maximum tool execution iterations

    Example:
        >>> from gluellm.executors import SimpleExecutor
        >>> import asyncio
        >>>
        >>> async def main():
        ...     executor = SimpleExecutor(
        ...         system_prompt="You are a helpful assistant.",
        ...         tools=[]
        ...     )
        ...     response = await executor.execute("What is 2+2?")
        ...     print(response)
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        max_tool_iterations: int | None = None,
        hook_registry=None,
    ):
        """Initialize a SimpleExecutor.

        Args:
            model: LLM model to use (defaults to settings.default_model)
            system_prompt: System prompt for the LLM (optional)
            tools: List of callable tools (defaults to empty list)
            max_tool_iterations: Maximum tool execution iterations (optional)
            hook_registry: Optional hook registry for this executor
        """
        super().__init__(hook_registry=hook_registry)
        self.model = model or settings.default_model
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_tool_iterations = max_tool_iterations

    async def _execute_internal(self, query: str) -> ExecutionResult:
        """Execute a query using the configured LLM.

        Args:
            query: The query string to process

        Returns:
            ExecutionResult: The result of the execution
        """
        client = GlueLLM(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            max_tool_iterations=self.max_tool_iterations,
        )
        return await client.complete(query)


class AgentExecutor(Executor):
    """Executor that uses a configured Agent for query processing.

    This executor wraps an Agent instance and uses its configuration
    to execute queries. This is useful when you have pre-configured
    agents with specific capabilities.

    Attributes:
        agent: The Agent instance to use for execution

    Example:
        >>> from gluellm.executors import AgentExecutor
        >>> from gluellm.models.agent import Agent
        >>> from gluellm.models.prompt import SystemPrompt
        >>> import asyncio
        >>>
        >>> agent = Agent(
        ...     name="Assistant",
        ...     description="A helpful agent",
        ...     system_prompt=SystemPrompt(content="You are helpful."),
        ...     tools=[],
        ...     max_tool_iterations=5
        ... )
        >>>
        >>> async def main():
        ...     executor = AgentExecutor(agent=agent)
        ...     response = await executor.execute("Hello!")
        ...     print(response)
        >>>
        >>> asyncio.run(main())
    """

    def __init__(self, agent: Agent, hook_registry=None):
        """Initialize an AgentExecutor.

        Args:
            agent: The Agent instance to use for query execution
            hook_registry: Optional hook registry for this executor
        """
        super().__init__(hook_registry=hook_registry)
        self.agent = agent

    async def _execute_internal(self, query: str) -> ExecutionResult:
        """Execute a query using the agent's configuration.

        Args:
            query: The query string to process

        Returns:
            str: The LLM's final response
        """
        from gluellm.api import _current_agent

        # Set agent in context for automatic recording
        token = _current_agent.set(self.agent)
        try:
            client = GlueLLM(
                model=self.agent.model,
                system_prompt=self.agent.system_prompt.content if self.agent.system_prompt else None,
                tools=self.agent.tools,
                max_tool_iterations=self.agent.max_tool_iterations,
            )
            return await client.complete(query)
        finally:
            # Reset context variable
            _current_agent.reset(token)


class AgentStructuredExecutor(Executor):
    """Executor that uses a configured Agent for query processing and returns structured output."""

    def __init__(self, agent: Agent, response_format: type[T], hook_registry=None):
        super().__init__(hook_registry=hook_registry)
        self.agent = agent
        self.response_format = response_format

    async def _execute_internal(self, query: str) -> ExecutionResult:
        """Execute a query using the agent's configuration and return structured output."""
        from gluellm.api import _current_agent

        # Set agent in context for automatic recording
        token = _current_agent.set(self.agent)
        try:
            client = GlueLLM(
                model=self.agent.model,
                system_prompt=self.agent.system_prompt.content if self.agent.system_prompt else None,
                tools=self.agent.tools,
                max_tool_iterations=self.agent.max_tool_iterations,
            )
            return await client.structured_complete(query, self.response_format)
        finally:
            # Reset context variable
            _current_agent.reset(token)


__all__ = [
    "Executor",
    "SimpleExecutor",
    "AgentExecutor",
]


# Trigger model rebuild now that Executor is fully defined
def _trigger_model_rebuild():
    """Trigger rebuild of workflow models that use Executor."""
    try:
        from gluellm.models import workflow

        workflow._rebuild_models()
    except Exception:
        # Models might not be imported yet, which is fine
        pass


_trigger_model_rebuild()


if __name__ == "__main__":
    import asyncio

    async def main():
        """Demo script for SimpleExecutor."""
        executor = SimpleExecutor(
            system_prompt="You are a simple executor that can execute a query",
            tools=[],
        )
        print(await executor.execute("What is the weather in Tokyo?"))

    asyncio.run(main())
