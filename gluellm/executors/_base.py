"""Base executor interface for GlueLLM.

This module defines the abstract base class for all executors,
which are responsible for executing queries using LLM agents.
"""

from abc import ABC, abstractmethod


class Executor(ABC):
    """Abstract base class for query executors.

    Executors are responsible for processing queries and returning responses.
    Subclasses must implement the execute method to define their specific
    execution strategy (e.g., simple execution, agent-based execution, etc.).

    Example:
        >>> class MyExecutor(Executor):
        ...     async def execute(self, query: str) -> str:
        ...         # Custom execution logic
        ...         return "Response"
    """

    @abstractmethod
    async def execute(self, query: str) -> str:
        """Execute a query and return the response.

        Args:
            query: The query string to execute

        Returns:
            str: The response from the LLM

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
