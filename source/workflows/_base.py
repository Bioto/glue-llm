"""Base workflow interface for GlueLLM.

This module defines the abstract base class for all workflows,
which orchestrate multi-agent interactions and complex execution patterns.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class WorkflowResult(BaseModel):
    """Result from a workflow execution.

    Attributes:
        final_output: The final output from the workflow
        iterations: Number of iterations/rounds completed
        agent_interactions: Detailed history of all agent interactions
        metadata: Additional metadata about the workflow execution
    """

    final_output: str = Field(description="The final output from the workflow")
    iterations: int = Field(description="Number of iterations/rounds completed")
    agent_interactions: list[dict[str, Any]] = Field(
        default_factory=list, description="Detailed history of all agent interactions"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the workflow execution"
    )


class Workflow(ABC):
    """Abstract base class for multi-agent workflows.

    Workflows orchestrate multiple agents to accomplish complex tasks
    through patterns like iterative refinement, pipelines, debates, etc.

    Subclasses must implement the execute method to define their specific
    workflow pattern.

    Example:
        >>> class MyWorkflow(Workflow):
        ...     async def execute(self, initial_input: str, context: dict | None = None) -> WorkflowResult:
        ...         # Custom workflow logic
        ...         return WorkflowResult(
        ...             final_output="Result",
        ...             iterations=1,
        ...             agent_interactions=[]
        ...         )
    """

    @abstractmethod
    async def execute(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute the workflow with initial input.

        Args:
            initial_input: The initial input/query to process
            context: Optional context dictionary for workflow execution

        Returns:
            WorkflowResult: The result of the workflow execution

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            bool: True if configuration is valid, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
