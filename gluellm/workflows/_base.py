"""Base workflow interface for GlueLLM.

This module defines the abstract base class for all workflows,
which orchestrate multi-agent interactions and complex execution patterns.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from gluellm.hooks.manager import GLOBAL_HOOK_REGISTRY, HookManager
from gluellm.models.hook import HookRegistry, HookStage
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)


class WorkflowResult(BaseModel):
    """Result from a workflow execution.

    Attributes:
        final_output: The final output from the workflow
        iterations: Number of iterations/rounds completed
        agent_interactions: Detailed history of all agent interactions
        metadata: Additional metadata about the workflow execution
        hooks_executed: Count of hooks executed during workflow
        hook_errors: List of any hook errors encountered
    """

    final_output: str = Field(description="The final output from the workflow")
    iterations: int = Field(description="Number of iterations/rounds completed")
    agent_interactions: list[dict[str, Any]] = Field(
        default_factory=list, description="Detailed history of all agent interactions"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the workflow execution"
    )
    hooks_executed: int = Field(default=0, description="Count of hooks executed during workflow")
    hook_errors: list[dict[str, Any]] = Field(default_factory=list, description="List of any hook errors encountered")


class Workflow(ABC):
    """Abstract base class for multi-agent workflows.

    Workflows orchestrate multiple agents to accomplish complex tasks
    through patterns like iterative refinement, pipelines, debates, etc.

    Subclasses must implement the _execute_internal method to define their specific
    workflow pattern.

    Example:
        >>> class MyWorkflow(Workflow):
        ...     async def _execute_internal(self, initial_input: str, context: dict | None = None) -> WorkflowResult:
        ...         # Custom workflow logic
        ...         return WorkflowResult(
        ...             final_output="Result",
        ...             iterations=1,
        ...             agent_interactions=[]
        ...         )
    """

    def __init__(self, hook_registry: HookRegistry | None = None):
        """Initialize a Workflow.

        Args:
            hook_registry: Optional hook registry for this workflow instance
        """
        self.hook_registry = hook_registry
        self._hook_manager = HookManager()

    async def execute(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute the workflow with webhook support.

        This method wraps the internal execution with pre/post-workflow webhooks.

        Args:
            initial_input: The initial input/query to process
            context: Optional context dictionary for workflow execution

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        workflow_start_time = time.time()
        workflow_type = self.__class__.__name__
        logger.info(f"Starting workflow execution: type={workflow_type}, input_length={len(initial_input)}")

        # Merge global and instance hooks
        merged_registry = self._get_merged_registry()
        pre_hooks = merged_registry.get_hooks(HookStage.PRE_WORKFLOW)
        post_hooks = merged_registry.get_hooks(HookStage.POST_WORKFLOW)
        logger.debug(f"Workflow hooks: pre={len(pre_hooks)}, post={len(post_hooks)}")

        # Execute pre-workflow hooks
        metadata: dict[str, Any] = {
            "workflow_type": workflow_type,
            "context": context or {},
        }
        processed_input = await self._hook_manager.execute_hooks(
            initial_input,
            HookStage.PRE_WORKFLOW,
            metadata,
            pre_hooks,
        )
        logger.debug(f"Pre-workflow hooks completed: input_length={len(processed_input)}")

        # Execute the actual workflow
        logger.debug(f"Executing internal workflow logic: {workflow_type}")
        result = await self._execute_internal(processed_input, context)
        logger.info(
            f"Workflow internal execution completed: iterations={result.iterations}, "
            f"output_length={len(result.final_output)}"
        )

        # Execute post-workflow hooks
        metadata["original_input"] = initial_input
        metadata["processed_input"] = processed_input
        metadata["iterations"] = result.iterations
        processed_output = await self._hook_manager.execute_hooks(
            result.final_output,
            HookStage.POST_WORKFLOW,
            metadata,
            post_hooks,
        )
        logger.debug(f"Post-workflow hooks completed: output_length={len(processed_output)}")

        # Update result with processed output and hook metadata
        result.final_output = processed_output
        # Note: hooks_executed and hook_errors would be tracked in a more
        # sophisticated implementation, but for now we'll leave them at defaults

        workflow_elapsed = time.time() - workflow_start_time
        logger.info(
            f"Workflow execution completed: type={workflow_type}, duration={workflow_elapsed:.3f}s, "
            f"iterations={result.iterations}"
        )

        return result

    @abstractmethod
    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute the workflow with initial input.

        This is the internal implementation that subclasses must provide.
        This method is called by execute() after pre-workflow hooks have run.

        Args:
            initial_input: The initial input/query to process (may have been modified by webhooks)
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

    def _get_merged_registry(self) -> HookRegistry:
        """Get merged hook registry (global + instance).

        Returns:
            Merged HookRegistry
        """
        from gluellm.models.hook import HookRegistry

        global_registry = GLOBAL_HOOK_REGISTRY or HookRegistry()
        instance_registry = self.hook_registry or HookRegistry()
        return global_registry.merge(instance_registry)
