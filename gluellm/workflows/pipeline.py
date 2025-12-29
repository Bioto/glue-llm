"""Pipeline workflow for sequential agent execution.

This module provides the PipelineWorkflow, which executes agents sequentially
where the output of one agent becomes the input to the next.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.workflows._base import Workflow, WorkflowResult


class PipelineWorkflow(Workflow):
    """Workflow for sequential agent pipeline execution.

    This workflow executes agents in sequence, where the output of one
    agent becomes the input to the next agent in the pipeline.

    Attributes:
        stages: List of (stage_name, executor) tuples defining the pipeline

    Example:
        >>> from gluellm.workflows.pipeline import PipelineWorkflow
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = PipelineWorkflow(
        ...     stages=[
        ...         ("research", AgentExecutor(research_agent)),
        ...         ("write", AgentExecutor(writer_agent)),
        ...         ("edit", AgentExecutor(editor_agent)),
        ...     ]
        ... )
        >>>
        >>> result = await workflow.execute("Topic: Climate Change")
    """

    def __init__(self, stages: list[tuple[str, Executor]], hook_registry: HookRegistry | None = None):
        """Initialize a PipelineWorkflow.

        Args:
            stages: List of (stage_name, executor) tuples defining the pipeline
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.stages = stages

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute pipeline workflow.

        Args:
            initial_input: The initial input for the first stage
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        current_output = initial_input

        for stage_name, executor in self.stages:
            output = await executor.execute(current_output)
            interactions.append(
                {
                    "stage": stage_name,
                    "input": current_output,
                    "output": output,
                }
            )
            current_output = output

        return WorkflowResult(
            final_output=current_output,
            iterations=len(self.stages),
            agent_interactions=interactions,
            metadata={"stages": [name for name, _ in self.stages]},
        )

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.stages) > 0
