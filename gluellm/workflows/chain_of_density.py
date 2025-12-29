"""Chain of Density workflow for iteratively increasing detail.

This module provides the ChainOfDensityWorkflow, which iteratively increases
the density and detail of content through multiple passes.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ChainOfDensityConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class ChainOfDensityWorkflow(Workflow):
    """Workflow for iteratively increasing content density and detail.

    This workflow starts with sparse content and iteratively adds entities,
    details, or examples to increase density while optionally preserving length.

    Attributes:
        generator: The executor for generating/refining content
        config: Configuration for the density chain process

    Example:
        >>> from gluellm.workflows.chain_of_density import ChainOfDensityWorkflow, ChainOfDensityConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = ChainOfDensityWorkflow(
        ...     generator=AgentExecutor(generator_agent),
        ...     config=ChainOfDensityConfig(num_iterations=5, density_increment="entities")
        ... )
        >>>
        >>> result = await workflow.execute("Summarize the article")
    """

    def __init__(
        self, generator: Executor, config: ChainOfDensityConfig | None = None, hook_registry: HookRegistry | None = None
    ):
        """Initialize a ChainOfDensityWorkflow.

        Args:
            generator: The executor for generating/refining content
            config: Optional configuration for density chain process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.generator = generator
        self.config = config or ChainOfDensityConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute chain of density workflow.

        Args:
            initial_input: The initial input/query
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        current_output = None
        target_length = None

        for iteration in range(self.config.num_iterations):
            # Build prompt for this iteration
            if iteration == 0:
                # Initial generation - start sparse
                prompt = f"""{initial_input}

Generate a concise summary with minimal detail. Focus on key points only."""
            else:
                # Increase density
                density_instruction = self._get_density_instruction(iteration)
                length_instruction = (
                    f"Maintain approximately {target_length} characters in length."
                    if self.config.preserve_length and target_length
                    else "You may expand the content as needed."
                )

                prompt = f"""{initial_input}

Previous version (iteration {iteration}):
{current_output}

{density_instruction}

{length_instruction}

Generate an improved version with increased density."""

            # Generate/refine content
            current_output = await self.generator.execute(prompt)

            # Record target length from first iteration if preserving length
            if iteration == 0 and self.config.preserve_length:
                target_length = len(current_output)

            interactions.append(
                {
                    "iteration": iteration + 1,
                    "agent": "generator",
                    "input": prompt,
                    "output": current_output,
                    "length": len(current_output),
                }
            )

        return WorkflowResult(
            final_output=current_output or "",
            iterations=self.config.num_iterations,
            agent_interactions=interactions,
            metadata={
                "density_increment": self.config.density_increment,
                "preserve_length": self.config.preserve_length,
                "final_length": len(current_output) if current_output else 0,
            },
        )

    def _get_density_instruction(self, iteration: int) -> str:
        """Get the density increment instruction for this iteration.

        Args:
            iteration: Current iteration number (1-indexed)

        Returns:
            Instruction string for increasing density
        """
        if self.config.density_increment == "entities":
            return f"""Add more named entities (people, places, organizations, concepts) to iteration {iteration + 1}.
Include specific names, dates, locations, and other concrete details."""
        if self.config.density_increment == "details":
            return f"""Add more descriptive details and specifics to iteration {iteration + 1}.
Include more context, explanations, and supporting information."""
        # examples
        return f"""Add more examples, evidence, and concrete instances to iteration {iteration + 1}.
Include specific cases, data points, and illustrative examples."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return self.config.num_iterations > 0
