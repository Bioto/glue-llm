"""Reflection workflow for self-critique and improvement.

This module provides the ReflectionWorkflow, which enables an agent to
critique and improve its own output through iterative reflection cycles.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ReflectionConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class ReflectionWorkflow(Workflow):
    """Workflow for self-critique and improvement through reflection.

    This workflow enables an agent to generate content, then reflect on it,
    identify improvements, and revise. The process repeats until satisfaction
    or max reflections is reached.

    Attributes:
        generator: The executor for generating initial content
        reflector: The executor for reflecting on and critiquing content
        config: Configuration for the reflection process

    Example:
        >>> from gluellm.workflows.reflection import ReflectionWorkflow, ReflectionConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = ReflectionWorkflow(
        ...     generator=AgentExecutor(generator_agent),
        ...     reflector=AgentExecutor(reflector_agent),
        ...     config=ReflectionConfig(max_reflections=3)
        ... )
        >>>
        >>> result = await workflow.execute("Write an article about Python")
    """

    def __init__(
        self,
        generator: Executor,
        reflector: Executor | None = None,
        config: ReflectionConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a ReflectionWorkflow.

        Args:
            generator: The executor for generating content
            reflector: Optional executor for reflection (defaults to generator if None)
            config: Optional configuration for reflection process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.generator = generator
        self.reflector = reflector or generator
        self.config = config or ReflectionConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute reflection workflow.

        Args:
            initial_input: The initial input/query for content generation
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        current_output = None
        previous_output = None

        for reflection_num in range(self.config.max_reflections):
            # Generate or refine content
            if reflection_num == 0:
                # Initial generation
                prompt = initial_input
            else:
                # Refinement based on reflection
                reflection_prompt = self._build_reflection_prompt(previous_output, current_output)
                reflection = await self.reflector.execute(reflection_prompt)
                interactions.append(
                    {
                        "reflection": reflection_num,
                        "agent": "reflector",
                        "input": reflection_prompt,
                        "output": reflection,
                    }
                )

                # Build refinement prompt
                prompt = f"""{initial_input}

Previous version:
{previous_output}

Reflection and feedback:
{reflection}

Please revise the content based on this reflection."""

            # Generate/refine content
            current_output = await self.generator.execute(prompt)
            interactions.append(
                {
                    "reflection": reflection_num + 1,
                    "agent": "generator",
                    "input": prompt,
                    "output": current_output,
                }
            )

            # Check improvement threshold if configured
            if self.config.min_improvement_threshold is not None and previous_output is not None:
                # Simple heuristic: compare lengths (can be replaced with custom evaluator)
                improvement = abs(len(current_output) - len(previous_output)) / max(len(previous_output), 1)
                if improvement < self.config.min_improvement_threshold:
                    # Improvement is below threshold, stop
                    break

            previous_output = current_output

        return WorkflowResult(
            final_output=current_output or "",
            iterations=reflection_num + 1,
            agent_interactions=interactions,
            metadata={
                "max_reflections": self.config.max_reflections,
                "reflections_completed": reflection_num + 1,
            },
        )

    def _build_reflection_prompt(self, previous_output: str | None, current_output: str) -> str:
        """Build a prompt for reflection.

        Args:
            previous_output: The previous version of the output (if any)
            current_output: The current output to reflect on

        Returns:
            Formatted reflection prompt
        """
        if self.config.reflection_prompt_template:
            return self.config.reflection_prompt_template.format(
                previous_output=previous_output or "N/A", current_output=current_output
            )

        prompt = f"""Review and critique the following content. Identify areas for improvement,
strengths, and weaknesses. Be specific and constructive.

Content to review:
{current_output}"""

        if previous_output:
            prompt += f"""

Previous version for comparison:
{previous_output}"""

        prompt += "\n\nProvide your reflection and suggestions for improvement:"

        return prompt

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return self.config.max_reflections > 0
