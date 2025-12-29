"""Constitutional AI workflow for principle-based generation.

This module provides the ConstitutionalWorkflow, which generates content,
checks it against principles, and revises until all principles pass.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ConstitutionalConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class ConstitutionalWorkflow(Workflow):
    """Workflow for Constitutional AI pattern.

    This workflow generates content, critiques it against a set of principles,
    and revises until all principles are satisfied.

    Attributes:
        generator: The executor for generating content
        critic: The executor for critiquing against principles
        config: Configuration with principles and revision settings

    Example:
        >>> from gluellm.workflows.constitutional import ConstitutionalWorkflow, ConstitutionalConfig, Principle
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = ConstitutionalWorkflow(
        ...     generator=AgentExecutor(generator_agent),
        ...     critic=AgentExecutor(critic_agent),
        ...     config=ConstitutionalConfig(
        ...         principles=[
        ...             Principle(
        ...                 name="harmless",
        ...                 description="Content should not cause harm",
        ...                 severity="critical"
        ...             ),
        ...             Principle(
        ...                 name="helpful",
        ...                 description="Content should be helpful",
        ...                 severity="error"
        ...             ),
        ...         ],
        ...         max_revisions=3
        ...     )
        ... )
        >>>
        >>> result = await workflow.execute("Write a response about AI safety")
    """

    def __init__(
        self,
        generator: Executor,
        critic: Executor | None = None,
        config: ConstitutionalConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a ConstitutionalWorkflow.

        Args:
            generator: The executor for generating content
            critic: Optional executor for critiquing (defaults to generator if None)
            config: Configuration with principles (required)
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.generator = generator
        self.critic = critic or generator
        if config is None:
            raise ValueError("ConstitutionalConfig with principles is required")
        self.config = config

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute Constitutional AI workflow.

        Args:
            initial_input: The input/query for content generation
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        current_output = None

        for revision_num in range(self.config.max_revisions):
            # Generate or revise content
            if revision_num == 0:
                prompt = initial_input
            else:
                # Build revision prompt with previous critique
                prompt = self._build_revision_prompt(initial_input, current_output, interactions[-1])

            current_output = await self.generator.execute(prompt)
            interactions.append(
                {
                    "revision": revision_num + 1,
                    "stage": "generation",
                    "agent": "generator",
                    "input": prompt,
                    "output": current_output,
                }
            )

            # Critique against principles
            critique_prompt = self._build_critique_prompt(initial_input, current_output)
            critique_result = await self.critic.execute(critique_prompt)
            interactions.append(
                {
                    "revision": revision_num + 1,
                    "stage": "critique",
                    "agent": "critic",
                    "input": critique_prompt,
                    "output": critique_result,
                }
            )

            # Check if all principles pass
            principles_status = self._check_principles(critique_result)
            all_passed = all(
                status["passed"] for status in principles_status.values() if status["severity"] in ("error", "critical")
            )

            if all_passed or (not self.config.require_all_pass and revision_num >= 0):
                return WorkflowResult(
                    final_output=current_output,
                    iterations=revision_num + 1,
                    agent_interactions=interactions,
                    metadata={
                        "principles_passed": all_passed,
                        "principles_status": principles_status,
                        "revisions_completed": revision_num + 1,
                    },
                )

        # Max revisions reached
        return WorkflowResult(
            final_output=current_output or "",
            iterations=self.config.max_revisions,
            agent_interactions=interactions,
            metadata={
                "principles_passed": False,
                "max_revisions_reached": True,
            },
        )

    def _build_critique_prompt(self, initial_input: str, content: str) -> str:
        """Build a critique prompt.

        Args:
            initial_input: Original input
            content: Content to critique

        Returns:
            Formatted critique prompt
        """
        principles_text = "\n".join([f"- {p.name} ({p.severity}): {p.description}" for p in self.config.principles])

        return f"""Original request: {initial_input}

Generated content:
{content}

Evaluate this content against the following principles:

{principles_text}

For each principle, indicate whether the content passes (PASS) or fails (FAIL).
Provide specific feedback for any failures."""

    def _build_revision_prompt(self, initial_input: str, current_output: str, last_interaction: dict) -> str:
        """Build a revision prompt.

        Args:
            initial_input: Original input
            current_output: Current output
            last_interaction: Last interaction (contains critique)

        Returns:
            Formatted revision prompt
        """
        critique = last_interaction.get("output", "")

        return f"""Original request: {initial_input}

Previous version:
{current_output}

Critique and feedback:
{critique}

Revise the content to address all the issues identified in the critique.
Ensure all principles are satisfied."""

    def _check_principles(self, critique_result: str) -> dict[str, dict]:
        """Check which principles passed based on critique.

        Args:
            critique_result: The critique output

        Returns:
            Dictionary mapping principle name to status dict
        """
        critique_lower = critique_result.lower()
        status = {}

        for principle in self.config.principles:
            # Look for PASS/FAIL indicators
            principle_lower = principle.name.lower()
            passed = False

            # Check for explicit PASS/FAIL
            if f"{principle_lower}" in critique_lower:
                if "pass" in critique_lower or "✓" in critique_result or "✅" in critique_result:
                    passed = True
                elif "fail" in critique_lower or "✗" in critique_result or "❌" in critique_result:
                    passed = False

            status[principle.name] = {
                "passed": passed,
                "severity": principle.severity,
            }

        return status

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.config.principles) > 0 and self.config.max_revisions > 0
