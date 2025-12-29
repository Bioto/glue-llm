"""Round-robin workflow for collaborative agent contributions.

This module provides the RoundRobinWorkflow, which enables multiple agents
to take turns contributing to a shared output.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import RoundRobinConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class RoundRobinWorkflow(Workflow):
    """Workflow for round-robin collaborative contributions.

    This workflow orchestrates multiple agents taking turns contributing
    to a shared output. Each agent builds on previous contributions.

    Attributes:
        agents: List of (agent_name, executor) tuples
        config: Configuration for the round-robin process

    Example:
        >>> from gluellm.workflows.round_robin import RoundRobinWorkflow, RoundRobinConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = RoundRobinWorkflow(
        ...     agents=[
        ...         ("Writer", AgentExecutor(writer_agent)),
        ...         ("Editor", AgentExecutor(editor_agent)),
        ...         ("Reviewer", AgentExecutor(reviewer_agent)),
        ...     ],
        ...     config=RoundRobinConfig(max_rounds=3)
        ... )
        >>>
        >>> result = await workflow.execute("Write an article about AI")
    """

    def __init__(
        self,
        agents: list[tuple[str, Executor]],
        config: RoundRobinConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a RoundRobinWorkflow.

        Args:
            agents: List of (agent_name, executor) tuples
            config: Optional configuration for round-robin process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.agents = agents
        self.config = config or RoundRobinConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute round-robin workflow.

        Args:
            initial_input: The initial input/task
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        contributions = []
        current_state = initial_input

        for round_num in range(self.config.max_rounds):
            round_contributions = []

            for agent_name, executor in self.agents:
                # Build prompt based on contribution style
                prompt = self._build_contribution_prompt(initial_input, current_state, contributions, agent_name)

                # Get contribution
                contribution = await executor.execute(prompt)
                round_contributions.append((agent_name, contribution))
                contributions.append((round_num + 1, agent_name, contribution))

                interactions.append(
                    {
                        "round": round_num + 1,
                        "agent": agent_name,
                        "input": prompt,
                        "output": contribution,
                    }
                )

                # Update current state
                current_state = self._update_state(current_state, contribution, round_num + 1)

        # Final synthesis if configured
        if self.config.final_synthesis:
            synthesis_prompt = self._build_synthesis_prompt(initial_input, contributions)
            # Use first agent for synthesis
            synthesizer = self.agents[0][1]
            final_output = await synthesizer.execute(synthesis_prompt)
            interactions.append(
                {
                    "stage": "synthesis",
                    "agent": "synthesizer",
                    "input": synthesis_prompt,
                    "output": final_output,
                }
            )
        else:
            # Concatenate all contributions
            final_output = self._format_contributions(contributions)

        return WorkflowResult(
            final_output=final_output,
            iterations=self.config.max_rounds,
            agent_interactions=interactions,
            metadata={
                "agents": [name for name, _ in self.agents],
                "contribution_style": self.config.contribution_style,
                "synthesized": self.config.final_synthesis,
            },
        )

    def _build_contribution_prompt(
        self, initial_input: str, current_state: str, contributions: list[tuple], agent_name: str
    ) -> str:
        """Build a prompt for an agent's contribution.

        Args:
            initial_input: The original input/task
            current_state: Current state of the output
            contributions: List of (round, agent, contribution) tuples
            agent_name: Name of the contributing agent

        Returns:
            Formatted contribution prompt
        """
        if self.config.contribution_style == "extend":
            instruction = "Extend and build upon the current content. Add new ideas, details, or sections."
        elif self.config.contribution_style == "refine":
            instruction = "Refine and improve the current content. Polish, clarify, and enhance what's there."
        else:  # challenge
            instruction = "Challenge and critique the current content. Identify weaknesses and suggest improvements."

        history_text = ""
        if contributions:
            history_parts = []
            for round_num, contrib_agent, contrib_text in contributions[-5:]:  # Last 5 contributions
                history_parts.append(f"Round {round_num} - {contrib_agent}: {contrib_text[:100]}...")
            history_text = "\n".join(history_parts)

        return f"""Original task: {initial_input}

Current state:
{current_state}

{instruction}

Previous contributions:
{history_text}

Provide your contribution as {agent_name}:"""

    def _update_state(self, current_state: str, contribution: str, round_num: int) -> str:
        """Update the current state with a new contribution.

        Args:
            current_state: Current state
            contribution: New contribution
            round_num: Round number

        Returns:
            Updated state
        """
        if self.config.contribution_style == "extend":
            return f"{current_state}\n\n{contribution}"
        if self.config.contribution_style == "refine":
            # For refine, replace current state (simplified)
            return contribution
        # challenge
        return f"{current_state}\n\n[Challenge]: {contribution}"

    def _format_contributions(self, contributions: list[tuple]) -> str:
        """Format all contributions into final output.

        Args:
            contributions: List of (round, agent, contribution) tuples

        Returns:
            Formatted output string
        """
        parts = []
        for round_num, agent_name, contrib in contributions:
            parts.append(f"[Round {round_num} - {agent_name}]\n{contrib}")
        return "\n\n".join(parts)

    def _build_synthesis_prompt(self, initial_input: str, contributions: list[tuple]) -> str:
        """Build a synthesis prompt.

        Args:
            initial_input: Original input/task
            contributions: All contributions

        Returns:
            Formatted synthesis prompt
        """
        contrib_text = self._format_contributions(contributions)

        return f"""Original task: {initial_input}

All contributions from the round-robin process:
{contrib_text}

Synthesize these contributions into a cohesive, final output that incorporates
the best elements from all contributions."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.agents) > 0 and self.config.max_rounds > 0
