"""Debate workflow for multi-agent debates.

This module provides the DebateWorkflow, which orchestrates multiple agents
debating a topic with optional judge agent for final decision.
"""

from typing import Any

from pydantic import BaseModel, Field

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.workflows._base import Workflow, WorkflowResult


class DebateConfig(BaseModel):
    """Configuration for debate workflow.

    Attributes:
        max_rounds: Maximum number of debate rounds
        require_consensus: Whether consensus is required to end debate
        judge_decides: Whether judge should make final decision
    """

    max_rounds: int = Field(default=3, description="Maximum number of debate rounds", gt=0)
    require_consensus: bool = Field(default=False, description="Whether consensus is required to end debate")
    judge_decides: bool = Field(default=True, description="Whether judge should make final decision")


class DebateWorkflow(Workflow):
    """Workflow for multi-agent debates with optional judge.

    This workflow orchestrates multiple agents debating a topic. Each
    participant provides arguments, and optionally a judge agent makes
    a final decision.

    Attributes:
        participants: List of (participant_name, executor) tuples
        judge: Optional judge executor for final decision
        config: Configuration for the debate

    Example:
        >>> from gluellm.workflows.debate import DebateWorkflow, DebateConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = DebateWorkflow(
        ...     participants=[
        ...         ("Pro", AgentExecutor(pro_agent)),
        ...         ("Con", AgentExecutor(con_agent)),
        ...     ],
        ...     judge=AgentExecutor(judge_agent),
        ...     config=DebateConfig(max_rounds=3)
        ... )
        >>>
        >>> result = await workflow.execute("Should AI be regulated?")
    """

    def __init__(
        self,
        participants: list[tuple[str, Executor]],
        judge: Executor | None = None,
        config: DebateConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a DebateWorkflow.

        Args:
            participants: List of (participant_name, executor) tuples
            judge: Optional judge executor for final decision
            config: Optional configuration for the debate
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.participants = participants
        self.judge = judge
        self.config = config or DebateConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute debate workflow.

        Args:
            initial_input: The debate topic/question
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        debate_history = []

        for round_num in range(self.config.max_rounds):
            for participant_name, executor in self.participants:
                # Build context of previous arguments
                debate_context = "\n\n".join([f"{name}: {arg}" for name, arg in debate_history])

                prompt = f"Topic: {initial_input}\n\n"
                if debate_context:
                    prompt += f"Previous arguments:\n{debate_context}\n\n"
                prompt += "Your argument:"

                response = await executor.execute(prompt)
                debate_history.append((participant_name, response))
                interactions.append(
                    {
                        "round": round_num + 1,
                        "participant": participant_name,
                        "argument": response,
                    }
                )

        # Judge makes final decision if configured
        final_output = None
        if self.judge and self.config.judge_decides:
            judge_input = (
                f"Topic: {initial_input}\n\nDebate:\n"
                + "\n\n".join([f"{name}: {arg}" for name, arg in debate_history])
                + "\n\nProvide your final judgment:"
            )

            final_output = await self.judge.execute(judge_input)
            interactions.append(
                {
                    "stage": "judgment",
                    "output": final_output,
                }
            )
        else:
            final_output = "\n\n".join([f"{name}: {arg}" for name, arg in debate_history])

        return WorkflowResult(
            final_output=final_output,
            iterations=self.config.max_rounds,
            agent_interactions=interactions,
            metadata={
                "participants": [name for name, _ in self.participants],
                "judge_used": self.judge is not None and self.config.judge_decides,
            },
        )

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.participants) >= 2 and self.config.max_rounds > 0
