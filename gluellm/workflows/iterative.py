"""Iterative refinement workflow for multi-agent content improvement.

This module provides the IterativeRefinementWorkflow, which orchestrates
a producer agent and one or more critic agents in an iterative refinement loop.
Critics can execute in parallel and provide specialized feedback based on
their configured specialty and goal.
"""

import asyncio
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import CriticConfig, IterativeConfig
from gluellm.workflows._base import Workflow, WorkflowResult

# Rebuild CriticConfig model now that Executor is imported
CriticConfig.model_rebuild()


class IterativeRefinementWorkflow(Workflow):
    """Workflow for iterative refinement with producer and critic agents.

    This workflow orchestrates a producer agent that creates/refines content
    and one or more critic agents that provide feedback. The workflow iterates
    until convergence criteria are met or max iterations are reached.

    Supports both single and multiple critics. Multiple critics execute in
    parallel, each providing specialized feedback based on their configured
    specialty and goal.

    Attributes:
        producer: The executor for the producer agent
        critics: List of critic configurations (can be single or multiple)
        config: Configuration for the iterative refinement process

    Example:
        >>> from gluellm.workflows.iterative import IterativeRefinementWorkflow, IterativeConfig, CriticConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = IterativeRefinementWorkflow(
        ...     producer=AgentExecutor(writer_agent),
        ...     critics=[
        ...         CriticConfig(
        ...             executor=AgentExecutor(grammar_critic),
        ...             specialty="grammar",
        ...             goal="Optimize for correctness"
        ...         ),
        ...         CriticConfig(
        ...             executor=AgentExecutor(style_critic),
        ...             specialty="style",
        ...             goal="Optimize for engagement"
        ...         ),
        ...     ],
        ...     config=IterativeConfig(max_iterations=3)
        ... )
        >>>
        >>> result = await workflow.execute("Write an article about Python")
    """

    def __init__(
        self,
        producer: Executor,
        critics: CriticConfig | list[CriticConfig],
        config: IterativeConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize an IterativeRefinementWorkflow.

        Args:
            producer: The executor for the producer agent
            critics: Single CriticConfig or list of CriticConfig instances
            config: Optional configuration for iterative refinement
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.producer = producer

        # Normalize critics to always be a list
        if isinstance(critics, CriticConfig):
            self.critics = [critics]
        else:
            self.critics = critics

        self.config = config or IterativeConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute iterative refinement workflow.

        Args:
            initial_input: The initial input/query for the producer
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        current_output = None
        critique = None

        for iteration in range(self.config.max_iterations):
            # Producer creates/refines content
            if iteration == 0:
                producer_input = initial_input
            else:
                # Build feedback summary for producer
                feedback_summary = self._format_feedback(critique)
                producer_input = f"""{initial_input}

Previous attempt:
{current_output}

Feedback from critics:
{feedback_summary}

Please revise the content based on this feedback."""

            # Execute producer
            current_output = await self.producer.execute(producer_input)
            interactions.append(
                {
                    "iteration": iteration + 1,
                    "agent": "producer",
                    "input": producer_input,
                    "output": current_output.final_response,
                }
            )

            # Execute critics in parallel
            critique = await self._execute_critics_parallel(current_output.final_response, iteration + 1, interactions)

            # Check convergence criteria
            should_stop = False
            if self.config.quality_evaluator and self.config.min_quality_score is not None:
                try:
                    score = self.config.quality_evaluator(current_output.final_response, critique)
                    if score >= self.config.min_quality_score:
                        should_stop = True
                except Exception:
                    # If evaluator fails, continue iteration
                    pass

            if should_stop:
                break

        return WorkflowResult(
            final_output=current_output.final_response or "",
            iterations=iteration + 1,
            agent_interactions=interactions,
            metadata={
                "converged": iteration < self.config.max_iterations - 1,
                "num_critics": len(self.critics),
                "config": {
                    "max_iterations": self.config.max_iterations,
                    "min_quality_score": self.config.min_quality_score,
                },
            },
        )

    async def _execute_critics_parallel(
        self, content: str, iteration: int, interactions: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Execute all critics in parallel and return their feedback.

        Args:
            content: The content to be reviewed
            iteration: Current iteration number
            interactions: List to append interaction records to

        Returns:
            Dictionary mapping critic specialty to feedback
        """
        # Create tasks for all critics
        critic_tasks = []
        for critic_config in self.critics:
            prompt = self._build_critic_prompt(critic_config, content)
            task = self._execute_critic(critic_config, prompt, iteration)
            critic_tasks.append((critic_config, task))

        # Execute all critics in parallel
        results = await asyncio.gather(*[task for _, task in critic_tasks], return_exceptions=True)

        # Process results and build feedback dictionary
        feedback_dict = {}
        for (critic_config, _), result in zip(critic_tasks, results, strict=False):
            feedback = f"Error: {type(result).__name__}: {str(result)}" if isinstance(result, Exception) else result

            feedback_dict[critic_config.specialty] = feedback

            # Record interaction
            interactions.append(
                {
                    "iteration": iteration,
                    "agent": f"critic_{critic_config.specialty}",
                    "specialty": critic_config.specialty,
                    "goal": critic_config.goal,
                    "input": self._build_critic_prompt(critic_config, content),
                    "output": feedback,
                }
            )

        return feedback_dict

    async def _execute_critic(self, critic_config: CriticConfig, prompt: str, iteration: int) -> str:
        """Execute a single critic.

        Args:
            critic_config: Configuration for the critic
            prompt: The prompt to send to the critic
            iteration: Current iteration number

        Returns:
            The critic's feedback
        """
        return await critic_config.executor.execute(prompt)

    def _build_critic_prompt(self, critic_config: CriticConfig, content: str) -> str:
        """Build a specialized prompt for a critic.

        Args:
            critic_config: Configuration for the critic
            content: The content to review

        Returns:
            Formatted prompt for the critic
        """
        return f"""You are a {critic_config.specialty} critic.

Goal: {critic_config.goal}

Review this content:
{content}

Provide specific feedback focused on {critic_config.specialty}. Be constructive and actionable."""

    def _format_feedback(self, feedback_dict: dict[str, str] | None) -> str:
        """Format feedback from multiple critics into a single string.

        Args:
            feedback_dict: Dictionary mapping specialty to feedback

        Returns:
            Formatted feedback string with headers
        """
        if not feedback_dict:
            return "No feedback available."

        feedback_parts = []
        for specialty, feedback in feedback_dict.items():
            feedback_parts.append(f"=== {specialty.title()} Critic ===\n{feedback}\n")

        return "\n".join(feedback_parts)

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.critics:
            return False
        return not self.config.max_iterations <= 0
