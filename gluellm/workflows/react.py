"""ReAct (Reasoning + Acting) workflow.

This module provides the ReActWorkflow, which interleaves reasoning and
action steps to solve problems using tools.
"""

import re
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ReActConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class ReActWorkflow(Workflow):
    """Workflow for ReAct (Reasoning + Acting) pattern.

    This workflow interleaves reasoning steps (thoughts) with action steps
    (tool calls), following the ReAct pattern for tool-using agents.

    Attributes:
        reasoner: The executor for reasoning and acting
        config: Configuration for the ReAct process

    Example:
        >>> from gluellm.workflows.react import ReActWorkflow, ReActConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = ReActWorkflow(
        ...     reasoner=AgentExecutor(reasoner_agent),
        ...     config=ReActConfig(max_steps=10, stop_on_final_answer=True)
        ... )
        >>>
        >>> result = await workflow.execute("What is the weather in Paris?")
    """

    def __init__(
        self, reasoner: Executor, config: ReActConfig | None = None, hook_registry: HookRegistry | None = None
    ):
        """Initialize a ReActWorkflow.

        Args:
            reasoner: The executor for reasoning and acting
            config: Optional configuration for ReAct process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.reasoner = reasoner
        self.config = config or ReActConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute ReAct workflow.

        Args:
            initial_input: The question/problem to solve
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        step_history = []

        for step_num in range(self.config.max_steps):
            # Build prompt with history
            prompt = self._build_step_prompt(initial_input, step_history, step_num + 1)

            # Execute reasoner
            response = await self.reasoner.execute(prompt)
            interactions.append(
                {
                    "step": step_num + 1,
                    "agent": "reasoner",
                    "input": prompt,
                    "output": response,
                }
            )

            # Parse response for thought/action/observation
            parsed = self._parse_response(response)
            step_history.append(parsed)

            # Check for final answer
            if self.config.stop_on_final_answer and parsed.get("final_answer"):
                return WorkflowResult(
                    final_output=parsed["final_answer"],
                    iterations=step_num + 1,
                    agent_interactions=interactions,
                    metadata={
                        "steps_taken": step_num + 1,
                        "final_answer_detected": True,
                    },
                )

            # If action was taken, simulate observation (in real implementation, execute tool)
            if parsed.get("action"):
                # Simulate observation from action
                observation = f"Action '{parsed['action']}' executed. Result: [Simulated result]"
                step_history[-1]["observation"] = observation
                interactions.append(
                    {
                        "step": step_num + 1,
                        "stage": "observation",
                        "output": observation,
                    }
                )

        # Max steps reached
        final_answer = self._extract_final_answer(step_history) or step_history[-1].get("thought", "")

        return WorkflowResult(
            final_output=final_answer,
            iterations=self.config.max_steps,
            agent_interactions=interactions,
            metadata={
                "steps_taken": self.config.max_steps,
                "final_answer_detected": False,
            },
        )

    def _build_step_prompt(self, initial_input: str, step_history: list[dict], step_num: int) -> str:
        """Build a prompt for a ReAct step.

        Args:
            initial_input: Original question/problem
            step_history: History of previous steps
            step_num: Current step number

        Returns:
            Formatted step prompt
        """
        prompt = f"""Question: {initial_input}

You can use the following format:
{self.config.thought_prefix} [your reasoning]
{self.config.action_prefix} [action to take]
{self.config.observation_prefix} [result from action]
Final Answer: [your final answer]

"""

        if step_history:
            prompt += "Previous steps:\n"
            for i, step in enumerate(step_history, 1):
                if step.get("thought"):
                    prompt += f"\nStep {i}:\n{self.config.thought_prefix} {step['thought']}\n"
                if step.get("action"):
                    prompt += f"{self.config.action_prefix} {step['action']}\n"
                if step.get("observation"):
                    prompt += f"{self.config.observation_prefix} {step['observation']}\n"

        prompt += f"\nStep {step_num}:"

        return prompt

    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse a ReAct response into components.

        Args:
            response: The response text

        Returns:
            Dictionary with thought, action, observation, final_answer keys
        """
        parsed = {}

        # Extract thought
        thought_match = re.search(
            rf"{re.escape(self.config.thought_prefix)}\s*(.+?)(?={re.escape(self.config.action_prefix)}|{re.escape(self.config.observation_prefix)}|Final Answer:|$)",
            response,
            re.DOTALL,
        )
        if thought_match:
            parsed["thought"] = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(
            rf"{re.escape(self.config.action_prefix)}\s*(.+?)(?={re.escape(self.config.observation_prefix)}|Final Answer:|$)",
            response,
            re.DOTALL,
        )
        if action_match:
            parsed["action"] = action_match.group(1).strip()

        # Extract observation
        obs_match = re.search(
            rf"{re.escape(self.config.observation_prefix)}\s*(.+?)(?={re.escape(self.config.action_prefix)}|Final Answer:|$)",
            response,
            re.DOTALL,
        )
        if obs_match:
            parsed["observation"] = obs_match.group(1).strip()

        # Extract final answer
        final_match = re.search(r"Final Answer:\s*(.+?)(?=\n|$)", response, re.DOTALL)
        if final_match:
            parsed["final_answer"] = final_match.group(1).strip()

        return parsed

    def _extract_final_answer(self, step_history: list[dict]) -> str | None:
        """Extract final answer from step history.

        Args:
            step_history: History of steps

        Returns:
            Final answer if found, None otherwise
        """
        for step in reversed(step_history):
            if step.get("final_answer"):
                return step["final_answer"]
        return None

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return self.config.max_steps > 0
