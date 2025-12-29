"""Mixture of Experts workflow for specialized agent routing.

This module provides the MixtureOfExpertsWorkflow, which routes queries to
specialized expert agents and combines their outputs.
"""

import asyncio
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ExpertConfig, MoEConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class MixtureOfExpertsWorkflow(Workflow):
    """Workflow for Mixture of Experts pattern.

    This workflow routes queries to specialized expert agents based on
    routing strategy, then combines their outputs.

    Attributes:
        experts: List of expert configurations
        router: Optional executor for routing (uses first expert if None)
        combiner: Optional executor for combining outputs (uses router if None)
        config: Configuration for the MoE process

    Example:
        >>> from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow, ExpertConfig, MoEConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = MixtureOfExpertsWorkflow(
        ...     experts=[
        ...         ExpertConfig(
        ...             executor=AgentExecutor(math_expert),
        ...             specialty="mathematics",
        ...             description="Expert in math and calculations",
        ...             activation_keywords=["calculate", "math", "equation"]
        ...         ),
        ...         ExpertConfig(
        ...             executor=AgentExecutor(code_expert),
        ...             specialty="programming",
        ...             description="Expert in coding and software",
        ...             activation_keywords=["code", "program", "algorithm"]
        ...         ),
        ...     ],
        ...     config=MoEConfig(routing_strategy="keyword", top_k=2)
        ... )
        >>>
        >>> result = await workflow.execute("Calculate the factorial of 10")
    """

    def __init__(
        self,
        experts: list[ExpertConfig],
        router: Executor | None = None,
        combiner: Executor | None = None,
        config: MoEConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a MixtureOfExpertsWorkflow.

        Args:
            experts: List of expert configurations
            router: Optional executor for routing (uses first expert if None)
            combiner: Optional executor for combining outputs (uses router if None)
            config: Optional configuration for MoE process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.experts = experts
        self.router = router or (experts[0].executor if experts else None)
        self.combiner = combiner or self.router
        self.config = config or MoEConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute Mixture of Experts workflow.

        Args:
            initial_input: The query to route to experts
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []

        # Route to experts
        selected_experts = self._route_to_experts(initial_input)
        interactions.append(
            {
                "stage": "routing",
                "input": initial_input,
                "selected_experts": [e.specialty for e in selected_experts],
                "routing_strategy": self.config.routing_strategy,
            }
        )

        # Execute selected experts in parallel
        expert_tasks = []
        for expert_config in selected_experts:
            prompt = self._build_expert_prompt(initial_input, expert_config)
            task = self._execute_expert(expert_config.executor, prompt, expert_config.specialty)
            expert_tasks.append((expert_config, task))

        results = await asyncio.gather(*[task for _, task in expert_tasks], return_exceptions=True)

        expert_outputs = []
        for (expert_config, _), result in zip(expert_tasks, results, strict=False):
            expert_output = (
                f"Error: {type(result).__name__}: {str(result)}" if isinstance(result, Exception) else result
            )
            expert_outputs.append((expert_config.specialty, expert_output))
            interactions.append(
                {
                    "stage": "expert_execution",
                    "expert": expert_config.specialty,
                    "input": self._build_expert_prompt(initial_input, expert_config),
                    "output": expert_output,
                }
            )

        # Combine expert outputs
        if len(expert_outputs) == 1:
            final_output = expert_outputs[0][1]
        else:
            combine_prompt = self._build_combine_prompt(initial_input, expert_outputs)
            final_output = await self.combiner.execute(combine_prompt)
            interactions.append(
                {
                    "stage": "combination",
                    "agent": "combiner",
                    "input": combine_prompt,
                    "output": final_output,
                }
            )

        return WorkflowResult(
            final_output=final_output,
            iterations=len(selected_experts),
            agent_interactions=interactions,
            metadata={
                "experts_used": len(selected_experts),
                "routing_strategy": self.config.routing_strategy,
                "combine_strategy": self.config.combine_strategy,
            },
        )

    def _route_to_experts(self, query: str) -> list[ExpertConfig]:
        """Route query to appropriate experts.

        Args:
            query: The query to route

        Returns:
            List of selected expert configurations
        """
        query_lower = query.lower()

        if self.config.routing_strategy == "all":
            return self.experts

        if self.config.routing_strategy == "keyword":
            # Match by keywords
            selected = []
            for expert in self.experts:
                if any(keyword.lower() in query_lower for keyword in expert.activation_keywords):
                    selected.append(expert)
            return selected if selected else self.experts[:1]

        if self.config.routing_strategy == "top_k":
            # Select top k experts (simplified - uses first k)
            return self.experts[: self.config.top_k]

        # semantic
        # Semantic routing (simplified - uses keyword matching as fallback)
        selected = []
        for expert in self.experts:
            # Check if query relates to expert's specialty or keywords
            if expert.specialty.lower() in query_lower or any(
                kw.lower() in query_lower for kw in expert.activation_keywords
            ):
                selected.append(expert)
        return selected[: self.config.top_k] if selected else self.experts[: self.config.top_k]

    def _build_expert_prompt(self, query: str, expert_config: ExpertConfig) -> str:
        """Build a prompt for an expert.

        Args:
            query: The query
            expert_config: Expert configuration

        Returns:
            Formatted expert prompt
        """
        return f"""You are an expert in {expert_config.specialty}.

{expert_config.description}

Query: {query}

Provide your expert response:"""

    async def _execute_expert(self, executor: Executor, prompt: str, specialty: str) -> str:
        """Execute an expert.

        Args:
            executor: The expert executor
            prompt: The expert prompt
            specialty: Expert specialty

        Returns:
            Expert output
        """
        return await executor.execute(prompt)

    def _build_combine_prompt(self, query: str, expert_outputs: list[tuple[str, str]]) -> str:
        """Build a prompt for combining expert outputs.

        Args:
            query: Original query
            expert_outputs: List of (specialty, output) tuples

        Returns:
            Formatted combine prompt
        """
        outputs_text = "\n\n".join([f"[{specialty} Expert]\n{output}" for specialty, output in expert_outputs])

        if self.config.combine_strategy == "concatenate":
            instruction = "Combine all expert outputs into a single document."
        elif self.config.combine_strategy == "vote":
            instruction = "Compare the expert outputs and provide a consensus or best answer."
        else:  # synthesize
            instruction = "Synthesize all expert outputs into a cohesive, comprehensive response."

        return f"""Original query: {query}

Expert outputs:
{outputs_text}

{instruction}
Create a final response that integrates insights from all experts."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.experts) > 0
