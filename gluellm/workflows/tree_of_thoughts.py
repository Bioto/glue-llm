"""Tree of Thoughts workflow for exploring multiple reasoning paths.

This module provides the TreeOfThoughtsWorkflow, which explores multiple
reasoning paths in parallel, evaluates them, and selects the best.
"""

import asyncio
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import TreeOfThoughtsConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class TreeOfThoughtsWorkflow(Workflow):
    """Workflow for Tree of Thoughts pattern.

    This workflow explores multiple reasoning paths in parallel, evaluates
    them at each level, and selects the best paths to continue exploring.

    Attributes:
        thinker: The executor for generating thoughts
        evaluator: Optional executor for evaluating thoughts (defaults to thinker)
        config: Configuration for the ToT process

    Example:
        >>> from gluellm.workflows.tree_of_thoughts import TreeOfThoughtsWorkflow, TreeOfThoughtsConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = TreeOfThoughtsWorkflow(
        ...     thinker=AgentExecutor(thinker_agent),
        ...     evaluator=AgentExecutor(evaluator_agent),
        ...     config=TreeOfThoughtsConfig(
        ...         branching_factor=3,
        ...         max_depth=3,
        ...         evaluation_strategy="score"
        ...     )
        ... )
        >>>
        >>> result = await workflow.execute("Solve this puzzle: ...")
    """

    def __init__(
        self,
        thinker: Executor,
        evaluator: Executor | None = None,
        config: TreeOfThoughtsConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a TreeOfThoughtsWorkflow.

        Args:
            thinker: The executor for generating thoughts
            evaluator: Optional executor for evaluating thoughts (defaults to thinker)
            config: Optional configuration for ToT process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.thinker = thinker
        self.evaluator = evaluator or thinker
        self.config = config or TreeOfThoughtsConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute Tree of Thoughts workflow.

        Args:
            initial_input: The problem/question to solve
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        tree = {}  # depth -> list of (thought, score, parent) tuples
        current_level = [("", initial_input, None)]  # (thought, state, parent)

        for depth in range(self.config.max_depth):
            # Generate thoughts for current level
            thought_tasks = []
            for thought, state, parent in current_level:
                prompt = self._build_thought_prompt(initial_input, state, depth + 1, thought)
                task = self._generate_thoughts(self.thinker, prompt, thought, state, parent)
                thought_tasks.append(task)

            # Execute thought generation in parallel
            results = await asyncio.gather(*thought_tasks, return_exceptions=True)

            # Collect all thoughts
            all_thoughts = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                all_thoughts.extend(result)

            # Limit to branching factor
            all_thoughts = all_thoughts[: self.config.branching_factor]

            # Evaluate thoughts
            evaluation_tasks = []
            for thought_text, state, parent in all_thoughts:
                eval_prompt = self._build_evaluation_prompt(initial_input, thought_text, state, depth + 1)
                task = self._evaluate_thought(self.evaluator, eval_prompt, thought_text, state, parent)
                evaluation_tasks.append(task)

            evaluations = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

            # Combine thoughts with scores
            scored_thoughts = []
            for (thought_text, state, parent), score in zip(all_thoughts, evaluations, strict=False):
                if isinstance(score, Exception):
                    score = 0.0
                scored_thoughts.append((thought_text, state, parent, score))

            # Prune low-scoring thoughts
            threshold = self.config.prune_threshold
            scored_thoughts = [t for t in scored_thoughts if t[3] >= threshold]

            # Sort by score
            scored_thoughts.sort(key=lambda x: x[3], reverse=True)

            # Keep top thoughts for next level
            current_level = [(t[0], t[1], t[2]) for t in scored_thoughts[: self.config.branching_factor]]

            # Store in tree
            tree[depth + 1] = scored_thoughts

            interactions.append(
                {
                    "depth": depth + 1,
                    "thoughts_generated": len(all_thoughts),
                    "thoughts_after_pruning": len(scored_thoughts),
                    "top_thoughts": [{"thought": t[0][:100], "score": t[3]} for t in scored_thoughts[:3]],
                }
            )

            # Check if we have a final answer
            if depth + 1 >= self.config.max_depth:
                break

        # Select best path
        best_thought = self._select_best_thought(tree)
        final_output = best_thought[1] if best_thought else initial_input

        return WorkflowResult(
            final_output=final_output,
            iterations=self.config.max_depth,
            agent_interactions=interactions,
            metadata={
                "max_depth": self.config.max_depth,
                "branching_factor": self.config.branching_factor,
                "evaluation_strategy": self.config.evaluation_strategy,
            },
        )

    def _build_thought_prompt(self, initial_input: str, current_state: str, depth: int, parent_thought: str) -> str:
        """Build a prompt for generating thoughts.

        Args:
            initial_input: Original problem
            current_state: Current state/context
            depth: Current depth
            parent_thought: Parent thought (if any)

        Returns:
            Formatted thought prompt
        """
        prompt = f"""Problem: {initial_input}

Current state: {current_state}

"""
        if parent_thought:
            prompt += f"Previous thought: {parent_thought}\n\n"

        prompt += f"""Generate {self.config.branching_factor} different reasoning steps or approaches
to progress toward solving this problem. Be creative and explore different angles."""

        return prompt

    async def _generate_thoughts(
        self, executor: Executor, prompt: str, parent_thought: str, state: str, parent: Any
    ) -> list[tuple[str, str, Any]]:
        """Generate multiple thoughts.

        Args:
            executor: The thinker executor
            prompt: The thought prompt
            parent_thought: Parent thought
            state: Current state
            parent: Parent node

        Returns:
            List of (thought_text, new_state, parent) tuples
        """
        response = await executor.execute(prompt)

        # Parse multiple thoughts (simplified - split by numbering or bullets)
        import re

        # Try to extract numbered or bulleted items
        thoughts = re.split(r"\n\s*\d+[\.\)]\s*|\n\s*[-*]\s*", response)
        thoughts = [t.strip() for t in thoughts if t.strip() and len(t.strip()) > 10]

        if not thoughts:
            thoughts = [response]

        # Limit to branching factor
        thoughts = thoughts[: self.config.branching_factor]

        return [(thought, f"{state}\n{thought}", parent) for thought in thoughts]

    def _build_evaluation_prompt(self, initial_input: str, thought: str, state: str, depth: int) -> str:
        """Build an evaluation prompt.

        Args:
            initial_input: Original problem
            thought: The thought to evaluate
            state: Current state
            depth: Current depth

        Returns:
            Formatted evaluation prompt
        """
        if self.config.evaluation_strategy == "vote":
            instruction = "Vote on whether this thought is promising (1-10 scale)."
        elif self.config.evaluation_strategy == "best_first":
            instruction = "Evaluate if this is the best next step (1-10 scale)."
        else:  # score
            instruction = "Score this thought on a scale of 0.0 to 1.0 based on how promising it is."

        return f"""Problem: {initial_input}

Current state: {state}

Thought to evaluate: {thought}

{instruction}
Provide only a numeric score."""

    async def _evaluate_thought(self, executor: Executor, prompt: str, thought: str, state: str, parent: Any) -> float:
        """Evaluate a thought.

        Args:
            executor: The evaluator executor
            prompt: The evaluation prompt
            thought: The thought being evaluated
            state: Current state
            parent: Parent node

        Returns:
            Score (0.0 to 1.0)
        """
        response = await executor.execute(prompt)

        # Extract numeric score
        import re

        numbers = re.findall(r"\d+\.?\d*", response)
        if numbers:
            score = float(numbers[0])
            # Normalize to 0-1 range
            if score > 1.0:
                score = score / 10.0 if score <= 10 else score / 100.0
            return max(0.0, min(1.0, score))

        return 0.5  # Default score

    def _select_best_thought(self, tree: dict[int, list]) -> tuple | None:
        """Select the best thought from the tree.

        Args:
            tree: Dictionary mapping depth to list of (thought, state, parent, score) tuples

        Returns:
            Best thought tuple or None
        """
        if not tree:
            return None

        # Get deepest level
        max_depth = max(tree.keys())
        deepest_thoughts = tree[max_depth]

        if not deepest_thoughts:
            return None

        # Return highest scoring thought
        return max(deepest_thoughts, key=lambda x: x[3])

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return self.config.branching_factor > 0 and self.config.max_depth > 0
