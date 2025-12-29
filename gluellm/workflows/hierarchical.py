"""Hierarchical task workflow for manager-worker pattern.

This module provides the HierarchicalWorkflow, which enables a manager agent
to break down tasks, worker agents to execute subtasks, and the manager to
synthesize results.
"""

import asyncio
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import HierarchicalConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class HierarchicalWorkflow(Workflow):
    """Workflow for hierarchical task decomposition and execution.

    This workflow orchestrates a manager agent that breaks down tasks into
    subtasks, worker agents that execute those subtasks, and the manager
    synthesizing the results.

    Attributes:
        manager: The executor for the manager agent
        workers: List of (worker_name, executor) tuples
        config: Configuration for the hierarchical process

    Example:
        >>> from gluellm.workflows.hierarchical import HierarchicalWorkflow, HierarchicalConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = HierarchicalWorkflow(
        ...     manager=AgentExecutor(manager_agent),
        ...     workers=[
        ...         ("Worker1", AgentExecutor(worker1)),
        ...         ("Worker2", AgentExecutor(worker2)),
        ...     ],
        ...     config=HierarchicalConfig(max_subtasks=5, parallel_workers=True)
        ... )
        >>>
        >>> result = await workflow.execute("Research and write a report on AI")
    """

    def __init__(
        self,
        manager: Executor,
        workers: list[tuple[str, Executor]],
        config: HierarchicalConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a HierarchicalWorkflow.

        Args:
            manager: The executor for the manager agent
            workers: List of (worker_name, executor) tuples
            config: Optional configuration for hierarchical process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.manager = manager
        self.workers = workers
        self.config = config or HierarchicalConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute hierarchical workflow.

        Args:
            initial_input: The initial task/problem
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []

        # Manager breaks down task
        decomposition_prompt = self._build_decomposition_prompt(initial_input)
        subtasks_text = await self.manager.execute(decomposition_prompt)
        interactions.append(
            {
                "stage": "decomposition",
                "agent": "manager",
                "input": decomposition_prompt,
                "output": subtasks_text,
            }
        )

        # Parse subtasks (simplified - assumes numbered list)
        subtasks = self._parse_subtasks(subtasks_text)

        # Limit subtasks
        subtasks = subtasks[: self.config.max_subtasks]

        # Workers execute subtasks
        worker_results = []
        if self.config.parallel_workers:
            # Execute in parallel
            worker_tasks = []
            for i, subtask in enumerate(subtasks):
                worker_name, worker_executor = self.workers[i % len(self.workers)]
                prompt = self._build_worker_prompt(subtask, initial_input)
                task = self._execute_worker(worker_executor, prompt, worker_name, subtask)
                worker_tasks.append((worker_name, subtask, task))

            results = await asyncio.gather(*[task for _, _, task in worker_tasks], return_exceptions=True)

            for (worker_name, subtask, _), result in zip(worker_tasks, results, strict=False):
                worker_result = (
                    f"Error: {type(result).__name__}: {str(result)}" if isinstance(result, Exception) else result
                )
                worker_results.append((worker_name, subtask, worker_result))
                interactions.append(
                    {
                        "stage": "execution",
                        "agent": worker_name,
                        "subtask": subtask,
                        "input": self._build_worker_prompt(subtask, initial_input),
                        "output": worker_result,
                    }
                )
        else:
            # Execute sequentially
            for i, subtask in enumerate(subtasks):
                worker_name, worker_executor = self.workers[i % len(self.workers)]
                prompt = self._build_worker_prompt(subtask, initial_input)
                worker_result = await worker_executor.execute(prompt)
                worker_results.append((worker_name, subtask, worker_result))
                interactions.append(
                    {
                        "stage": "execution",
                        "agent": worker_name,
                        "subtask": subtask,
                        "input": prompt,
                        "output": worker_result,
                    }
                )

        # Manager synthesizes results
        synthesis_prompt = self._build_synthesis_prompt(initial_input, subtasks, worker_results)
        final_output = await self.manager.execute(synthesis_prompt)
        interactions.append(
            {
                "stage": "synthesis",
                "agent": "manager",
                "input": synthesis_prompt,
                "output": final_output,
            }
        )

        return WorkflowResult(
            final_output=final_output,
            iterations=len(subtasks),
            agent_interactions=interactions,
            metadata={
                "subtasks_created": len(subtasks),
                "workers_used": len({name for name, _, _ in worker_results}),
                "parallel_execution": self.config.parallel_workers,
                "synthesis_strategy": self.config.synthesis_strategy,
            },
        )

    def _build_decomposition_prompt(self, task: str) -> str:
        """Build a prompt for task decomposition.

        Args:
            task: The original task

        Returns:
            Formatted decomposition prompt
        """
        return f"""Break down the following task into {self.config.max_subtasks} or fewer subtasks.
Each subtask should be specific, actionable, and independent.

Task: {task}

Provide a numbered list of subtasks. Each subtask should be clear and focused."""

    def _parse_subtasks(self, subtasks_text: str) -> list[str]:
        """Parse subtasks from text.

        Args:
            subtasks_text: Text containing subtasks

        Returns:
            List of subtask strings
        """
        # Simple parsing - look for numbered items
        import re

        # Match patterns like "1. Task", "1) Task", "- Task"
        patterns = [
            r"\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)",
            r"\d+\)\s*(.+?)(?=\n\d+\)|\n\n|$)",
            r"-\s*(.+?)(?=\n-|\n\n|$)",
        ]

        subtasks = []
        for pattern in patterns:
            matches = re.findall(pattern, subtasks_text, re.MULTILINE | re.DOTALL)
            if matches:
                subtasks = [m.strip() for m in matches]
                break

        # Fallback: split by newlines and filter
        if not subtasks:
            lines = [line.strip() for line in subtasks_text.split("\n") if line.strip()]
            subtasks = [line for line in lines if len(line) > 10]  # Filter short lines

        return subtasks[: self.config.max_subtasks]

    def _build_worker_prompt(self, subtask: str, original_task: str) -> str:
        """Build a prompt for a worker.

        Args:
            subtask: The subtask to execute
            original_task: The original task context

        Returns:
            Formatted worker prompt
        """
        return f"""Original task: {original_task}

Your subtask: {subtask}

Execute this subtask and provide a detailed result. Be thorough and specific."""

    async def _execute_worker(self, executor: Executor, prompt: str, worker_name: str, subtask: str) -> str:
        """Execute a worker task.

        Args:
            executor: The worker executor
            prompt: The worker prompt
            worker_name: Name of the worker
            subtask: The subtask being executed

        Returns:
            Worker result
        """
        return await executor.execute(prompt)

    def _build_synthesis_prompt(
        self, original_task: str, subtasks: list[str], worker_results: list[tuple[str, str, str]]
    ) -> str:
        """Build a synthesis prompt.

        Args:
            original_task: Original task
            subtasks: List of subtasks
            worker_results: List of (worker_name, subtask, result) tuples

        Returns:
            Formatted synthesis prompt
        """
        results_text = ""
        for worker_name, subtask, result in worker_results:
            results_text += f"\n\nSubtask: {subtask}\nWorker: {worker_name}\nResult: {result}"

        if self.config.synthesis_strategy == "concatenate":
            instruction = "Combine all results into a single document."
        elif self.config.synthesis_strategy == "merge":
            instruction = "Merge and integrate all results into a cohesive whole."
        else:  # summarize
            instruction = "Synthesize and summarize all results into a comprehensive final output."

        return f"""Original task: {original_task}

Subtasks executed:
{chr(10).join(f"{i + 1}. {st}" for i, st in enumerate(subtasks))}

Worker results:
{results_text}

{instruction}
Create a final output that addresses the original task using all the worker results."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.workers) > 0 and self.config.max_subtasks > 0
