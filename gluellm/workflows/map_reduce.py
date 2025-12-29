"""MapReduce workflow for parallel processing and aggregation.

This module provides the MapReduceWorkflow, which splits input into chunks,
processes them in parallel, and reduces the results.
"""

import asyncio
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import MapReduceConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class MapReduceWorkflow(Workflow):
    """Workflow for MapReduce pattern processing.

    This workflow splits input into chunks, processes each chunk in parallel
    (map phase), then aggregates the results (reduce phase).

    Attributes:
        mapper: The executor for processing chunks
        reducer: Optional executor for reducing results (defaults to mapper)
        config: Configuration for the MapReduce process

    Example:
        >>> from gluellm.workflows.map_reduce import MapReduceWorkflow, MapReduceConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = MapReduceWorkflow(
        ...     mapper=AgentExecutor(mapper_agent),
        ...     reducer=AgentExecutor(reducer_agent),
        ...     config=MapReduceConfig(chunk_size=1000, reduce_strategy="summarize")
        ... )
        >>>
        >>> result = await workflow.execute("Process this long document...")
    """

    def __init__(
        self,
        mapper: Executor,
        reducer: Executor | None = None,
        config: MapReduceConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a MapReduceWorkflow.

        Args:
            mapper: The executor for processing chunks
            reducer: Optional executor for reducing results (defaults to mapper)
            config: Optional configuration for MapReduce process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.mapper = mapper
        self.reducer = reducer or mapper
        self.config = config or MapReduceConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute MapReduce workflow.

        Args:
            initial_input: The input to process
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []

        # Split into chunks
        chunks = self._split_into_chunks(initial_input)
        interactions.append(
            {
                "stage": "chunking",
                "input": initial_input,
                "chunks_created": len(chunks),
            }
        )

        # Map phase: process chunks in parallel
        map_tasks = []
        for i, chunk in enumerate(chunks):
            map_prompt = self._build_map_prompt(chunk, i + 1, len(chunks))
            task = self._execute_map(self.mapper, map_prompt, i)
            map_tasks.append((i, chunk, task))

        # Execute map tasks
        max_parallel = self.config.max_parallel_chunks or len(chunks)
        map_results = []
        for i in range(0, len(map_tasks), max_parallel):
            batch = map_tasks[i : i + max_parallel]
            results = await asyncio.gather(*[task for _, _, task in batch], return_exceptions=True)

            for (chunk_idx, chunk, _), result in zip(batch, results, strict=False):
                map_result = (
                    f"Error: {type(result).__name__}: {str(result)}" if isinstance(result, Exception) else result
                )
                map_results.append((chunk_idx, chunk, map_result))
                interactions.append(
                    {
                        "stage": "map",
                        "chunk_index": chunk_idx,
                        "input": self._build_map_prompt(chunk, chunk_idx + 1, len(chunks)),
                        "output": map_result,
                    }
                )

        # Sort by chunk index
        map_results.sort(key=lambda x: x[0])

        # Reduce phase: aggregate results
        reduce_prompt = self._build_reduce_prompt([result for _, _, result in map_results])
        final_output = await self.reducer.execute(reduce_prompt)
        interactions.append(
            {
                "stage": "reduce",
                "agent": "reducer",
                "input": reduce_prompt,
                "output": final_output,
            }
        )

        return WorkflowResult(
            final_output=final_output,
            iterations=len(chunks),
            agent_interactions=interactions,
            metadata={
                "chunks_processed": len(chunks),
                "reduce_strategy": self.config.reduce_strategy,
                "parallel_chunks": min(max_parallel, len(chunks)),
            },
        )

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if self.config.chunk_size is None:
            # No chunking specified, return as single chunk
            return [text]

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        i = 0
        while i < len(text):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
            i += chunk_size - overlap

        return chunks if chunks else [text]

    def _build_map_prompt(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """Build a prompt for the map phase.

        Args:
            chunk: The chunk to process
            chunk_num: Chunk number (1-indexed)
            total_chunks: Total number of chunks

        Returns:
            Formatted map prompt
        """
        return f"""Process the following chunk ({chunk_num} of {total_chunks}):

{chunk}

Provide your analysis/processing of this chunk."""

    async def _execute_map(self, executor: Executor, prompt: str, chunk_idx: int) -> str:
        """Execute a map task.

        Args:
            executor: The mapper executor
            prompt: The map prompt
            chunk_idx: Chunk index

        Returns:
            Map result
        """
        return await executor.execute(prompt)

    def _build_reduce_prompt(self, map_results: list[str]) -> str:
        """Build a prompt for the reduce phase.

        Args:
            map_results: List of results from map phase

        Returns:
            Formatted reduce prompt
        """
        results_text = "\n\n".join([f"[Chunk {i + 1}]\n{result}" for i, result in enumerate(map_results)])

        if self.config.reduce_strategy == "concatenate":
            instruction = "Combine all results into a single document, maintaining the order."
        elif self.config.reduce_strategy == "hierarchical":
            instruction = "Create a hierarchical summary that organizes the results by themes or topics."
        else:  # summarize
            instruction = "Synthesize and summarize all results into a cohesive final output."

        return f"""You have processed {len(map_results)} chunks. Here are the results:

{results_text}

{instruction}
Create a final output that integrates all the chunk results."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return True  # Always valid
