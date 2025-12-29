"""RAG (Retrieval-Augmented Generation) workflow.

This module provides the RAGWorkflow, which combines retrieval of relevant
context with generation of responses using that context.
"""

from collections.abc import Callable
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import RAGConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class RAGWorkflow(Workflow):
    """Workflow for Retrieval-Augmented Generation.

    This workflow retrieves relevant context/documentation and uses it to
    generate informed responses. Supports optional fact verification.

    Attributes:
        retriever: Callable that retrieves relevant chunks (query: str) -> list[dict]
        generator: The executor for generating responses
        verifier: Optional executor for verifying facts
        config: Configuration for the RAG process

    Example:
        >>> from gluellm.workflows.rag import RAGWorkflow, RAGConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> def my_retriever(query: str) -> list[dict]:
        ...     # Your retrieval logic
        ...     return [{"content": "...", "source": "doc1"}]
        >>>
        >>> workflow = RAGWorkflow(
        ...     retriever=my_retriever,
        ...     generator=AgentExecutor(generator_agent),
        ...     config=RAGConfig(max_retrieved_chunks=5)
        ... )
        >>>
        >>> result = await workflow.execute("What is Python?")
    """

    def __init__(
        self,
        retriever: Callable[[str], list[dict[str, Any]]],
        generator: Executor,
        verifier: Executor | None = None,
        config: RAGConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a RAGWorkflow.

        Args:
            retriever: Callable that retrieves relevant chunks given a query
            generator: The executor for generating responses
            verifier: Optional executor for verifying facts in generated response
            config: Optional configuration for RAG process
            hook_registry: Optional hook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.config = config or RAGConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute RAG workflow.

        Args:
            initial_input: The query/question to answer
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []

        # Retrieve relevant context
        retrieved_chunks = self.retriever(initial_input)
        interactions.append(
            {
                "stage": "retrieval",
                "agent": "retriever",
                "input": initial_input,
                "output": f"Retrieved {len(retrieved_chunks)} chunks",
                "chunks": retrieved_chunks[: self.config.max_retrieved_chunks],
            }
        )

        # Limit chunks
        retrieved_chunks = retrieved_chunks[: self.config.max_retrieved_chunks]

        # Check if we have context
        if not retrieved_chunks and not self.config.fallback_on_no_context:
            return WorkflowResult(
                final_output="No relevant context found.",
                iterations=1,
                agent_interactions=interactions,
                metadata={"retrieved_chunks": 0, "verified": False},
            )

        # Build context string
        context_parts = []
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            content = chunk.get("content", str(chunk))
            source = chunk.get("source", f"chunk_{i}")
            context_parts.append(f"[{i}] {content}")
            if self.config.include_sources:
                sources.append(source)

        context_text = "\n\n".join(context_parts)

        # Generate response with context
        if retrieved_chunks:
            generation_prompt = self._build_generation_prompt(initial_input, context_text, sources)
        else:
            # Fallback: generate without context
            generation_prompt = initial_input

        generated_response = await self.generator.execute(generation_prompt)
        interactions.append(
            {
                "stage": "generation",
                "agent": "generator",
                "input": generation_prompt,
                "output": generated_response,
            }
        )

        # Verify facts if configured
        verified = False
        if self.config.verify_facts and self.verifier and retrieved_chunks:
            verification_prompt = self._build_verification_prompt(initial_input, generated_response, context_text)
            verification_result = await self.verifier.execute(verification_prompt)
            interactions.append(
                {
                    "stage": "verification",
                    "agent": "verifier",
                    "input": verification_prompt,
                    "output": verification_result,
                }
            )
            verified = True

            # Append verification to response if needed
            if self.config.include_sources:
                generated_response += f"\n\n[Verification: {verification_result}]"

        # Add sources if configured
        if self.config.include_sources and sources:
            source_text = ", ".join(set(sources))
            generated_response += f"\n\nSources: {source_text}"

        return WorkflowResult(
            final_output=generated_response,
            iterations=1,
            agent_interactions=interactions,
            metadata={
                "retrieved_chunks": len(retrieved_chunks),
                "verified": verified,
                "sources_included": self.config.include_sources,
            },
        )

    def _build_generation_prompt(self, query: str, context: str, sources: list[str]) -> str:
        """Build the generation prompt with context.

        Args:
            query: The original query
            context: Retrieved context text
            sources: List of source identifiers

        Returns:
            Formatted generation prompt
        """
        return f"""Answer the following question using the provided context.
If the context doesn't contain enough information, say so.

Question: {query}

Context:
{context}

Answer:"""

    def _build_verification_prompt(self, query: str, response: str, context: str) -> str:
        """Build the verification prompt.

        Args:
            query: The original query
            response: The generated response
            context: The retrieved context

        Returns:
            Formatted verification prompt
        """
        return f"""Verify the factual accuracy of the following response against the provided context.

Query: {query}

Generated Response:
{response}

Context:
{context}

Review the response and identify any factual inaccuracies or unsupported claims.
Provide a verification report."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return self.config.max_retrieved_chunks > 0
