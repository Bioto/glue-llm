"""Batch processing for GlueLLM.

This module provides functionality for processing multiple LLM requests
in parallel with configurable concurrency and error handling.
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from gluellm.api import GlueLLM
from gluellm.models.batch import (
    APIKeyConfig,
    BatchConfig,
    BatchErrorStrategy,
    BatchRequest,
    BatchResponse,
    BatchResult,
)
from gluellm.observability.logging_config import get_logger
from gluellm.rate_limiting.api_key_pool import APIKeyPool, extract_provider_from_model

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class BatchProcessor:
    """Processor for handling batched LLM requests.

    This class manages the parallel execution of multiple LLM requests
    with configurable concurrency limits and error handling strategies.

    Example:
        >>> processor = BatchProcessor(
        ...     model="openai:gpt-4o-mini",
        ...     config=BatchConfig(max_concurrent=3)
        ... )
        >>> requests = [
        ...     BatchRequest(user_message="What is 2+2?"),
        ...     BatchRequest(user_message="What is 3+3?"),
        ... ]
        >>> response = await processor.process(requests)
        >>> print(f"Processed {response.successful_requests} requests")
    """

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        max_tool_iterations: int | None = None,
        config: BatchConfig | None = None,
    ):
        """Initialize the batch processor.

        Args:
            model: Default model identifier
            system_prompt: Default system prompt
            tools: Default tools
            max_tool_iterations: Default max tool iterations
            config: Batch processing configuration
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_tool_iterations = max_tool_iterations
        self.config = config or BatchConfig()
        # Initialize API key pool if keys are provided
        self.key_pool: APIKeyPool | None = None
        if self.config.api_keys:
            from gluellm.rate_limiting.api_key_pool import APIKeyConfig as PoolAPIKeyConfig

            key_configs = [
                PoolAPIKeyConfig.from_batch_config(k) if isinstance(k, APIKeyConfig) else k
                for k in self.config.api_keys
            ]
            self.key_pool = APIKeyPool(keys=key_configs)
            logger.info(f"Initialized API key pool with {len(self.config.api_keys)} keys")

    async def process(self, requests: list[BatchRequest]) -> BatchResponse:
        """Process a batch of requests.

        Args:
            requests: List of batch requests to process

        Returns:
            BatchResponse with results for all requests

        Raises:
            Exception: If error_strategy is FAIL_FAST and a request fails
        """
        if not requests:
            logger.warning("Empty batch request received")
            return BatchResponse(
                results=[],
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_elapsed_time=0.0,
                total_tokens_used=None,
            )

        # Assign IDs to requests that don't have them
        for i, request in enumerate(requests):
            if request.id is None:
                request.id = f"batch-{uuid.uuid4()}-{i}"

        logger.info(
            f"Starting batch processing: {len(requests)} requests, "
            f"max_concurrent={self.config.max_concurrent}, "
            f"error_strategy={self.config.error_strategy.value}"
        )

        start_time = time.time()

        # Process requests with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        tasks = [self._process_single(request, semaphore) for request in requests]

        # Gather results based on error strategy
        if self.config.error_strategy == BatchErrorStrategy.FAIL_FAST:
            results = await asyncio.gather(*tasks)
        else:
            # Continue on errors, collect exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Convert exceptions to failed BatchResult objects
            results = [
                self._exception_to_result(requests[i], exc) if isinstance(exc, Exception) else exc
                for i, exc in enumerate(results)
            ]

        total_elapsed_time = time.time() - start_time

        # Filter results based on error strategy
        if self.config.error_strategy == BatchErrorStrategy.SKIP:
            results = [r for r in results if r.success]

        # Calculate statistics
        successful_requests = sum(1 for r in results if r.success)
        failed_requests = sum(1 for r in results if not r.success)

        # Aggregate token usage
        total_tokens_used = None
        if any(r.tokens_used for r in results):
            total_tokens_used = {
                "prompt": sum(r.tokens_used.get("prompt", 0) for r in results if r.tokens_used),
                "completion": sum(r.tokens_used.get("completion", 0) for r in results if r.tokens_used),
                "total": sum(r.tokens_used.get("total", 0) for r in results if r.tokens_used),
            }

        logger.info(
            f"Batch processing completed: {successful_requests}/{len(requests)} successful, "
            f"elapsed={total_elapsed_time:.3f}s"
        )

        return BatchResponse(
            results=results,
            total_requests=len(requests),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_elapsed_time=total_elapsed_time,
            total_tokens_used=total_tokens_used,
        )

    async def _process_single(self, request: BatchRequest, semaphore: asyncio.Semaphore) -> BatchResult:
        """Process a single request with concurrency control.

        Args:
            request: The batch request to process
            semaphore: Semaphore for concurrency control

        Returns:
            BatchResult for this request
        """
        async with semaphore:
            start_time = time.time()
            request_id = request.id or f"batch-{uuid.uuid4()}"

            logger.debug(f"Processing request {request_id}: {request.user_message[:50]}...")

            # Retry logic: attempt once, retry once if enabled
            max_attempts = 2 if self.config.retry_failed else 1
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    # Get API key from pool if available
                    api_key = None
                    if self.key_pool:
                        model_to_use = self.model or "openai:gpt-4o-mini"  # Default fallback
                        provider = extract_provider_from_model(model_to_use)
                        api_key = await self.key_pool.acquire_key(provider=provider, model=model_to_use)
                        if api_key:
                            logger.debug(f"Using API key from pool for request {request_id}")

                    # Create client with request-specific or default settings
                    client = GlueLLM(
                        model=self.model,
                        system_prompt=request.system_prompt or self.system_prompt,
                        tools=request.tools if request.tools is not None else self.tools,
                        max_tool_iterations=request.max_tool_iterations or self.max_tool_iterations,
                    )

                    # Execute the request
                    result = await client.complete(
                        user_message=request.user_message,
                        execute_tools=request.execute_tools,
                        correlation_id=request_id,
                        timeout=request.timeout,
                        api_key=api_key,
                    )

                    elapsed_time = time.time() - start_time

                    logger.debug(
                        f"Request {request_id} succeeded: elapsed={elapsed_time:.3f}s, tool_calls={result.tool_calls_made}"
                    )

                    return BatchResult(
                        id=request_id,
                        success=True,
                        response=result.final_response,
                        tool_calls_made=result.tool_calls_made,
                        tool_execution_history=result.tool_execution_history,
                        tokens_used=result.tokens_used,
                        metadata=request.metadata,
                        elapsed_time=elapsed_time,
                    )

                except Exception as e:
                    last_exception = e
                    elapsed_time = time.time() - start_time
                    error_type = type(e).__name__
                    error_msg = str(e)

                    if attempt < max_attempts - 1:
                        logger.info(f"Request {request_id} failed (attempt {attempt + 1}/{max_attempts}), retrying...")
                    else:
                        logger.error(
                            f"Request {request_id} failed after {elapsed_time:.3f}s: {error_type}: {error_msg}"
                        )

            # All attempts failed
            elapsed_time = time.time() - start_time
            return BatchResult(
                id=request_id,
                success=False,
                error=str(last_exception) if last_exception else "Unknown error",
                error_type=type(last_exception).__name__ if last_exception else "UnknownError",
                metadata=request.metadata,
                elapsed_time=elapsed_time,
            )

    def _exception_to_result(self, request: BatchRequest, exc: Exception) -> BatchResult:
        """Convert an exception to a BatchResult.

        Args:
            request: The original request
            exc: The exception that occurred

        Returns:
            BatchResult representing the failure
        """
        return BatchResult(
            id=request.id or f"batch-{uuid.uuid4()}",
            success=False,
            error=str(exc),
            error_type=type(exc).__name__,
            metadata=request.metadata,
            elapsed_time=0.0,
        )


# Convenience functions for batch processing


async def batch_complete(
    requests: list[BatchRequest],
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    max_tool_iterations: int | None = None,
    config: BatchConfig | None = None,
) -> BatchResponse:
    """Process a batch of completion requests.

    This is a convenience function for processing multiple requests in parallel.

    Args:
        requests: List of batch requests to process
        model: Default model identifier
        system_prompt: Default system prompt
        tools: Default tools
        max_tool_iterations: Default max tool iterations
        config: Batch processing configuration

    Returns:
        BatchResponse with results for all requests

    Example:
        >>> requests = [
        ...     BatchRequest(user_message="What is 2+2?"),
        ...     BatchRequest(user_message="What is the capital of France?"),
        ...     BatchRequest(user_message="Explain quantum computing"),
        ... ]
        >>> response = await batch_complete(
        ...     requests,
        ...     config=BatchConfig(max_concurrent=3)
        ... )
        >>> for result in response.results:
        ...     if result.success:
        ...         print(f"{result.id}: {result.response}")
    """
    processor = BatchProcessor(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
        config=config,
    )
    return await processor.process(requests)


async def batch_complete_simple(
    messages: list[str],
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list[Callable] | None = None,
    config: BatchConfig | None = None,
) -> list[str]:
    """Process a batch of simple text messages and return responses.

    This is a simplified version of batch_complete that takes a list of strings
    and returns a list of response strings (failed requests return error messages).

    Args:
        messages: List of user messages to process
        model: Model identifier
        system_prompt: System prompt
        tools: Tools to use
        config: Batch configuration

    Returns:
        List of response strings (one per input message)

    Example:
        >>> messages = [
        ...     "What is 2+2?",
        ...     "What is the capital of France?",
        ...     "Explain quantum computing",
        ... ]
        >>> responses = await batch_complete_simple(messages)
        >>> for msg, resp in zip(messages, responses):
        ...     print(f"Q: {msg}")
        ...     print(f"A: {resp}")
    """
    requests = [BatchRequest(user_message=msg) for msg in messages]
    response = await batch_complete(
        requests=requests,
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        config=config,
    )

    # Return responses in the same order as input
    return [result.response if result.success else f"Error: {result.error}" for result in response.results]
