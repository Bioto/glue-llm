"""Batch request and result models for GlueLLM.

This module provides models for batching multiple LLM requests
and processing them efficiently in parallel.
"""

from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class BatchErrorStrategy(str, Enum):
    """Strategy for handling errors in batch requests.

    - FAIL_FAST: Stop processing on first error and raise
    - CONTINUE: Continue processing all requests, collect errors
    - SKIP: Skip failed requests and return successful ones only
    """

    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    SKIP = "skip"


class APIKeyConfig(BaseModel):
    """Configuration for a single API key in a pool.

    Attributes:
        key: The API key value
        provider: Provider name (e.g., "openai", "anthropic", "xai")
        requests_per_minute: Optional per-key rate limit override
        burst: Optional per-key burst capacity override
    """

    key: Annotated[str, Field(description="The API key value")]
    provider: Annotated[str, Field(description="Provider name (e.g., 'openai', 'anthropic', 'xai')")]
    requests_per_minute: Annotated[
        int | None, Field(description="Optional per-key rate limit override", default=None, gt=0)
    ] = None
    burst: Annotated[int | None, Field(description="Optional per-key burst capacity override", default=None, gt=0)] = (
        None
    )


class BatchRequest(BaseModel):
    """A single request in a batch.

    Attributes:
        id: Unique identifier for this request (optional, auto-generated if not provided)
        user_message: The user's message/request
        system_prompt: Optional system prompt override
        tools: Optional tools override for this specific request
        execute_tools: Whether to execute tools for this request
        max_tool_iterations: Optional max tool iterations override
        timeout: Optional timeout override for this request
        metadata: Optional metadata to attach to this request
    """

    id: Annotated[str | None, Field(description="Unique identifier for this request", default=None)]
    user_message: Annotated[str, Field(description="The user's message/request")]
    system_prompt: Annotated[str | None, Field(description="Optional system prompt override", default=None)]
    tools: Annotated[list[Callable] | None, Field(description="Optional tools override", default=None)]
    execute_tools: Annotated[bool, Field(description="Whether to execute tools", default=True)]
    max_tool_iterations: Annotated[int | None, Field(description="Optional max tool iterations override", default=None)]
    timeout: Annotated[float | None, Field(description="Optional timeout override", default=None)]
    metadata: Annotated[dict[str, Any], Field(description="Optional metadata", default_factory=dict)]

    model_config = {"arbitrary_types_allowed": True}


class BatchResult(BaseModel):
    """Result from a single request in a batch.

    Attributes:
        id: The request ID this result corresponds to
        success: Whether the request succeeded
        response: The response text (if successful)
        tool_calls_made: Number of tool calls made (if successful)
        tool_execution_history: Tool execution history (if successful)
        tokens_used: Token usage information (if available)
        error: Error message (if failed)
        error_type: Type of error that occurred (if failed)
        metadata: Metadata from the original request
        elapsed_time: Time taken to process this request in seconds
    """

    id: Annotated[str, Field(description="The request ID")]
    success: Annotated[bool, Field(description="Whether the request succeeded")]
    response: Annotated[str | None, Field(description="The response text", default=None)]
    tool_calls_made: Annotated[int, Field(description="Number of tool calls made", default=0)]
    tool_execution_history: Annotated[
        list[dict[str, Any]], Field(description="Tool execution history", default_factory=list)
    ]
    tokens_used: Annotated[dict[str, int] | None, Field(description="Token usage information", default=None)]
    error: Annotated[str | None, Field(description="Error message", default=None)]
    error_type: Annotated[str | None, Field(description="Type of error", default=None)]
    metadata: Annotated[dict[str, Any], Field(description="Request metadata", default_factory=dict)]
    elapsed_time: Annotated[float, Field(description="Time taken to process in seconds", default=0.0)]


class BatchResponse(BaseModel):
    """Response from a batch request.

    Attributes:
        results: List of results for each request
        total_requests: Total number of requests processed
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        total_elapsed_time: Total time taken to process the batch in seconds
        total_tokens_used: Total tokens used across all requests (if available)
    """

    results: Annotated[list[BatchResult], Field(description="List of results")]
    total_requests: Annotated[int, Field(description="Total number of requests")]
    successful_requests: Annotated[int, Field(description="Number of successful requests")]
    failed_requests: Annotated[int, Field(description="Number of failed requests")]
    total_elapsed_time: Annotated[float, Field(description="Total time taken in seconds")]
    total_tokens_used: Annotated[
        dict[str, int] | None, Field(description="Total tokens used across all requests", default=None)
    ]


class BatchConfig(BaseModel):
    """Configuration for batch processing.

    Attributes:
        max_concurrent: Maximum number of concurrent requests to process
        error_strategy: How to handle errors (fail_fast, continue, skip)
        show_progress: Whether to show progress during processing
        retry_failed: Whether to retry failed requests once
        api_keys: Optional list of API keys to use for rate limit distribution
    """

    max_concurrent: Annotated[int, Field(description="Maximum concurrent requests", default=5, gt=0)]
    error_strategy: Annotated[
        BatchErrorStrategy, Field(description="Error handling strategy", default=BatchErrorStrategy.CONTINUE)
    ]
    show_progress: Annotated[bool, Field(description="Show progress during processing", default=False)]
    retry_failed: Annotated[bool, Field(description="Retry failed requests once", default=False)]
    api_keys: Annotated[
        list[APIKeyConfig] | None,
        Field(description="Optional list of API keys for rate limit distribution", default=None),
    ] = None
