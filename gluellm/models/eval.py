"""Evaluation data models for GlueLLM.

This module provides models for capturing complete request/response lifecycle
data for LLM evaluation purposes.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalRecord(BaseModel):
    """Complete evaluation record capturing request/response lifecycle.

    This model captures all relevant data from an LLM interaction for
    evaluation, debugging, and analysis purposes.

    Attributes:
        id: Unique identifier for this record (UUID)
        correlation_id: Correlation ID for request tracking
        timestamp: When the request was made
        user_message: The user's input message
        system_prompt: System prompt used for this request
        model: Model identifier (e.g., "openai:gpt-4o-mini")
        messages_snapshot: Full conversation state at time of request
        final_response: The final text response from the model
        structured_output: Serialized structured output (if applicable)
        raw_response: Serialized raw ChatCompletion response
        tool_calls_made: Number of tool calls executed
        tool_execution_history: Complete history of tool calls and results
        tools_available: List of tool names available to the model
        latency_ms: Total request latency in milliseconds
        tokens_used: Token usage dictionary with 'prompt', 'completion', 'total'
        estimated_cost_usd: Estimated cost in USD
        success: Whether the request succeeded
        error_type: Type of error if request failed (e.g., "TokenLimitError")
        error_message: Error message if request failed
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique record identifier")
    correlation_id: str | None = Field(default=None, description="Correlation ID for request tracking")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Request timestamp")

    # Request
    user_message: str = Field(description="The user's input message")
    system_prompt: str = Field(description="System prompt used for this request")
    model: str = Field(description="Model identifier (provider:model_name)")
    messages_snapshot: list[dict[str, Any]] = Field(
        default_factory=list, description="Full conversation state at time of request"
    )

    # Response
    final_response: str = Field(default="", description="The final text response from the model")
    structured_output: Any | None = Field(default=None, description="Serialized structured output (if applicable)")
    raw_response: dict[str, Any] | None = Field(default=None, description="Serialized raw ChatCompletion response")

    # Tool execution
    tool_calls_made: int = Field(default=0, description="Number of tool calls executed")
    tool_execution_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Complete history of tool calls and results"
    )
    tools_available: list[str] = Field(default_factory=list, description="List of tool names available to the model")

    # Metrics
    latency_ms: float = Field(default=0.0, description="Total request latency in milliseconds")
    tokens_used: dict[str, int] | None = Field(
        default=None, description="Token usage dictionary with 'prompt', 'completion', 'total'"
    )
    estimated_cost_usd: float | None = Field(default=None, description="Estimated cost in USD")

    # Outcome
    success: bool = Field(default=True, description="Whether the request succeeded")
    error_type: str | None = Field(default=None, description="Type of error if request failed")
    error_message: str | None = Field(default=None, description="Error message if request failed")

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON string with proper handling of complex types."""
        # Use Pydantic's default serialization but ensure datetime is ISO format
        data = self.model_dump(mode="json", **kwargs)
        # Ensure timestamp is serialized as ISO string
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data, default=str)

    def model_dump_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with proper handling of complex types."""
        data = self.model_dump(mode="json")
        # Ensure timestamp is serialized as ISO string
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data
