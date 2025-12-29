"""OpenTelemetry Tracing Example with MLflow.

This example demonstrates how to enable OpenTelemetry tracing for GlueLLM
using MLflow for observability and monitoring of LLM interactions.

Features Demonstrated:
- Configuring OpenTelemetry tracing with MLflow
- Automatic tracing of LLM calls
- Tool execution tracing
- Token usage tracking
- Viewing traces in MLflow UI

Prerequisites:
    1. Install required packages:
       pip install gluellm mlflow>=3.6.0 opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

    2. Start MLflow tracking server (in a separate terminal):
       mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

    3. Set environment variables:
       export GLUELLM_ENABLE_TRACING=true
       export GLUELLM_MLFLOW_TRACKING_URI=http://localhost:5000
       export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5000/v1/traces
       export OPENAI_API_KEY=your_key_here

Usage:
    python examples/opentelemetry_tracing.py
"""

import asyncio
import os
from datetime import datetime

from pydantic import BaseModel, Field

from gluellm.api import GlueLLM, complete
from gluellm.config import settings
from gluellm.telemetry import configure_tracing


# Tool functions for demonstration
def get_current_time() -> str:
    """Get the current time.

    Returns:
        Current time as a formatted string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


def get_weather(location: str) -> str:
    """Get weather information for a location (mock).

    Args:
        location: City name

    Returns:
        Weather description
    """
    # This is a mock function for demonstration
    return f"Weather in {location}: Sunny, 72°F"


# Structured output model
class MathResult(BaseModel):
    """Result of a mathematical calculation."""

    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Explanation of the calculation")


async def simple_completion_example():
    """Example 1: Simple completion with tracing."""
    print("\n" + "=" * 70)
    print("Example 1: Simple Completion with Tracing")
    print("=" * 70)

    result = await complete("What is the capital of France?", model="openai:gpt-4o-mini")

    print(f"\nResponse: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")


async def tool_execution_example():
    """Example 2: Completion with tool execution and tracing."""
    print("\n" + "=" * 70)
    print("Example 2: Tool Execution with Tracing")
    print("=" * 70)

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        tools=[get_current_time, calculate_sum, get_weather],
        system_prompt="You are a helpful assistant with access to various tools.",
    )

    # Example with multiple tool calls
    queries = [
        "What time is it?",
        "Calculate 123 + 456",
        "What's the weather like in San Francisco?",
    ]

    for query in queries:
        print(f"\n[Query] {query}")
        result = await client.complete(query)
        print(f"[Response] {result.final_response}")
        print(f"[Tools Used] {result.tool_calls_made}")

        if result.tool_execution_history:
            print("[Tool History]")
            for call in result.tool_execution_history:
                status = "✗ Error" if call["error"] else "✓ Success"
                print(f"  {status} - {call['tool_name']}: {call['result'][:50]}...")


async def structured_output_example():
    """Example 3: Structured output with tracing."""
    print("\n" + "=" * 70)
    print("Example 3: Structured Output with Tracing")
    print("=" * 70)

    client = GlueLLM(model="openai:gpt-4o-mini")

    result = await client.structured_complete("Calculate 15 * 8 and explain the result", response_format=MathResult)

    print(f"\nExpression: {result.expression}")
    print(f"Result: {result.result}")
    print(f"Explanation: {result.explanation}")


async def multi_turn_conversation_example():
    """Example 4: Multi-turn conversation with tracing."""
    print("\n" + "=" * 70)
    print("Example 4: Multi-Turn Conversation with Tracing")
    print("=" * 70)

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        tools=[calculate_sum],
        system_prompt="You are a math tutor. Help the student learn step by step.",
    )

    conversation = [
        "I need to add 25 and 37. Can you help?",
        "Now multiply the result by 2",
        "What would I get if I subtract 10 from that?",
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n[Turn {i}] User: {message}")
        result = await client.complete(message)
        print(f"[Turn {i}] Assistant: {result.final_response}")


async def error_handling_example():
    """Example 5: Error handling with tracing."""
    print("\n" + "=" * 70)
    print("Example 5: Error Handling with Tracing")
    print("=" * 70)

    def failing_tool() -> str:
        """A tool that always fails."""
        raise ValueError("This tool intentionally fails for demonstration")

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        tools=[failing_tool],
    )

    try:
        result = await client.complete("Use the failing_tool")
        print(f"\nResponse: {result.final_response}")

        if result.tool_execution_history:
            for call in result.tool_execution_history:
                if call["error"]:
                    print(f"[Tool Error] {call['tool_name']}: {call['result']}")
    except Exception as e:
        print(f"\n[Exception] {type(e).__name__}: {e}")


async def main():
    """Run all tracing examples."""
    print("=" * 70)
    print("GlueLLM OpenTelemetry Tracing Examples with MLflow")
    print("=" * 70)

    # Check configuration
    print("\nConfiguration:")
    print(f"  Tracing Enabled: {settings.enable_tracing}")
    print(f"  MLflow Tracking URI: {settings.mlflow_tracking_uri or 'Not set'}")
    print(f"  OTLP Endpoint: {settings.otel_exporter_endpoint or 'Not set'}")
    print(f"  Experiment Name: {settings.mlflow_experiment_name}")

    if not settings.enable_tracing:
        print("\n⚠️  WARNING: Tracing is disabled!")
        print("    Set GLUELLM_ENABLE_TRACING=true to enable tracing.")
        print("    Set GLUELLM_MLFLOW_TRACKING_URI to your MLflow server.")
        print("    Set OTEL_EXPORTER_OTLP_ENDPOINT to the OTLP endpoint.")
        print("\nRunning examples without tracing...\n")

    # Configure tracing
    configure_tracing()

    # Run examples
    try:
        await simple_completion_example()
        await tool_execution_example()
        await structured_output_example()
        await multi_turn_conversation_example()
        await error_handling_example()

        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)

        if settings.enable_tracing and settings.mlflow_tracking_uri:
            print(f"\nView traces in MLflow UI: {settings.mlflow_tracking_uri}")
            print("Navigate to the 'Traces' tab to see detailed execution traces.")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Ensure environment is configured
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  ERROR: OPENAI_API_KEY environment variable is not set!")
        print("    Please set your OpenAI API key before running this example.")
        exit(1)

    asyncio.run(main())
