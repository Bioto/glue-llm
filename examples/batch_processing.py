"""Batch processing examples for GlueLLM.

This example demonstrates how to process multiple LLM requests
efficiently using batch processing with configurable concurrency
and error handling strategies.
"""

import asyncio

from gluellm import (
    BatchConfig,
    BatchErrorStrategy,
    BatchRequest,
    batch_complete,
    batch_complete_simple,
)
from gluellm.observability.logging_config import setup_logging


async def example_1_simple_batch():
    """Example 1: Simple batch processing with list of messages."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Batch Processing")
    print("=" * 60)

    messages = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
    ]

    print(f"\nProcessing {len(messages)} messages...")

    responses = await batch_complete_simple(
        messages,
        config=BatchConfig(max_concurrent=3),
    )

    print("\nResults:")
    for i, (msg, resp) in enumerate(zip(messages, responses, strict=True), 1):
        print(f"\n{i}. Q: {msg}")
        print(f"   A: {resp[:100]}...")


async def example_2_batch_with_metadata():
    """Example 2: Batch processing with metadata and custom IDs."""
    print("\n" + "=" * 60)
    print("Example 2: Batch with Metadata")
    print("=" * 60)

    requests = [
        BatchRequest(
            id="math-1",
            user_message="Calculate 15 * 23",
            metadata={"category": "math", "priority": "high"},
        ),
        BatchRequest(
            id="geography-1",
            user_message="What is the largest ocean?",
            metadata={"category": "geography", "priority": "medium"},
        ),
        BatchRequest(
            id="history-1",
            user_message="When did World War II end?",
            metadata={"category": "history", "priority": "low"},
        ),
    ]

    print(f"\nProcessing {len(requests)} categorized requests...")

    response = await batch_complete(
        requests,
        config=BatchConfig(max_concurrent=2),
    )

    print(f"\n✓ Processed {response.successful_requests}/{response.total_requests} requests")
    print(f"⏱  Total time: {response.total_elapsed_time:.2f}s")
    if response.total_tokens_used:
        print(f"🔢 Total tokens: {response.total_tokens_used['total']}")

    print("\nResults by category:")
    for result in response.results:
        category = result.metadata.get("category", "unknown")
        priority = result.metadata.get("priority", "unknown")
        print(f"\n[{category.upper()}] ({priority} priority) - {result.id}")
        if result.success:
            print(f"  Response: {result.response[:80]}...")
            print(f"  Time: {result.elapsed_time:.2f}s")
        else:
            print(f"  ERROR: {result.error}")


async def example_3_error_handling():
    """Example 3: Different error handling strategies."""
    print("\n" + "=" * 60)
    print("Example 3: Error Handling Strategies")
    print("=" * 60)

    # Create requests with one that will likely timeout
    requests = [
        BatchRequest(
            id="req-1",
            user_message="What is 2+2?",
            timeout=30.0,
        ),
        BatchRequest(
            id="req-2",
            user_message="Explain the entire history of the universe.",
            timeout=0.001,  # Very short timeout to cause failure
        ),
        BatchRequest(
            id="req-3",
            user_message="What is the capital of Japan?",
            timeout=30.0,
        ),
    ]

    # Strategy 1: CONTINUE (process all, collect errors)
    print("\nStrategy 1: CONTINUE (process all requests)")
    response = await batch_complete(
        requests,
        config=BatchConfig(
            max_concurrent=3,
            error_strategy=BatchErrorStrategy.CONTINUE,
        ),
    )
    print(f"✓ Successful: {response.successful_requests}")
    print(f"✗ Failed: {response.failed_requests}")

    # Strategy 2: SKIP (only return successful)
    print("\nStrategy 2: SKIP (only return successful results)")
    response = await batch_complete(
        requests,
        config=BatchConfig(
            max_concurrent=3,
            error_strategy=BatchErrorStrategy.SKIP,
        ),
    )
    print(f"✓ Results returned: {len(response.results)}")
    print(f"✓ All successful: {all(r.success for r in response.results)}")


async def example_4_tools_in_batch():
    """Example 4: Batch processing with tools."""
    print("\n" + "=" * 60)
    print("Example 4: Batch with Tools")
    print("=" * 60)

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location."""
        # Mock implementation
        temperatures = {
            "tokyo": 18,
            "paris": 12,
            "new york": 5,
            "london": 8,
        }
        temp = temperatures.get(location.lower(), 20)
        return f"The weather in {location} is {temp}°{unit[0].upper()} and partly cloudy."

    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            result = eval(expression)  # Note: unsafe in production!
            return f"The result is {result}"
        except Exception as e:
            return f"Error calculating: {e}"

    requests = [
        BatchRequest(
            user_message="What's the weather in Tokyo?",
            tools=[get_weather],
        ),
        BatchRequest(
            user_message="Calculate 123 * 456",
            tools=[calculate],
        ),
        BatchRequest(
            user_message="What's the weather in Paris and how much is 50 + 50?",
            tools=[get_weather, calculate],
        ),
    ]

    print(f"\nProcessing {len(requests)} requests with tools...")

    response = await batch_complete(
        requests,
        config=BatchConfig(max_concurrent=2),
    )

    print("\nResults:")
    for i, result in enumerate(response.results, 1):
        print(f"\n{i}. Request: {requests[i - 1].user_message}")
        if result.success:
            print(f"   Tool calls made: {result.tool_calls_made}")
            print(f"   Response: {result.response}")
        else:
            print(f"   ERROR: {result.error}")


async def example_5_high_concurrency():
    """Example 5: High concurrency batch processing."""
    print("\n" + "=" * 60)
    print("Example 5: High Concurrency Processing")
    print("=" * 60)

    # Generate many requests
    num_requests = 20
    requests = [
        BatchRequest(
            id=f"req-{i}",
            user_message=f"Generate a random fact about the number {i}",
        )
        for i in range(1, num_requests + 1)
    ]

    print(f"\nProcessing {num_requests} requests with max_concurrent=10...")

    import time

    start = time.time()
    response = await batch_complete(
        requests,
        config=BatchConfig(
            max_concurrent=10,
            error_strategy=BatchErrorStrategy.CONTINUE,
        ),
    )
    elapsed = time.time() - start

    print(f"\n✓ Processed {response.successful_requests}/{response.total_requests} requests")
    print(f"⏱  Wall time: {elapsed:.2f}s")
    print(f"⏱  Total processing time: {response.total_elapsed_time:.2f}s")
    print(f"📊 Concurrency benefit: {response.total_elapsed_time / elapsed:.1f}x")
    if response.total_tokens_used:
        print(f"🔢 Total tokens: {response.total_tokens_used['total']}")


async def example_6_retry_failed():
    """Example 6: Automatic retry of failed requests."""
    print("\n" + "=" * 60)
    print("Example 6: Automatic Retry")
    print("=" * 60)

    requests = [
        BatchRequest(
            user_message="What is 2+2?",
            timeout=30.0,
        ),
        BatchRequest(
            user_message="Explain quantum physics.",
            timeout=0.01,  # Will likely fail and retry
        ),
    ]

    print("\nProcessing with retry_failed=True...")

    response = await batch_complete(
        requests,
        config=BatchConfig(
            max_concurrent=2,
            error_strategy=BatchErrorStrategy.CONTINUE,
            retry_failed=True,
        ),
    )

    print(f"\n✓ Successful: {response.successful_requests}")
    print(f"✗ Failed: {response.failed_requests}")

    for result in response.results:
        print(f"\n{result.id}:")
        print(f"  Success: {result.success}")
        print(f"  Retried: {result.metadata.get('_retried', False)}")


async def main():
    """Run all examples."""
    # Setup logging with force=True to ensure filter is applied to all loggers
    setup_logging(console_output=True, force=True)  # Enable console for demonstration

    print("\n" + "=" * 60)
    print("GlueLLM Batch Processing Examples")
    print("=" * 60)

    await example_1_simple_batch()
    await example_2_batch_with_metadata()
    await example_3_error_handling()
    await example_4_tools_in_batch()
    await example_5_high_concurrency()
    await example_6_retry_failed()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
