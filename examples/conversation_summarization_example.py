"""Example: Conversation Summarization

Demonstrates GlueLLM's automatic context summarization for long multi-turn
conversations. When the message count exceeds the configured threshold (see
``SummarizeContextConfig.threshold``), older messages are compressed into a
single [Conversation Summary] message while the most recent messages are kept
verbatim.

This keeps context size bounded regardless of conversation length, preventing
token bloat and context-window exhaustion in production chatbots.
"""

import asyncio

from gluellm import GlueLLM, SummarizeContextConfig


def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return str(a + b)


def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return str(a * b)


# Example 1: Basic summarization on a long conversation
async def example_basic_summarization():
    """Summarization kicks in automatically after the threshold is exceeded."""
    print("=" * 70)
    print("Example 1: Basic Summarization (enabled on client)")
    print("=" * 70)

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Keep answers brief.",
        summarize_context=SummarizeContextConfig(
            enabled=True,
            threshold=8,  # low threshold to trigger quickly
            keep_recent=4,
        ),
    )

    exchanges = [
        "My name is Alice and I'm a software engineer.",
        "I work primarily with Python and Go.",
        "I'm based in Seattle.",
        "My favourite project right now is a distributed task queue.",
        "It uses Redis as a broker.",
        "We process about 50,000 jobs per day.",
        "What do you remember about me so far?",
    ]

    for msg in exchanges:
        result = await client.complete(msg)
        print(f"User:      {msg}")
        print(f"Assistant: {result.final_response}\n")

    print(
        "Note: older turns were automatically summarized once the message\n"
        "count exceeded the threshold. The final question was answered using\n"
        "a [Conversation Summary] in place of the full history.\n"
    )


# Example 2: Per-call override of summarization parameters
async def example_per_call_override():
    """summarize_context can be overridden on each complete() call."""
    print("=" * 70)
    print("Example 2: Per-call Override")
    print("=" * 70)

    # Client has summarization off by default …
    client = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
    )

    # … but we turn it on (with a custom threshold) for a specific call.
    result = await client.complete(
        "Briefly explain what Python is.",
        summarize_context=SummarizeContextConfig(enabled=True, threshold=20, keep_recent=6),
    )

    print(f"Response: {result.final_response}\n")
    print(
        "Note: summarize_context was enabled only for this single call;\n"
        "the client default remains False.\n"
    )


# Example 3: Cheaper summarization model
async def example_custom_summarization_model():
    """Use a fast, cheap model for summarization instead of the primary model."""
    print("=" * 70)
    print("Example 3: Custom Summarization Model")
    print("=" * 70)

    client = GlueLLM(
        model="openai:gpt-4o",  # expensive primary model
        system_prompt="You are a helpful assistant. Keep answers brief.",
        summarize_context=SummarizeContextConfig(
            enabled=True,
            threshold=6,
            model="openai:gpt-4o-mini",  # cheap summarizer
        ),
    )

    for msg in [
        "Tell me a fun fact about the ocean.",
        "Tell me a fun fact about space.",
        "Tell me a fun fact about ancient history.",
        "Tell me a fun fact about mathematics.",
        "What were the four topics you just covered?",
    ]:
        result = await client.complete(msg)
        print(f"User:      {msg}")
        print(f"Assistant: {result.final_response}\n")

    print(
        "Note: the summarization LLM call used gpt-4o-mini while the main\n"
        "conversation ran on gpt-4o, keeping summarization costs low.\n"
    )


# Example 4: Summarization with tools
async def example_summarization_with_tools():
    """Summarization works transparently alongside tool use."""
    print("=" * 70)
    print("Example 4: Summarization with Tools")
    print("=" * 70)

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="You are a math assistant. Use tools for all calculations.",
        tools=[add, multiply],
        summarize_context=SummarizeContextConfig(
            enabled=True,
            threshold=8,
            keep_recent=4,
        ),
    )

    exchanges = [
        "What is 12 + 7?",
        "Now multiply that by 3.",
        "Add 100 to the previous result.",
        "Multiply the last result by 2.",
        "What was the first calculation you did, and what's the running total now?",
    ]

    for msg in exchanges:
        result = await client.complete(msg)
        print(f"User:      {msg}")
        print(f"Assistant: {result.final_response}")
        if result.tool_calls_made:
            print(f"           (used {result.tool_calls_made} tool call(s))")
        print()

    print(
        "Note: tool call results are included in the summarization transcript,\n"
        "so the model retains awareness of prior calculations.\n"
    )


async def main():
    await example_basic_summarization()
    await example_per_call_override()
    await example_custom_summarization_model()
    await example_summarization_with_tools()

    print("=" * 70)
    print("All conversation summarization examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
