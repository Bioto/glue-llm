"""
Example: Reasoning Effort and Logprobs

Demonstrates any_llm 1.11.0 parameters for:
- reasoning_effort: Control thinking depth for o3, o4-mini, Claude 3.7+ models
- logprobs: Token-level probabilities for eval and confidence scoring
"""

import asyncio

from gluellm.api import complete
from gluellm.config import settings


async def example_reasoning_effort():
    """Use reasoning_effort with thinking/reasoning models (o3, o4-mini, Claude 3.7+)."""
    print("=" * 70)
    print("Example 1: Reasoning Effort (Thinking Models)")
    print("=" * 70)

    # reasoning_effort: "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"auto"
    # Supported by o1, o3, o4-mini only. With gpt-5.4-2026-03-05 it is auto-omitted.
    result = await complete(
        user_message="What is the next number in: 2, 4, 8, 16, ?",
        model="openai:gpt-5.4-2026-03-05",  # Use "openai:o4-mini" for reasoning_effort
        reasoning_effort="high",
        system_prompt="You are a logic puzzle assistant. Think step by step.",
    )

    print(f"Response: {result.final_response}")
    print(f"Tokens used: {result.tokens_used}")
    print()


async def example_reasoning_effort_on_client():
    """Set reasoning_effort on the client for all calls."""
    print("=" * 70)
    print("Example 2: Client-Level Reasoning Effort")
    print("=" * 70)

    from gluellm.api import GlueLLM

    client = GlueLLM(
        model="openai:gpt-5.4-2026-03-05",
        reasoning_effort="medium",
        system_prompt="You are a helpful assistant.",
    )

    result = await client.complete("Explain why 2+2=4 in one sentence.")
    print(f"Response: {result.final_response}")
    print()


async def example_logprobs():
    """Enable logprobs for token-level confidence (eval, scoring)."""
    print("=" * 70)
    print("Example 3: Logprobs for Confidence Scoring")
    print("=" * 70)

    # logprobs=True returns token-level log probabilities
    # top_logprobs=N returns top N alternatives per token
    result = await complete(
        user_message="Is Paris the capital of France? Reply with only: Yes or No.",
        system_prompt="Answer briefly with Yes or No only.",
        logprobs=True,
        top_logprobs=3,
    )

    print(f"Response: {result.final_response}")
    print(f"Tokens used: {result.tokens_used}")
    print("(Logprobs are in the raw provider response; use for eval pipelines)")
    print()


async def example_default_from_config():
    """Use default_reasoning_effort from config (env: GLUELLM_DEFAULT_REASONING_EFFORT)."""
    print("=" * 70)
    print("Example 4: Config Defaults")
    print("=" * 70)

    print(f"Current default_reasoning_effort: {settings.default_reasoning_effort}")
    print(f"Current default_parallel_tool_calls: {settings.default_parallel_tool_calls}")
    print("Set GLUELLM_DEFAULT_REASONING_EFFORT or GLUELLM_DEFAULT_PARALLEL_TOOL_CALLS in .env")
    print()


async def main():
    """Run all examples."""
    print("\n🧙 Reasoning Effort and Logprobs Examples\n")

    await example_reasoning_effort()
    await example_reasoning_effort_on_client()
    await example_logprobs()
    await example_default_from_config()

    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
