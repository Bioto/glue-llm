"""
Example: Reasoning Effort, Traces, and Logprobs

Demonstrates:
- reasoning_effort: Control thinking depth for o3, o4-mini, Claude 3.7+ models
- reasoning_summary / reasoning_trace: Readable reasoning on the Responses API path
- stream_response + reasoning_chunk: Live reasoning deltas via on_status
- logprobs: Token-level probabilities for eval and confidence scoring
"""

import asyncio

from gluellm.api import GlueLLM, complete, response
from gluellm.config import settings
from gluellm.events import ProcessEvent


async def example_reasoning_effort():
    """Use reasoning_effort with thinking/reasoning models (o3, o4-mini, Claude 3.7+)."""
    print("=" * 70)
    print("Example 1: Reasoning Effort (Thinking Models)")
    print("=" * 70)

    # reasoning_effort: "none"|"minimal"|"low"|"medium"|"high"|"xhigh"
    # Use a model that supports it (e.g. o4-mini); unsupported models error at the provider.
    result = await complete(
        user_message="What is the next number in: 2, 4, 8, 16, ?",
        model="openai:o4-mini",
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

    client = GlueLLM(
        model="openai:gpt-5.4-2026-03-05",
        reasoning_effort="medium",
        system_prompt="You are a helpful assistant.",
    )

    result = await client.complete("Explain why 2+2=4 in one sentence.")
    print(f"Response: {result.final_response}")
    print()


async def example_reasoning_trace():
    """Opt into Responses API reasoning summaries via reasoning_summary."""
    print("=" * 70)
    print("Example 3: Reasoning Trace (Responses API)")
    print("=" * 70)

    # reasoning_summary opts into readable summaries; result.reasoning_trace holds them.
    # Prefer "detailed" on o4-mini — "auto"/"concise" can occasionally return empty summaries.
    result = await response(
        "What is the next number in: 2, 4, 8, 16, ?",
        model="openai:o4-mini",
        reasoning_effort="high",
        reasoning_summary="detailed",
        system_prompt="You are a logic puzzle assistant.",
        execute_tools=False,
    )

    print(f"Reasoning trace: {result.reasoning_trace}")
    print(f"Response: {result.final_response}")
    print()


async def example_streaming_reasoning_trace():
    """Stream answer text and reasoning summaries on separate channels.

    Answer deltas arrive as StreamingChunk / stream_chunk events.
    Reasoning summary deltas arrive only as ProcessEvent(kind="reasoning_chunk")
    via on_status (or sinks) — they are not mixed into StreamingChunk.content.
    """
    print("=" * 70)
    print("Example 4: Streaming Reasoning Trace (Responses API)")
    print("=" * 70)

    client = GlueLLM(
        model="openai:o4-mini",
        system_prompt="You are a logic puzzle assistant.",
    )
    reasoning_parts: list[str] = []
    in_reasoning = False

    def on_status(event: ProcessEvent) -> None:
        nonlocal in_reasoning
        if event.kind == "reasoning_chunk" and event.content:
            if not in_reasoning:
                print("[think] ", end="", flush=True)
                in_reasoning = True
            reasoning_parts.append(event.content)
            print(event.content, end="", flush=True)
        elif event.kind == "stream_chunk" and event.content:
            if in_reasoning:
                print("\n", end="", flush=True)
                in_reasoning = False
            print(event.content, end="", flush=True)

    print("(reasoning under [think]; answer streams after)\n")
    async for chunk in client.stream_response(
        "What is the next number in: 2, 4, 8, 16, ?",
        reasoning_effort="high",
        reasoning_summary="detailed",
        on_status=on_status,
        execute_tools=False,
    ):
        if chunk.done:
            break

    print()
    full_reasoning = "".join(reasoning_parts)
    print(f"Full reasoning: {full_reasoning or '(none received — provider returned no summary events)'}")
    print()


async def example_logprobs():
    """Enable logprobs for token-level confidence (eval, scoring)."""
    print("=" * 70)
    print("Example 5: Logprobs for Confidence Scoring")
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
    print("Example 6: Config Defaults")
    print("=" * 70)

    print(f"Current default_reasoning_effort: {settings.default_reasoning_effort}")
    print(f"Current default_parallel_tool_calls: {settings.default_parallel_tool_calls}")
    print("Set GLUELLM_DEFAULT_REASONING_EFFORT or GLUELLM_DEFAULT_PARALLEL_TOOL_CALLS in .env")
    print()


async def main():
    """Run all examples."""
    print("\n🧙 Reasoning Effort, Traces, and Logprobs Examples\n")

    await example_reasoning_effort()
    await example_reasoning_effort_on_client()
    await example_reasoning_trace()
    await example_streaming_reasoning_trace()
    await example_logprobs()
    await example_default_from_config()

    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
