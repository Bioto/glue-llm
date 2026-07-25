"""
Example: response() — plain-text completion via the OpenAI Responses API.

``response()`` is the Responses-API twin of ``complete()``. It supports the
same tool-execution loop, dynamic tool routing, guardrails, conversation
summarisation, AAAK condensing/compression, hooks, eval recording, cost
tracking, status events and sinks. Differences are confined to the wire
protocol:

* It calls ``provider.aresponses`` instead of ``provider.acompletion``.
* ``user_input`` may be a string OR a Responses ``ResponseInputParam`` list
  (multimodal, prefilled tool history, ...).
* Tools are flattened to the Responses tool shape under the hood; the same
  Python callables work unchanged.

Reach for ``response()`` when you need Responses-only features (multimodal
``input_image``, native reasoning summaries, server-side tools, the typed
``ResponseInputParam`` shape) without giving up GlueLLM's orchestration.

On providers without Responses support (e.g. direct Anthropic), GlueLLM
automatically falls back to ``complete()`` for string inputs. Multimodal list
inputs require a Responses-capable provider.
"""

import asyncio

from gluellm import GlueLLM, response


# Example 1: Simple Responses-API completion
async def example_simple_response():
    """Simple one-off Responses-API completion without tools."""
    print("=" * 80)
    print("Example 1: Simple Response")
    print("=" * 80)

    result = await response(
        user_input="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    print()


# Example 2: Tool execution loop over the Responses API
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Paris"
        unit: Temperature unit, either "celsius" or "fahrenheit"
    """
    temps = {
        "Paris": 22,
        "Tokyo": 25,
        "London": 15,
    }
    temp = temps.get(city, 20)
    return f"The weather in {city} is {temp} degrees {unit} and sunny."


async def example_tool_execution():
    """Tools work transparently — same Python callables, different wire protocol."""
    print("=" * 80)
    print("Example 2: Automatic Tool Execution via Responses API")
    print("=" * 80)

    result = await response(
        user_input="What's the weather like in Paris and Tokyo?",
        system_prompt="You are a helpful weather assistant. Use the get_weather tool.",
        tools=[get_weather],
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}\n")
    print(f"Tool calls made: {result.tool_calls_made}")
    print("\nTool execution history:")
    for i, exec_info in enumerate(result.tool_execution_history, 1):
        print(f"  {i}. {exec_info['tool_name']}({exec_info['arguments']}) -> {exec_info['result']}")
    print()


# Example 3: Multi-turn conversation with the GlueLLM client
async def example_multi_turn_with_client():
    """A single GlueLLM instance may interleave complete() and response() across turns."""
    print("=" * 80)
    print("Example 3: Multi-turn Conversation Using GlueLLM.response()")
    print("=" * 80)

    client = GlueLLM(
        model="openai:gpt-5.4-2026-03-05",
        system_prompt="You are concise and friendly.",
    )

    result1 = await client.response("Remember: my favourite colour is teal.")
    print("User: Remember: my favourite colour is teal.")
    print(f"Assistant: {result1.final_response}\n")

    result2 = await client.response("What's my favourite colour?")
    print("User: What's my favourite colour?")
    print(f"Assistant: {result2.final_response}\n")


# Example 4: ResponseInputParam list — multimodal/prefilled input
async def example_response_input_param_list():
    """Pass a typed Responses input list directly when you need richer input items.

    The Responses-API ``ResponseInputParam`` shape lets you mix text, images,
    and prefilled tool outputs in a single request. ``response()`` accepts the
    list shape verbatim — useful for multimodal or replaying a prior tool
    transcript. See ``multimodal_image_example.py`` for ``input_image`` parts.
    """
    print("=" * 80)
    print("Example 4: ResponseInputParam List Input")
    print("=" * 80)

    typed_input = [
        {
            "role": "user",
            "content": "Pretend you can see images and describe a rainbow in one sentence.",
        }
    ]

    result = await response(
        user_input=typed_input,
        system_prompt="You are an evocative writer.",
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}")
    print()


# Example 5: Cost and token usage tracking
async def example_observability():
    """Token usage from input_tokens/output_tokens is normalised to prompt/completion."""
    print("=" * 80)
    print("Example 5: Token Usage and Cost Tracking")
    print("=" * 80)

    result = await response(
        user_input="Explain photosynthesis in two sentences.",
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}\n")
    print(f"Tokens used: {result.tokens_used}")
    if result.estimated_cost_usd is not None:
        print(f"Estimated cost: ${result.estimated_cost_usd:.6f}")
    print(f"Model: {result.model}")
    print()


async def main():
    """Run all examples."""
    print("\nResponse API Examples (Plain Text via OpenAI Responses)\n")

    await example_simple_response()
    await example_tool_execution()
    await example_multi_turn_with_client()
    await example_response_input_param_list()
    await example_observability()

    print("=" * 80)
    print("All examples completed.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
