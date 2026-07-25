"""
Example: structured_response() — structured output via the OpenAI Responses API.

``structured_response()`` is the Responses-API twin of ``structured_complete()``.
It returns an ``ExecutionResult`` whose ``structured_output`` field holds the
parsed Pydantic instance, and supports the full feature set:

* Tool execution loop (tools may run BEFORE the final structured call so the
  model can gather facts first).
* Validation-aware retries: a Pydantic ``ValidationError`` is fed back to the
  model so it can self-correct, up to ``max_validation_retries`` times.
* Guardrails (input + output with retry), conversation summarisation, AAAK
  condensing/compression, hooks, eval recording, cost tracking, status events.
* ``user_input`` may be a string OR a Responses ``ResponseInputParam`` list.

Reach for ``structured_response()`` when you need both Pydantic-validated
output AND Responses-only features (multimodal ``input_image``, native
reasoning summaries, server-side tools, ...).

On providers without Responses support (e.g. direct Anthropic), GlueLLM
automatically falls back to ``structured_complete()`` for string inputs.
"""

import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from gluellm import GlueLLM, structured_response


# Example 1: Simple structured output
class PersonInfo(BaseModel):
    """Information about a person extracted from text."""

    name: Annotated[str, Field(description="Full name of the person")]
    age: Annotated[int, Field(description="Age in years")]
    occupation: Annotated[str, Field(description="Current occupation")]
    city: Annotated[str, Field(description="City of residence")]


async def example_simple_structured():
    """Extract structured data — no tools, single call."""
    print("=" * 80)
    print("Example 1: Simple Structured Output")
    print("=" * 80)

    result = await structured_response(
        user_input="Extract: John Smith is a 35 year old software engineer living in Seattle.",
        response_format=PersonInfo,
        system_prompt="You are a precise data extraction assistant.",
        model="openai:gpt-5.4-2026-03-05",
    )

    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")
    person = result.structured_output
    print(f"Name:       {person.name}")
    print(f"Age:        {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"City:       {person.city}")
    print(f"\nTokens used: {result.tokens_used}")
    if result.estimated_cost_usd is not None:
        print(f"Cost:        ${result.estimated_cost_usd:.6f}")
    print()


# Example 2: Tool-augmented structured output
class WeatherReport(BaseModel):
    """A weather report with temperature and recommendation."""

    location: str = Field(description="The location for the weather report")
    temperature_celsius: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions (e.g., sunny, rainy)")
    recommendation: str = Field(description="A recommendation based on the weather")


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: Name of the city
    """
    weather_data = {
        "Paris": {"temperature": 22, "conditions": "sunny"},
        "Tokyo": {"temperature": 25, "conditions": "cloudy"},
        "London": {"temperature": 15, "conditions": "rainy"},
    }
    return weather_data.get(city, {"temperature": 20, "conditions": "unknown"})


async def example_tool_then_structured():
    """The model uses a tool to gather data, then emits the structured answer."""
    print("=" * 80)
    print("Example 2: Tool-Augmented Structured Output")
    print("=" * 80)

    result = await structured_response(
        user_input="What's the weather in Paris? Use the tool, then give a structured report.",
        response_format=WeatherReport,
        tools=[get_weather],
        system_prompt="You are a weather reporter. Use tools, then return structured data.",
        model="openai:gpt-5.4-2026-03-05",
    )

    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")
    report = result.structured_output
    print(f"Location:       {report.location}")
    print(f"Temperature:    {report.temperature_celsius}C")
    print(f"Conditions:     {report.conditions}")
    print(f"Recommendation: {report.recommendation}")
    print(f"\nTool calls made: {result.tool_calls_made}")
    print("Tool execution history:")
    for i, tool_exec in enumerate(result.tool_execution_history, 1):
        print(f"  {i}. {tool_exec['tool_name']}({tool_exec['arguments']}) = {tool_exec['result']}")
    print()


# Example 3: Validation retries — schema mismatch self-correction
class StrictNumber(BaseModel):
    """A simple structured number with strict bounds."""

    value: Annotated[int, Field(ge=1, le=10, description="Integer between 1 and 10")]
    explanation: str = Field(description="Why this number was chosen")


async def example_validation_retries():
    """Validation errors are fed back so the model can self-correct.

    If the model returns ``value=42`` (out of range), the Pydantic
    ``ValidationError`` is converted into a follow-up user message and the
    model is asked to try again — up to ``max_validation_retries`` times.
    """
    print("=" * 80)
    print("Example 3: Validation-Aware Retries")
    print("=" * 80)

    result = await structured_response(
        user_input="Pick your favourite number between 1 and 10 and explain why.",
        response_format=StrictNumber,
        system_prompt="You follow schemas strictly.",
        model="openai:gpt-5.4-2026-03-05",
        max_validation_retries=3,
    )

    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")
    pick = result.structured_output
    print(f"Value:       {pick.value}")
    print(f"Explanation: {pick.explanation}")
    print()


# Example 4: ResponseInputParam list — typed input
class Sentiment(BaseModel):
    """Sentiment analysis of a piece of text."""

    label: Annotated[str, Field(description="One of 'positive', 'neutral', 'negative'")]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence between 0 and 1")]
    reasoning: str = Field(description="Why this label was chosen")


async def example_response_input_param_list():
    """Pass a Responses ``ResponseInputParam`` list directly.

    The list shape is the Responses-API native input format. ``structured_response()``
    accepts it verbatim and routes it through the same pipeline as a string input,
    so all context-management features still apply.
    """
    print("=" * 80)
    print("Example 4: ResponseInputParam List Input")
    print("=" * 80)

    typed_input = [
        {
            "role": "user",
            "content": "I absolutely love this new feature, it's a delight to use!",
        }
    ]

    result = await structured_response(
        user_input=typed_input,
        response_format=Sentiment,
        system_prompt="You are a sentiment analyst.",
        model="openai:gpt-5.4-2026-03-05",
    )

    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")
    s = result.structured_output
    print(f"Label:      {s.label}")
    print(f"Confidence: {s.confidence:.2f}")
    print(f"Reasoning:  {s.reasoning}")
    print()


# Example 5: Multi-turn with the GlueLLM client
async def example_multi_turn_with_client():
    """A single GlueLLM client can interleave structured_response with other helpers.

    Here we use ``client.complete()`` to plant context, then
    ``client.structured_response()`` to extract a typed answer. The
    conversation history is shared.
    """
    print("=" * 80)
    print("Example 5: Multi-turn With GlueLLM.structured_response()")
    print("=" * 80)

    client = GlueLLM(
        model="openai:gpt-5.4-2026-03-05",
        system_prompt="You are concise and accurate.",
    )

    setup = await client.complete(
        "I'm planning a trip and visited these cities: Lisbon, Porto, Faro."
    )
    print(f"(setup) Assistant: {setup.final_response}\n")

    class TripSummary(BaseModel):
        country: str = Field(description="The country these cities are in")
        city_count: int = Field(description="How many cities were mentioned")
        cities: list[str] = Field(description="The cities mentioned")

    result = await client.structured_response(
        "Summarise the trip context as structured data.",
        response_format=TripSummary,
    )

    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")
    summary = result.structured_output
    print(f"Country:    {summary.country}")
    print(f"City count: {summary.city_count}")
    print(f"Cities:     {', '.join(summary.cities)}")
    print()


async def main():
    """Run all examples."""
    print("\nStructured Response Examples (Pydantic via OpenAI Responses)\n")

    await example_simple_structured()
    await example_tool_then_structured()
    await example_validation_retries()
    await example_response_input_param_list()
    await example_multi_turn_with_client()

    print("=" * 80)
    print("All examples completed.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
