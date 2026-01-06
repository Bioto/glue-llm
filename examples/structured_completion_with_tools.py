"""
Example: Structured Completions with Tool Support

Demonstrates how to use structured completions with tools. The LLM can call
tools to gather information before returning the final structured output.
"""

import asyncio

from pydantic import BaseModel, Field

from gluellm.api import structured_complete


# Define structured output format
class WeatherReport(BaseModel):
    """A weather report with temperature and conditions."""

    location: str = Field(description="The location for the weather report")
    temperature_celsius: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions (e.g., sunny, cloudy, rainy)")
    recommendation: str = Field(description="A recommendation based on the weather")


class CalculationResult(BaseModel):
    """Result of a mathematical calculation."""

    expression: str = Field(description="The mathematical expression evaluated")
    result: float = Field(description="The numerical result")
    explanation: str = Field(description="Step-by-step explanation of the calculation")


# Define tools that the LLM can use
def get_weather(city: str) -> dict:
    """
    Get the current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Dictionary with temperature and conditions
    """
    # Mock weather data - in production, this would call a real weather API
    weather_data = {
        "San Francisco": {"temperature": 18, "conditions": "foggy"},
        "New York": {"temperature": 22, "conditions": "sunny"},
        "London": {"temperature": 15, "conditions": "rainy"},
        "Tokyo": {"temperature": 25, "conditions": "cloudy"},
    }
    return weather_data.get(city, {"temperature": 20, "conditions": "unknown"})


def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation
    """
    # In production, use a safer evaluation method
    # This is just for demonstration purposes
    try:
        # Only allow basic math operations for safety
        allowed_chars = set("0123456789+-*/(). ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return float(result)
        raise ValueError("Invalid characters in expression")
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}") from e


async def example_weather_with_tool():
    """Example: Get structured weather report using a weather tool."""
    print("=" * 80)
    print("Example 1: Weather Report with Tool")
    print("=" * 80)

    result = await structured_complete(
        user_message="What's the weather like in San Francisco? Use the weather tool and give me a structured report.",
        response_format=WeatherReport,
        tools=[get_weather],
        model="openai:gpt-4o-mini",
    )

    print(f"\n📍 Location: {result.structured_output.location}")
    print(f"🌡️  Temperature: {result.structured_output.temperature_celsius}°C")
    print(f"☁️  Conditions: {result.structured_output.conditions}")
    print(f"💡 Recommendation: {result.structured_output.recommendation}")
    print(f"\n📊 Tool calls made: {result.tool_calls_made}")
    print(f"💰 Cost: ${result.estimated_cost_usd:.6f}")
    print("🔧 Tool execution history:")
    for i, tool_exec in enumerate(result.tool_execution_history, 1):
        print(f"   {i}. {tool_exec['tool_name']}({tool_exec['arguments']}) = {tool_exec['result']}")


async def example_calculation_with_tool():
    """Example: Get structured calculation result using a calculator tool."""
    print("\n" + "=" * 80)
    print("Example 2: Mathematical Calculation with Tool")
    print("=" * 80)

    result = await structured_complete(
        user_message="Calculate (15 + 25) * 3 using the calculator tool and explain the steps.",
        response_format=CalculationResult,
        tools=[calculate],
        model="openai:gpt-4o-mini",
    )

    print(f"\n📐 Expression: {result.structured_output.expression}")
    print(f"🔢 Result: {result.structured_output.result}")
    print(f"📝 Explanation: {result.structured_output.explanation}")
    print(f"\n📊 Tool calls made: {result.tool_calls_made}")
    print(f"💰 Cost: ${result.estimated_cost_usd:.6f}")


async def example_without_tools():
    """Example: Structured completion without tools (direct response)."""
    print("\n" + "=" * 80)
    print("Example 3: Calculation without Tools (Direct)")
    print("=" * 80)

    result = await structured_complete(
        user_message="What is 2 + 2? Just give me the answer.",
        response_format=CalculationResult,
        model="openai:gpt-4o-mini",
    )

    print(f"\n📐 Expression: {result.structured_output.expression}")
    print(f"🔢 Result: {result.structured_output.result}")
    print(f"📝 Explanation: {result.structured_output.explanation}")
    print(f"\n📊 Tool calls made: {result.tool_calls_made}")
    print(f"💰 Cost: ${result.estimated_cost_usd:.6f}")


async def example_multiple_cities():
    """Example: Compare weather in multiple cities using multiple tool calls."""
    print("\n" + "=" * 80)
    print("Example 4: Weather Comparison (Multiple Tool Calls)")
    print("=" * 80)

    class WeatherComparison(BaseModel):
        """Comparison of weather in multiple cities."""

        cities_compared: list[str] = Field(description="List of cities compared")
        warmest_city: str = Field(description="The warmest city")
        coolest_city: str = Field(description="The coolest city")
        recommendation: str = Field(description="Travel recommendation based on weather")

    result = await structured_complete(
        user_message="Compare the weather in San Francisco, New York, and London. Which city is warmest?",
        response_format=WeatherComparison,
        tools=[get_weather],
        model="openai:gpt-4o-mini",
    )

    print(f"\n🌍 Cities compared: {', '.join(result.structured_output.cities_compared)}")
    print(f"🔥 Warmest: {result.structured_output.warmest_city}")
    print(f"❄️  Coolest: {result.structured_output.coolest_city}")
    print(f"✈️  Recommendation: {result.structured_output.recommendation}")
    print(f"\n📊 Tool calls made: {result.tool_calls_made}")
    print(f"💰 Cost: ${result.estimated_cost_usd:.6f}")


async def main():
    """Run all examples."""
    print("\n🧙‍♂️ Structured Completions with Tool Support Examples\n")

    await example_weather_with_tool()
    await example_calculation_with_tool()
    await example_without_tools()
    await example_multiple_cities()

    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
