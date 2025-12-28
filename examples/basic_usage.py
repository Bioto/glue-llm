"""Examples of basic GlueLLM usage."""

from typing import Annotated

from pydantic import BaseModel, Field

from source.api import GlueLLM, complete, structured_complete


# Example 1: Simple completion
async def example_simple_completion():
    """Simple one-off completion without tools."""
    print("=" * 60)
    print("Example 1: Simple Completion")
    print("=" * 60)

    result = await complete(
        user_message="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
    )

    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    print()


# Example 2: Completion with automatic tool execution
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: The city and country, e.g. "San Francisco, CA"
        unit: Temperature unit, either "celsius" or "fahrenheit"
    """
    # Simulated weather response
    temps = {
        "Tokyo, Japan": 22,
        "San Francisco, CA": 18,
        "London, UK": 12,
    }
    temp = temps.get(location, 20)
    return f"The weather in {location} is {temp} degrees {unit} and sunny."


async def example_tool_execution():
    """Example with automatic tool execution loop."""
    print("=" * 60)
    print("Example 2: Automatic Tool Execution")
    print("=" * 60)

    result = await complete(
        user_message="What's the weather like in Tokyo, Japan? Also check San Francisco.",
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to get current weather.",
        tools=[get_weather],
    )

    print(f"Response: {result.final_response}\n")
    print(f"Tool calls made: {result.tool_calls_made}")
    print("\nTool execution history:")
    for i, exec_info in enumerate(result.tool_execution_history, 1):
        print(f"  {i}. {exec_info['tool_name']}({exec_info['arguments']}) -> {exec_info['result']}")
    print()


# Example 3: Structured output
class PersonInfo(BaseModel):
    """Information about a person."""

    name: Annotated[str, Field(description="Full name of the person")]
    age: Annotated[int, Field(description="Age in years")]
    occupation: Annotated[str, Field(description="Current occupation")]
    city: Annotated[str, Field(description="City of residence")]


async def example_structured_output():
    """Example with structured output using Pydantic models."""
    print("=" * 60)
    print("Example 3: Structured Output")
    print("=" * 60)

    person = await structured_complete(
        user_message="Extract information about: John Smith is a 35 year old software engineer living in Seattle.",
        response_format=PersonInfo,
        system_prompt="You are a data extraction assistant. Extract structured information from the text.",
    )

    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"City: {person.city}")
    print(f"\nFull object: {person}")
    print()


# Example 4: Multi-turn conversation with tools
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"
    """
    try:
        # Simple safe evaluation (only numbers and basic operators)
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


async def example_conversation_with_tools():
    """Example of multi-turn conversation with tool usage."""
    print("=" * 60)
    print("Example 4: Multi-turn Conversation with Tools")
    print("=" * 60)

    # Create a client to maintain conversation state
    client = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful math assistant. Use the calculate tool for any math operations.",
        tools=[calculate],
    )

    # First turn
    result1 = await client.complete("What is 25 * 4?")
    print("User: What is 25 * 4?")
    print(f"Assistant: {result1.final_response}\n")

    # Second turn (references previous context)
    result2 = await client.complete("Now add 50 to that result")
    print("User: Now add 50 to that result")
    print(f"Assistant: {result2.final_response}\n")

    # Show total tool calls across conversation
    total_tools = result1.tool_calls_made + result2.tool_calls_made
    print(f"Total tool calls in conversation: {total_tools}")
    print()


# Example 5: Multiple tools working together
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a symbol.

    Args:
        symbol: Stock ticker symbol like "AAPL" or "GOOGL"
    """
    # Simulated stock prices
    prices = {
        "AAPL": 175.50,
        "GOOGL": 142.30,
        "MSFT": 380.20,
    }
    price = prices.get(symbol.upper(), 100.00)
    return f"Current price of {symbol}: ${price:.2f}"


def get_company_name(symbol: str) -> str:
    """Get the company name for a stock symbol.

    Args:
        symbol: Stock ticker symbol like "AAPL" or "GOOGL"
    """
    # Simulated company names
    companies = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
    }
    return companies.get(symbol.upper(), "Unknown Company")


async def example_multiple_tools():
    """Example using multiple tools in a single request."""
    print("=" * 60)
    print("Example 5: Multiple Tools Working Together")
    print("=" * 60)

    result = await complete(
        user_message="Tell me about AAPL stock - what company is it and what's the current price?",
        system_prompt="You are a financial assistant. Use available tools to answer questions about stocks.",
        tools=[get_stock_price, get_company_name],
    )

    print(f"Response: {result.final_response}\n")
    print(f"Tools used: {result.tool_calls_made}")
    print("\nExecution details:")
    for exec_info in result.tool_execution_history:
        print(f"  - {exec_info['tool_name']}: {exec_info['result']}")
    print()


if __name__ == "__main__":
    import asyncio

    async def main():
        # Run all examples
        await example_simple_completion()
        await example_tool_execution()
        await example_structured_output()
        await example_conversation_with_tools()
        await example_multiple_tools()

    asyncio.run(main())
