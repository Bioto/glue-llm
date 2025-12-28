"""GlueLLM Command-Line Interface.

This module provides CLI commands for testing, demonstrating, and running
GlueLLM functionality from the command line using Click.

Available Commands:
    - test_completion: Test basic completion functionality
    - test_tool_call: Test tool calling with weather example
    - run_tests: Run the test suite with various options
    - demo: Run interactive demos of core features
    - examples: Run example scripts from examples/
"""

import click
from rich.console import Console

console = Console()


@click.group()
def cli() -> None:
    """GlueLLM CLI - Command-line interface for GlueLLM operations.

    Provides commands for testing, demonstrations, and running examples.
    Use --help with any command for more information.
    """
    pass


def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    This is a mock function used for testing and demonstration purposes.

    Args:
        location: The city and country, e.g. "San Francisco, CA"
        unit: Temperature unit, either "celsius" or "fahrenheit"

    Returns:
        str: A simulated weather response string
    """
    # Simulated weather response
    return f"The weather in {location} is 22 degrees {unit} and sunny."


@cli.command()
def test_completion() -> None:
    """Test basic completion functionality.

    Demonstrates a simple completion request using the default model
    and configuration, with structured response format.

    Returns:
        The completion response object
    """
    from typing import Annotated

    from any_llm import completion
    from pydantic import BaseModel, Field

    from source.config import settings
    from source.models.config import RequestConfig
    from source.models.conversation import Role
    from source.models.prompt import SystemPrompt

    class DefaultResponseFormat(BaseModel):
        response: Annotated[str, Field(description="The response to the request")]

    request_config = RequestConfig(
        model=settings.default_model,
        system_prompt=SystemPrompt(
            content=settings.default_system_prompt,
        ),
        response_format=DefaultResponseFormat,
        tools=[get_weather],
    )
    request_config.add_message_to_conversation(Role.USER, "Get weather for Tokyo, Japan")

    response = completion(
        messages=request_config.get_conversation(),
        model=request_config.model,
        response_format=request_config.response_format if not request_config.tools else None,
        tools=request_config.tools,
    )

    console.print(response)

    return response


@cli.command()
def test_tool_call() -> None:
    """Test completion with automatic tool calling.

    Demonstrates the full tool execution flow:
    1. Model receives a query requiring tool use
    2. Model calls the appropriate tool
    3. Tool executes and returns results
    4. Model processes results and provides final response

    Uses the get_weather tool as an example.
    """
    from typing import Annotated

    from any_llm import completion
    from pydantic import BaseModel, Field

    from source.config import settings
    from source.models.config import RequestConfig
    from source.models.conversation import Role
    from source.models.prompt import SystemPrompt

    class DefaultResponseFormat(BaseModel):
        response: Annotated[str, Field(description="The response to the request")]

    request_config = RequestConfig(
        model=settings.default_model,
        system_prompt=SystemPrompt(
            content="You are a helpful assistant. Use the get_weather tool when asked about weather.",
        ),
        response_format=DefaultResponseFormat,
        tools=[get_weather],
    )
    request_config.add_message_to_conversation(Role.USER, "What's the weather like in Tokyo, Japan?")

    response = completion(
        messages=request_config.get_conversation(),
        model=request_config.model,
        tools=request_config.tools,
    )

    console.print("[bold]Initial Response:[/bold]")
    console.print(response)

    # Check if the model wants to call a tool
    if response.choices[0].message.tool_calls:
        import json

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        console.print(f"\n[bold]Tool Call:[/bold] {tool_call.function.name}")
        console.print(f"[bold]Arguments:[/bold] {args}")

        # Execute the tool
        result = get_weather(**args)
        console.print(f"[bold]Tool Result:[/bold] {result}")

        # Build messages with tool call and result for follow-up
        messages = request_config.get_conversation() + [
            response.choices[0].message,  # Assistant message with tool_calls
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            },
        ]

        final_response = completion(
            messages=messages,
            model=request_config.model,
            tools=request_config.tools,
        )

        console.print("\n[bold]Final Response:[/bold]")
        console.print(final_response.choices[0].message.content)


@cli.command()
@click.option("--test", "-t", help="Specific test to run (e.g., test_single_tool_call)")
@click.option("--class-name", "-c", help="Test class to run (e.g., TestBasicToolCalling)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-integration", is_flag=True, help="Skip integration tests")
def run_tests(test: str | None, class_name: str | None, verbose: bool, no_integration: bool) -> None:
    """Run the GlueLLM test suite using pytest.

    Provides flexible test execution with options to:
    - Run specific tests or test classes
    - Enable verbose output
    - Skip integration tests

    Args:
        test: Specific test method name to run
        class_name: Specific test class to run
        verbose: Enable verbose pytest output
        no_integration: Skip tests marked with @pytest.mark.integration

    Examples:
        glm run-tests --verbose
        glm run-tests -c TestBasicToolCalling
        glm run-tests -t test_single_tool_call
    """
    import subprocess
    import sys

    args = ["pytest"]

    if verbose:
        args.append("-v")

    # Always show output for these tests
    args.append("-s")

    if no_integration:
        args.extend(["-m", "not integration"])

    if class_name:
        args.append(f"tests/test_llm_edge_cases.py::{class_name}")
    elif test:
        # Try to find the test
        args.append(f"tests/test_llm_edge_cases.py::*::{test}")
    else:
        args.append("tests/")

    console.print(f"[bold cyan]Running:[/bold cyan] {' '.join(args)}")
    result = subprocess.run(args, cwd="/home/nick/Projects/gluellm")
    sys.exit(result.returncode)


@cli.command()
def demo() -> None:
    """Run interactive GlueLLM API demonstrations.

    Executes four demo scenarios:
    1. Simple completion - Basic text generation
    2. Tool execution - Automatic tool calling and execution
    3. Structured output - Pydantic model-based responses
    4. Multi-turn conversation - Conversational memory

    This command is useful for understanding core GlueLLM features
    and verifying that the installation is working correctly.
    """
    import asyncio

    async def run_demos():
        console.print("[bold cyan]GlueLLM API Demos[/bold cyan]\n")

        from typing import Annotated

        from pydantic import BaseModel, Field

        from source.api import GlueLLM, complete, structured_complete

        # Demo 1: Simple completion
        console.print("[bold]Demo 1: Simple Completion[/bold]")
        result = await complete(
            user_message="What is 2+2? Answer briefly.",
            system_prompt="You are a helpful math assistant.",
        )
        console.print(f"Response: {result.final_response}\n")

        # Demo 2: Tool execution
        console.print("[bold]Demo 2: Automatic Tool Execution[/bold]")

        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: 22°C, sunny ☀️"

        result = await complete(
            user_message="What's the weather in Tokyo?",
            system_prompt="You are a weather assistant. Use get_weather for queries.",
            tools=[get_weather],
        )
        console.print(f"Response: {result.final_response}")
        console.print(f"Tool calls: {result.tool_calls_made}\n")

        # Demo 3: Structured output
        console.print("[bold]Demo 3: Structured Output[/bold]")

        class CityInfo(BaseModel):
            city: Annotated[str, Field(description="City name")]
            country: Annotated[str, Field(description="Country name")]
            population: Annotated[int, Field(description="Population estimate")]

        city = await structured_complete(
            user_message="Extract: Tokyo, Japan has a population of about 14 million",
            response_format=CityInfo,
        )
        console.print(f"City: {city.city}")
        console.print(f"Country: {city.country}")
        console.print(f"Population: {city.population:,}\n")

        # Demo 4: Multi-turn conversation
        console.print("[bold]Demo 4: Multi-turn Conversation[/bold]")
        client = GlueLLM(system_prompt="You are a helpful assistant with memory.")

        result1 = await client.complete("My favorite number is 42")
        console.print(f"Turn 1: {result1.final_response}")

        result2 = await client.complete("What's my favorite number?")
        console.print(f"Turn 2: {result2.final_response}\n")

        console.print("[bold green]✓ All demos completed![/bold green]")

    asyncio.run(run_demos())


@cli.command()
def examples() -> None:
    """Run example scripts from the examples/ directory.

    Executes examples/basic_usage.py which contains comprehensive
    usage examples for the GlueLLM library.

    This is useful for:
    - Learning by example
    - Verifying functionality
    - Understanding best practices
    """
    import subprocess
    import sys

    console.print("[bold cyan]Running GlueLLM Examples[/bold cyan]\n")
    result = subprocess.run(["python", "examples/basic_usage.py"], cwd="/home/nick/Projects/gluellm")
    sys.exit(result.returncode)


if __name__ == "__main__":
    cli()
