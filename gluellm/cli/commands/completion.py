"""Completion-related CLI commands.

Commands for testing basic completion, streaming, and structured output.
"""

import click

from gluellm.cli.utils import (
    console,
    print_error,
    print_header,
    print_step,
    print_success,
    run_async,
)


@click.command("test-completion")
def test_completion() -> None:
    """Test basic completion functionality.

    Demonstrates a simple completion request using the default model
    and configuration, with structured response format.
    """
    from typing import Annotated

    from any_llm import completion
    from pydantic import BaseModel, Field

    from gluellm.config import settings
    from gluellm.models.config import RequestConfig
    from gluellm.models.conversation import Role
    from gluellm.models.prompt import SystemPrompt

    class DefaultResponseFormat(BaseModel):
        response: Annotated[str, Field(description="The response to the request")]

    print_header("Test Completion", f"Model: {settings.default_model}")

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is 22 degrees {unit} and sunny."

    request_config = RequestConfig(
        model=settings.default_model,
        system_prompt=SystemPrompt(content=settings.default_system_prompt),
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
    print_success("Completion test passed")


@click.command("test-streaming")
@click.option("--message", "-m", default="Tell me a short story about a robot", help="Message to stream")
def test_streaming(message: str) -> None:
    """Test streaming completion functionality."""
    from gluellm.api import stream_complete

    print_header("Test Streaming Completion")
    print_step(1, 2, f"Streaming response for: {message[:50]}...")

    async def run_stream():
        full_response = ""
        async for chunk in stream_complete(message, execute_tools=False):
            if chunk.content:
                console.print(chunk.content, end="")
                full_response += chunk.content
            if chunk.done:
                console.print()
                return full_response
        return full_response

    try:
        result = run_async(run_stream())
        print_success(f"Streaming completed ({len(result)} chars)")
    except Exception as e:
        print_error(f"Streaming failed: {e}")


@click.command("test-structured-output")
@click.option("--data", "-d", default="Tokyo, Japan has 14 million people", help="Data to extract")
def test_structured_output(data: str) -> None:
    """Test structured output with Pydantic models."""
    from typing import Annotated

    from pydantic import BaseModel, Field

    from gluellm.api import structured_complete

    class CityInfo(BaseModel):
        city: Annotated[str, Field(description="City name")]
        country: Annotated[str, Field(description="Country name")]
        population: Annotated[int, Field(description="Population in millions")]

    print_header("Test Structured Output")
    print_step(1, 2, f"Extracting structured data from: {data}")

    async def run_extraction():
        return await structured_complete(
            f"Extract information: {data}",
            response_format=CityInfo,
        )

    try:
        result = run_async(run_extraction())
        console.print(f"Extracted: {result}")
        print_success("Structured output test passed")
    except Exception as e:
        print_error(f"Extraction failed: {e}")


@click.command("test-multi-turn-conversation")
@click.option("--turns", "-t", default=3, type=int, help="Number of conversation turns")
def test_multi_turn_conversation(turns: int) -> None:
    """Test multi-turn conversation memory."""
    from gluellm.api import GlueLLM

    print_header("Test Multi-Turn Conversation", f"Turns: {turns}")

    prompts = [
        "Hi! My name is Alex.",
        "What's my name?",
        "What was the first thing I said?",
        "Can you summarize our conversation?",
    ]

    async def run_conversation():
        client = GlueLLM(system_prompt="You are a helpful assistant with good memory.")

        for i, prompt in enumerate(prompts[:turns], 1):
            print_step(i, turns, f"User: {prompt}")
            result = await client.complete(prompt)
            console.print(f"  Assistant: {result.final_response[:200]}")

    try:
        run_async(run_conversation())
        print_success("Multi-turn conversation test passed")
    except Exception as e:
        print_error(f"Conversation failed: {e}")


@click.command("test-embedding")
@click.option("--text", "-t", default="Hello, world!", help="Text to embed")
@click.option(
    "--model", "-m", default=None, help="Embedding model to use (defaults to settings.default_embedding_model)"
)
@click.option("--batch", "-b", is_flag=True, help="Treat input as comma-separated batch")
def test_embedding(text: str, model: str | None, batch: bool) -> None:
    """Test embedding generation functionality."""
    from gluellm import embed
    from gluellm.config import settings

    print_header("Test Embedding Generation")

    # Determine model
    embedding_model = model or settings.default_embedding_model
    print_step(1, 3, f"Model: {embedding_model}")

    # Parse inputs
    if batch:
        texts = [t.strip() for t in text.split(",")]
        print_step(2, 3, f"Generating embeddings for {len(texts)} texts")
    else:
        texts = text
        print_step(2, 3, f"Generating embedding for: {text[:50]}...")

    async def run_embedding():
        return await embed(texts=texts, model=embedding_model)

    try:
        result = run_async(run_embedding())

        console.print("\n[bold]Results:[/bold]")
        console.print(f"  Model: {result.model}")
        console.print(f"  Embeddings: {result.count}")
        console.print(f"  Dimension: {result.dimension}")
        console.print(f"  Tokens used: {result.tokens_used:,}")
        if result.estimated_cost_usd is not None:
            console.print(f"  Estimated cost: ${result.estimated_cost_usd:.6f}")
        else:
            console.print("  Estimated cost: N/A (pricing not available)")

        if isinstance(texts, list):
            console.print("\n[bold]Sample embeddings (first 5 dimensions):[/bold]")
            for i, emb in enumerate(result.embeddings[:3]):  # Show first 3
                sample = emb[:5]
                console.print(f"  [{i}]: {sample}...")
        else:
            sample = result.embeddings[0][:5]
            console.print("\n[bold]Sample embedding (first 5 dimensions):[/bold]")
            console.print(f"  {sample}...")

        print_success("Embedding test passed")
    except Exception as e:
        print_error(f"Embedding failed: {e}")


# Export all commands
completion_commands = [
    test_completion,
    test_streaming,
    test_structured_output,
    test_multi_turn_conversation,
    test_embedding,
]
