"""Tool-related CLI commands.

Commands for testing tool calling and execution functionality.
"""

import json

import click

from gluellm.cli.utils import (
    console,
    get_weather,
    print_error,
    print_header,
    print_result,
    print_step,
    print_success,
    run_async,
)


@click.command("test-tool-call")
def test_tool_call() -> None:
    """Test completion with automatic tool calling."""
    from typing import Annotated

    from any_llm import completion
    from pydantic import BaseModel, Field

    from gluellm.config import settings
    from gluellm.models.config import RequestConfig
    from gluellm.models.conversation import Role
    from gluellm.models.prompt import SystemPrompt

    class DefaultResponseFormat(BaseModel):
        response: Annotated[str, Field(description="The response to the request")]

    print_header("Test Tool Calling", f"Model: {settings.default_model}")

    request_config = RequestConfig(
        model=settings.default_model,
        system_prompt=SystemPrompt(content=settings.default_system_prompt, tools=[get_weather]),
        response_format=DefaultResponseFormat,
        tools=[get_weather],
    )
    request_config.add_message_to_conversation(Role.USER, "Get weather for Tokyo, Japan")

    print_step(1, 3, "Sending initial request with tool...")

    # Build initial messages
    messages = request_config.get_conversation()

    response = completion(
        messages=messages,
        model=request_config.model,
        response_format=request_config.response_format if not request_config.tools else None,
        tools=request_config.tools,
    )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print_step(2, 3, f"Tool called: {tool_call.function.name}")

        tool_args = json.loads(tool_call.function.arguments)
        tool_result = get_weather(**tool_args)

        # Append assistant message and tool result to messages list
        messages.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

        print_step(3, 3, "Getting final response...")
        response = completion(
            messages=messages,
            model=request_config.model,
            response_format=request_config.response_format if not request_config.tools else None,
            tools=request_config.tools,
        )

    print_result("Response", response.choices[0].message.content or "")
    print_success("Tool calling test passed")


@click.command("test-tool-without-execution")
def test_tool_without_execution() -> None:
    """Test tool registration without auto-execution."""
    from gluellm.api import complete

    print_header("Test Tool Without Execution")

    async def run_test():
        return await complete(
            "What's the weather in Paris?",
            tools=[get_weather],
            execute_tools=False,
        )

    try:
        result = run_async(run_test())
        print_result("Response", result.final_response)
        console.print(f"Tool calls detected: {result.tool_calls_made}")
        print_success("Test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-batch-processing")
@click.option("--count", "-c", default=5, type=int, help="Number of requests to process")
def test_batch_processing(count: int) -> None:
    """Test batch processing with parallel requests."""
    from gluellm.batch import batch_complete

    print_header("Test Batch Processing", f"Requests: {count}")

    requests = [{"user_message": f"What is {i} * {i}?", "request_id": f"req_{i}"} for i in range(1, count + 1)]

    async def run_batch():
        return await batch_complete(requests, max_concurrent=3)

    try:
        results = run_async(run_batch())
        for req_id, result in results.items():
            if result.get("success"):
                console.print(f"  {req_id}: {result.get('response', '')[:50]}...")
            else:
                console.print(f"  {req_id}: [red]Error[/red]")
        print_success(f"Batch processing completed: {len(results)} results")
    except Exception as e:
        print_error(f"Batch processing failed: {e}")


# Export all commands
tools_commands = [
    test_tool_call,
    test_tool_without_execution,
    test_batch_processing,
]
