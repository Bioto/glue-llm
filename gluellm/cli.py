"""GlueLLM Command-Line Interface.

This module provides CLI commands for testing, demonstrating, and running
GlueLLM functionality from the command line using Click.

Available Commands:
    Basic Tests:
    - test_completion: Test basic completion functionality
    - test_tool_call: Test tool calling with weather example
    - test_streaming: Test streaming completion functionality
    - test_structured_output: Test structured output with Pydantic models
    - test_multi_turn_conversation: Test conversation memory

    Advanced Features:
    - test_batch_processing: Test parallel batch processing
    - test_hooks: Test hooks system for lifecycle events
    - test_correlation_ids: Test correlation ID tracking
    - test_api_key_pool: Test API key pooling
    - test_telemetry: Test OpenTelemetry integration

    Performance & Reliability:
    - test_error_handling: Test error handling and retries
    - test_rate_limiting: Test rate limiting functionality
    - test_timeout: Test timeout handling
    - test_tool_without_execution: Test tool registration without execution
    - test_different_models: Test switching between models

    Other:
    - run_tests: Run the test suite with various options
    - demo: Run interactive demos of core features
    - examples: Run example scripts from examples/
    - test_iterative_workflow: Test iterative refinement workflow
    - test_pipeline_workflow: Test pipeline workflow
    - test_debate_workflow: Test debate workflow
    - test_reflection_workflow: Test reflection workflow
    - test_chain_of_density_workflow: Test chain of density workflow
    - test_socratic_workflow: Test Socratic workflow
    - test_rag_workflow: Test RAG workflow
    - test_round_robin_workflow: Test round-robin workflow
    - test_consensus_workflow: Test consensus workflow
    - test_hierarchical_workflow: Test hierarchical workflow
    - test_map_reduce_workflow: Test MapReduce workflow
    - test_react_workflow: Test ReAct workflow
    - test_mixture_of_experts_workflow: Test Mixture of Experts workflow
    - test_constitutional_workflow: Test Constitutional workflow
    - test_tree_of_thoughts_workflow: Test Tree of Thoughts workflow
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

    from gluellm.config import settings
    from gluellm.models.config import RequestConfig
    from gluellm.models.conversation import Role
    from gluellm.models.prompt import SystemPrompt

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

    from gluellm.config import settings
    from gluellm.models.config import RequestConfig
    from gluellm.models.conversation import Role
    from gluellm.models.prompt import SystemPrompt

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
@click.option("--message", "-m", default="Tell me a short story about a robot", help="Message to stream")
def test_streaming(message: str) -> None:
    """Test streaming completion functionality.

    Demonstrates real-time streaming of LLM responses, displaying
    tokens as they are generated by the model.

    Args:
        message: The prompt message to send to the model

    Examples:
        glm test-streaming
        glm test-streaming -m "Count to 10"
    """
    import asyncio

    async def run_streaming_test():
        from gluellm.api import GlueLLM

        console.print("[bold cyan]Testing Streaming Completion[/bold cyan]\n")
        console.print(f"[bold]Message:[/bold] {message}\n")
        console.print("[yellow]Streaming response...[/yellow]\n")

        client = GlueLLM(system_prompt="You are a helpful assistant. Be concise and engaging.")

        console.print("[bold]Response:[/bold] ", end="")

        full_response = ""
        async for chunk in client.stream_complete(message):
            # Print each chunk's content as it arrives
            if chunk.content:
                console.print(chunk.content, end="", markup=False)
                full_response += chunk.content

            # Check if streaming is done
            if chunk.done:
                break

        console.print("\n")  # New line after streaming completes
        console.print("\n[bold green]✓ Streaming completed![/bold green]")
        console.print(f"[bold]Total characters:[/bold] {len(full_response)}")

    asyncio.run(run_streaming_test())


@cli.command()
@click.option("--data", "-d", default="Tokyo, Japan has 14 million people", help="Data to extract")
def test_structured_output(data: str) -> None:
    """Test structured output with Pydantic models.

    Demonstrates extracting structured data from text using Pydantic models
    for type-safe, validated responses.

    Args:
        data: Text data to extract structured information from

    Examples:
        glm test-structured-output
        glm test-structured-output -d "Paris, France has 2 million people"
    """
    import asyncio

    async def run_structured_test():
        from typing import Annotated

        from pydantic import BaseModel, Field

        from gluellm.api import structured_complete

        console.print("[bold cyan]Testing Structured Output[/bold cyan]\n")
        console.print(f"[bold]Input data:[/bold] {data}\n")

        class CityInfo(BaseModel):
            city: Annotated[str, Field(description="City name")]
            country: Annotated[str, Field(description="Country name")]
            population: Annotated[int, Field(description="Population in millions")]

        console.print("[yellow]Extracting structured data...[/yellow]\n")

        result = await structured_complete(
            user_message=f"Extract city information from: {data}",
            response_format=CityInfo,
        )

        console.print("[bold green]✓ Extraction completed![/bold green]\n")
        console.print(f"[bold]City:[/bold] {result.city}")
        console.print(f"[bold]Country:[/bold] {result.country}")
        console.print(f"[bold]Population:[/bold] {result.population:,} million")
        console.print(f"\n[bold]Type:[/bold] {type(result).__name__}")

    asyncio.run(run_structured_test())


@cli.command()
@click.option("--count", "-c", default=5, type=int, help="Number of requests to process")
@click.option("--concurrent", "-n", default=3, type=int, help="Max concurrent requests")
def test_batch_processing(count: int, concurrent: int) -> None:
    """Test batch processing with parallel requests.

    Demonstrates processing multiple LLM requests in parallel with
    configurable concurrency limits and error handling.

    Args:
        count: Number of requests in the batch
        concurrent: Maximum number of concurrent requests

    Examples:
        glm test-batch-processing
        glm test-batch-processing -c 10 -n 5
    """
    import asyncio

    async def run_batch_test():
        from gluellm.batch import batch_complete
        from gluellm.models.batch import BatchConfig, BatchRequest

        console.print("[bold cyan]Testing Batch Processing[/bold cyan]\n")
        console.print(f"[bold]Total requests:[/bold] {count}")
        console.print(f"[bold]Max concurrent:[/bold] {concurrent}\n")

        # Create batch requests
        requests = [BatchRequest(user_message=f"What is {i} + {i}? Answer briefly.") for i in range(1, count + 1)]

        console.print("[yellow]Processing batch...[/yellow]\n")

        response = await batch_complete(
            requests=requests,
            config=BatchConfig(max_concurrent=concurrent),
        )

        console.print("[bold green]✓ Batch completed![/bold green]\n")
        console.print(f"[bold]Total requests:[/bold] {response.total_requests}")
        console.print(f"[bold]Successful:[/bold] {response.successful_requests}")
        console.print(f"[bold]Failed:[/bold] {response.failed_requests}")
        console.print(f"[bold]Total time:[/bold] {response.total_elapsed_time:.2f}s")

        if response.total_tokens_used:
            console.print(f"[bold]Total tokens:[/bold] {response.total_tokens_used.get('total', 0):,}")

        console.print("\n[bold]Sample results:[/bold]")
        for i, result in enumerate(response.results[:3], 1):
            if result.success:
                preview = result.response[:60] if result.response else ""
                console.print(f"  {i}. {preview}...")
            else:
                console.print(f"  {i}. [red]Error: {result.error}[/red]")

    asyncio.run(run_batch_test())


@cli.command()
@click.option("--turns", "-t", default=3, type=int, help="Number of conversation turns")
def test_multi_turn_conversation(turns: int) -> None:
    """Test multi-turn conversation with memory.

    Demonstrates maintaining conversation context across multiple
    interactions, allowing the model to reference previous messages.

    Args:
        turns: Number of conversation turns to simulate

    Examples:
        glm test-multi-turn-conversation
        glm test-multi-turn-conversation -t 5
    """
    import asyncio

    async def run_conversation_test():
        from gluellm.api import GlueLLM

        console.print("[bold cyan]Testing Multi-turn Conversation[/bold cyan]\n")
        console.print(f"[bold]Conversation turns:[/bold] {turns}\n")

        client = GlueLLM(system_prompt="You are a helpful assistant with a good memory.")

        conversation_prompts = [
            "My name is Alice and I love Python programming.",
            "What's my name?",
            "What programming language do I love?",
            "Can you write me a short Python function?",
            "What was the first thing I told you?",
        ]

        for i in range(min(turns, len(conversation_prompts))):
            prompt = conversation_prompts[i]
            console.print(f"[bold]Turn {i + 1}:[/bold] {prompt}")

            result = await client.complete(prompt)

            console.print(f"[dim]Response:[/dim] {result.final_response}\n")

        console.print("[bold green]✓ Conversation completed![/bold green]")
        console.print(f"[bold]Total messages in history:[/bold] {len(client._conversation.messages)}")

    asyncio.run(run_conversation_test())


@cli.command()
def test_error_handling() -> None:
    """Test error handling and retry mechanisms.

    Demonstrates how GlueLLM handles various error conditions including
    invalid requests, rate limits (simulated), and error classification.

    Examples:
        glm test-error-handling
    """
    import asyncio

    async def run_error_test():
        from gluellm.api import GlueLLM

        console.print("[bold cyan]Testing Error Handling[/bold cyan]\n")

        # Test 1: Invalid model
        console.print("[bold]Test 1: Invalid model[/bold]")
        try:
            client = GlueLLM(model="invalid:nonexistent-model")
            await client.complete("Hello")
            console.print("[red]✗ Should have raised an error[/red]")
        except Exception as e:
            console.print(f"[green]✓ Caught error:[/green] {type(e).__name__}: {str(e)[:80]}...")

        # Test 2: Empty message handling
        console.print("\n[bold]Test 2: Empty message[/bold]")
        try:
            client = GlueLLM()
            result = await client.complete("")
            console.print(f"[green]✓ Handled empty message:[/green] {result.final_response[:60]}...")
        except Exception as e:
            console.print(f"[yellow]⚠ Error with empty message:[/yellow] {type(e).__name__}")

        # Test 3: Very long message (token limit)
        console.print("\n[bold]Test 3: Very long message[/bold]")
        try:
            client = GlueLLM()
            long_message = "test " * 100000  # Very long message
            console.print(f"[dim]Message length:[/dim] {len(long_message)} characters")
            result = await client.complete(long_message[:1000])  # Truncate for safety
            console.print("[green]✓ Handled long message[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error:[/yellow] {type(e).__name__}: {str(e)[:60]}...")

        console.print("\n[bold green]✓ Error handling tests completed![/bold green]")

    asyncio.run(run_error_test())


@cli.command()
def test_hooks() -> None:
    """Test hooks system for request lifecycle events.

    Demonstrates registering and using hooks to intercept and monitor
    requests at various stages (pre/post executor, pre/post workflow).

    Examples:
        glm test-hooks
    """
    import asyncio

    async def run_hooks_test():
        from gluellm.executors import SimpleExecutor
        from gluellm.hooks.utils import normalize_whitespace, remove_pii
        from gluellm.models.hook import HookConfig, HookErrorStrategy, HookRegistry, HookStage

        console.print("[bold cyan]Testing Hooks System[/bold cyan]\n")

        # Create hook registry
        registry = HookRegistry()

        # Add PII removal hook (pre-executor)
        registry.add_hook(
            HookStage.PRE_EXECUTOR,
            HookConfig(
                handler=remove_pii,
                name="remove_pii",
                error_strategy=HookErrorStrategy.SKIP,
            ),
        )

        # Add whitespace normalization hook (post-executor)
        registry.add_hook(
            HookStage.POST_EXECUTOR,
            HookConfig(
                handler=normalize_whitespace,
                name="normalize_whitespace",
                error_strategy=HookErrorStrategy.SKIP,
            ),
        )

        console.print("[bold]Hooks configured:[/bold]")
        console.print("  1. PRE_EXECUTOR: remove_pii")
        console.print("  2. POST_EXECUTOR: normalize_whitespace\n")

        # Create executor with hooks
        executor = SimpleExecutor(
            system_prompt="You are a helpful assistant.",
            hook_registry=registry,
        )

        # Test query with PII
        query = "My email is john@example.com and phone is 555-1234. What is 2+2?"
        console.print(f"[bold]Query (with PII):[/bold] {query}\n")
        console.print("[yellow]Executing with hooks...[/yellow]\n")

        result = await executor.execute(query)

        console.print(f"[bold]Response:[/bold] {result}")
        console.print("\n[bold green]✓ Hooks test completed![/bold green]")
        console.print("[dim]PII should have been removed before sending to LLM[/dim]")

    asyncio.run(run_hooks_test())


@cli.command()
def test_correlation_ids() -> None:
    """Test correlation ID tracking for request tracing.

    Demonstrates using correlation IDs to track requests through
    the system for debugging and observability.

    Examples:
        glm test-correlation-ids
    """
    import asyncio

    async def run_correlation_test():
        from gluellm.api import complete

        console.print("[bold cyan]Testing Correlation IDs[/bold cyan]\n")

        # Test 1: Auto-generated correlation ID
        console.print("[bold]Test 1: Auto-generated correlation ID[/bold]")
        result1 = await complete("What is 2+2?", correlation_id=None)
        console.print("[green]✓ Request completed[/green]")
        console.print(f"[dim]Response:[/dim] {result1.final_response[:60]}...")

        # Test 2: Custom correlation ID
        console.print("\n[bold]Test 2: Custom correlation ID[/bold]")
        custom_id = "my-custom-request-123"
        console.print(f"[dim]Using correlation ID:[/dim] {custom_id}")
        result2 = await complete("What is 3+3?", correlation_id=custom_id)
        console.print("[green]✓ Request completed with custom ID[/green]")
        console.print(f"[dim]Response:[/dim] {result2.final_response[:60]}...")

        # Test 3: Multiple requests with different IDs
        console.print("\n[bold]Test 3: Multiple requests[/bold]")
        requests = ["What is 4+4?", "What is 5+5?"]
        for i, msg in enumerate(requests, 1):
            corr_id = f"batch-request-{i}"
            _result = await complete(msg, correlation_id=corr_id)
            console.print(f"[green]✓ Request {i}:[/green] {corr_id}")

        console.print("\n[bold green]✓ Correlation ID tests completed![/bold green]")
        console.print("[dim]Check logs for correlation ID traces[/dim]")

    asyncio.run(run_correlation_test())


@cli.command()
@click.option("--keys", "-k", default=2, type=int, help="Number of API keys to simulate")
def test_api_key_pool(keys: int) -> None:
    """Test API key pooling for load distribution.

    Demonstrates using multiple API keys to distribute load and
    increase rate limits across keys.

    Args:
        keys: Number of API keys to simulate in the pool

    Examples:
        glm test-api-key-pool
        glm test-api-key-pool -k 3
    """
    import asyncio

    async def run_key_pool_test():
        console.print("[bold cyan]Testing API Key Pool[/bold cyan]\n")
        console.print(f"[bold]API keys in pool:[/bold] {keys}\n")

        console.print("[yellow]Note:[/yellow] This test demonstrates the API key pool structure.")
        console.print("[yellow]To use with real keys, set them in BatchConfig.api_keys[/yellow]\n")

        from gluellm.api_key_pool import APIKeyConfig, APIKeyPool

        # Create simulated key pool
        key_configs = [
            APIKeyConfig(
                key=f"sk-test-key-{i}",
                provider="openai",
                requests_per_minute=3500,
                burst=10,
            )
            for i in range(1, keys + 1)
        ]

        pool = APIKeyPool(keys=key_configs)

        console.print("[bold]Pool statistics:[/bold]")
        console.print(f"  Total keys: {len(pool._keys_by_provider.get('openai', []))}")
        console.print("  Provider: openai")

        # Simulate key acquisition
        console.print("\n[bold]Simulating key acquisition:[/bold]")
        for i in range(min(keys + 1, 5)):
            key = pool.get_key("openai")
            if key:
                console.print(f"  {i + 1}. Acquired: {key[:20]}...")
            else:
                console.print(f"  {i + 1}. [yellow]No keys available[/yellow]")

        console.print("\n[bold green]✓ API key pool test completed![/bold green]")

    asyncio.run(run_key_pool_test())


@cli.command()
def test_telemetry() -> None:
    """Test OpenTelemetry integration for tracing.

    Demonstrates OpenTelemetry tracing integration for monitoring
    and observability of LLM requests.

    Examples:
        glm test-telemetry
    """
    import asyncio

    async def run_telemetry_test():
        console.print("[bold cyan]Testing OpenTelemetry Integration[/bold cyan]\n")

        try:
            from gluellm.telemetry import configure_tracing, is_mlflow_enabled, is_tracing_enabled

            # Check current tracing status
            console.print(f"[bold]Tracing enabled:[/bold] {is_tracing_enabled()}")
            console.print(f"[bold]MLflow enabled:[/bold] {is_mlflow_enabled()}")

            if not is_tracing_enabled():
                console.print("\n[yellow]⚠ OpenTelemetry tracing not enabled[/yellow]")
                console.print("[dim]Set GLUELLM_ENABLE_TRACING=true to enable[/dim]")

                # Try to configure tracing
                console.print("\n[yellow]Attempting to configure tracing...[/yellow]")
                configure_tracing()

                if is_tracing_enabled():
                    console.print("[green]✓ Tracing configured successfully[/green]")
                else:
                    console.print("[yellow]⚠ Tracing could not be enabled[/yellow]")

            if is_tracing_enabled():
                # Make a traced request
                from gluellm.api import complete

                console.print("\n[yellow]Making traced request...[/yellow]\n")

                result = await complete("What is 2+2? Answer briefly.")

                console.print(f"[bold]Response:[/bold] {result.final_response}")
                console.print("\n[green]✓ Request traced successfully[/green]")
                console.print("[dim]Check your OpenTelemetry backend for traces[/dim]")

            console.print("\n[bold]Configuration:[/bold]")
            console.print("  Enable tracing: GLUELLM_ENABLE_TRACING=true")
            console.print("  OTLP endpoint: OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318")
            console.print("  MLflow URI: GLUELLM_MLFLOW_TRACKING_URI=http://localhost:5000")

        except ImportError as e:
            console.print(f"[red]✗ Required packages not installed:[/red] {e}")
            console.print("[dim]Install with: pip install opentelemetry-api opentelemetry-sdk mlflow[/dim]")

        console.print("\n[bold green]✓ Telemetry test completed![/bold green]")

    asyncio.run(run_telemetry_test())


@cli.command()
@click.option("--requests", "-r", default=5, type=int, help="Number of rapid requests")
def test_rate_limiting(requests: int) -> None:
    """Test rate limiting functionality.

    Demonstrates built-in rate limiting to prevent exceeding
    provider rate limits.

    Args:
        requests: Number of rapid requests to make

    Examples:
        glm test-rate-limiting
        glm test-rate-limiting -r 10
    """
    import asyncio
    import time

    async def run_rate_limit_test():
        from gluellm.api import GlueLLM

        console.print("[bold cyan]Testing Rate Limiting[/bold cyan]\n")
        console.print(f"[bold]Rapid requests:[/bold] {requests}\n")

        client = GlueLLM()

        console.print("[yellow]Making rapid requests...[/yellow]\n")

        start_time = time.time()
        request_times = []

        for i in range(requests):
            request_start = time.time()
            _result = await client.complete(f"What is {i}+1? Answer with just the number.")
            request_end = time.time()

            request_times.append(request_end - request_start)
            console.print(f"  {i + 1}. Completed in {request_end - request_start:.2f}s")

        total_time = time.time() - start_time

        console.print("\n[bold green]✓ Rate limiting test completed![/bold green]")
        console.print(f"[bold]Total time:[/bold] {total_time:.2f}s")
        console.print(f"[bold]Average time per request:[/bold] {sum(request_times) / len(request_times):.2f}s")
        console.print(f"[bold]Requests per second:[/bold] {requests / total_time:.2f}")

    asyncio.run(run_rate_limit_test())


@cli.command()
@click.option("--timeout", "-t", default=5.0, type=float, help="Timeout in seconds")
def test_timeout(timeout: float) -> None:
    """Test timeout handling for requests.

    Demonstrates configurable timeouts for LLM requests to prevent
    hanging requests.

    Args:
        timeout: Timeout value in seconds

    Examples:
        glm test-timeout
        glm test-timeout -t 10.0
    """
    import asyncio

    async def run_timeout_test():
        from gluellm.api import complete

        console.print("[bold cyan]Testing Timeout Handling[/bold cyan]\n")
        console.print(f"[bold]Timeout:[/bold] {timeout}s\n")

        # Test 1: Normal request within timeout
        console.print("[bold]Test 1: Normal request[/bold]")
        try:
            result = await complete("What is 2+2? Answer briefly.", timeout=timeout)
            console.print(f"[green]✓ Completed within timeout:[/green] {result.final_response[:60]}...")
        except TimeoutError:
            console.print("[red]✗ Request timed out[/red]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error:[/yellow] {type(e).__name__}")

        # Test 2: Very short timeout (might timeout)
        console.print("\n[bold]Test 2: Very short timeout (1s)[/bold]")
        try:
            result = await complete(
                "Write a long essay about the history of computing.",
                timeout=1.0,
            )
            console.print(f"[green]✓ Completed:[/green] {result.final_response[:60]}...")
        except TimeoutError:
            console.print("[yellow]⚠ Request timed out (expected for short timeout)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error:[/yellow] {type(e).__name__}")

        console.print("\n[bold green]✓ Timeout test completed![/bold green]")

    asyncio.run(run_timeout_test())


@cli.command()
def test_tool_without_execution() -> None:
    """Test tool registration without automatic execution.

    Demonstrates registering tools but controlling execution manually,
    useful for custom tool execution logic or approval workflows.

    Examples:
        glm test-tool-without-execution
    """
    import asyncio

    async def run_tool_no_exec_test():
        from gluellm.api import GlueLLM

        console.print("[bold cyan]Testing Tool Without Execution[/bold cyan]\n")

        def calculate(a: int, b: int, operation: str = "add") -> int:
            """Perform a calculation."""
            if operation == "add":
                return a + b
            if operation == "multiply":
                return a * b
            return 0

        client = GlueLLM(
            system_prompt="You are a calculator assistant. Use the calculate tool when needed.",
            tools=[calculate],
        )

        console.print("[bold]Making request with execute_tools=False[/bold]\n")

        result = await client.complete(
            "What is 5 plus 3?",
            execute_tools=False,  # Don't execute automatically
        )

        console.print(f"[bold]Tool calls requested:[/bold] {result.tool_calls_made}")
        console.print(f"[bold]Response:[/bold] {result.final_response}")

        if result.tool_execution_history:
            console.print("\n[bold]Tool execution history:[/bold]")
            for i, execution in enumerate(result.tool_execution_history, 1):
                console.print(f"  {i}. Tool: {execution.get('tool_name')}")
                console.print(f"     Args: {execution.get('arguments')}")

        console.print("\n[bold green]✓ Test completed![/bold green]")
        console.print("[dim]Note: With execute_tools=False, tools are registered but not called[/dim]")

    asyncio.run(run_tool_no_exec_test())


@cli.command()
@click.option("--models", "-m", multiple=True, default=["openai:gpt-4o-mini", "openai:gpt-3.5-turbo"])
def test_different_models(models: tuple[str, ...]) -> None:
    """Test switching between different models/providers.

    Demonstrates using different models and providers, showing how
    to switch models for different use cases.

    Args:
        models: List of model identifiers to test

    Examples:
        glm test-different-models
        glm test-different-models -m openai:gpt-4o -m anthropic:claude-3-sonnet
    """
    import asyncio

    async def run_models_test():
        from gluellm.api import GlueLLM

        console.print("[bold cyan]Testing Different Models[/bold cyan]\n")
        console.print(f"[bold]Models to test:[/bold] {len(models)}\n")

        prompt = "What is 2+2? Answer in exactly 3 words."

        for i, model in enumerate(models, 1):
            console.print(f"[bold]Test {i}: {model}[/bold]")

            try:
                client = GlueLLM(model=model)
                result = await client.complete(prompt)

                console.print("[green]✓ Success[/green]")
                console.print(f"[dim]Response:[/dim] {result.final_response}")

                if result.tokens_used:
                    console.print(f"[dim]Tokens:[/dim] {result.tokens_used.get('total', 0)}")

            except Exception as e:
                console.print(f"[red]✗ Error:[/red] {type(e).__name__}: {str(e)[:60]}...")

            console.print()

        console.print("[bold green]✓ Model comparison completed![/bold green]")

    asyncio.run(run_models_test())


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

        from gluellm.api import GlueLLM, complete, structured_complete

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


@cli.command()
@click.option(
    "--input",
    "-i",
    default="Write a short article about Python async programming",
    help="Input prompt for the workflow",
)
@click.option("--iterations", "-n", default=2, type=int, help="Maximum number of iterations")
@click.option("--critics", "-c", default=1, type=int, help="Number of critics to use (1-3)")
def test_iterative_workflow(input: str, iterations: int, critics: int) -> None:
    """Test iterative refinement workflow with producer and critic agents.

    Demonstrates the iterative refinement workflow where a producer agent
    creates content and one or more critic agents provide feedback in parallel.
    The workflow iterates until convergence or max iterations.

    Examples:
        glm test-iterative-workflow
        glm test-iterative-workflow -i "Write about AI" -n 3 -c 2
    """
    import asyncio

    async def run_iterative_workflow():
        from gluellm.executors import AgentExecutor
        from gluellm.models.agent import Agent
        from gluellm.models.prompt import SystemPrompt
        from gluellm.models.workflow import CriticConfig, IterativeConfig
        from gluellm.workflows.iterative import IterativeRefinementWorkflow

        console.print("[bold cyan]Testing Iterative Refinement Workflow[/bold cyan]\n")

        # Define the producer agent (writer)
        writer = Agent(
            name="Writer",
            description="Technical writer who creates content",
            system_prompt=SystemPrompt(
                content="""You are a technical writer. You write clear, engaging, and well-structured content.
                When given feedback, you carefully consider it and revise your work to address the concerns raised."""
            ),
            tools=[],
            max_tool_iterations=5,
        )

        # Define critic agents
        critic_configs = []
        if critics >= 1:
            grammar_critic = Agent(
                name="Grammar Critic",
                description="Reviews grammar, spelling, and clarity",
                system_prompt=SystemPrompt(
                    content="""You are a grammar and clarity critic. You focus on:
                    - Grammar and spelling errors
                    - Sentence structure and clarity
                    - Readability and flow
                    Provide specific, actionable feedback."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            critic_configs.append(
                CriticConfig(
                    executor=AgentExecutor(grammar_critic),
                    specialty="grammar and clarity",
                    goal="Optimize for readability and eliminate errors",
                )
            )

        if critics >= 2:
            style_critic = Agent(
                name="Style Critic",
                description="Reviews writing style and engagement",
                system_prompt=SystemPrompt(
                    content="""You are a writing style critic. You focus on:
                    - Writing style and tone
                    - Engagement and narrative flow
                    - Structure and organization
                    Provide specific, actionable feedback."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            critic_configs.append(
                CriticConfig(
                    executor=AgentExecutor(style_critic),
                    specialty="writing style and flow",
                    goal="Ensure engaging, well-structured narrative",
                )
            )

        if critics >= 3:
            accuracy_critic = Agent(
                name="Accuracy Critic",
                description="Verifies technical accuracy",
                system_prompt=SystemPrompt(
                    content="""You are a technical accuracy critic. You focus on:
                    - Technical correctness of claims
                    - Factual accuracy
                    - Logical consistency
                    Provide specific, actionable feedback."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            critic_configs.append(
                CriticConfig(
                    executor=AgentExecutor(accuracy_critic),
                    specialty="technical accuracy",
                    goal="Verify all technical claims are correct and logical",
                )
            )

        # Create workflow
        workflow = IterativeRefinementWorkflow(
            producer=AgentExecutor(writer),
            critics=critic_configs,
            config=IterativeConfig(max_iterations=iterations),
        )

        console.print(f"[bold]Input:[/bold] {input}")
        console.print(f"[bold]Iterations:[/bold] {iterations}")
        console.print(f"[bold]Critics:[/bold] {len(critic_configs)}\n")
        console.print("[yellow]Executing workflow...[/yellow]\n")

        # Execute workflow
        result = await workflow.execute(input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Iterations completed:[/bold] {result.iterations}")
        console.print(f"[bold]Converged:[/bold] {result.metadata.get('converged', False)}")
        console.print(f"[bold]Number of critics:[/bold] {result.metadata.get('num_critics', 0)}\n")
        console.print("[bold]Final Output:[/bold]")
        console.print("─" * 60)
        console.print(result.final_output)
        console.print("─" * 60)
        console.print(f"\n[bold]Total interactions:[/bold] {len(result.agent_interactions)}")

    asyncio.run(run_iterative_workflow())


@cli.command()
@click.option("--input", "-i", default="Topic: The benefits of async programming", help="Input prompt for the workflow")
@click.option("--stages", "-s", default=3, type=int, help="Number of pipeline stages (1-3)")
def test_pipeline_workflow(input: str, stages: int) -> None:
    """Test pipeline workflow with sequential agent stages.

    Demonstrates the pipeline workflow where agents execute sequentially,
    with the output of one agent becoming the input to the next.

    Examples:
        glm test-pipeline-workflow
        glm test-pipeline-workflow -i "Write about Python" -s 2
    """
    import asyncio

    async def run_pipeline_workflow():
        from gluellm.executors import AgentExecutor
        from gluellm.models.agent import Agent
        from gluellm.models.prompt import SystemPrompt
        from gluellm.workflows.pipeline import PipelineWorkflow

        console.print("[bold cyan]Testing Pipeline Workflow[/bold cyan]\n")

        # Define stage agents
        stage_list = []

        if stages >= 1:
            researcher = Agent(
                name="Researcher",
                description="Researches and gathers information",
                system_prompt=SystemPrompt(
                    content="""You are a research assistant. You gather and organize information
                    about topics. Provide a comprehensive research summary."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            stage_list.append(("research", AgentExecutor(researcher)))

        if stages >= 2:
            writer = Agent(
                name="Writer",
                description="Writes content based on research",
                system_prompt=SystemPrompt(
                    content="""You are a technical writer. You write clear, engaging content
                    based on research materials provided to you."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            stage_list.append(("write", AgentExecutor(writer)))

        if stages >= 3:
            editor = Agent(
                name="Editor",
                description="Edits and polishes content",
                system_prompt=SystemPrompt(
                    content="""You are an editor. You review, edit, and polish written content
                    to improve clarity, flow, and quality."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            stage_list.append(("edit", AgentExecutor(editor)))

        # Create workflow
        workflow = PipelineWorkflow(stages=stage_list)

        console.print(f"[bold]Input:[/bold] {input}")
        console.print(f"[bold]Stages:[/bold] {len(stage_list)}")
        for i, (stage_name, _) in enumerate(stage_list, 1):
            console.print(f"  {i}. {stage_name}")
        console.print("\n[yellow]Executing workflow...[/yellow]\n")

        # Execute workflow
        result = await workflow.execute(input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Stages completed:[/bold] {result.iterations}")
        console.print("\n[bold]Final Output:[/bold]")
        console.print("─" * 60)
        console.print(result.final_output)
        console.print("─" * 60)
        console.print(f"\n[bold]Total interactions:[/bold] {len(result.agent_interactions)}")

        # Show stage outputs
        console.print("\n[bold]Stage Outputs:[/bold]")
        for i, interaction in enumerate(result.agent_interactions, 1):
            stage_name = interaction.get("stage", "unknown")
            output_preview = interaction.get("output", "")[:100]
            console.print(f"  {i}. {stage_name}: {output_preview}...")

    asyncio.run(run_pipeline_workflow())


@cli.command()
@click.option("--topic", "-t", default="Should AI be regulated?", help="Debate topic/question")
@click.option("--rounds", "-r", default=2, type=int, help="Number of debate rounds")
@click.option("--judge/--no-judge", default=True, help="Include judge for final decision")
def test_debate_workflow(topic: str, rounds: int, judge: bool) -> None:
    """Test debate workflow with multiple participants and optional judge.

    Demonstrates the debate workflow where multiple agents argue different
    perspectives on a topic, with an optional judge agent making a final decision.

    Examples:
        glm test-debate-workflow
        glm test-debate-workflow -t "Is remote work better?" -r 3 --no-judge
    """
    import asyncio

    async def run_debate_workflow():
        from gluellm.executors import AgentExecutor
        from gluellm.models.agent import Agent
        from gluellm.models.prompt import SystemPrompt
        from gluellm.workflows.debate import DebateConfig, DebateWorkflow

        console.print("[bold cyan]Testing Debate Workflow[/bold cyan]\n")

        # Define participant agents
        pro_agent = Agent(
            name="Proponent",
            description="Argues in favor of the topic",
            system_prompt=SystemPrompt(
                content="""You are a persuasive debater arguing in favor of positions.
                You present strong, logical arguments with supporting evidence.
                Be respectful but firm in your position."""
            ),
            tools=[],
            max_tool_iterations=5,
        )

        con_agent = Agent(
            name="Opponent",
            description="Argues against the topic",
            system_prompt=SystemPrompt(
                content="""You are a persuasive debater arguing against positions.
                You present strong, logical counter-arguments with supporting evidence.
                Be respectful but firm in your position."""
            ),
            tools=[],
            max_tool_iterations=5,
        )

        participants = [
            ("Pro", AgentExecutor(pro_agent)),
            ("Con", AgentExecutor(con_agent)),
        ]

        # Optional judge agent
        judge_executor = None
        if judge:
            judge_agent = Agent(
                name="Judge",
                description="Evaluates arguments and makes final decision",
                system_prompt=SystemPrompt(
                    content="""You are an impartial judge evaluating a debate.
                    You consider the strength of arguments, evidence, and logic
                    from both sides. Provide a fair, reasoned judgment."""
                ),
                tools=[],
                max_tool_iterations=5,
            )
            judge_executor = AgentExecutor(judge_agent)

        # Create workflow
        workflow = DebateWorkflow(
            participants=participants,
            judge=judge_executor,
            config=DebateConfig(max_rounds=rounds, judge_decides=judge),
        )

        console.print(f"[bold]Topic:[/bold] {topic}")
        console.print(f"[bold]Rounds:[/bold] {rounds}")
        console.print(f"[bold]Participants:[/bold] {len(participants)}")
        console.print(f"[bold]Judge:[/bold] {'Yes' if judge else 'No'}\n")
        console.print("[yellow]Executing workflow...[/yellow]\n")

        # Execute workflow
        result = await workflow.execute(topic)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Rounds completed:[/bold] {result.iterations}")
        console.print(f"[bold]Judge used:[/bold] {result.metadata.get('judge_used', False)}\n")
        console.print("[bold]Final Output:[/bold]")
        console.print("─" * 60)
        console.print(result.final_output)
        console.print("─" * 60)
        console.print(f"\n[bold]Total interactions:[/bold] {len(result.agent_interactions)}")

        # Show debate structure
        console.print("\n[bold]Debate Structure:[/bold]")
        for _i, interaction in enumerate(result.agent_interactions, 1):
            if "round" in interaction:
                round_num = interaction.get("round", "?")
                participant = interaction.get("participant", "unknown")
                arg_preview = interaction.get("argument", "")[:80]
                console.print(f"  Round {round_num} - {participant}: {arg_preview}...")
            elif "stage" in interaction and interaction.get("stage") == "judgment":
                console.print(f"  [bold]Judgment:[/bold] {interaction.get('output', '')[:100]}...")

    asyncio.run(run_debate_workflow())


@cli.command()
@click.option("--input", "-i", default="Write an article about Python", help="Input prompt for the workflow")
@click.option("--reflections", "-r", default=2, help="Number of reflection iterations")
def test_reflection_workflow(input: str, reflections: int) -> None:
    """Test reflection workflow with self-critique and improvement.

    Demonstrates the reflection workflow where an agent critiques and
    improves its own output through iterative self-reflection.

    Examples:
        glm test-reflection-workflow
        glm test-reflection-workflow -i "Write about AI" -r 3
    """
    import asyncio

    async def run_reflection_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import ReflectionConfig
        from gluellm.workflows.reflection import ReflectionWorkflow

        console.print("[bold cyan]Testing Reflection Workflow[/bold cyan]\n")
        console.print(f"[dim]Input:[/dim] {input}")
        console.print(f"[dim]Reflections:[/dim] {reflections}\n")

        generator = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful writer.",
        )
        reflector = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a critical reviewer.",
        )

        workflow = ReflectionWorkflow(
            generator=generator,
            reflector=reflector,
            config=ReflectionConfig(max_reflections=reflections),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output}\n")
        console.print(f"[bold]Iterations:[/bold] {result.iterations}")
        console.print(f"[bold]Total interactions:[/bold] {len(result.agent_interactions)}")

    asyncio.run(run_reflection_workflow())


@cli.command()
@click.option("--input", "-i", default="Summarize this article", help="Input prompt for the workflow")
@click.option("--iterations", "-n", default=3, help="Number of density iterations")
def test_chain_of_density_workflow(input: str, iterations: int) -> None:
    """Test chain of density workflow.

    Demonstrates iteratively increasing content density.

    Examples:
        glm test-chain-of-density-workflow
        glm test-chain-of-density-workflow -i "Summarize" -n 5
    """
    import asyncio

    async def run_chain_of_density_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import ChainOfDensityConfig
        from gluellm.workflows.chain_of_density import ChainOfDensityWorkflow

        console.print("[bold cyan]Testing Chain of Density Workflow[/bold cyan]\n")
        console.print(f"[dim]Input:[/dim] {input}")
        console.print(f"[dim]Iterations:[/dim] {iterations}\n")

        generator = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful summarizer.",
        )

        workflow = ChainOfDensityWorkflow(
            generator=generator,
            config=ChainOfDensityConfig(num_iterations=iterations),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output}\n")
        console.print(f"[bold]Iterations:[/bold] {result.iterations}")

    asyncio.run(run_chain_of_density_workflow())


@cli.command()
@click.option("--topic", "-t", default="What is artificial intelligence?", help="Topic to explore")
@click.option("--exchanges", "-e", default=3, help="Number of question-answer exchanges")
def test_socratic_workflow(topic: str, exchanges: int) -> None:
    """Test Socratic workflow with question-answer dialogue.

    Demonstrates two agents engaging in Socratic dialogue.

    Examples:
        glm test-socratic-workflow
        glm test-socratic-workflow -t "What is Python?" -e 5
    """
    import asyncio

    async def run_socratic_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import SocraticConfig
        from gluellm.workflows.socratic import SocraticWorkflow

        console.print("[bold cyan]Testing Socratic Workflow[/bold cyan]\n")
        console.print(f"[dim]Topic:[/dim] {topic}")
        console.print(f"[dim]Exchanges:[/dim] {exchanges}\n")

        questioner = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a Socratic questioner.",
        )
        responder = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a thoughtful responder.",
        )

        workflow = SocraticWorkflow(
            questioner=questioner,
            responder=responder,
            config=SocraticConfig(max_exchanges=exchanges),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(topic)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output}\n")
        console.print(f"[bold]Exchanges:[/bold] {result.iterations}")

    asyncio.run(run_socratic_workflow())


@cli.command()
@click.option("--query", "-q", default="What is Python?", help="Query to answer")
def test_rag_workflow(query: str) -> None:
    """Test RAG (Retrieval-Augmented Generation) workflow.

    Demonstrates retrieval-augmented generation with context.

    Examples:
        glm test-rag-workflow
        glm test-rag-workflow -q "What is machine learning?"
    """
    import asyncio

    async def run_rag_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import RAGConfig
        from gluellm.workflows.rag import RAGWorkflow

        console.print("[bold cyan]Testing RAG Workflow[/bold cyan]\n")
        console.print(f"[dim]Query:[/dim] {query}\n")

        def mock_retriever(q: str) -> list[dict]:
            return [
                {"content": f"Context about {q}", "source": "doc1"},
                {"content": f"More information about {q}", "source": "doc2"},
            ]

        generator = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You answer questions using provided context.",
        )

        workflow = RAGWorkflow(
            retriever=mock_retriever,
            generator=generator,
            config=RAGConfig(max_retrieved_chunks=2),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(query)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output}\n")
        console.print(f"[bold]Retrieved chunks:[/bold] {result.metadata.get('retrieved_chunks', 0)}")

    asyncio.run(run_rag_workflow())


@cli.command()
@click.option("--input", "-i", default="Write an article", help="Input task")
@click.option("--rounds", "-r", default=2, help="Number of rounds")
@click.option("--agents", "-a", default=2, help="Number of agents")
def test_round_robin_workflow(input: str, rounds: int, agents: int) -> None:
    """Test round-robin workflow with collaborative agents.

    Demonstrates agents taking turns contributing.

    Examples:
        glm test-round-robin-workflow
        glm test-round-robin-workflow -i "Write about AI" -r 3 -a 3
    """
    import asyncio

    async def run_round_robin_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import RoundRobinConfig
        from gluellm.workflows.round_robin import RoundRobinWorkflow

        console.print("[bold cyan]Testing Round-Robin Workflow[/bold cyan]\n")
        console.print(f"[dim]Input:[/dim] {input}")
        console.print(f"[dim]Rounds:[/dim] {rounds}")
        console.print(f"[dim]Agents:[/dim] {agents}\n")

        agent_list = []
        for i in range(agents):
            agent_list.append(
                (
                    f"Agent{i + 1}",
                    SimpleExecutor(
                        model="openai:gpt-4o-mini",
                        system_prompt=f"You are Agent {i + 1}, a helpful contributor.",
                    ),
                )
            )

        workflow = RoundRobinWorkflow(
            agents=agent_list,
            config=RoundRobinConfig(max_rounds=rounds),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Rounds:[/bold] {result.iterations}")

    asyncio.run(run_round_robin_workflow())


@cli.command()
@click.option("--problem", "-p", default="Design a solution", help="Problem to solve")
@click.option("--rounds", "-r", default=2, help="Number of consensus rounds")
@click.option("--agents", "-a", default=3, help="Number of proposing agents")
def test_consensus_workflow(problem: str, rounds: int, agents: int) -> None:
    """Test consensus workflow with voting agents.

    Demonstrates agents proposing solutions and reaching consensus.

    Examples:
        glm test-consensus-workflow
        glm test-consensus-workflow -p "Solve X" -r 3 -a 4
    """
    import asyncio

    async def run_consensus_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import ConsensusConfig
        from gluellm.workflows.consensus import ConsensusWorkflow

        console.print("[bold cyan]Testing Consensus Workflow[/bold cyan]\n")
        console.print(f"[dim]Problem:[/dim] {problem}")
        console.print(f"[dim]Rounds:[/dim] {rounds}")
        console.print(f"[dim]Agents:[/dim] {agents}\n")

        proposers = []
        for i in range(agents):
            proposers.append(
                (
                    f"Agent{i + 1}",
                    SimpleExecutor(
                        model="openai:gpt-4o-mini",
                        system_prompt=f"You are Agent {i + 1}, proposing solutions.",
                    ),
                )
            )

        workflow = ConsensusWorkflow(
            proposers=proposers,
            config=ConsensusConfig(max_rounds=rounds, min_agreement_ratio=0.6),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(problem)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Consensus reached:[/bold] {result.metadata.get('consensus_reached', False)}")

    asyncio.run(run_consensus_workflow())


@cli.command()
@click.option("--task", "-t", default="Research and write a report", help="Task to decompose")
@click.option("--subtasks", "-s", default=3, help="Maximum subtasks")
def test_hierarchical_workflow(task: str, subtasks: int) -> None:
    """Test hierarchical workflow with manager and workers.

    Demonstrates task decomposition and parallel execution.

    Examples:
        glm test-hierarchical-workflow
        glm test-hierarchical-workflow -t "Complex task" -s 5
    """
    import asyncio

    async def run_hierarchical_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import HierarchicalConfig
        from gluellm.workflows.hierarchical import HierarchicalWorkflow

        console.print("[bold cyan]Testing Hierarchical Workflow[/bold cyan]\n")
        console.print(f"[dim]Task:[/dim] {task}")
        console.print(f"[dim]Max subtasks:[/dim] {subtasks}\n")

        manager = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a manager who breaks down tasks.",
        )
        worker1 = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a worker executing subtasks.",
        )
        worker2 = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a worker executing subtasks.",
        )

        workflow = HierarchicalWorkflow(
            manager=manager,
            workers=[("Worker1", worker1), ("Worker2", worker2)],
            config=HierarchicalConfig(max_subtasks=subtasks, parallel_workers=True),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(task)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Subtasks created:[/bold] {result.metadata.get('subtasks_created', 0)}")

    asyncio.run(run_hierarchical_workflow())


@cli.command()
@click.option("--input", "-i", default="Process this long document", help="Input to process")
@click.option("--chunk-size", "-c", default=500, help="Chunk size for splitting")
def test_map_reduce_workflow(input: str, chunk_size: int) -> None:
    """Test MapReduce workflow with parallel processing.

    Demonstrates parallel chunk processing and aggregation.

    Examples:
        glm test-map-reduce-workflow
        glm test-map-reduce-workflow -i "Long text..." -c 1000
    """
    import asyncio

    async def run_map_reduce_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import MapReduceConfig
        from gluellm.workflows.map_reduce import MapReduceWorkflow

        console.print("[bold cyan]Testing MapReduce Workflow[/bold cyan]\n")

        # Create long input if needed
        workflow_input = input
        if len(workflow_input) < chunk_size:
            workflow_input = (workflow_input + " ") * (chunk_size // len(workflow_input) + 1)

        console.print(f"[dim]Input length:[/dim] {len(workflow_input)} characters")
        console.print(f"[dim]Chunk size:[/dim] {chunk_size}\n")

        mapper = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You process text chunks.",
        )
        reducer = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You synthesize results.",
        )

        workflow = MapReduceWorkflow(
            mapper=mapper,
            reducer=reducer,
            config=MapReduceConfig(chunk_size=chunk_size, reduce_strategy="summarize"),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(workflow_input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Chunks processed:[/bold] {result.metadata.get('chunks_processed', 0)}")

    asyncio.run(run_map_reduce_workflow())


@cli.command()
@click.option("--question", "-q", default="What is the weather in Paris?", help="Question to solve")
@click.option("--steps", "-s", default=5, help="Maximum reasoning steps")
def test_react_workflow(question: str, steps: int) -> None:
    """Test ReAct (Reasoning + Acting) workflow.

    Demonstrates interleaved reasoning and action steps.

    Examples:
        glm test-react-workflow
        glm test-react-workflow -q "Solve this problem" -s 10
    """
    import asyncio

    async def run_react_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import ReActConfig
        from gluellm.workflows.react import ReActWorkflow

        console.print("[bold cyan]Testing ReAct Workflow[/bold cyan]\n")
        console.print(f"[dim]Question:[/dim] {question}")
        console.print(f"[dim]Max steps:[/dim] {steps}\n")

        reasoner = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You reason step by step and take actions.",
        )

        workflow = ReActWorkflow(
            reasoner=reasoner,
            config=ReActConfig(max_steps=steps, stop_on_final_answer=True),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(question)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Answer:[/bold]\n{result.final_output}\n")
        console.print(f"[bold]Steps taken:[/bold] {result.iterations}")

    asyncio.run(run_react_workflow())


@cli.command()
@click.option("--query", "-q", default="Calculate something", help="Query to route")
def test_mixture_of_experts_workflow(query: str) -> None:
    """Test Mixture of Experts workflow.

    Demonstrates routing to specialized experts.

    Examples:
        glm test-mixture-of-experts-workflow
        glm test-mixture-of-experts-workflow -q "Write code"
    """
    import asyncio

    async def run_moe_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import ExpertConfig, MoEConfig
        from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow

        console.print("[bold cyan]Testing Mixture of Experts Workflow[/bold cyan]\n")
        console.print(f"[dim]Query:[/dim] {query}\n")

        math_expert = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a math expert.",
        )
        code_expert = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You are a coding expert.",
        )

        workflow = MixtureOfExpertsWorkflow(
            experts=[
                ExpertConfig(
                    executor=math_expert,
                    specialty="mathematics",
                    description="Expert in math",
                    activation_keywords=["calculate", "math"],
                ),
                ExpertConfig(
                    executor=code_expert,
                    specialty="programming",
                    description="Expert in coding",
                    activation_keywords=["code", "program"],
                ),
            ],
            config=MoEConfig(routing_strategy="keyword", top_k=2),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(query)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Experts used:[/bold] {result.metadata.get('experts_used', 0)}")

    asyncio.run(run_moe_workflow())


@cli.command()
@click.option("--input", "-i", default="Write about AI safety", help="Input to generate")
def test_constitutional_workflow(input: str) -> None:
    """Test Constitutional AI workflow.

    Demonstrates principle-based generation and critique.

    Examples:
        glm test-constitutional-workflow
        glm test-constitutional-workflow -i "Write a response"
    """
    import asyncio

    async def run_constitutional_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import ConstitutionalConfig, Principle
        from gluellm.workflows.constitutional import ConstitutionalWorkflow

        console.print("[bold cyan]Testing Constitutional Workflow[/bold cyan]\n")
        console.print(f"[dim]Input:[/dim] {input}\n")

        generator = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You generate helpful content.",
        )
        critic = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You critique content against principles.",
        )

        workflow = ConstitutionalWorkflow(
            generator=generator,
            critic=critic,
            config=ConstitutionalConfig(
                principles=[
                    Principle(name="harmless", description="Should not harm", severity="critical"),
                    Principle(name="helpful", description="Should be helpful", severity="error"),
                ],
                max_revisions=2,
            ),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(input)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Revisions:[/bold] {result.iterations}")

    asyncio.run(run_constitutional_workflow())


@cli.command()
@click.option("--problem", "-p", default="Solve this problem", help="Problem to solve")
@click.option("--depth", "-d", default=2, help="Maximum tree depth")
@click.option("--branching", "-b", default=2, help="Branching factor")
def test_tree_of_thoughts_workflow(problem: str, depth: int, branching: int) -> None:
    """Test Tree of Thoughts workflow.

    Demonstrates exploring multiple reasoning paths.

    Examples:
        glm test-tree-of-thoughts-workflow
        glm test-tree-of-thoughts-workflow -p "Complex problem" -d 3 -b 3
    """
    import asyncio

    async def run_tot_workflow():
        from gluellm.executors import SimpleExecutor
        from gluellm.models.workflow import TreeOfThoughtsConfig
        from gluellm.workflows.tree_of_thoughts import TreeOfThoughtsWorkflow

        console.print("[bold cyan]Testing Tree of Thoughts Workflow[/bold cyan]\n")
        console.print(f"[dim]Problem:[/dim] {problem}")
        console.print(f"[dim]Depth:[/dim] {depth}")
        console.print(f"[dim]Branching:[/dim] {branching}\n")

        thinker = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You think through problems step by step.",
        )
        evaluator = SimpleExecutor(
            model="openai:gpt-4o-mini",
            system_prompt="You evaluate reasoning paths.",
        )

        workflow = TreeOfThoughtsWorkflow(
            thinker=thinker,
            evaluator=evaluator,
            config=TreeOfThoughtsConfig(
                branching_factor=branching,
                max_depth=depth,
                evaluation_strategy="score",
            ),
        )

        console.print("[yellow]Executing workflow...[/yellow]\n")
        result = await workflow.execute(problem)

        console.print("[bold green]✓ Workflow completed![/bold green]\n")
        console.print(f"[bold]Final Output:[/bold]\n{result.final_output[:200]}...\n")
        console.print(f"[bold]Depth explored:[/bold] {result.iterations}")

    asyncio.run(run_tot_workflow())


if __name__ == "__main__":
    cli()
