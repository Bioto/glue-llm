"""GlueLLM Command-Line Interface.

This module provides CLI commands for testing, demonstrating, and running
GlueLLM functionality from the command line using Click.

Available Commands:
    - test_completion: Test basic completion functionality
    - test_tool_call: Test tool calling with weather example
    - run_tests: Run the test suite with various options
    - demo: Run interactive demos of core features
    - examples: Run example scripts from examples/
    - test_iterative_workflow: Test iterative refinement workflow
    - test_pipeline_workflow: Test pipeline workflow
    - test_debate_workflow: Test debate workflow
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
        from source.executors import AgentExecutor
        from source.models.agent import Agent
        from source.models.prompt import SystemPrompt
        from source.models.workflow import CriticConfig, IterativeConfig
        from source.workflows.iterative import IterativeRefinementWorkflow

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
        from source.executors import AgentExecutor
        from source.models.agent import Agent
        from source.models.prompt import SystemPrompt
        from source.workflows.pipeline import PipelineWorkflow

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
        from source.executors import AgentExecutor
        from source.models.agent import Agent
        from source.models.prompt import SystemPrompt
        from source.workflows.debate import DebateConfig, DebateWorkflow

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


if __name__ == "__main__":
    cli()
