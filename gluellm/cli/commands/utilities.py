"""Utility CLI commands.

Commands for demos, examples, running tests, and other utilities.
"""

import subprocess
import sys

import click

from gluellm.cli.utils import (
    console,
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    run_async,
)


@click.command("run-tests")
@click.option("--test", "-t", help="Specific test to run")
@click.option("--class-name", "-c", help="Specific test class to run")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-integration", is_flag=True, help="Skip integration tests")
def run_tests(test: str | None, class_name: str | None, verbose: bool, no_integration: bool) -> None:
    """Run the test suite."""
    print_header("Running Tests")

    cmd = [sys.executable, "-m", "pytest", "tests/", "-s"]

    if verbose:
        cmd.append("-v")

    if test:
        cmd.extend(["-k", test])

    if class_name:
        cmd.extend(["-k", class_name])

    if no_integration:
        cmd.extend(["-m", "not integration"])

    console.print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd="/home/nick/Projects/gluellm")
    sys.exit(result.returncode)


@click.command("demo")
def demo() -> None:
    """Run interactive demos of core features."""
    from pydantic import BaseModel

    from gluellm.api import complete, get_session_summary, structured_complete

    def get_time(timezone: str = "UTC") -> str:
        """Get the current time in a specified timezone.

        Args:
            timezone: The timezone to get the time for (default: UTC)

        Returns:
            Current time as a formatted string
        """
        from datetime import datetime

        return f"Current time in {timezone}: {datetime.now().strftime('%H:%M:%S')}"

    class Color(BaseModel):
        name: str
        hex_code: str

    async def run_all_demos():
        """Run all demos in a single async context to avoid event loop issues."""
        print_header("GlueLLM Demo", "Interactive demonstration of core features")

        # Demo 1: Simple completion
        print_step(1, 3, "Simple Completion")
        try:
            result = await complete("What is the capital of France? Answer briefly.")
            console.print(f"  Response: {result.final_response}")
            if result.estimated_cost_usd:
                console.print(f"  Cost: ${result.estimated_cost_usd:.6f}")
            print_success("Simple completion works!")
        except Exception as e:
            print_error(f"Demo failed: {e}")

        # Demo 2: Tool calling
        print_step(2, 3, "Tool Calling")
        try:
            result = await complete("What time is it?", tools=[get_time])
            console.print(f"  Response: {result.final_response}")
            console.print(f"  Tool calls: {result.tool_calls_made}")
            if result.estimated_cost_usd:
                console.print(f"  Cost: ${result.estimated_cost_usd:.6f}")
            print_success("Tool calling works!")
        except Exception as e:
            print_error(f"Demo failed: {e}")

        # Demo 3: Structured output
        print_step(3, 3, "Structured Output")
        try:
            color = await structured_complete(
                "Give me the color red",
                response_format=Color,
            )
            console.print(f"  Color: {color.name} ({color.hex_code})")
            print_success("Structured output works!")
        except Exception as e:
            print_error(f"Demo failed: {e}")

        # Print session summary
        summary = get_session_summary()
        if summary["request_count"] > 0:
            console.print()
            console.print("[bold]Session Summary:[/bold]")
            console.print(f"  Total requests: {summary['request_count']}")
            console.print(f"  Total tokens: {summary['total_tokens']:,}")
            console.print(f"  Total cost: ${summary['total_cost_usd']:.6f}")

    run_async(run_all_demos())

    print_success("Demo completed!")


@click.command("examples")
@click.option("--example", "-e", help="Specific example to run")
def examples(example: str | None) -> None:
    """Run example scripts from examples/."""
    from pathlib import Path

    print_header("GlueLLM Examples")

    examples_dir = Path(__file__).parent.parent.parent.parent / "examples"

    if not examples_dir.exists():
        print_error(f"Examples directory not found: {examples_dir}")
        return

    # List available examples
    available = [f.stem for f in examples_dir.glob("*.py") if not f.name.startswith("_")]

    if not example:
        console.print("Available examples:")
        for ex in sorted(available):
            console.print(f"  - {ex}")
        console.print("\nRun with: gluellm examples -e <example_name>")
        return

    if example not in available:
        print_error(f"Example '{example}' not found")
        console.print(f"Available: {', '.join(available)}")
        return

    example_path = examples_dir / f"{example}.py"
    print_info(f"Running example: {example}")

    result = subprocess.run([sys.executable, str(example_path)])
    if result.returncode == 0:
        print_success("Example completed")
    else:
        print_error(f"Example failed with code {result.returncode}")


@click.command("version")
def version() -> None:
    """Show GlueLLM version information."""
    try:
        import importlib.metadata

        ver = importlib.metadata.version("gluellm")
    except Exception:
        ver = "unknown"

    print_header("GlueLLM", f"Version: {ver}")

    from gluellm.config import settings

    console.print(f"  Default model: {settings.default_model}")
    console.print(f"  Tracing enabled: {settings.enable_tracing}")


@click.command("config")
def show_config() -> None:
    """Show current GlueLLM configuration."""
    from gluellm.config import settings

    print_header("GlueLLM Configuration")

    config_items = [
        ("Default Model", settings.default_model),
        ("Max Tool Iterations", settings.max_tool_iterations),
        ("Default Timeout", f"{settings.default_request_timeout}s"),
        ("Max Timeout", f"{settings.max_request_timeout}s"),
        ("Retry Max Attempts", settings.retry_max_attempts),
        ("Rate Limit (req/s)", settings.rate_limit_requests_per_second),
        ("Tracing Enabled", settings.enable_tracing),
        ("Log Level", settings.log_level),
    ]

    for name, value in config_items:
        console.print(f"  {name}: {value}")


# Export all commands
utilities_commands = [
    run_tests,
    demo,
    examples,
    version,
    show_config,
]
