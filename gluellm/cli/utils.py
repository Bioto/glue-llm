"""Shared CLI utilities and helper functions.

This module provides common utilities used across CLI commands.
"""

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_async(coro):
    """Run an async function in the event loop.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    return asyncio.run(coro)


def print_header(title: str, subtitle: str = "") -> None:
    """Print a styled header.

    Args:
        title: Main title text
        subtitle: Optional subtitle
    """
    text = f"[bold blue]{title}[/bold blue]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text))


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Message to print
    """
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Message to print
    """
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Message to print
    """
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Message to print
    """
    console.print(f"[blue]ℹ[/blue] {message}")


def print_step(step: int, total: int, message: str) -> None:
    """Print a step progress message.

    Args:
        step: Current step number
        total: Total steps
        message: Step description
    """
    console.print(f"[dim]({step}/{total})[/dim] {message}")


def print_result(title: str, content: str, truncate: int = 500) -> None:
    """Print a result in a panel.

    Args:
        title: Panel title
        content: Content to display
        truncate: Maximum characters to show
    """
    if len(content) > truncate:
        content = content[:truncate] + "..."
    console.print(Panel(content, title=title))


def print_table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    """Print a formatted table.

    Args:
        title: Table title
        columns: Column headers
        rows: List of row data
    """
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def format_tokens(tokens: dict[str, int] | None) -> str:
    """Format token usage for display.

    Args:
        tokens: Token usage dict

    Returns:
        Formatted string
    """
    if not tokens:
        return "N/A"
    return f"{tokens.get('total', 0)} (prompt: {tokens.get('prompt', 0)}, completion: {tokens.get('completion', 0)})"


# Sample tool for demonstrations
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    This is a mock function used for testing and demonstration purposes.

    Args:
        location: The city and country, e.g. "San Francisco, CA"
        unit: Temperature unit, either "celsius" or "fahrenheit"

    Returns:
        str: A simulated weather response string
    """
    return f"The weather in {location} is 22 degrees {unit} and sunny."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate

    Returns:
        Result as string
    """
    try:
        # Safe eval with limited builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
