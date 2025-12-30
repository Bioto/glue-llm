"""GlueLLM CLI Package.

This package provides the command-line interface for GlueLLM, organized into
logical command groups for better maintainability.

Command Groups:
    - completion: Basic completion and streaming tests
    - tools: Tool calling and execution tests
    - infrastructure: Error handling, hooks, telemetry, rate limiting tests
    - workflows: Multi-agent workflow tests
    - utilities: Demo, examples, and test runner

The main CLI is assembled from these groups in the main module.
"""

import click

from gluellm.cli.commands.completion import completion_commands
from gluellm.cli.commands.infrastructure import infrastructure_commands
from gluellm.cli.commands.tools import tools_commands
from gluellm.cli.commands.utilities import utilities_commands
from gluellm.cli.commands.workflows import workflows_commands
from gluellm.cli.utils import get_weather


@click.group()
def cli() -> None:
    """GlueLLM CLI - Command-line interface for GlueLLM operations.

    Provides commands for testing, demonstrations, and running examples.
    Use --help with any command for more information.
    """
    pass


# Register all command groups
for cmd in completion_commands:
    cli.add_command(cmd)

for cmd in tools_commands:
    cli.add_command(cmd)

for cmd in infrastructure_commands:
    cli.add_command(cmd)

for cmd in workflows_commands:
    cli.add_command(cmd)

for cmd in utilities_commands:
    cli.add_command(cmd)


__all__ = ["cli", "get_weather"]
