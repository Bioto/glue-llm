"""Prompt management and templating for system prompts.

This module provides classes and utilities for creating, formatting,
and managing system prompts with tool integration using Jinja2 templates.
"""

from collections.abc import Callable
from typing import Annotated

from jinja2 import Template
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Basic prompt model.

    Attributes:
        system_prompt: The system prompt text
    """

    system_prompt: Annotated[str, Field(description="The system prompt for the prompt")]


def _flatten_xml(xml_str: str) -> str:
    """Flatten XML by removing extraneous indentation and line breaks.

    This function cleans up XML formatting by removing leading/trailing
    whitespace from each line and joining them into a single line.

    Args:
        xml_str: The XML string to flatten

    Returns:
        str: The flattened XML string with minimal whitespace

    Example:
        >>> xml = '''
        ... <tag>
        ...     <nested>content</nested>
        ... </tag>
        ... '''
        >>> _flatten_xml(xml)
        '<tag><nested>content</nested></tag>'
    """
    # Remove leading/trailing whitespace on each line, then join lines
    lines = [line.strip() for line in xml_str.strip().splitlines()]
    return "".join(lines)


BASE_SYSTEM_PROMPT_TEMPLATE = r"""
<system_prompt>
    <system_instructions>
        {{instructions}}
    </system_instructions>
    {% if tools %}
        <tools>
            {% for tool in tools %}
                <tool>
                    <name>{{ tool.__name__ }}</name>
                    <description>{{ tool.__doc__ if tool.__doc__ else '' }}</description>
                </tool>
            {% endfor %}
        </tools>
    {% endif %}
</system_prompt>
"""


class FlattenedTemplate(Template):
    """Jinja2 template that automatically flattens XML output.

    This subclass of Jinja2's Template automatically flattens the
    rendered output, making it suitable for compact XML system prompts.
    """

    def render(self, *args, **kwargs):
        """Render the template and flatten the result.

        Args:
            *args: Positional arguments for template rendering
            **kwargs: Keyword arguments for template variables

        Returns:
            str: The flattened rendered template
        """
        rendered = super().render(*args, **kwargs)
        return _flatten_xml(rendered)


# Global template instance for system prompts
BASE_SYSTEM_PROMPT = FlattenedTemplate(BASE_SYSTEM_PROMPT_TEMPLATE, autoescape=True)


class SystemPrompt(BaseModel):
    """System prompt with tool integration support.

    Represents a system prompt that can be formatted with available tools
    using the BASE_SYSTEM_PROMPT template. The prompt is wrapped in XML
    structure and includes tool definitions when applicable.

    Attributes:
        content: The instruction text for the system prompt

    Example:
        >>> from gluellm.models.prompt import SystemPrompt
        >>>
        >>> def calculate(x: int, y: int) -> int:
        ...     '''Add two numbers together.'''
        ...     return x + y
        >>>
        >>> prompt = SystemPrompt(content="You are a math assistant.")
        >>> formatted = prompt.to_formatted_string(tools=[calculate])
        >>> print(formatted)
        <system_prompt><system_instructions>You are a math assistant...</system_instructions>...
    """

    content: Annotated[str, Field(description="The content of the system prompt")]

    def to_formatted_string(self, tools: list[Callable]) -> str:
        """Format the system prompt with tools using the template.

        Renders the system prompt using the BASE_SYSTEM_PROMPT template,
        including tool information if tools are provided.

        Args:
            tools: List of callable functions to include in the prompt

        Returns:
            str: The formatted XML system prompt with instructions and tools
        """
        return _flatten_xml(
            BASE_SYSTEM_PROMPT.render(
                instructions=self.content,
                tools=tools,
            )
        )


if __name__ == "__main__":

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location.

        Args:
            location: The city and country, e.g. "San Francisco, CA"
            unit: Temperature unit, either "celsius" or "fahrenheit"
        """
        return f"The weather in {location} is 22 degrees {unit} and sunny."

    print(
        SystemPrompt(
            content="You are a helpful assistant. Use the get_weather tool when asked about weather."
        ).to_formatted_string(tools=[get_weather]),
        end="\n\n",
    )
