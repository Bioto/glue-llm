from pydantic import BaseModel, Field
from typing import Annotated
from jinja2 import Template

from typing import Callable

class Prompt(BaseModel):
    system_prompt: Annotated[str, Field(description="The system prompt for the prompt")]



def _flatten_xml(xml_str: str) -> str:
    """Flatten XML by removing extraneous indentation and line breaks."""
    import re
    # Remove leading/trailing whitespace on each line, then join lines
    lines = [line.strip() for line in xml_str.strip().splitlines()]
    text = "".join(lines)
    return text

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
                    <description>{{ tool.__doc__ }}</description>
                </tool>
            {% endfor %}
        </tools>
    {% endif %}
</system_prompt>
"""

class FlattenedTemplate(Template):
    def render(self, *args, **kwargs):
        rendered = super().render(*args, **kwargs)
        return _flatten_xml(rendered)

BASE_SYSTEM_PROMPT = FlattenedTemplate(BASE_SYSTEM_PROMPT_TEMPLATE, autoescape=True)


class SystemPrompt(BaseModel):
    content: Annotated[str, Field(description="The content of the system prompt")]

    def to_formatted_string(self, tools: list[Callable]) -> str:
        return _flatten_xml(BASE_SYSTEM_PROMPT.render(
            instructions=self.content,
            tools=tools,
        ))

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