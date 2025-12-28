from typing import Callable

from source.models.prompt import SystemPrompt

class Agent:
    def __init__(self,
        name: str,
        description: str,
        system_prompt: SystemPrompt,
        tools: list[Callable],
        max_tool_iterations: int = 10,
        model: str = "openai:gpt-4o-mini",
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model
