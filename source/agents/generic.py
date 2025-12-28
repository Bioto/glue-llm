from source.models.agent import Agent
from source.models.prompt import SystemPrompt
    
class GenericAgent(Agent):
    def __init__(self, ):
        super().__init__(
            name="Generic Agent",
            description="A generic agent that can use any tool",
            system_prompt=SystemPrompt(content="You are a generic agent that can use any tool. You are a pirate"),
            tools=[],
            max_tool_iterations=10,
            model="openai:gpt-4o-mini",
        )
