from typing import Callable

from source.api import GlueLLM
from source.models.agent import Agent

from ._base import Executor

class SimpleExecutor(Executor):
    def __init__(self,
        model: str = "openai:gpt-4o-mini",
        system_prompt: str = None,
        tools: list[Callable] = None,
        max_tool_iterations: int = 10,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_tool_iterations = max_tool_iterations
        
    def execute(self, query: str) -> str:   
        client = GlueLLM(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            max_tool_iterations=self.max_tool_iterations,
        )
        result = client.complete(query)
        return result.final_response


class AgentExecutor(Executor):
    def __init__(self, agent: Agent):
        self.agent = agent

    def execute(self, query: str) -> str:
        client = GlueLLM(
            model=self.agent.model,
            system_prompt=self.agent.system_prompt.content if self.agent.system_prompt else None,
            tools=self.agent.tools,
            max_tool_iterations=getattr(self.agent, 'max_tool_iterations', 10),
        )
        result = client.complete(query)
        return result.final_response

__all__ = [
    "Executor",
    "SimpleExecutor",
    "AgentExecutor",
]

if __name__ == "__main__":
    executor = SimpleExecutor(
        model="openai:gpt-4o-mini",
        system_prompt="You are a simple executor that can execute a query",
        tools=[],
    )
    print(executor.execute("What is the weather in Tokyo?"))