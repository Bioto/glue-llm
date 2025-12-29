"""Generic agent implementation.

This module provides a GenericAgent that can be used as a starting point
for building custom agents with specific capabilities.
"""

from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt


class GenericAgent(Agent):
    """A generic agent with customizable behavior.

    This agent provides a basic template for creating agents with
    custom system prompts and tool sets. It can be subclassed or
    used directly for general-purpose tasks.

    Example:
        >>> from gluellm.agents.generic import GenericAgent
        >>>
        >>> agent = GenericAgent()
        >>> print(agent.name)
        'Generic Agent'
    """

    def __init__(
        self,
    ):
        """Initialize a GenericAgent with default configuration.

        The agent is configured with:
        - Name: "Generic Agent"
        - A generic description
        - A pirate-themed system prompt (for demonstration)
        - Empty tools list (can be customized)
        - 10 max tool iterations
        """
        super().__init__(
            name="Generic Agent",
            description="A generic agent that can use any tool",
            system_prompt=SystemPrompt(content="You are a generic agent that can use any tool. You are a pirate"),
            tools=[],
            max_tool_iterations=10,
        )
