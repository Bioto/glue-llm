"""Agent model for defining LLM agents with specific capabilities.

This module provides the Agent class, which represents a configured
LLM agent with a specific role, tools, and behavior.
"""

from collections.abc import Callable
from typing import Literal

from gluellm.models.prompt import SystemPrompt

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class Agent:
    """Represents an LLM agent with specific capabilities and configuration.

    An Agent encapsulates all configuration needed for a specialized LLM
    agent, including its identity, system prompt, available tools, and
    execution parameters.

    Attributes:
        name: The agent's name/identifier
        description: A description of the agent's purpose and capabilities
        system_prompt: The system prompt defining agent behavior
        tools: List of callable tools the agent can use
        model: LLM model identifier (provider:model_name format)
        max_tool_iterations: Maximum tool execution iterations
        reasoning_effort: Optional reasoning effort for reasoning-capable models

    Example:
        >>> from gluellm.models.agent import Agent
        >>> from gluellm.models.prompt import SystemPrompt
        >>>
        >>> def search_web(query: str) -> str:
        ...     return f"Results for: {query}"
        >>>
        >>> agent = Agent(
        ...     name="Research Assistant",
        ...     description="Helps with research tasks",
        ...     system_prompt=SystemPrompt(content="You are a research assistant."),
        ...     tools=[search_web],
        ...     max_tool_iterations=5,
        ...     reasoning_effort="high",
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: SystemPrompt,
        tools: list[Callable],
        max_tool_iterations: int = 10,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ):
        """Initialize an Agent.

        Args:
            name: Unique name for the agent
            description: Description of agent's purpose and capabilities
            system_prompt: SystemPrompt defining agent behavior
            tools: List of callable functions the agent can use
            max_tool_iterations: Maximum number of tool call iterations
                (defaults to 10)
            model: LLM model to use (defaults to settings.default_model)
            max_tokens: Maximum number of tokens to generate (defaults to settings.default_max_tokens).
            reasoning_effort: Reasoning effort for o-series / gpt-5 models.
        """
        from gluellm.config import settings

        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model or settings.default_model
        self.max_tool_iterations = max_tool_iterations
        self.max_tokens = max_tokens if max_tokens is not None else settings.default_max_tokens
        self.reasoning_effort = reasoning_effort
