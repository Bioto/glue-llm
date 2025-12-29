"""Request configuration model for LLM interactions.

This module provides the RequestConfig class, which encapsulates all
configuration needed for a single LLM request including model selection,
prompts, tools, and conversation history.
"""

from collections.abc import Callable
from typing import Annotated

from pydantic import BaseModel, Field, PrivateAttr

from gluellm.models.conversation import Conversation, Role
from gluellm.models.prompt import SystemPrompt


class RequestConfig(BaseModel):
    """Configuration for a single LLM request.

    This class bundles all parameters needed to make an LLM request,
    including the model, system prompt, response format, and available tools.
    It also manages conversation history internally.

    Attributes:
        model: LLM model identifier in format "provider:model_name"
            (e.g., "openai:gpt-4", "anthropic:claude-3-sonnet")
        system_prompt: SystemPrompt defining model behavior
        response_format: Optional Pydantic model for structured output
        tools: List of callable functions the model can use

    Private Attributes:
        _conversation: Internal conversation history manager

    Example:
        >>> from gluellm.models.config import RequestConfig
        >>> from gluellm.models.prompt import SystemPrompt
        >>> from gluellm.models.conversation import Role
        >>> from pydantic import BaseModel
        >>>
        >>> class OutputFormat(BaseModel):
        ...     answer: str
        ...
        >>> config = RequestConfig(
        ...     model="openai:gpt-4o-mini",
        ...     system_prompt=SystemPrompt(content="You are helpful."),
        ...     response_format=OutputFormat,
        ...     tools=[]
        ... )
        >>> config.add_message_to_conversation(Role.USER, "Hello!")
        >>> messages = config.get_conversation()
    """

    model: Annotated[str, Field(description="The model to use for the request provider:model_name")]
    system_prompt: Annotated[SystemPrompt, Field(description="The system prompt to use for the request")]
    response_format: Annotated[
        type[BaseModel] | None, Field(default=None, description="The response format to use for the request")
    ]
    tools: Annotated[list[Callable], Field(description="The tools to use for the request")]

    _conversation: Conversation = PrivateAttr(default_factory=Conversation)

    def add_message_to_conversation(self, role: Role, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: The role of the message (USER, ASSISTANT, SYSTEM, or TOOL)
            content: The message content
        """
        self._conversation.add_message(role, content)

    def get_conversation(self) -> list[dict]:
        """Get the full conversation including system message.

        Returns:
            list[dict]: List of message dictionaries in OpenAI format,
                starting with the system message followed by conversation history
        """
        system_message = {
            "role": "system",
            "content": self.system_prompt.to_formatted_string(tools=self.tools),
        }
        return [system_message] + self._conversation.messages_dict
