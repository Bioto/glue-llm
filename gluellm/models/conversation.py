"""Conversation and message models for managing chat history.

This module provides classes for managing conversation state and messages
in LLM interactions, including role enumeration and message tracking.
"""

import uuid
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role enumeration for chat interactions.

    Defines the four standard roles in LLM conversations:
    - SYSTEM: System/instruction messages
    - USER: User input messages
    - ASSISTANT: LLM response messages
    - TOOL: Tool execution result messages
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Represents a single message in a conversation.

    Attributes:
        id: Unique identifier for the message (auto-generated UUID)
        role: The role of the message sender (USER, ASSISTANT, etc.)
        content: The text content of the message

    Example:
        >>> from gluellm.models.conversation import Message, Role
        >>> msg = Message(
        ...     id="123e4567-e89b-12d3-a456-426614174000",
        ...     role=Role.USER,
        ...     content="Hello, AI!"
        ... )
    """

    id: Annotated[str, Field(description="The unique identifier for the message")]
    role: Annotated[Role, Field(description="The role of the message")]
    content: Annotated[str, Field(description="The content of the message")]


class Conversation(BaseModel):
    """Manages a conversation history with multiple messages.

    Tracks an ordered sequence of messages and provides utilities
    for converting to dictionary format for LLM API calls.

    Attributes:
        id: Unique identifier for the conversation (auto-generated UUID)
        messages: List of Message objects in chronological order

    Example:
        >>> from gluellm.models.conversation import Conversation, Role
        >>> conv = Conversation()
        >>> conv.add_message(Role.USER, "What is 2+2?")
        >>> conv.add_message(Role.ASSISTANT, "2+2 equals 4")
        >>> print(conv.messages_dict)
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4"}
        ]
    """

    id: Annotated[
        str, Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the conversation")
    ]
    messages: Annotated[list[Message], Field(default_factory=list, description="The messages in the conversation")]

    @property
    def messages_dict(self) -> list[dict]:
        """Convert messages to dictionary format for LLM APIs.

        Returns:
            list[dict]: List of message dictionaries with 'role' and 'content' keys,
                suitable for passing to LLM API calls
        """
        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in self.messages
        ]

    def add_message(self, role: Role, content: str) -> None:
        """Add a new message to the conversation.

        Args:
            role: The role of the message sender
            content: The text content of the message
        """
        self.messages.append(Message(id=str(uuid.uuid4()), role=role, content=content))
