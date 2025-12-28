import uuid

from typing import Annotated

from pydantic import BaseModel, Field
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    id: Annotated[str, Field(description="The unique identifier for the message")]
    role: Annotated[Role, Field(description="The role of the message")]
    content: Annotated[str, Field(description="The content of the message")]

class Conversation(BaseModel):
    id: Annotated[str, Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the conversation")]
    messages: Annotated[list[Message], Field(default_factory=list, description="The messages in the conversation")]

    @property
    def messages_dict(self) -> list[dict]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
            } for message in self.messages
        ]
    
    def add_message(self, role: Role, content: str) -> None:
        self.messages.append(Message(id=str(uuid.uuid4()), role=role, content=content))
