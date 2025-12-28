from typing import Annotated, Type, Callable
from pydantic import BaseModel, Field, PrivateAttr

from source.models.prompt import Prompt
from source.models.prompt import BASE_SYSTEM_PROMPT, SystemPrompt
from source.models.conversation import Conversation, Role

class RequestConfig(BaseModel):
    model: Annotated[str, Field(description="The model to use for the request provider:model_name")]
    system_prompt: Annotated[SystemPrompt, Field(description="The system prompt to use for the request")]
    response_format: Annotated[Type[BaseModel], Field(description="The response format to use for the request")]
    tools: Annotated[list[Callable], Field(description="The tools to use for the request")]

    _conversation: Conversation = PrivateAttr(default_factory=Conversation)

    def add_message_to_conversation(self, role: Role, content: str) -> None:
        self._conversation.add_message(role, content)

    def get_conversation(self) -> Conversation:
        system_message = {
            "role": "system",
            "content": self.system_prompt.to_formatted_string(tools=self.tools),
        }
        return [system_message] + self._conversation.messages_dict