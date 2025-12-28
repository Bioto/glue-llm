"""GlueLLM - A high-level LLM SDK with automatic tool execution and structured outputs."""

from source.api import (
    GlueLLM,
    complete,
    structured_complete,
    ToolExecutionResult,
)

from source.models.config import RequestConfig
from source.models.conversation import Conversation, Message, Role
from source.models.prompt import SystemPrompt, Prompt

__all__ = [
    # High-level API
    "GlueLLM",
    "complete",
    "structured_complete",
    "ToolExecutionResult",
    
    # Models
    "RequestConfig",
    "Conversation",
    "Message",
    "Role",
    "SystemPrompt",
    "Prompt",
]

__version__ = "0.1.0"

