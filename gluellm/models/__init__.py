"""Data models for GlueLLM.

This module contains Pydantic models and dataclasses used throughout
GlueLLM for configuration, conversation management, and prompt handling.

Available Models:
    - Agent: Configured LLM agent with specific capabilities
    - RequestConfig: Configuration for individual LLM requests
    - Conversation: Conversation history manager
     - Message: Individual message in a conversation
     - Role: Message role enumeration
     - SystemPrompt: System prompt with tool integration
     - Prompt: Basic prompt model
     - CriticConfig: Configuration for specialized critics in workflows
     - IterativeConfig: Configuration for iterative refinement workflows
"""

from gluellm.models.hook import HookConfig, HookContext, HookErrorStrategy, HookRegistry, HookStage
from gluellm.models.workflow import CriticConfig, IterativeConfig

__all__ = [
    "CriticConfig",
    "IterativeConfig",
    "HookConfig",
    "HookContext",
    "HookErrorStrategy",
    "HookRegistry",
    "HookStage",
]
