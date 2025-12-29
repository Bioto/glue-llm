"""Hook models for GlueLLM.

This module provides models for configuring and managing hooks that can
intercept and transform data before and after LLM processing.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HookStage(str, Enum):
    """Enumeration of hook execution stages."""

    PRE_WORKFLOW = "pre_workflow"
    POST_WORKFLOW = "post_workflow"
    PRE_EXECUTOR = "pre_executor"
    POST_EXECUTOR = "post_executor"


class HookErrorStrategy(str, Enum):
    """Enumeration of error handling strategies for hooks."""

    ABORT = "abort"
    SKIP = "skip"
    FALLBACK = "fallback"


class HookContext(BaseModel):
    """Context data structure passed to hooks.

    Attributes:
        content: The text being processed
        stage: The execution stage (pre_workflow, post_workflow, pre_executor, post_executor)
        metadata: Dictionary with workflow/executor info, iteration counts, etc.
        original_content: Reference to the unmodified original content
    """

    content: str = Field(description="The text being processed")
    stage: HookStage = Field(description="The execution stage")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    original_content: str | None = Field(default=None, description="Reference to unmodified original content")

    model_config = {"arbitrary_types_allowed": True}


class HookConfig(BaseModel):
    """Configuration for an individual hook.

    Attributes:
        handler: Callable that processes the hook context (sync or async)
        name: Human-readable identifier for the hook
        error_strategy: How to handle errors (abort, skip, fallback)
        fallback_value: Optional fallback content if error_strategy is FALLBACK
        enabled: Whether the hook is enabled
        timeout: Optional timeout in seconds for hook execution
    """

    handler: Callable[[HookContext], HookContext | str] | Callable[[HookContext], Any] = Field(
        description="Callable that processes the hook context"
    )
    name: str = Field(description="Human-readable identifier for the hook")
    error_strategy: HookErrorStrategy = Field(default=HookErrorStrategy.SKIP, description="How to handle errors")
    fallback_value: str | None = Field(default=None, description="Optional fallback content")
    enabled: bool = Field(default=True, description="Whether the hook is enabled")
    timeout: float | None = Field(default=None, description="Optional timeout in seconds", gt=0)

    model_config = {"arbitrary_types_allowed": True}


class HookRegistry(BaseModel):
    """Container for organizing hooks by stage.

    Attributes:
        pre_workflow: List of pre-workflow hooks
        post_workflow: List of post-workflow hooks
        pre_executor: List of pre-executor hooks
        post_executor: List of post-executor hooks
    """

    pre_workflow: list[HookConfig] = Field(default_factory=list, description="Pre-workflow hooks")
    post_workflow: list[HookConfig] = Field(default_factory=list, description="Post-workflow hooks")
    pre_executor: list[HookConfig] = Field(default_factory=list, description="Pre-executor hooks")
    post_executor: list[HookConfig] = Field(default_factory=list, description="Post-executor hooks")

    def get_hooks(self, stage: HookStage) -> list[HookConfig]:
        """Get hooks for a specific stage.

        Args:
            stage: The hook stage

        Returns:
            List of hook configs for the stage
        """
        stage_map = {
            HookStage.PRE_WORKFLOW: self.pre_workflow,
            HookStage.POST_WORKFLOW: self.post_workflow,
            HookStage.PRE_EXECUTOR: self.pre_executor,
            HookStage.POST_EXECUTOR: self.post_executor,
        }
        return stage_map.get(stage, [])

    def add_hook(self, stage: HookStage, config: HookConfig) -> None:
        """Add a hook to a specific stage.

        Args:
            stage: The hook stage
            config: The hook configuration
        """
        stage_map = {
            HookStage.PRE_WORKFLOW: self.pre_workflow,
            HookStage.POST_WORKFLOW: self.post_workflow,
            HookStage.PRE_EXECUTOR: self.pre_executor,
            HookStage.POST_EXECUTOR: self.post_executor,
        }
        if stage in stage_map:
            stage_map[stage].append(config)

    def remove_hook(self, stage: HookStage, name: str) -> bool:
        """Remove a hook by name from a specific stage.

        Args:
            stage: The hook stage
            name: The name of the hook to remove

        Returns:
            True if hook was found and removed, False otherwise
        """
        hooks = self.get_hooks(stage)
        for i, hook in enumerate(hooks):
            if hook.name == name:
                hooks.pop(i)
                return True
        return False

    def clear_stage(self, stage: HookStage) -> None:
        """Clear all hooks from a specific stage.

        Args:
            stage: The hook stage to clear
        """
        stage_map = {
            HookStage.PRE_WORKFLOW: self.pre_workflow,
            HookStage.POST_WORKFLOW: self.post_workflow,
            HookStage.PRE_EXECUTOR: self.pre_executor,
            HookStage.POST_EXECUTOR: self.post_executor,
        }
        if stage in stage_map:
            stage_map[stage].clear()

    def merge(self, other: "HookRegistry") -> "HookRegistry":
        """Merge another registry into this one.

        Args:
            other: The other registry to merge

        Returns:
            A new registry with merged hooks (other's hooks appended)
        """
        merged = HookRegistry()
        for stage in HookStage:
            merged_hooks = self.get_hooks(stage) + other.get_hooks(stage)
            for hook in merged_hooks:
                merged.add_hook(stage, hook)
        return merged
