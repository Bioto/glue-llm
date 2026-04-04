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
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    PRE_ITERATION = "pre_iteration"
    POST_ITERATION = "post_iteration"
    PRE_GUARDRAIL = "pre_guardrail"
    POST_GUARDRAIL = "post_guardrail"
    ON_LLM_RETRY = "on_llm_retry"
    PRE_TOOL_ROUTE = "pre_tool_route"
    POST_TOOL_ROUTE = "post_tool_route"
    ON_VALIDATION_RETRY = "on_validation_retry"
    PRE_BATCH_ITEM = "pre_batch_item"
    POST_BATCH_ITEM = "post_batch_item"
    PRE_EVAL_RECORD = "pre_eval_record"


class HookErrorStrategy(str, Enum):
    """Enumeration of error handling strategies for hooks."""

    ABORT = "abort"
    SKIP = "skip"
    FALLBACK = "fallback"


class HookContext(BaseModel):
    """Context data structure passed to hooks.

    Attributes:
        content: The text being processed. Semantics vary by stage:
            - pre_workflow / post_workflow: workflow input/output string
            - pre_executor / post_executor: executor query / final response string
            - pre_tool: JSON-serialised tool arguments (modifiable; parsed before invocation)
            - post_tool: tool result string (modifiable; used as the result sent to the LLM)
            - pre_iteration: last user message string for the current LLM call
            - post_iteration: LLM response text before tool dispatch
            - pre_guardrail / post_guardrail: text entering/exiting the guardrail chain (modifiable)
            - on_llm_retry: stringified exception that triggered the retry
            - pre_tool_route: user context string sent to the router LLM
            - post_tool_route: JSON array of matched tool names (modifiable to override selection)
            - on_validation_retry: raw bad JSON string that failed Pydantic validation
            - pre_batch_item / post_batch_item: user message / result or error string
            - pre_eval_record: user_message field of the record (modifiable for PII scrubbing)
        stage: The execution stage
        metadata: Dictionary with stage-specific context. Common keys per stage:
            - pre_tool / post_tool: ``tool_name``, ``call_index``, ``iteration``
            - post_tool: additionally ``duration_seconds``, ``error`` (bool)
            - pre_iteration / post_iteration: ``iteration``, ``tool_count``
            - post_iteration: additionally ``has_tool_calls``
            - pre_guardrail / post_guardrail: ``direction`` ("input"|"output"), ``attempt``
            - on_llm_retry: ``attempt``, ``max_attempts``, ``wait_seconds``, ``exception_type``
            - pre_tool_route / post_tool_route: ``route_query``, ``available_tool_count``
            - post_tool_route: additionally ``matched_tool_names``, ``fallback_to_all``
            - on_validation_retry: ``validation_attempt``, ``max_validation_retries``, ``error``, ``response_format_name``
            - pre_batch_item / post_batch_item: ``batch_request_id``, ``index``, ``attempt``
            - post_batch_item: additionally ``success``
            - pre_eval_record: ``correlation_id``, ``success``, ``model``
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
        pre_tool: List of pre-tool hooks (fire before each tool function is invoked)
        post_tool: List of post-tool hooks (fire after each tool function returns)
        pre_iteration: List of pre-iteration hooks (fire before each LLM API call in the loop)
        post_iteration: List of post-iteration hooks (fire after each LLM response, before tool dispatch)
        pre_guardrail: List of pre-guardrail hooks (fire before the guardrail chain runs)
        post_guardrail: List of post-guardrail hooks (fire after the guardrail chain completes)
        on_llm_retry: List of hooks that fire before each LLM retry sleep
        pre_tool_route: List of hooks that fire before the tool-routing LLM call
        post_tool_route: List of hooks that fire after tool routing resolves
        on_validation_retry: List of hooks that fire before each structured-output validation retry
        pre_batch_item: List of hooks that fire before each BatchProcessor item
        post_batch_item: List of hooks that fire after each BatchProcessor item
        pre_eval_record: List of hooks that fire before writing to the eval store
    """

    pre_workflow: list[HookConfig] = Field(default_factory=list, description="Pre-workflow hooks")
    post_workflow: list[HookConfig] = Field(default_factory=list, description="Post-workflow hooks")
    pre_executor: list[HookConfig] = Field(default_factory=list, description="Pre-executor hooks")
    post_executor: list[HookConfig] = Field(default_factory=list, description="Post-executor hooks")
    pre_tool: list[HookConfig] = Field(default_factory=list, description="Pre-tool hooks")
    post_tool: list[HookConfig] = Field(default_factory=list, description="Post-tool hooks")
    pre_iteration: list[HookConfig] = Field(default_factory=list, description="Pre-iteration hooks")
    post_iteration: list[HookConfig] = Field(default_factory=list, description="Post-iteration hooks")
    pre_guardrail: list[HookConfig] = Field(default_factory=list, description="Pre-guardrail hooks")
    post_guardrail: list[HookConfig] = Field(default_factory=list, description="Post-guardrail hooks")
    on_llm_retry: list[HookConfig] = Field(default_factory=list, description="LLM retry hooks")
    pre_tool_route: list[HookConfig] = Field(default_factory=list, description="Pre-tool-route hooks")
    post_tool_route: list[HookConfig] = Field(default_factory=list, description="Post-tool-route hooks")
    on_validation_retry: list[HookConfig] = Field(default_factory=list, description="Validation retry hooks")
    pre_batch_item: list[HookConfig] = Field(default_factory=list, description="Pre-batch-item hooks")
    post_batch_item: list[HookConfig] = Field(default_factory=list, description="Post-batch-item hooks")
    pre_eval_record: list[HookConfig] = Field(default_factory=list, description="Pre-eval-record hooks")

    def _stage_map(self) -> dict[HookStage, list[HookConfig]]:
        return {
            HookStage.PRE_WORKFLOW: self.pre_workflow,
            HookStage.POST_WORKFLOW: self.post_workflow,
            HookStage.PRE_EXECUTOR: self.pre_executor,
            HookStage.POST_EXECUTOR: self.post_executor,
            HookStage.PRE_TOOL: self.pre_tool,
            HookStage.POST_TOOL: self.post_tool,
            HookStage.PRE_ITERATION: self.pre_iteration,
            HookStage.POST_ITERATION: self.post_iteration,
            HookStage.PRE_GUARDRAIL: self.pre_guardrail,
            HookStage.POST_GUARDRAIL: self.post_guardrail,
            HookStage.ON_LLM_RETRY: self.on_llm_retry,
            HookStage.PRE_TOOL_ROUTE: self.pre_tool_route,
            HookStage.POST_TOOL_ROUTE: self.post_tool_route,
            HookStage.ON_VALIDATION_RETRY: self.on_validation_retry,
            HookStage.PRE_BATCH_ITEM: self.pre_batch_item,
            HookStage.POST_BATCH_ITEM: self.post_batch_item,
            HookStage.PRE_EVAL_RECORD: self.pre_eval_record,
        }

    def get_hooks(self, stage: HookStage) -> list[HookConfig]:
        """Get hooks for a specific stage.

        Args:
            stage: The hook stage

        Returns:
            List of hook configs for the stage
        """
        return self._stage_map().get(stage, [])

    def add_hook(self, stage: HookStage, config: HookConfig) -> None:
        """Add a hook to a specific stage.

        Args:
            stage: The hook stage
            config: The hook configuration
        """
        stage_list = self._stage_map().get(stage)
        if stage_list is not None:
            stage_list.append(config)

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
        stage_list = self._stage_map().get(stage)
        if stage_list is not None:
            stage_list.clear()

    def merge(self, other: "HookRegistry") -> "HookRegistry":
        """Merge another registry into this one.

        Args:
            other: The other registry to merge

        Returns:
            A new registry with merged hooks (other's hooks appended)
        """
        merged = HookRegistry()
        for stage in HookStage:
            for hook in self.get_hooks(stage) + other.get_hooks(stage):
                merged.add_hook(stage, hook)
        return merged
