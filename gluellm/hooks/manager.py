"""Hook manager for executing hooks.

This module provides the HookManager class that handles hook execution,
error handling, and timeout management.
"""

import asyncio
import inspect
import time
from typing import Any

from gluellm.models.hook import HookConfig, HookContext, HookErrorStrategy, HookRegistry, HookStage
from gluellm.observability.logging_config import get_logger

logger = get_logger(__name__)


class HookManager:
    """Manages hook execution with error handling and timeout support.

    This class handles the execution of hooks, automatically detecting
    sync vs async callables and chaining their outputs.
    """

    def __init__(self):
        """Initialize the HookManager."""
        pass

    async def execute_hooks(
        self,
        content: str,
        stage: HookStage,
        metadata: dict[str, Any] | None = None,
        hooks: list[HookConfig] | None = None,
    ) -> str:
        """Execute a list of hooks in sequence.

        Args:
            content: The content to process
            stage: The execution stage
            metadata: Optional metadata dictionary
            hooks: List of hook configs to execute

        Returns:
            The processed content after all hooks have been applied

        Raises:
            Exception: If a hook raises an exception and error_strategy is ABORT
        """
        if not hooks:
            return content

        metadata = metadata or {}
        current_content = content
        original_content = content
        enabled_hooks = [h for h in hooks if h.enabled]

        logger.debug(f"Executing {len(enabled_hooks)} hook(s) for stage '{stage.value}'")

        for hook_config in hooks:
            if not hook_config.enabled:
                logger.debug(f"Skipping disabled hook '{hook_config.name}'")
                continue

            hook_start_time = time.time()
            logger.debug(f"Executing hook '{hook_config.name}' for stage '{stage.value}'")

            try:
                # Create hook context
                context = HookContext(
                    content=current_content,
                    stage=stage,
                    metadata=metadata,
                    original_content=original_content,
                )

                # Execute hook with timeout if specified
                if hook_config.timeout:
                    logger.debug(f"Hook '{hook_config.name}' has timeout of {hook_config.timeout}s")
                    result = await self._execute_with_timeout(hook_config, context, hook_config.timeout)
                else:
                    result = await self._execute_hook(hook_config, context)

                hook_elapsed = time.time() - hook_start_time

                # Update current content based on result type
                if isinstance(result, HookContext):
                    current_content = result.content
                    logger.debug(
                        f"Hook '{hook_config.name}' completed in {hook_elapsed:.3f}s, "
                        f"returned HookContext, content_length={len(current_content)}"
                    )
                elif isinstance(result, str):
                    current_content = result
                    logger.debug(
                        f"Hook '{hook_config.name}' completed in {hook_elapsed:.3f}s, "
                        f"returned string, content_length={len(current_content)}"
                    )
                else:
                    logger.warning(
                        f"Hook '{hook_config.name}' returned unexpected type {type(result)}, using original content"
                    )

            except TimeoutError:
                hook_elapsed = time.time() - hook_start_time
                logger.warning(
                    f"Hook '{hook_config.name}' timed out after {hook_elapsed:.3f}s (limit: {hook_config.timeout}s)"
                )
                current_content = self._handle_error(
                    hook_config, current_content, TimeoutError(f"Hook '{hook_config.name}' timed out")
                )
            except Exception as e:
                hook_elapsed = time.time() - hook_start_time
                logger.error(f"Error executing hook '{hook_config.name}' after {hook_elapsed:.3f}s: {e}", exc_info=True)
                current_content = self._handle_error(hook_config, current_content, e)

        logger.debug(f"Completed hook execution for stage '{stage.value}', final_content_length={len(current_content)}")
        return current_content

    async def _execute_hook(self, hook_config: HookConfig, context: HookContext) -> HookContext | str:
        """Execute a single hook, handling sync/async automatically.

        Args:
            hook_config: The hook configuration
            context: The hook context

        Returns:
            The result from the hook (HookContext or str)
        """
        handler = hook_config.handler

        # Check if handler is async
        if inspect.iscoroutinefunction(handler):
            logger.debug(f"Hook '{hook_config.name}' is async")
            result = await handler(context)
        else:
            logger.debug(f"Hook '{hook_config.name}' is sync, running in executor")
            # Run sync handler in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, handler, context)

        return result

    async def _execute_with_timeout(
        self, hook_config: HookConfig, context: HookContext, timeout: float
    ) -> HookContext | str:
        """Execute a hook with a timeout.

        Args:
            hook_config: The hook configuration
            context: The hook context
            timeout: Timeout in seconds

        Returns:
            The result from the hook

        Raises:
            asyncio.TimeoutError: If the hook exceeds the timeout
        """
        return await asyncio.wait_for(self._execute_hook(hook_config, context), timeout=timeout)

    def _handle_error(self, hook_config: HookConfig, current_content: str, error: Exception) -> str:
        """Handle hook errors based on error strategy.

        Args:
            hook_config: The hook configuration
            current_content: The current content being processed
            error: The exception that occurred

        Returns:
            The content to use (original, fallback, or raises exception)

        Raises:
            Exception: If error_strategy is ABORT
        """
        if hook_config.error_strategy == HookErrorStrategy.ABORT:
            raise error
        if hook_config.error_strategy == HookErrorStrategy.FALLBACK:
            if hook_config.fallback_value is not None:
                return hook_config.fallback_value
            logger.warning(
                f"Hook '{hook_config.name}' error strategy is FALLBACK but no fallback_value provided, "
                "using original content"
            )
            return current_content
        # SKIP
        return current_content


# Global hook registry instance
GLOBAL_HOOK_REGISTRY: HookRegistry | None = None


def _get_global_registry() -> "HookRegistry":
    """Get or create the global hook registry.

    Returns:
        The global hook registry instance
    """
    global GLOBAL_HOOK_REGISTRY
    if GLOBAL_HOOK_REGISTRY is None:
        from gluellm.models.hook import HookRegistry

        GLOBAL_HOOK_REGISTRY = HookRegistry()
    return GLOBAL_HOOK_REGISTRY


def register_global_hook(stage: HookStage, config: HookConfig) -> None:
    """Register a global hook.

    Args:
        stage: The hook stage
        config: The hook configuration
    """
    registry = _get_global_registry()
    registry.add_hook(stage, config)
    logger.info(
        f"Registered global hook '{config.name}' for stage '{stage.value}' "
        f"(timeout={config.timeout}s, error_strategy={config.error_strategy.value})"
    )


def unregister_global_hook(stage: HookStage, name: str) -> bool:
    """Unregister a global hook by name.

    Args:
        stage: The hook stage
        name: The name of the hook to unregister

    Returns:
        True if hook was found and removed, False otherwise
    """
    registry = _get_global_registry()
    result = registry.remove_hook(stage, name)
    if result:
        logger.info(f"Unregistered global hook '{name}' from stage '{stage.value}'")
    return result


def clear_global_hooks(stage: HookStage | None = None) -> None:
    """Clear global hooks.

    Args:
        stage: Optional stage to clear. If None, clears all stages.
    """
    registry = _get_global_registry()
    if stage:
        registry.clear_stage(stage)
        logger.info(f"Cleared global hooks for stage '{stage.value}'")
    else:
        for s in HookStage:
            registry.clear_stage(s)
        logger.info("Cleared all global hooks")
