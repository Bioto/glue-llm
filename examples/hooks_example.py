"""Examples demonstrating hook functionality in GlueLLM.

This module provides comprehensive examples of using hooks for:
- PII removal before sending to LLM
- Output validation after LLM response
- Chaining multiple hooks
- Global vs instance-level configuration
- Error handling strategies
- Custom async hook implementation
- Content filtering with workflows
"""

import asyncio

from gluellm.executors import SimpleExecutor
from gluellm.hooks import register_global_hook
from gluellm.hooks.utils import (
    check_toxicity_factory,
    normalize_whitespace,
    remove_emails,
    remove_pii,
    require_citations,
    truncate_output_factory,
    validate_length_factory,
)
from gluellm.models.hook import HookConfig, HookErrorStrategy, HookRegistry, HookStage
from gluellm.workflows.reflection import ReflectionWorkflow


# Example 1: PII removal before sending to LLM
async def example_1_pii_removal():
    """Example: Remove PII before sending query to LLM."""
    print("\n=== Example 1: PII Removal ===")

    # Create hook registry with PII removal
    registry = HookRegistry()
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=remove_pii,
            name="remove_pii",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    # Create executor with hook registry
    executor = SimpleExecutor(
        system_prompt="You are a helpful assistant.",
        hook_registry=registry,
    )

    # Query with PII
    query = "My email is john@example.com and my phone is 555-123-4567. What is AI?"
    result = await executor.execute(query)
    print(f"Query: {query}")
    print(f"Result: {result[:100]}...")


# Example 2: Output validation after LLM response
async def example_2_output_validation():
    """Example: Validate output length and require citations."""
    print("\n=== Example 2: Output Validation ===")

    registry = HookRegistry()
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=validate_length_factory(min_len=50, max_len=500),
            name="validate_length",
            error_strategy=HookErrorStrategy.ABORT,
        ),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=require_citations,
            name="require_citations",
            error_strategy=HookErrorStrategy.SKIP,  # Skip if no citations
        ),
    )

    executor = SimpleExecutor(
        system_prompt="You are a helpful assistant. Always cite sources.",
        hook_registry=registry,
    )

    query = "Explain quantum computing"
    try:
        result = await executor.execute(query)
        print(f"Result: {result[:200]}...")
    except ValueError as e:
        print(f"Validation error: {e}")


# Example 3: Chaining multiple hooks
async def example_3_chaining():
    """Example: Chain multiple hooks together."""
    print("\n=== Example 3: Chaining Hooks ===")

    registry = HookRegistry()
    # Pre-executor: Remove PII and normalize whitespace
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(handler=remove_pii, name="remove_pii", error_strategy=HookErrorStrategy.SKIP),
    )
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(handler=normalize_whitespace, name="normalize", error_strategy=HookErrorStrategy.SKIP),
    )

    # Post-executor: Normalize, truncate, and check toxicity
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(handler=normalize_whitespace, name="normalize_output", error_strategy=HookErrorStrategy.SKIP),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=truncate_output_factory(max_chars=200),
            name="truncate",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=check_toxicity_factory(threshold=0.3),
            name="check_toxicity",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)
    result = await executor.execute("Tell me about artificial intelligence")
    print(f"Result: {result}")


# Example 4: Global vs instance-level configuration
async def example_4_global_vs_instance():
    """Example: Global hooks vs instance-specific hooks."""
    print("\n=== Example 4: Global vs Instance Hooks ===")

    # Register a global hook (applies to all executors/workflows)
    register_global_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=remove_emails,
            name="global_email_removal",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    # Create executor without instance hooks (uses global only)
    executor1 = SimpleExecutor()
    result1 = await executor1.execute("My email is test@example.com. What is Python?")
    print(f"Executor 1 (global only): {result1[:100]}...")

    # Create executor with instance hooks (global + instance)
    instance_registry = HookRegistry()
    instance_registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=normalize_whitespace,
            name="instance_normalize",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor2 = SimpleExecutor(hook_registry=instance_registry)
    result2 = await executor2.execute("My email is test@example.com. What is Python?")
    print(f"Executor 2 (global + instance): {result2[:100]}...")


# Example 5: Error handling strategies
async def example_5_error_handling():
    """Example: Different error handling strategies."""
    print("\n=== Example 5: Error Handling Strategies ===")

    # ABORT strategy: Raises exception on error
    registry_abort = HookRegistry()
    registry_abort.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=validate_length_factory(min_len=1000),  # Will fail for short responses
            name="validate_length_abort",
            error_strategy=HookErrorStrategy.ABORT,
        ),
    )

    executor_abort = SimpleExecutor(hook_registry=registry_abort)
    try:
        await executor_abort.execute("Short query")
    except ValueError as e:
        print(f"ABORT strategy raised: {e}")

    # SKIP strategy: Logs warning and continues
    registry_skip = HookRegistry()
    registry_skip.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=validate_length_factory(min_len=1000),
            name="validate_length_skip",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor_skip = SimpleExecutor(hook_registry=registry_skip)
    result = await executor_skip.execute("Short query")
    print(f"SKIP strategy continued: {result[:50]}...")

    # FALLBACK strategy: Uses fallback value
    registry_fallback = HookRegistry()
    registry_fallback.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=validate_length_factory(min_len=1000),
            name="validate_length_fallback",
            error_strategy=HookErrorStrategy.FALLBACK,
            fallback_value="[Response too short - validation failed]",
        ),
    )

    executor_fallback = SimpleExecutor(hook_registry=registry_fallback)
    result = await executor_fallback.execute("Short query")
    print(f"FALLBACK strategy used fallback: {result}")


# Example 6: Custom async hook implementation
async def example_6_custom_async():
    """Example: Custom async hook implementation."""
    print("\n=== Example 6: Custom Async Hook ===")

    async def custom_async_hook(context):
        """Custom async hook that adds a prefix."""
        # Simulate async operation (e.g., API call, database query)
        await asyncio.sleep(0.1)
        context.content = f"[PROCESSED] {context.content}"
        return context

    registry = HookRegistry()
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=custom_async_hook,
            name="custom_async",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)
    result = await executor.execute("Test query")
    print(f"Result with custom async hook: {result}")


# Example 7: Content filtering with Constitutional AI workflow
async def example_7_workflow_hooks():
    """Example: Using hooks with workflows."""
    print("\n=== Example 7: Workflow Hooks ===")

    # Create hook registry for workflow
    workflow_registry = HookRegistry()
    workflow_registry.add_hook(
        HookStage.PRE_WORKFLOW,
        HookConfig(
            handler=remove_pii,
            name="workflow_pii_removal",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )
    workflow_registry.add_hook(
        HookStage.POST_WORKFLOW,
        HookConfig(
            handler=normalize_whitespace,
            name="workflow_normalize",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    # Create executors with their own hooks
    executor_registry = HookRegistry()
    executor_registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=truncate_output_factory(max_chars=300),
            name="executor_truncate",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    generator = SimpleExecutor(hook_registry=executor_registry)
    reflector = SimpleExecutor(hook_registry=executor_registry)

    # Create workflow with hooks
    workflow = ReflectionWorkflow(
        generator=generator,
        reflector=reflector,
        hook_registry=workflow_registry,
    )

    query = "My email is user@example.com. Write a short article about AI."
    result = await workflow.execute(query)
    print(f"Workflow result: {result.final_output[:200]}...")


async def main():
    """Run all examples."""
    print("GlueLLM Hook Examples")
    print("=" * 50)

    await example_1_pii_removal()
    await example_2_output_validation()
    await example_3_chaining()
    await example_4_global_vs_instance()
    await example_5_error_handling()
    await example_6_custom_async()
    await example_7_workflow_hooks()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
