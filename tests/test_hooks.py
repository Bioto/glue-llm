"""Tests for hook system."""

import asyncio

import pytest

from gluellm.executors import SimpleExecutor
from gluellm.hooks import (
    HookManager,
    clear_global_hooks,
    register_global_hook,
    unregister_global_hook,
)
from gluellm.hooks import manager as hooks_manager
from gluellm.hooks.utils import (
    normalize_whitespace,
    remove_emails,
    remove_pii,
    truncate_output_factory,
    validate_length_factory,
)
from gluellm.models.hook import (
    HookConfig,
    HookContext,
    HookErrorStrategy,
    HookRegistry,
    HookStage,
)
from gluellm.workflows.reflection import ReflectionWorkflow


class MockExecutor(SimpleExecutor):
    """Mock executor for testing hooks."""

    async def _execute_internal(self, query: str) -> str:
        """Return a simple response."""
        return f"Response to: {query}"


class TestHookContext:
    """Tests for HookContext."""

    def test_hook_context_creation(self):
        """Test creating a HookContext."""
        context = HookContext(
            content="test content",
            stage=HookStage.PRE_EXECUTOR,
            metadata={"key": "value"},
        )
        assert context.content == "test content"
        assert context.stage == HookStage.PRE_EXECUTOR
        assert context.metadata == {"key": "value"}


class TestHookConfig:
    """Tests for HookConfig."""

    def test_hook_config_creation(self):
        """Test creating a HookConfig."""

        def handler(ctx):
            return ctx

        config = HookConfig(
            handler=handler,
            name="test_hook",
            error_strategy=HookErrorStrategy.SKIP,
        )
        assert config.name == "test_hook"
        assert config.error_strategy == HookErrorStrategy.SKIP
        assert config.enabled is True


class TestHookRegistry:
    """Tests for HookRegistry."""

    def test_registry_creation(self):
        """Test creating a HookRegistry."""
        registry = HookRegistry()
        assert len(registry.pre_workflow) == 0
        assert len(registry.post_workflow) == 0
        assert len(registry.pre_executor) == 0
        assert len(registry.post_executor) == 0

    def test_add_hook(self):
        """Test adding a hook to registry."""
        registry = HookRegistry()
        config = HookConfig(handler=lambda x: x, name="test")
        registry.add_hook(HookStage.PRE_EXECUTOR, config)
        assert len(registry.pre_executor) == 1
        assert registry.pre_executor[0].name == "test"

    def test_get_hooks(self):
        """Test getting hooks for a stage."""
        registry = HookRegistry()
        config = HookConfig(handler=lambda x: x, name="test")
        registry.add_hook(HookStage.PRE_EXECUTOR, config)
        hooks = registry.get_hooks(HookStage.PRE_EXECUTOR)
        assert len(hooks) == 1
        assert hooks[0].name == "test"

    def test_remove_hook(self):
        """Test removing a hook from registry."""
        registry = HookRegistry()
        config = HookConfig(handler=lambda x: x, name="test")
        registry.add_hook(HookStage.PRE_EXECUTOR, config)
        assert len(registry.pre_executor) == 1
        result = registry.remove_hook(HookStage.PRE_EXECUTOR, "test")
        assert result is True
        assert len(registry.pre_executor) == 0

    def test_merge_registries(self):
        """Test merging two registries."""
        registry1 = HookRegistry()
        registry2 = HookRegistry()
        config1 = HookConfig(handler=lambda x: x, name="hook1")
        config2 = HookConfig(handler=lambda x: x, name="hook2")
        registry1.add_hook(HookStage.PRE_EXECUTOR, config1)
        registry2.add_hook(HookStage.PRE_EXECUTOR, config2)
        merged = registry1.merge(registry2)
        assert len(merged.pre_executor) == 2


class TestHookManager:
    """Tests for HookManager."""

    @pytest.mark.asyncio
    async def test_execute_hooks_empty(self):
        """Test executing empty hook list."""
        manager = HookManager()
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [])
        assert result == "test"

    @pytest.mark.asyncio
    async def test_execute_hooks_sync(self):
        """Test executing sync hook."""
        manager = HookManager()

        def add_prefix(context: HookContext) -> HookContext:
            context.content = f"PREFIX_{context.content}"
            return context

        config = HookConfig(handler=add_prefix, name="add_prefix")
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [config])
        assert result == "PREFIX_test"

    @pytest.mark.asyncio
    async def test_execute_hooks_async(self):
        """Test executing async hook."""
        manager = HookManager()

        async def add_prefix_async(context: HookContext) -> HookContext:
            await asyncio.sleep(0.01)
            context.content = f"ASYNC_{context.content}"
            return context

        config = HookConfig(handler=add_prefix_async, name="add_prefix_async")
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [config])
        assert result == "ASYNC_test"

    @pytest.mark.asyncio
    async def test_execute_hooks_chaining(self):
        """Test chaining multiple hooks."""
        manager = HookManager()

        def add_a(context: HookContext) -> HookContext:
            context.content = f"A_{context.content}"
            return context

        def add_b(context: HookContext) -> HookContext:
            context.content = f"{context.content}_B"
            return context

        configs = [
            HookConfig(handler=add_a, name="add_a"),
            HookConfig(handler=add_b, name="add_b"),
        ]
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, configs)
        assert result == "A_test_B"

    @pytest.mark.asyncio
    async def test_execute_hooks_error_abort(self):
        """Test error handling with ABORT strategy."""
        manager = HookManager()

        def raise_error(context: HookContext) -> HookContext:
            raise ValueError("Test error")

        config = HookConfig(
            handler=raise_error,
            name="error_hook",
            error_strategy=HookErrorStrategy.ABORT,
        )
        with pytest.raises(ValueError, match="Test error"):
            await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [config])

    @pytest.mark.asyncio
    async def test_execute_hooks_error_skip(self):
        """Test error handling with SKIP strategy."""
        manager = HookManager()

        def raise_error(context: HookContext) -> HookContext:
            raise ValueError("Test error")

        config = HookConfig(
            handler=raise_error,
            name="error_hook",
            error_strategy=HookErrorStrategy.SKIP,
        )
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [config])
        assert result == "test"  # Original content preserved

    @pytest.mark.asyncio
    async def test_execute_hooks_error_fallback(self):
        """Test error handling with FALLBACK strategy."""
        manager = HookManager()

        def raise_error(context: HookContext) -> HookContext:
            raise ValueError("Test error")

        config = HookConfig(
            handler=raise_error,
            name="error_hook",
            error_strategy=HookErrorStrategy.FALLBACK,
            fallback_value="fallback_value",
        )
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [config])
        assert result == "fallback_value"

    @pytest.mark.asyncio
    async def test_execute_hooks_disabled(self):
        """Test that disabled hooks are skipped."""
        manager = HookManager()

        def add_prefix(context: HookContext) -> HookContext:
            context.content = f"PREFIX_{context.content}"
            return context

        config = HookConfig(handler=add_prefix, name="add_prefix", enabled=False)
        result = await manager.execute_hooks("test", HookStage.PRE_EXECUTOR, None, [config])
        assert result == "test"  # No change because hook is disabled


class TestExecutorHooks:
    """Tests for hook integration with executors."""

    @pytest.mark.asyncio
    async def test_executor_pre_hook(self):
        """Test pre-executor hook."""
        registry = HookRegistry()
        registry.add_hook(
            HookStage.PRE_EXECUTOR,
            HookConfig(
                handler=lambda ctx: HookContext(content=f"PRE_{ctx.content}", stage=ctx.stage, metadata=ctx.metadata),
                name="pre_hook",
            ),
        )

        executor = MockExecutor(hook_registry=registry)
        result = await executor.execute("test")
        assert "PRE_test" in result

    @pytest.mark.asyncio
    async def test_executor_post_hook(self):
        """Test post-executor hook."""
        registry = HookRegistry()
        registry.add_hook(
            HookStage.POST_EXECUTOR,
            HookConfig(
                handler=lambda ctx: HookContext(content=f"{ctx.content}_POST", stage=ctx.stage, metadata=ctx.metadata),
                name="post_hook",
            ),
        )

        executor = MockExecutor(hook_registry=registry)
        result = await executor.execute("test")
        assert result.endswith("_POST")

    @pytest.mark.asyncio
    async def test_executor_both_hooks(self):
        """Test both pre and post-executor hooks."""
        registry = HookRegistry()
        registry.add_hook(
            HookStage.PRE_EXECUTOR,
            HookConfig(
                handler=lambda ctx: HookContext(content=f"PRE_{ctx.content}", stage=ctx.stage, metadata=ctx.metadata),
                name="pre_hook",
            ),
        )
        registry.add_hook(
            HookStage.POST_EXECUTOR,
            HookConfig(
                handler=lambda ctx: HookContext(content=f"{ctx.content}_POST", stage=ctx.stage, metadata=ctx.metadata),
                name="post_hook",
            ),
        )

        executor = MockExecutor(hook_registry=registry)
        result = await executor.execute("test")
        assert "PRE_" in result
        assert result.endswith("_POST")


class TestWorkflowHooks:
    """Tests for hook integration with workflows."""

    @pytest.mark.asyncio
    async def test_workflow_pre_hook(self):
        """Test pre-workflow hook."""
        registry = HookRegistry()
        registry.add_hook(
            HookStage.PRE_WORKFLOW,
            HookConfig(
                handler=lambda ctx: HookContext(content=f"PRE_{ctx.content}", stage=ctx.stage, metadata=ctx.metadata),
                name="pre_workflow_hook",
            ),
        )

        executor = MockExecutor()
        workflow = ReflectionWorkflow(generator=executor, reflector=executor, hook_registry=registry)
        result = await workflow.execute("test")
        assert "PRE_" in result.final_output

    @pytest.mark.asyncio
    async def test_workflow_post_hook(self):
        """Test post-workflow hook."""
        registry = HookRegistry()
        registry.add_hook(
            HookStage.POST_WORKFLOW,
            HookConfig(
                handler=lambda ctx: HookContext(content=f"{ctx.content}_POST", stage=ctx.stage, metadata=ctx.metadata),
                name="post_workflow_hook",
            ),
        )

        executor = MockExecutor()
        workflow = ReflectionWorkflow(generator=executor, reflector=executor, hook_registry=registry)
        result = await workflow.execute("test")
        assert result.final_output.endswith("_POST")


class TestUtilityHooks:
    """Tests for utility hooks."""

    def test_remove_emails(self):
        """Test email removal hook."""
        context = HookContext(content="Contact me at test@example.com", stage=HookStage.PRE_EXECUTOR)
        result = remove_emails(context)
        assert "[EMAIL_REDACTED]" in result.content
        assert "test@example.com" not in result.content

    def test_remove_pii(self):
        """Test PII removal hook."""
        context = HookContext(
            content="Email: test@example.com, Phone: 555-123-4567",
            stage=HookStage.PRE_EXECUTOR,
        )
        result = remove_pii(context)
        assert "[EMAIL_REDACTED]" in result.content
        assert "[PHONE_REDACTED]" in result.content

    def test_normalize_whitespace(self):
        """Test whitespace normalization hook."""
        context = HookContext(content="  multiple   spaces\n\n\nnewlines  ", stage=HookStage.PRE_EXECUTOR)
        result = normalize_whitespace(context)
        assert "  " not in result.content  # No double spaces
        assert result.content.startswith("multiple")  # Leading whitespace removed

    def test_validate_length_pass(self):
        """Test length validation hook that passes."""
        validator = validate_length_factory(min_len=5, max_len=100)
        context = HookContext(content="This is a valid length string", stage=HookStage.POST_EXECUTOR)
        result = validator(context)
        assert result.content == context.content

    def test_validate_length_fail(self):
        """Test length validation hook that fails."""
        validator = validate_length_factory(min_len=100)
        context = HookContext(content="Too short", stage=HookStage.POST_EXECUTOR)
        with pytest.raises(ValueError, match="too short"):
            validator(context)

    def test_truncate_output(self):
        """Test output truncation hook."""
        truncator = truncate_output_factory(max_chars=10)
        context = HookContext(content="This is a very long string", stage=HookStage.POST_EXECUTOR)
        result = truncator(context)
        assert len(result.content) <= 13  # 10 chars + "..."
        assert result.content.endswith("...")


class TestGlobalHooks:
    """Tests for global hook registry."""

    def setup_method(self):
        """Clear global hooks before each test."""
        clear_global_hooks()

    def test_register_global_hook(self):
        """Test registering a global hook."""
        config = HookConfig(handler=lambda x: x, name="global_test")
        register_global_hook(HookStage.PRE_EXECUTOR, config)
        assert hooks_manager.GLOBAL_HOOK_REGISTRY is not None
        hooks = hooks_manager.GLOBAL_HOOK_REGISTRY.get_hooks(HookStage.PRE_EXECUTOR)
        assert len(hooks) == 1
        assert hooks[0].name == "global_test"

    def test_unregister_global_hook(self):
        """Test unregistering a global hook."""
        config = HookConfig(handler=lambda x: x, name="global_test")
        register_global_hook(HookStage.PRE_EXECUTOR, config)
        result = unregister_global_hook(HookStage.PRE_EXECUTOR, "global_test")
        assert result is True
        hooks = hooks_manager.GLOBAL_HOOK_REGISTRY.get_hooks(HookStage.PRE_EXECUTOR)
        assert len(hooks) == 0

    def test_clear_global_hooks(self):
        """Test clearing all global hooks."""
        config = HookConfig(handler=lambda x: x, name="global_test")
        register_global_hook(HookStage.PRE_EXECUTOR, config)
        register_global_hook(HookStage.POST_EXECUTOR, config)
        clear_global_hooks()
        assert len(hooks_manager.GLOBAL_HOOK_REGISTRY.get_hooks(HookStage.PRE_EXECUTOR)) == 0
        assert len(hooks_manager.GLOBAL_HOOK_REGISTRY.get_hooks(HookStage.POST_EXECUTOR)) == 0

    @pytest.mark.asyncio
    async def test_global_hook_applies_to_executor(self):
        """Test that global hooks apply to executors."""
        config = HookConfig(
            handler=lambda ctx: HookContext(content=f"GLOBAL_{ctx.content}", stage=ctx.stage, metadata=ctx.metadata),
            name="global_hook",
        )
        register_global_hook(HookStage.PRE_EXECUTOR, config)

        executor = MockExecutor()  # No instance registry
        result = await executor.execute("test")
        assert "GLOBAL_" in result
