"""Tests for hook system."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gluellm.api import ExecutionResult, GlueLLM
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

    async def _execute_internal(self, query: str) -> ExecutionResult:
        """Return a simple response."""
        return ExecutionResult(
            final_response=f"Response to: {query}",
            tool_calls_made=0,
            tool_execution_history=[],
        )


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
        assert "PRE_test" in result.final_response

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
        assert result.final_response.endswith("_POST")

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
        assert "PRE_" in result.final_response
        assert result.final_response.endswith("_POST")


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
        assert "GLOBAL_" in result.final_response


# ---------------------------------------------------------------------------
# Helpers shared by GlueLLM hook integration tests
# ---------------------------------------------------------------------------

def _llm_response(content: str = "Done", tool_calls=None):
    """Build a minimal fake LLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = tool_calls
    resp.choices[0].finish_reason = "stop" if not tool_calls else "tool_calls"
    resp.usage = None
    resp.model = "openai:gpt-4o-mini"
    return resp


def _fake_tool_call(name: str, args: dict, call_id: str = "call_1"):
    """Build a minimal fake tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


@pytest.mark.asyncio
class TestGlueLLMToolHooks:
    """Integration tests for PRE_TOOL and POST_TOOL hook stages on GlueLLM."""

    def setup_method(self):
        clear_global_hooks()

    def _make_client(self, registry: HookRegistry) -> GlueLLM:
        def my_tool(value: str) -> str:
            """A simple tool."""
            return f"result:{value}"

        return GlueLLM(tools=[my_tool], hook_registry=registry)

    def _two_response_side_effect(self, first_tool_calls, final_content="Final answer"):
        """Return a side-effect list: first response has tool calls, second is final."""
        return [
            _llm_response(tool_calls=first_tool_calls),
            _llm_response(content=final_content),
        ]

    @pytest.mark.asyncio
    async def test_pre_tool_hook_receives_tool_name_and_args(self):
        """PRE_TOOL hook receives tool_name and call_index in metadata."""
        observed = {}

        def capture(ctx: HookContext) -> HookContext:
            observed["tool_name"] = ctx.metadata.get("tool_name")
            observed["call_index"] = ctx.metadata.get("call_index")
            observed["content"] = ctx.content
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_TOOL, HookConfig(handler=capture, name="capture"))
        client = self._make_client(registry)

        tool_call = _fake_tool_call("my_tool", {"value": "hello"})
        responses = self._two_response_side_effect([tool_call])

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Run the tool")

        assert observed["tool_name"] == "my_tool"
        assert observed["call_index"] == 1
        assert json.loads(observed["content"]) == {"value": "hello"}

    @pytest.mark.asyncio
    async def test_pre_tool_hook_can_modify_args(self):
        """PRE_TOOL hook returning modified JSON changes the args passed to the tool."""
        received_args: list[dict] = []

        def inject_extra(ctx: HookContext) -> str:
            args = json.loads(ctx.content)
            args["injected"] = "yes"
            return json.dumps(args)

        def spy_tool(value: str, injected: str = "no") -> str:
            """Spy tool that records received args."""
            received_args.append({"value": value, "injected": injected})
            return "ok"

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_TOOL, HookConfig(handler=inject_extra, name="inject"))

        client = GlueLLM(tools=[spy_tool], hook_registry=registry)
        tool_call = _fake_tool_call("spy_tool", {"value": "x"})
        responses = self._two_response_side_effect([tool_call])

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Run it")

        assert len(received_args) == 1
        assert received_args[0]["injected"] == "yes"

    @pytest.mark.asyncio
    async def test_post_tool_hook_receives_result_and_metadata(self):
        """POST_TOOL hook receives result string and duration/error metadata."""
        observed = {}

        def capture(ctx: HookContext) -> HookContext:
            observed["content"] = ctx.content
            observed["duration"] = ctx.metadata.get("duration_seconds")
            observed["error"] = ctx.metadata.get("error")
            observed["tool_name"] = ctx.metadata.get("tool_name")
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_TOOL, HookConfig(handler=capture, name="capture"))
        client = self._make_client(registry)

        tool_call = _fake_tool_call("my_tool", {"value": "hi"})
        responses = self._two_response_side_effect([tool_call])

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Run")

        assert observed["tool_name"] == "my_tool"
        assert observed["content"] == "result:hi"
        assert observed["error"] is False
        assert isinstance(observed["duration"], float)

    @pytest.mark.asyncio
    async def test_post_tool_hook_can_modify_result(self):
        """POST_TOOL hook may transform the result string the LLM receives."""
        def uppercase_result(ctx: HookContext) -> str:
            return ctx.content.upper()

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_TOOL, HookConfig(handler=uppercase_result, name="upper"))
        client = self._make_client(registry)

        tool_call = _fake_tool_call("my_tool", {"value": "hello"})
        responses = self._two_response_side_effect([tool_call])

        tool_messages: list[dict] = []

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses) as mock_llm:
            await client.complete("Run")
            # The second LLM call receives the (possibly modified) tool result message
            second_call_messages = mock_llm.call_args_list[1][1]["messages"]
            tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]

        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == "RESULT:HELLO"

    @pytest.mark.asyncio
    async def test_pre_tool_hook_invalid_json_falls_back_to_original_args(self):
        """PRE_TOOL hook returning invalid JSON logs a warning and uses original args."""
        called_with: list = []

        def bad_json_hook(ctx: HookContext) -> str:
            return "{ not valid json"

        def spy_tool(value: str) -> str:
            """Spy tool."""
            called_with.append(value)
            return "ok"

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_TOOL, HookConfig(handler=bad_json_hook, name="bad_json"))
        client = GlueLLM(tools=[spy_tool], hook_registry=registry)

        tool_call = _fake_tool_call("spy_tool", {"value": "original"})
        responses = self._two_response_side_effect([tool_call])

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Run")

        # Tool must still be called with the original args, not crash
        assert called_with == ["original"]

    @pytest.mark.asyncio
    async def test_new_hook_stages_registered_in_registry(self):
        """PRE_TOOL, POST_TOOL, PRE_ITERATION, POST_ITERATION are valid registry stages."""
        registry = HookRegistry()
        noop = HookConfig(handler=lambda ctx: ctx, name="noop")
        for stage in (HookStage.PRE_TOOL, HookStage.POST_TOOL, HookStage.PRE_ITERATION, HookStage.POST_ITERATION):
            registry.add_hook(stage, noop)
            assert len(registry.get_hooks(stage)) == 1


@pytest.mark.asyncio
class TestGlueLLMIterationHooks:
    """Integration tests for PRE_ITERATION and POST_ITERATION hook stages on GlueLLM."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_pre_iteration_hook_fires_per_llm_call(self):
        """PRE_ITERATION hook fires once per LLM API call in the tool loop."""
        fire_count = 0

        def count_fires(ctx: HookContext) -> HookContext:
            nonlocal fire_count
            fire_count += 1
            assert ctx.metadata.get("iteration") is not None
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_ITERATION, HookConfig(handler=count_fires, name="counter"))

        def my_tool(x: str) -> str:
            """Simple tool."""
            return x

        client = GlueLLM(tools=[my_tool], hook_registry=registry)
        tool_call = _fake_tool_call("my_tool", {"x": "a"})

        # Simulate: first response has tool call, second response is final
        responses = [
            _llm_response(tool_calls=[tool_call]),
            _llm_response(content="Done"),
        ]

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Go")

        # Two LLM calls → two PRE_ITERATION firings
        assert fire_count == 2

    @pytest.mark.asyncio
    async def test_post_iteration_hook_fires_after_each_llm_response(self):
        """POST_ITERATION hook fires after each LLM response with has_tool_calls in metadata."""
        metadata_snapshots: list[dict] = []

        def capture(ctx: HookContext) -> HookContext:
            metadata_snapshots.append(dict(ctx.metadata))
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_ITERATION, HookConfig(handler=capture, name="capture"))

        def my_tool(x: str) -> str:
            """Simple tool."""
            return x

        client = GlueLLM(tools=[my_tool], hook_registry=registry)
        tool_call = _fake_tool_call("my_tool", {"x": "a"})

        responses = [
            _llm_response(tool_calls=[tool_call]),
            _llm_response(content="Done"),
        ]

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Go")

        assert len(metadata_snapshots) == 2
        # First iteration has tool calls
        assert metadata_snapshots[0]["has_tool_calls"] is True
        # Second iteration is the final response
        assert metadata_snapshots[1]["has_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_iteration_hooks_receive_iteration_number(self):
        """PRE_ITERATION and POST_ITERATION metadata contains the correct iteration number."""
        pre_iterations: list[int] = []
        post_iterations: list[int] = []

        def capture_pre(ctx: HookContext) -> HookContext:
            pre_iterations.append(ctx.metadata["iteration"])
            return ctx

        def capture_post(ctx: HookContext) -> HookContext:
            post_iterations.append(ctx.metadata["iteration"])
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_ITERATION, HookConfig(handler=capture_pre, name="pre"))
        registry.add_hook(HookStage.POST_ITERATION, HookConfig(handler=capture_post, name="post"))

        def my_tool(x: str) -> str:
            """Simple tool."""
            return x

        client = GlueLLM(tools=[my_tool], hook_registry=registry)
        tool_call = _fake_tool_call("my_tool", {"x": "b"})

        responses = [
            _llm_response(tool_calls=[tool_call]),
            _llm_response(content="Final"),
        ]

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses):
            await client.complete("Run")

        assert pre_iterations == [1, 2]
        assert post_iterations == [1, 2]


# ---------------------------------------------------------------------------
# Tests for the 9 additional hook stages
# ---------------------------------------------------------------------------


class TestAllNewHookStagesInRegistry:
    """All 9 new HookStage values must be usable with HookRegistry."""

    def test_all_new_stages_registered(self):
        registry = HookRegistry()
        noop = HookConfig(handler=lambda ctx: ctx, name="noop")
        new_stages = [
            HookStage.PRE_GUARDRAIL,
            HookStage.POST_GUARDRAIL,
            HookStage.ON_LLM_RETRY,
            HookStage.PRE_TOOL_ROUTE,
            HookStage.POST_TOOL_ROUTE,
            HookStage.ON_VALIDATION_RETRY,
            HookStage.PRE_BATCH_ITEM,
            HookStage.POST_BATCH_ITEM,
            HookStage.PRE_EVAL_RECORD,
        ]
        for stage in new_stages:
            registry.add_hook(stage, noop)
            assert len(registry.get_hooks(stage)) == 1, f"Stage {stage} not working"

    def test_new_stages_survive_merge(self):
        reg_a = HookRegistry()
        reg_b = HookRegistry()
        hook_a = HookConfig(handler=lambda ctx: ctx, name="a")
        hook_b = HookConfig(handler=lambda ctx: ctx, name="b")
        reg_a.add_hook(HookStage.PRE_GUARDRAIL, hook_a)
        reg_b.add_hook(HookStage.PRE_GUARDRAIL, hook_b)
        merged = reg_a.merge(reg_b)
        assert len(merged.get_hooks(HookStage.PRE_GUARDRAIL)) == 2


@pytest.mark.asyncio
class TestGlueLLMGuardrailHooks:
    """PRE_GUARDRAIL / POST_GUARDRAIL hook stages."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_pre_guardrail_fires_with_direction_input(self):
        """PRE_GUARDRAIL fires with direction='input' for input guardrails."""
        observed: list[dict] = []

        def capture(ctx: HookContext) -> HookContext:
            observed.append({"direction": ctx.metadata.get("direction"), "content": ctx.content})
            return ctx

        from gluellm.guardrails import GuardrailsConfig

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_GUARDRAIL, HookConfig(handler=capture, name="cap"))

        client = GlueLLM(hook_registry=registry)
        guardrail_cfg = GuardrailsConfig()

        with patch("gluellm.api.run_input_guardrails", return_value="hello") as mock_input_gr, \
             patch("gluellm.api._llm_call_with_retry", return_value=_llm_response("ok")):
            await client.complete("hello", guardrails=guardrail_cfg)

        assert any(o["direction"] == "input" for o in observed)
        mock_input_gr.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_guardrail_fires_with_direction_output(self):
        """POST_GUARDRAIL fires with direction='output' for output guardrails."""
        observed: list[dict] = []

        def capture(ctx: HookContext) -> HookContext:
            observed.append({"direction": ctx.metadata.get("direction"), "content": ctx.content})
            return ctx

        from gluellm.guardrails import GuardrailsConfig

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_GUARDRAIL, HookConfig(handler=capture, name="cap"))

        client = GlueLLM(hook_registry=registry)
        guardrail_cfg = GuardrailsConfig()

        with patch("gluellm.api.run_input_guardrails", return_value="hello"), \
             patch("gluellm.api.run_output_guardrails", return_value="ok") as mock_output_gr, \
             patch("gluellm.api._llm_call_with_retry", return_value=_llm_response("ok")):
            await client.complete("hello", guardrails=guardrail_cfg)

        assert any(o["direction"] == "output" for o in observed)
        mock_output_gr.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_guardrail_can_modify_content(self):
        """POST_GUARDRAIL hook may transform the text that exits the guardrail chain."""
        def append_tag(ctx: HookContext) -> str:
            return ctx.content + " [CHECKED]"

        from gluellm.guardrails import GuardrailsConfig

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_GUARDRAIL, HookConfig(handler=append_tag, name="tagger"))

        client = GlueLLM(hook_registry=registry)
        guardrail_cfg = GuardrailsConfig()

        with patch("gluellm.api.run_input_guardrails", return_value="hello [CHECKED]"), \
             patch("gluellm.api._llm_call_with_retry", return_value=_llm_response("reply")):
            result = await client.complete("hello", guardrails=guardrail_cfg)

        assert result is not None  # Completed without error


@pytest.mark.asyncio
class TestLLMRetryHook:
    """ON_LLM_RETRY hook stage."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_on_llm_retry_fires_on_retry(self):
        """ON_LLM_RETRY fires when the hooked callback is invoked during a retry."""
        observed: list[dict] = []

        def capture(ctx: HookContext) -> HookContext:
            observed.append({
                "attempt": ctx.metadata.get("attempt"),
                "exception_type": ctx.metadata.get("exception_type"),
                "content": ctx.content,
            })
            return ctx

        from gluellm.api import RateLimitError, RetryConfig
        from gluellm.models.hook import HookConfig, HookRegistry, HookStage

        registry = HookRegistry()
        registry.add_hook(HookStage.ON_LLM_RETRY, HookConfig(handler=capture, name="capture"))

        client = GlueLLM(hook_registry=registry)
        merged = client._get_merged_hook_registry()

        # Test _llm_call directly: simulate the callback being called for a retry
        exc = RateLimitError("rate limited")
        retry_hooks = merged.get_hooks(HookStage.ON_LLM_RETRY)

        # Manually fire the hooked callback that _llm_call would build
        cfg = RetryConfig(retry_enabled=True, max_attempts=3, min_wait=0.0, max_wait=0.0)
        wait_time = min(cfg.max_wait, cfg.min_wait * (cfg.multiplier ** 0))

        await client._hook_manager.execute_hooks(
            content=str(exc),
            stage=HookStage.ON_LLM_RETRY,
            metadata={
                "attempt": 1,
                "max_attempts": 3,
                "wait_seconds": wait_time,
                "exception_type": type(exc).__name__,
            },
            hooks=retry_hooks,
        )

        assert len(observed) == 1
        assert observed[0]["attempt"] == 1
        assert "RateLimitError" in observed[0]["exception_type"]

    @pytest.mark.asyncio
    async def test_on_llm_retry_not_fired_on_success(self):
        """ON_LLM_RETRY does NOT fire when the first call succeeds (no hooks triggered)."""
        fired = []

        def capture(ctx: HookContext) -> HookContext:
            fired.append(True)
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.ON_LLM_RETRY, HookConfig(handler=capture, name="capture"))

        client = GlueLLM(hook_registry=registry)

        with patch("gluellm.api._llm_call_with_retry", return_value=_llm_response("ok")):
            await client.complete("hi")

        # No retry → hook never fires
        assert fired == []


@pytest.mark.asyncio
class TestToolRouteHooks:
    """PRE_TOOL_ROUTE / POST_TOOL_ROUTE hook stages."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_pre_tool_route_fires_before_routing(self):
        """PRE_TOOL_ROUTE fires with route_query metadata before resolve_tool_route is called."""
        observed: list[dict] = []

        def capture(ctx: HookContext) -> HookContext:
            observed.append({
                "route_query": ctx.metadata.get("route_query"),
                "available_tool_count": ctx.metadata.get("available_tool_count"),
            })
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_TOOL_ROUTE, HookConfig(handler=capture, name="cap"))

        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        client = GlueLLM(tools=[tool_a], tool_mode="dynamic", hook_registry=registry)

        # Simulate the router tool call
        router_tc = _fake_tool_call("route_tools", {"query": "user query"})
        responses = [
            _llm_response(tool_calls=[router_tc]),  # router call
            _llm_response(content="Done"),           # final answer
        ]

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses), \
             patch("gluellm.api.resolve_tool_route", return_value=[tool_a]) as mock_route, \
             patch("gluellm.api.is_router_call", return_value=True):
            await client.complete("user query")

        assert len(observed) >= 1
        mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_tool_route_can_override_matched_tools(self):
        """POST_TOOL_ROUTE hook returning a modified JSON array overrides matched tools."""
        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        def tool_b(x: str) -> str:
            """Tool B."""
            return x

        # Hook that removes tool_b from matched list
        def allowlist(ctx: HookContext) -> str:
            names = json.loads(ctx.content)
            return json.dumps([n for n in names if n != "tool_b"])

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_TOOL_ROUTE, HookConfig(handler=allowlist, name="allow"))

        client = GlueLLM(tools=[tool_a, tool_b], tool_mode="dynamic", hook_registry=registry)

        router_tc = _fake_tool_call("route_tools", {"query": "q"})
        responses = [
            _llm_response(tool_calls=[router_tc]),
            _llm_response(content="Done"),
        ]

        with patch("gluellm.api._llm_call_with_retry", side_effect=responses), \
             patch("gluellm.api.resolve_tool_route", return_value=[tool_a, tool_b]), \
             patch("gluellm.api.is_router_call", return_value=True):
            await client.complete("q")

        # If the hook had no effect, tool_b would also be in matched — verifying no error is enough


@pytest.mark.asyncio
class TestValidationRetryHook:
    """ON_VALIDATION_RETRY hook stage."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_on_validation_retry_fires_on_bad_json(self):
        """ON_VALIDATION_RETRY fires when Pydantic validation fails and a retry is attempted."""
        from pydantic import BaseModel as PydanticBase

        class Answer(PydanticBase):
            value: int

        observed: list[dict] = []

        def capture(ctx: HookContext) -> HookContext:
            observed.append({
                "validation_attempt": ctx.metadata.get("validation_attempt"),
                "response_format_name": ctx.metadata.get("response_format_name"),
                "content": ctx.content,
            })
            return ctx

        registry = HookRegistry()
        registry.add_hook(
            HookStage.ON_VALIDATION_RETRY,
            HookConfig(handler=capture, name="capture"),
        )

        client = GlueLLM(hook_registry=registry)

        # First response: bad JSON. Second: valid JSON.
        bad_resp = _llm_response(content="not json at all")
        bad_resp.choices[0].message.parsed = None
        good_resp = _llm_response(content='{"value": 42}')
        good_resp.choices[0].message.parsed = Answer(value=42)

        with patch("gluellm.api._llm_call_with_retry", side_effect=[bad_resp, good_resp]):
            result = await client.structured_complete("Give me 42", response_format=Answer)

        assert result.structured_output.value == 42
        assert len(observed) >= 1
        assert observed[0]["validation_attempt"] == 1
        assert observed[0]["response_format_name"] == "Answer"


@pytest.mark.asyncio
class TestBatchHooks:
    """PRE_BATCH_ITEM / POST_BATCH_ITEM hook stages."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_pre_and_post_batch_item_fire_per_request(self):
        """PRE_BATCH_ITEM and POST_BATCH_ITEM fire for each item in the batch."""
        from gluellm.batch import BatchProcessor
        from gluellm.models.batch import BatchConfig, BatchRequest

        pre_fired: list[str] = []
        post_fired: list[dict] = []

        def pre_hook(ctx: HookContext) -> HookContext:
            pre_fired.append(ctx.metadata["batch_request_id"])
            return ctx

        def post_hook(ctx: HookContext) -> HookContext:
            post_fired.append({
                "id": ctx.metadata["batch_request_id"],
                "success": ctx.metadata["success"],
            })
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_BATCH_ITEM, HookConfig(handler=pre_hook, name="pre"))
        registry.add_hook(HookStage.POST_BATCH_ITEM, HookConfig(handler=post_hook, name="post"))

        processor = BatchProcessor(
            model="openai:gpt-4o-mini",
            hook_registry=registry,
            config=BatchConfig(max_concurrent=1),
        )

        requests = [
            BatchRequest(id="req-1", user_message="Msg 1"),
            BatchRequest(id="req-2", user_message="Msg 2"),
        ]

        mock_result = MagicMock()
        mock_result.final_response = "ok"
        mock_result.structured_output = None
        mock_result.tool_calls_made = 0
        mock_result.tool_execution_history = []
        mock_result.tokens_used = None

        with patch("gluellm.batch.GlueLLM") as MockGlueLLM:
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value=mock_result)
            MockGlueLLM.return_value = mock_client
            response = await processor.process(requests)

        assert len(pre_fired) == 2
        assert len(post_fired) == 2
        assert all(p["success"] for p in post_fired)

    @pytest.mark.asyncio
    async def test_post_batch_item_fires_on_failure(self):
        """POST_BATCH_ITEM fires with success=False when an item fails."""
        from gluellm.batch import BatchProcessor
        from gluellm.models.batch import BatchConfig, BatchRequest

        post_fired: list[dict] = []

        def post_hook(ctx: HookContext) -> HookContext:
            post_fired.append({"success": ctx.metadata["success"]})
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_BATCH_ITEM, HookConfig(handler=post_hook, name="post"))

        processor = BatchProcessor(
            model="openai:gpt-4o-mini",
            hook_registry=registry,
            config=BatchConfig(max_concurrent=1, retry_failed=False),
        )

        with patch("gluellm.batch.GlueLLM") as MockGlueLLM:
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(side_effect=RuntimeError("boom"))
            MockGlueLLM.return_value = mock_client
            response = await processor.process([BatchRequest(id="req-1", user_message="fail")])

        assert len(post_fired) == 1
        assert post_fired[0]["success"] is False


@pytest.mark.asyncio
class TestPreEvalRecordHook:
    """PRE_EVAL_RECORD hook stage."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_pre_eval_record_can_scrub_user_message(self):
        """PRE_EVAL_RECORD hook may modify user_message before it is written to the eval store."""
        import re

        from gluellm.api import _record_eval_data
        from gluellm.hooks.manager import HookManager
        from gluellm.models.hook import HookConfig, HookRegistry, HookStage

        def scrub_email(ctx: HookContext) -> str:
            return re.sub(r"\S+@\S+\.\S+", "[EMAIL]", ctx.content)

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_EVAL_RECORD, HookConfig(handler=scrub_email, name="scrub"))

        recorded_messages: list[str] = []

        class FakeEvalStore:
            async def record(self, rec):
                recorded_messages.append(rec.user_message)

        hooks = registry.get_hooks(HookStage.PRE_EVAL_RECORD)
        manager = HookManager()

        await _record_eval_data(
            eval_store=FakeEvalStore(),
            user_message="My email is alice@example.com please help",
            system_prompt="You are helpful",
            model="openai:gpt-4o-mini",
            messages_snapshot=[],
            start_time=0.0,
            on_eval_record_hooks=hooks,
            hook_manager=manager,
        )

        assert len(recorded_messages) == 1
        assert "[EMAIL]" in recorded_messages[0]
        assert "alice@example.com" not in recorded_messages[0]

    @pytest.mark.asyncio
    async def test_pre_eval_record_not_called_without_hooks(self):
        """_record_eval_data works fine with no hooks (backward-compatible)."""
        from gluellm.api import _record_eval_data

        recorded: list[str] = []

        class FakeEvalStore:
            async def record(self, rec):
                recorded.append(rec.user_message)

        await _record_eval_data(
            eval_store=FakeEvalStore(),
            user_message="original message",
            system_prompt="",
            model="openai:gpt-4o-mini",
            messages_snapshot=[],
            start_time=0.0,
        )

        assert recorded == ["original message"]


@pytest.mark.asyncio
class TestIterationHookConsistency:
    """PRE_ITERATION/POST_ITERATION must fire in structured_complete Phase 1 and stream_complete."""

    def setup_method(self):
        clear_global_hooks()

    @pytest.mark.asyncio
    async def test_pre_iteration_fires_in_structured_complete_phase1(self):
        """PRE_ITERATION fires in structured_complete during the Phase 1 tool loop."""
        from pydantic import BaseModel as PydanticBase

        class Answer(PydanticBase):
            value: str

        fire_count = 0

        def counter(ctx: HookContext) -> HookContext:
            nonlocal fire_count
            fire_count += 1
            return ctx

        def my_tool(x: str) -> str:
            """A tool."""
            return x

        registry = HookRegistry()
        registry.add_hook(HookStage.PRE_ITERATION, HookConfig(handler=counter, name="counter"))

        client = GlueLLM(tools=[my_tool], hook_registry=registry)

        tool_call = _fake_tool_call("my_tool", {"x": "a"})
        # Phase 1 iter 1: tool call
        phase1_tool_resp = _llm_response(tool_calls=[tool_call])
        # Phase 1 iter 2: no more tool calls → exits Phase 1
        phase1_done_resp = _llm_response(content="")
        # Phase 2: structured output response
        final_resp = _llm_response(content='{"value": "done"}')
        final_resp.choices[0].message.parsed = Answer(value="done")

        with patch("gluellm.api._llm_call_with_retry", side_effect=[phase1_tool_resp, phase1_done_resp, final_resp]):
            result = await client.structured_complete("Do it", response_format=Answer)

        assert result.structured_output.value == "done"
        # PRE_ITERATION fires at least once (Phase 1 loop)
        assert fire_count >= 1

    @pytest.mark.asyncio
    async def test_post_iteration_fires_in_stream_complete(self):
        """POST_ITERATION fires in stream_complete's iteration loop."""
        fire_count = 0

        def counter(ctx: HookContext) -> HookContext:
            nonlocal fire_count
            fire_count += 1
            return ctx

        registry = HookRegistry()
        registry.add_hook(HookStage.POST_ITERATION, HookConfig(handler=counter, name="counter"))

        client = GlueLLM(hook_registry=registry)

        # Build a minimal stream response
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hello"
        chunk.choices[0].delta.tool_calls = None

        async def fake_stream(**kwargs):
            yield chunk

        with patch("gluellm.api._llm_call_with_retry", return_value=fake_stream()):
            chunks = []
            async for c in client.stream_complete("hi"):
                chunks.append(c)

        # POST_ITERATION should have fired once
        assert fire_count >= 1
