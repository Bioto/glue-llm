"""Tests for the telemetry module — tracing, span attributes, and metrics helpers."""

from unittest.mock import MagicMock

import gluellm.telemetry as telemetry_mod
from gluellm.telemetry import (
    get_default_mlflow_run,
    is_mlflow_enabled,
    is_tracing_enabled,
    record_cost,
    record_token_usage,
    record_tool_execution,
    set_span_attributes,
    shutdown_telemetry,
    trace_llm_call,
)


class TestTracingDefaults:
    """Tracing is disabled by default."""

    def test_tracing_disabled_by_default(self):
        assert is_tracing_enabled() is False

    def test_mlflow_disabled_by_default(self):
        assert is_mlflow_enabled() is False

    def test_default_mlflow_run_is_none(self):
        assert get_default_mlflow_run() is None


class TestTraceLlmCallNoOp:
    """When tracing is disabled, trace_llm_call yields a no-op span."""

    def test_yields_noop_span(self):
        with trace_llm_call("openai:gpt-4o-mini", [{"role": "user", "content": "hi"}]) as span:
            span.set_attribute("key", "value")
            span.set_status("ok")
            span.record_exception(Exception("test"))

    def test_noop_span_with_tools(self):
        tools = [lambda x: x]
        with trace_llm_call("openai:gpt-4o-mini", [], tools=tools, correlation_id="c1") as span:
            assert span is not None


class TestSetSpanAttributes:
    """set_span_attributes is a no-op when tracing is disabled."""

    def test_noop_when_disabled(self):
        mock_span = MagicMock()
        set_span_attributes(mock_span, key="value", count=42)
        mock_span.set_attribute.assert_not_called()


class TestRecordTokenUsage:
    """record_token_usage is a no-op when tracing is disabled."""

    def test_noop_when_disabled(self):
        mock_span = MagicMock()
        record_token_usage(mock_span, {"prompt": 10, "completion": 5, "total": 15}, cost_usd=0.001)
        mock_span.set_attribute.assert_not_called()


class TestRecordCost:
    def test_noop_when_disabled(self):
        mock_span = MagicMock()
        record_cost(mock_span, cost_usd=0.5, model="openai:gpt-4o")
        mock_span.set_attribute.assert_not_called()


class TestRecordToolExecution:
    def test_noop_when_disabled(self):
        mock_span = MagicMock()
        record_tool_execution(mock_span, "my_tool", {"arg": "val"}, "result", error=False)
        mock_span.set_attribute.assert_not_called()


class TestTracingEnabled:
    """Test behaviour when tracing is artificially enabled."""

    def setup_method(self):
        self._orig_enabled = telemetry_mod._tracing_enabled

    def teardown_method(self):
        telemetry_mod._tracing_enabled = self._orig_enabled

    def test_set_span_attributes_calls_span(self):
        telemetry_mod._tracing_enabled = True
        mock_span = MagicMock()
        set_span_attributes(mock_span, llm_model="gpt-4o", tokens=100)
        assert mock_span.set_attribute.call_count == 2

    def test_record_token_usage_sets_attributes(self):
        telemetry_mod._tracing_enabled = True
        mock_span = MagicMock()
        record_token_usage(mock_span, {"prompt": 10, "completion": 5, "total": 15}, cost_usd=0.01)
        calls = {call.args[0] for call in mock_span.set_attribute.call_args_list}
        assert "llm.tokens.prompt" in calls
        assert "llm.tokens.total" in calls
        assert "llm.cost.usd" in calls

    def test_record_cost_sets_attributes(self):
        telemetry_mod._tracing_enabled = True
        mock_span = MagicMock()
        record_cost(mock_span, 0.05, model="openai:gpt-4o")
        calls = {call.args[0] for call in mock_span.set_attribute.call_args_list}
        assert "llm.cost.usd" in calls
        assert "llm.cost.model" in calls

    def test_record_tool_execution_sets_attributes(self):
        telemetry_mod._tracing_enabled = True
        mock_span = MagicMock()
        record_tool_execution(mock_span, "get_weather", {"city": "NYC"}, "Sunny", error=False)
        calls = {call.args[0] for call in mock_span.set_attribute.call_args_list}
        assert "tool.get_weather.called" in calls
        assert "tool.get_weather.error" in calls
        assert "tool.get_weather.arg_count" in calls

    def test_set_span_attributes_handles_complex_types(self):
        telemetry_mod._tracing_enabled = True
        mock_span = MagicMock()
        set_span_attributes(mock_span, data={"nested": True}, items=[1, 2, 3])
        assert mock_span.set_attribute.call_count == 2


class TestShutdownTelemetry:
    """shutdown_telemetry resets global state."""

    def test_shutdown_resets_state(self):
        telemetry_mod._tracing_enabled = True
        telemetry_mod._mlflow_enabled = True
        shutdown_telemetry()
        assert telemetry_mod._tracing_enabled is False
        assert telemetry_mod._mlflow_enabled is False
        assert telemetry_mod._tracer is None
        assert telemetry_mod._mlflow_client is None
