"""Tests for the runtime module — correlation IDs, request metadata, and shutdown."""

import asyncio

import pytest

from gluellm.runtime.context import (
    clear_correlation_id,
    clear_request_metadata,
    get_context_dict,
    get_correlation_id,
    get_request_metadata,
    set_correlation_id,
    set_request_metadata,
    with_correlation_id,
)
from gluellm.runtime.shutdown import (
    ShutdownContext,
    _shutdown_event,
    decrement_in_flight,
    execute_shutdown_callbacks,
    get_in_flight_count,
    graceful_shutdown,
    increment_in_flight,
    is_shutting_down,
    register_shutdown_callback,
    unregister_shutdown_callback,
)

# ---------------------------------------------------------------------------
# Correlation ID tests
# ---------------------------------------------------------------------------


class TestCorrelationId:
    def setup_method(self):
        clear_correlation_id()

    def teardown_method(self):
        clear_correlation_id()

    def test_default_is_none(self):
        assert get_correlation_id() is None

    def test_set_and_get(self):
        set_correlation_id("req-abc")
        assert get_correlation_id() == "req-abc"

    def test_set_generates_uuid_when_none(self):
        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) > 0
        assert get_correlation_id() == cid

    def test_clear(self):
        set_correlation_id("req-1")
        clear_correlation_id()
        assert get_correlation_id() is None


class TestWithCorrelationId:
    def setup_method(self):
        clear_correlation_id()

    def teardown_method(self):
        clear_correlation_id()

    def test_context_manager_sets_and_restores(self):
        set_correlation_id("outer")
        with with_correlation_id("inner") as cid:
            assert cid == "inner"
            assert get_correlation_id() == "inner"
        assert get_correlation_id() == "outer"

    def test_context_manager_auto_generates_uuid(self):
        with with_correlation_id() as cid:
            assert cid is not None
            assert get_correlation_id() == cid
        assert get_correlation_id() is None

    def test_context_manager_restores_on_exception(self):
        set_correlation_id("original")
        with pytest.raises(ValueError), with_correlation_id("temp"):
            raise ValueError("boom")
        assert get_correlation_id() == "original"


# ---------------------------------------------------------------------------
# Request metadata tests
# ---------------------------------------------------------------------------


class TestRequestMetadata:
    def setup_method(self):
        clear_request_metadata()

    def teardown_method(self):
        clear_request_metadata()

    def test_default_is_empty_dict(self):
        assert get_request_metadata() == {}

    def test_set_and_get(self):
        set_request_metadata(user_id="u1", action="test")
        meta = get_request_metadata()
        assert meta["user_id"] == "u1"
        assert meta["action"] == "test"

    def test_accumulates_across_calls(self):
        set_request_metadata(a=1)
        set_request_metadata(b=2)
        meta = get_request_metadata()
        assert meta == {"a": 1, "b": 2}

    def test_later_values_overwrite(self):
        set_request_metadata(x="old")
        set_request_metadata(x="new")
        assert get_request_metadata()["x"] == "new"

    def test_returns_copy(self):
        set_request_metadata(key="val")
        m1 = get_request_metadata()
        m1["key"] = "mutated"
        assert get_request_metadata()["key"] == "val"

    def test_clear(self):
        set_request_metadata(key="val")
        clear_request_metadata()
        assert get_request_metadata() == {}


# ---------------------------------------------------------------------------
# Context dict tests
# ---------------------------------------------------------------------------


class TestGetContextDict:
    def setup_method(self):
        clear_correlation_id()
        clear_request_metadata()

    def teardown_method(self):
        clear_correlation_id()
        clear_request_metadata()

    def test_contains_both_fields(self):
        set_correlation_id("cid-1")
        set_request_metadata(env="test")
        ctx = get_context_dict()
        assert ctx["correlation_id"] == "cid-1"
        assert ctx["metadata"]["env"] == "test"

    def test_defaults_when_unset(self):
        ctx = get_context_dict()
        assert ctx["correlation_id"] is None
        assert ctx["metadata"] == {}


# ---------------------------------------------------------------------------
# Shutdown state tests
# ---------------------------------------------------------------------------


class TestShutdownState:
    """Tests for in-flight counting and shutdown event."""

    def setup_method(self):
        _shutdown_event.clear()
        # Reset in-flight to 0
        from gluellm.runtime import shutdown as _mod

        with _mod._shutdown_lock:
            _mod._in_flight_requests = 0

    def teardown_method(self):
        _shutdown_event.clear()
        from gluellm.runtime import shutdown as _mod

        with _mod._shutdown_lock:
            _mod._in_flight_requests = 0
            _mod._shutdown_callbacks.clear()

    def test_is_shutting_down_default(self):
        assert is_shutting_down() is False

    def test_in_flight_increment_decrement(self):
        assert get_in_flight_count() == 0
        increment_in_flight()
        assert get_in_flight_count() == 1
        increment_in_flight()
        assert get_in_flight_count() == 2
        decrement_in_flight()
        assert get_in_flight_count() == 1
        decrement_in_flight()
        assert get_in_flight_count() == 0

    def test_decrement_does_not_go_negative(self):
        decrement_in_flight()
        assert get_in_flight_count() == 0


class TestShutdownContext:
    def setup_method(self):
        _shutdown_event.clear()
        from gluellm.runtime import shutdown as _mod

        with _mod._shutdown_lock:
            _mod._in_flight_requests = 0

    def teardown_method(self):
        _shutdown_event.clear()
        from gluellm.runtime import shutdown as _mod

        with _mod._shutdown_lock:
            _mod._in_flight_requests = 0

    def test_sync_context_manager(self):
        with ShutdownContext():
            assert get_in_flight_count() == 1
        assert get_in_flight_count() == 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with ShutdownContext():
            assert get_in_flight_count() == 1
        assert get_in_flight_count() == 0

    def test_decrements_on_exception(self):
        with pytest.raises(RuntimeError), ShutdownContext():
            assert get_in_flight_count() == 1
            raise RuntimeError("fail")
        assert get_in_flight_count() == 0

    def test_raises_when_shutting_down(self):
        _shutdown_event.set()
        with pytest.raises(RuntimeError, match="shutdown in progress"), ShutdownContext():
            pass


class TestShutdownCallbacks:
    def setup_method(self):
        _shutdown_event.clear()
        from gluellm.runtime import shutdown as _mod

        with _mod._shutdown_lock:
            _mod._in_flight_requests = 0
            _mod._shutdown_callbacks.clear()

    def teardown_method(self):
        _shutdown_event.clear()
        from gluellm.runtime import shutdown as _mod

        with _mod._shutdown_lock:
            _mod._in_flight_requests = 0
            _mod._shutdown_callbacks.clear()

    def test_register_and_execute_sync_callback(self):
        called = []

        def cb():
            called.append(True)

        register_shutdown_callback(cb)
        asyncio.get_event_loop().run_until_complete(execute_shutdown_callbacks())
        assert called == [True]

    def test_register_and_execute_async_callback(self):
        called = []

        async def cb():
            called.append(True)

        register_shutdown_callback(cb)
        asyncio.get_event_loop().run_until_complete(execute_shutdown_callbacks())
        assert called == [True]

    def test_unregister_callback(self):
        called = []

        def cb():
            called.append(True)

        register_shutdown_callback(cb)
        unregister_shutdown_callback(cb)
        asyncio.get_event_loop().run_until_complete(execute_shutdown_callbacks())
        assert called == []

    def test_callback_error_does_not_propagate(self):
        def bad_cb():
            raise ValueError("oops")

        register_shutdown_callback(bad_cb)
        asyncio.get_event_loop().run_until_complete(execute_shutdown_callbacks())

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sets_event(self):
        assert not is_shutting_down()
        await graceful_shutdown(max_wait_time=0.1)
        assert is_shutting_down()
