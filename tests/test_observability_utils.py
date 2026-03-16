"""Tests for observability logging utilities — decorators and context managers."""

import logging

import pytest

from gluellm.observability.logging_utils import (
    log_async_function_call,
    log_function_call,
    log_operation,
    log_timing,
    timed_operation,
)

_UTILS_LOGGER = "gluellm.observability.logging_utils"


class TestLogFunctionCall:
    def test_decorated_function_returns_value(self):
        @log_function_call()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self):
        @log_function_call()
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    def test_propagates_exceptions(self):
        @log_function_call()
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

    def test_logs_call(self, caplog):
        @log_function_call(log_args=True, log_result=True)
        def multiply(x, y):
            return x * y

        with caplog.at_level(logging.DEBUG):
            result = multiply(3, 4)
        assert result == 12

    def test_no_args_logging(self, caplog):
        @log_function_call(log_args=False)
        def secret():
            return "hidden"

        with caplog.at_level(logging.DEBUG):
            secret()

    def test_custom_logger(self):
        custom_logger = logging.getLogger("test.custom")

        @log_function_call(logger=custom_logger)
        def greet():
            return "hi"

        assert greet() == "hi"


@pytest.mark.asyncio
class TestLogAsyncFunctionCall:
    async def test_async_decorated_function_returns_value(self):
        @log_async_function_call()
        async def async_add(a, b):
            return a + b

        assert await async_add(2, 3) == 5

    async def test_preserves_function_name(self):
        @log_async_function_call()
        async def my_async_func():
            pass

        assert my_async_func.__name__ == "my_async_func"

    async def test_propagates_exceptions(self):
        @log_async_function_call()
        async def async_fail():
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            await async_fail()

    async def test_logs_result(self, caplog):
        @log_async_function_call(log_args=True, log_result=True)
        async def compute(x):
            return x * 2

        with caplog.at_level(logging.DEBUG):
            result = await compute(5)
        assert result == 10


class TestTimedOperation:
    def test_basic_timing(self, caplog):
        test_logger = logging.getLogger("test.timed")
        with caplog.at_level(logging.DEBUG, logger="test.timed"):
            with timed_operation("test_op", logger=test_logger):
                pass
        assert any("test_op" in r.message for r in caplog.records)

    def test_logs_start_and_completion(self, caplog):
        test_logger = logging.getLogger("test.timed")
        with caplog.at_level(logging.DEBUG, logger="test.timed"):
            with timed_operation("my_op", logger=test_logger):
                pass
        messages = [r.message for r in caplog.records]
        assert any("Starting my_op" in m for m in messages)
        assert any("Completed my_op" in m for m in messages)

    def test_custom_log_level(self, caplog):
        test_logger = logging.getLogger("test.timed")
        with caplog.at_level(logging.WARNING, logger="test.timed"):
            with timed_operation("warn_op", logger=test_logger, log_level=logging.WARNING):
                pass
        assert any("warn_op" in r.message for r in caplog.records)

    def test_log_timing_is_alias(self):
        assert log_timing is timed_operation


class TestLogOperation:
    def test_logs_success(self, caplog):
        test_logger = logging.getLogger("test.logop")
        with caplog.at_level(logging.INFO, logger="test.logop"):
            with log_operation("good_op", logger=test_logger):
                pass
        messages = [r.message for r in caplog.records]
        assert any("Starting good_op" in m for m in messages)
        assert any("Completed good_op" in m for m in messages)

    def test_logs_failure(self, caplog):
        test_logger = logging.getLogger("test.logop")
        with caplog.at_level(logging.INFO, logger="test.logop"):
            with pytest.raises(ValueError):
                with log_operation("bad_op", logger=test_logger):
                    raise ValueError("fail")
        messages = [r.message for r in caplog.records]
        assert any("Failed bad_op" in m for m in messages)

    def test_propagates_exception(self):
        with pytest.raises(TypeError):
            with log_operation("err_op"):
                raise TypeError("type err")
