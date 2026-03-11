"""Tests for the GlueLLM API."""

from types import SimpleNamespace
from typing import Annotated
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from gluellm.api import (
    ExecutionResult,
    GlueLLM,
    _condense_tool_round,
    complete,
    stream_complete,
    structured_complete,
)
from gluellm.events import ProcessEvent

# Mark all tests as async
pytestmark = pytest.mark.asyncio


# Test fixtures


def dummy_tool(value: str) -> str:
    """A dummy tool for testing.

    Args:
        value: A test value to echo back
    """
    return f"Tool received: {value}"


def math_tool(a: int, b: int, operation: str = "add") -> str:
    """Perform a math operation.

    Args:
        a: First number
        b: Second number
        operation: Operation to perform (add, multiply, subtract)
    """
    if operation == "add":
        return str(a + b)
    if operation == "multiply":
        return str(a * b)
    if operation == "subtract":
        return str(a - b)
    return "Unknown operation"


# Test classes


class TestBasicCompletion:
    """Test basic completion functionality."""

    async def test_simple_completion_function(self):
        """Test the complete() convenience function."""
        result = await complete(
            user_message="Say hello",
            system_prompt="You are a friendly assistant. Always respond with 'Hello!'",
        )

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.final_response, str)
        assert len(result.final_response) > 0
        assert result.tool_calls_made == 0
        assert len(result.tool_execution_history) == 0

    async def test_client_completion(self):
        """Test completion using GlueLLM client."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )

        result = await client.complete("What is 2+2?")

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.final_response, str)
        assert result.tool_calls_made == 0

    async def test_custom_model(self):
        """Test completion with custom model."""
        result = await complete(
            user_message="Hello",
            model="openai:gpt-4o-mini",
        )

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.final_response, str)


class TestStructuredOutput:
    """Test structured output functionality."""

    async def test_simple_structured_output(self):
        """Test basic structured output."""

        class SimpleResponse(BaseModel):
            message: Annotated[str, Field(description="A simple message")]
            number: Annotated[int, Field(description="A number")]

        result = await structured_complete(
            user_message="Return a message 'test' and the number 42",
            response_format=SimpleResponse,
            system_prompt="Extract the requested information.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, SimpleResponse)
        assert isinstance(result.structured_output.message, str)
        assert isinstance(result.structured_output.number, int)

    async def test_nested_structured_output(self):
        """Test structured output with nested models."""

        class Address(BaseModel):
            street: Annotated[str, Field(description="Street address")]
            city: Annotated[str, Field(description="City name")]

        class Person(BaseModel):
            name: Annotated[str, Field(description="Person's name")]
            age: Annotated[int, Field(description="Person's age")]
            address: Annotated[Address, Field(description="Person's address")]

        result = await structured_complete(
            user_message="Extract: John Doe, 30 years old, lives at 123 Main St, Springfield",
            response_format=Person,
            system_prompt="Extract person information from the text.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, Person)
        assert isinstance(result.structured_output.name, str)
        assert isinstance(result.structured_output.age, int)
        assert isinstance(result.structured_output.address, Address)
        assert isinstance(result.structured_output.address.city, str)

    async def test_structured_output_with_client(self):
        """Test structured output using GlueLLM client."""

        class Color(BaseModel):
            name: Annotated[str, Field(description="Color name")]
            hex_code: Annotated[str, Field(description="Hex color code")]

        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You extract color information.",
        )

        result = await client.structured_complete(
            user_message="The color red has hex code #FF0000",
            response_format=Color,
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, Color)
        assert isinstance(result.structured_output.name, str)
        assert isinstance(result.structured_output.hex_code, str)

    async def test_structured_output_with_tools(self):
        """Test structured output with tool execution."""

        class WeatherInfo(BaseModel):
            city: Annotated[str, Field(description="City name")]
            temperature: Annotated[int, Field(description="Temperature in Celsius")]
            summary: Annotated[str, Field(description="Weather summary")]

        def get_temperature(city: str) -> int:
            """Get temperature for a city."""
            # Mock data for testing
            temps = {"Paris": 18, "Tokyo": 25, "London": 15}
            return temps.get(city, 20)

        result = await structured_complete(
            user_message="Get the temperature for Paris and provide a weather summary",
            response_format=WeatherInfo,
            tools=[get_temperature],
            system_prompt="Use the get_temperature tool to get weather data.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, WeatherInfo)
        assert isinstance(result.structured_output.city, str)
        assert isinstance(result.structured_output.temperature, int)
        assert isinstance(result.structured_output.summary, str)
        # Verify that tools were actually called
        assert result.tool_calls_made >= 1
        assert len(result.tool_execution_history) >= 1

    async def test_structured_output_with_multiple_tool_calls(self):
        """Test structured output with multiple tool calls."""

        class MathResult(BaseModel):
            calculation: Annotated[str, Field(description="The calculation performed")]
            result: Annotated[int, Field(description="The result")]
            explanation: Annotated[str, Field(description="Explanation of steps")]

        result = await structured_complete(
            user_message="Calculate 5 + 3 and 10 * 2 using the math_tool, then provide a summary",
            response_format=MathResult,
            tools=[math_tool],
            system_prompt="Use math_tool to perform calculations.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, MathResult)
        assert isinstance(result.structured_output.calculation, str)
        assert isinstance(result.structured_output.result, int)
        # Should have called the tool at least once
        assert result.tool_calls_made >= 1


class TestToolExecution:
    """Test automatic tool execution."""

    async def test_single_tool_execution(self):
        """Test execution of a single tool."""
        result = await complete(
            user_message="Use the dummy tool with value 'test123'",
            system_prompt="You are an assistant that uses tools. Use the dummy_tool when asked.",
            tools=[dummy_tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(result.tool_execution_history) >= 1
        assert result.tool_execution_history[0]["tool_name"] == "dummy_tool"

    async def test_multiple_tool_calls(self):
        """Test multiple tool calls in sequence."""
        result = await complete(
            user_message="First use dummy_tool with 'first', then use it again with 'second'",
            system_prompt="You are an assistant that uses tools as requested.",
            tools=[dummy_tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 2
        assert len(result.tool_execution_history) >= 2

    async def test_tool_with_parameters(self):
        """Test tool execution with multiple parameters."""
        result = await complete(
            user_message="Use the math tool to add 5 and 3",
            system_prompt="You are a math assistant. Use the math_tool for calculations.",
            tools=[math_tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1

        # Check tool was called with correct params
        history = result.tool_execution_history
        assert len(history) >= 1
        assert history[0]["tool_name"] == "math_tool"
        assert "a" in history[0]["arguments"]
        assert "b" in history[0]["arguments"]

    async def test_tool_execution_disabled(self):
        """Test that tool execution can be disabled."""
        result = await complete(
            user_message="Use the dummy tool",
            system_prompt="You are an assistant with tools.",
            tools=[dummy_tool],
            execute_tools=False,  # Disable execution
        )

        assert isinstance(result, ExecutionResult)
        # Tools should not be executed when disabled
        assert result.tool_calls_made == 0

    async def test_max_iterations(self):
        """Test max iterations limit."""
        # This test ensures we don't get stuck in infinite loops
        result = await complete(
            user_message="Test",
            system_prompt="You are an assistant.",
            tools=[dummy_tool],
            max_tool_iterations=2,  # Very low limit
        )

        assert isinstance(result, ExecutionResult)
        # Should complete without hanging
        assert result.tool_calls_made <= 2


class TestConversationState:
    """Test conversation state management."""

    async def test_conversation_persists(self):
        """Test that conversation history persists across calls."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant. Remember previous messages.",
        )

        # First message
        result1 = await client.complete("My name is Alice")
        assert isinstance(result1, ExecutionResult)

        # Second message referencing first
        result2 = await client.complete("What is my name?")
        assert isinstance(result2, ExecutionResult)
        # The response should reference Alice (though we can't assert exact text)
        assert len(result2.final_response) > 0

    async def test_conversation_reset(self):
        """Test conversation reset functionality."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )

        # Add some messages
        await client.complete("Remember the number 42")

        # Reset conversation
        client.reset_conversation()

        # Check conversation was reset
        assert len(client._conversation.messages) == 0

    async def test_tool_calls_persist_in_conversation(self):
        """Test that tool calls are part of conversation history."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a math assistant.",
            tools=[math_tool],
        )

        # First calculation
        result1 = await client.complete("What is 10 + 5?")
        initial_message_count = len(client._conversation.messages)

        # Second calculation (should have context)
        result2 = await client.complete("Now multiply that by 2")
        final_message_count = len(client._conversation.messages)

        # Conversation should grow
        assert final_message_count > initial_message_count


class TestMultipleTools:
    """Test scenarios with multiple tools."""

    async def test_multiple_tools_available(self):
        """Test completion with multiple tools available."""

        def tool_a(x: str) -> str:
            """Tool A.

            Args:
                x: Input for tool A
            """
            return f"Tool A: {x}"

        def tool_b(y: str) -> str:
            """Tool B.

            Args:
                y: Input for tool B
            """
            return f"Tool B: {y}"

        result = await complete(
            user_message="Use tool_a with 'test'",
            system_prompt="You have access to multiple tools. Use them as requested.",
            tools=[tool_a, tool_b],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        # Should have used tool_a
        assert any(h["tool_name"] == "tool_a" for h in result.tool_execution_history)

    async def test_using_different_tools_sequentially(self):
        """Test using different tools in the same request."""

        def get_weather(city: str) -> str:
            """Get weather.

            Args:
                city: City name
            """
            return f"Weather in {city}: Sunny"

        def get_time(timezone: str) -> str:
            """Get time.

            Args:
                timezone: Timezone
            """
            return f"Time in {timezone}: 12:00 PM"

        result = await complete(
            user_message="What's the weather in Tokyo and what time is it there?",
            system_prompt="Use available tools to answer questions.",
            tools=[get_weather, get_time],
        )

        assert isinstance(result, ExecutionResult)
        # Should use both tools
        tool_names = {h["tool_name"] for h in result.tool_execution_history}
        # At minimum one tool should be called
        assert len(tool_names) >= 1


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_tool_not_found(self):
        """Test handling when LLM tries to call non-existent tool."""
        # This is tricky to test since the LLM won't normally call non-existent tools
        # We'll test the internal _find_tool method instead
        client = GlueLLM(tools=[dummy_tool])

        found = client._find_tool("dummy_tool")
        assert found is not None

        not_found = client._find_tool("nonexistent_tool")
        assert not_found is None

    async def test_tool_execution_error(self):
        """Test handling of errors during tool execution."""

        def error_tool(x: str) -> str:
            """A tool that raises an error.

            Args:
                x: Input that causes error
            """
            raise ValueError("Intentional error for testing")

        result = await complete(
            user_message="Use error_tool with 'test'",
            system_prompt="Use the error_tool when asked.",
            tools=[error_tool],
        )

        assert isinstance(result, ExecutionResult)
        # Should handle error gracefully
        if result.tool_execution_history:
            # Error should be captured in result
            assert (
                "Error" in result.tool_execution_history[0]["result"]
                or "error" in result.tool_execution_history[0]["result"]
            )


class TestToolDiscovery:
    """Test tool discovery and validation edge cases.

    These tests are now async to match the module-level asyncio marker.
    """

    async def test_multiple_tools_same_name(self):
        """Test behavior when multiple tools have the same name."""

        # Create two tools with same name
        def tool1(x: str) -> str:
            """First tool."""
            return f"Tool 1: {x}"

        def tool2(x: str) -> str:
            """Second tool."""
            return f"Tool 2: {x}"

        # Rename second tool to match first
        tool2.__name__ = "tool1"

        client = GlueLLM(tools=[tool1, tool2])

        # _find_tool should return the first match
        found = client._find_tool("tool1")
        assert found is not None
        # Should return first tool in list
        assert found is tool1

    async def test_tool_with_no_docstring(self):
        """Test tool with no docstring."""

        def tool_no_doc(x: str) -> str:
            return f"No doc: {x}"

        # Ensure no docstring
        tool_no_doc.__doc__ = None

        client = GlueLLM(tools=[tool_no_doc])
        found = client._find_tool("tool_no_doc")
        assert found is not None
        assert found is tool_no_doc

    async def test_lambda_function_as_tool(self):
        """Test lambda function as tool."""
        lambda_tool = lambda x: f"Lambda: {x}"  # noqa: E731
        lambda_tool.__name__ = "lambda_tool"

        client = GlueLLM(tools=[lambda_tool])
        found = client._find_tool("lambda_tool")
        assert found is not None
        assert found is lambda_tool

    async def test_partial_function_as_tool(self):
        """Test partial function as tool."""
        from functools import partial

        def base_tool(x: str, y: int = 5) -> str:
            """Base tool."""
            return f"{x}:{y}"

        partial_tool = partial(base_tool, y=10)
        partial_tool.__name__ = "partial_tool"

        client = GlueLLM(tools=[partial_tool])
        found = client._find_tool("partial_tool")
        assert found is not None
        assert found is partial_tool

    async def test_tool_name_case_sensitivity(self):
        """Test that tool name lookup is case-sensitive."""

        def tool_with_caps(x: str) -> str:
            """Tool with capital letters."""
            return x

        client = GlueLLM(tools=[tool_with_caps])

        # Case-sensitive match
        found = client._find_tool("tool_with_caps")
        assert found is not None

        # Case-insensitive should not match
        not_found = client._find_tool("toolwithcaps")
        assert not_found is None

    async def test_class_method_as_tool(self):
        """Test class method as tool."""

        class ToolClass:
            def method_tool(self, x: str) -> str:
                """Method tool."""
                return f"Method: {x}"

        instance = ToolClass()
        method_tool = instance.method_tool
        # Bound methods already have __name__ from the underlying function

        client = GlueLLM(tools=[method_tool])
        found = client._find_tool("method_tool")
        assert found is not None
        assert found is method_tool


class TestStructuredOutputEdgeCases:
    """Test structured output edge cases."""

    async def test_nested_model_with_optional_fields(self):
        """Test nested model with optional fields."""

        class OptionalNested(BaseModel):
            value: str | None = None

        class ParentModel(BaseModel):
            name: str
            nested: OptionalNested | None = None

        result = await structured_complete(
            user_message="Return name 'test' with nested value 'nested'",
            response_format=ParentModel,
            system_prompt="Extract the information.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, ParentModel)
        assert result.structured_output.name == "test"
        # Nested may or may not be present depending on LLM
        if result.structured_output.nested is not None:
            assert isinstance(result.structured_output.nested, OptionalNested)

    async def test_list_fields_in_structured_output(self):
        """Test structured output with list/array fields."""

        class ListResponse(BaseModel):
            items: Annotated[list[str], Field(description="List of items")]
            count: Annotated[int, Field(description="Number of items")]

        result = await structured_complete(
            user_message="Return a list with items ['apple', 'banana', 'cherry']",
            response_format=ListResponse,
            system_prompt="Extract the list information.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, ListResponse)
        assert isinstance(result.structured_output.items, list)
        assert len(result.structured_output.items) > 0
        assert isinstance(result.structured_output.count, int)

    async def test_enum_fields_in_structured_output(self):
        """Test structured output with enum fields."""
        from enum import Enum

        class Status(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"

        class StatusResponse(BaseModel):
            status: Annotated[Status, Field(description="Status value")]

        result = await structured_complete(
            user_message="Return status 'active'",
            response_format=StatusResponse,
            system_prompt="Extract the status.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, StatusResponse)
        assert result.structured_output.status in Status

    async def test_nested_list_in_structured_output(self):
        """Test structured output with nested lists."""

        class NestedListResponse(BaseModel):
            matrix: Annotated[list[list[int]], Field(description="Matrix of numbers")]

        result = await structured_complete(
            user_message="Return a 2x2 matrix [[1, 2], [3, 4]]",
            response_format=NestedListResponse,
            system_prompt="Extract the matrix.",
        )

        assert isinstance(result, ExecutionResult)
        assert result.structured_output is not None
        assert isinstance(result.structured_output, NestedListResponse)
        assert isinstance(result.structured_output.matrix, list)
        if len(result.structured_output.matrix) > 0:
            assert isinstance(result.structured_output.matrix[0], list)


class TestToolResultSerialization:
    """Test tool result serialization with various return types."""

    async def test_tool_returning_int(self):
        """Test tool that returns an integer."""

        def int_tool(x: int) -> int:
            """Return an integer."""
            return x * 2

        result = await complete(
            user_message="Use int_tool with 5",
            system_prompt="Use the int_tool when asked.",
            tools=[int_tool],
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # Integer result should be converted to string
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert "10" in result.tool_execution_history[0]["result"]

    async def test_tool_returning_list(self):
        """Test tool that returns a list."""

        def list_tool() -> list[str]:
            """Return a list."""
            return ["item1", "item2", "item3"]

        result = await complete(
            user_message="Use list_tool",
            system_prompt="Use the list_tool when asked.",
            tools=[list_tool],
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # List result should be converted to string
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert "item1" in result.tool_execution_history[0]["result"]

    async def test_tool_returning_dict(self):
        """Test tool that returns a dictionary."""

        def dict_tool() -> dict[str, str]:
            """Return a dictionary."""
            return {"key1": "value1", "key2": "value2"}

        result = await complete(
            user_message="Use dict_tool",
            system_prompt="Use the dict_tool when asked.",
            tools=[dict_tool],
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # Dict result should be converted to string
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert (
                "key1" in result.tool_execution_history[0]["result"]
                or "value1" in result.tool_execution_history[0]["result"]
            )

    async def test_tool_returning_none(self):
        """Test tool that returns None."""

        def none_tool() -> None:
            """Return None."""
            return

        result = await complete(
            user_message="Use none_tool",
            system_prompt="Use the none_tool when asked.",
            tools=[none_tool],
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # None result should be converted to string
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert "None" in result.tool_execution_history[0]["result"]

    async def test_tool_returning_large_value(self):
        """Test tool that returns a large value (>1MB)."""
        large_string = "x" * (2 * 1024 * 1024)  # 2MB

        def large_tool() -> str:
            """Return a large string."""
            return large_string

        result = await complete(
            user_message="Use large_tool",
            system_prompt="Use the large_tool when asked.",
            tools=[large_tool],
            max_tool_iterations=1,  # Limit iterations for this test
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # Large result should still be serialized
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert len(result.tool_execution_history[0]["result"]) > 1000000

    async def test_tool_returning_special_characters(self):
        """Test tool that returns special characters."""

        def special_char_tool() -> str:
            """Return special characters."""
            return "Special: \n\t\r\"'\\<>{}[]()&|$@#%^*+=~`"

        result = await complete(
            user_message="Use special_char_tool",
            system_prompt="Use the special_char_tool when asked.",
            tools=[special_char_tool],
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # Special characters should be preserved in string conversion
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert "Special" in result.tool_execution_history[0]["result"]

    async def test_tool_returning_unicode(self):
        """Test tool that returns unicode characters."""

        def unicode_tool() -> str:
            """Return unicode characters."""
            return "Unicode: 你好 🌟 émojis 🎉"

        result = await complete(
            user_message="Use unicode_tool",
            system_prompt="Use the unicode_tool when asked.",
            tools=[unicode_tool],
        )

        assert isinstance(result, ExecutionResult)
        if result.tool_execution_history:
            # Unicode should be preserved
            assert isinstance(result.tool_execution_history[0]["result"], str)
            assert "Unicode" in result.tool_execution_history[0]["result"]


class TestProcessStatusEvents:
    """Test on_status callback and process events."""

    @pytest.mark.integration
    async def test_complete_emits_llm_and_complete_events(self):
        """complete() with on_status receives llm_call_start, llm_call_end, complete."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        events: list[ProcessEvent] = []

        def on_status(e: ProcessEvent) -> None:
            events.append(e)

        result = await complete(
            user_message="Reply with the word OK only.",
            system_prompt="You are a terse assistant. Reply with exactly: OK",
            on_status=on_status,
        )

        assert isinstance(result, ExecutionResult)
        kinds = [e.kind for e in events]
        assert "llm_call_start" in kinds
        assert "llm_call_end" in kinds
        assert "complete" in kinds
        assert kinds.index("llm_call_start") < kinds.index("llm_call_end")
        assert kinds.index("llm_call_end") < kinds.index("complete")

    @pytest.mark.integration
    async def test_stream_complete_emits_stream_events(self):
        """stream_complete() without tools emits stream_start, stream_chunk, stream_end."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        events: list[ProcessEvent] = []

        def on_status(e: ProcessEvent) -> None:
            events.append(e)

        chunks = []
        async for chunk in stream_complete(
            user_message="Say hi in one word.",
            system_prompt="Reply with one word: Hi",
            tools=[],
            on_status=on_status,
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        kinds = [e.kind for e in events]
        assert "stream_start" in kinds
        assert "stream_end" in kinds


class TestStreamCompleteWithTools:
    """Test stream_complete with execute_tools=True (streaming + tool loop)."""

    async def test_stream_complete_execute_tools_iteration2_messages_are_dicts(self):
        """Second LLM call in tool loop receives messages as list[dict], not SimpleNamespace.

        Regression test: _build_message_from_stream returns SimpleNamespace; we must convert
        to dict before appending to messages so any_llm validation does not fail on iteration 2.
        """
        call_count = 0
        second_call_messages = []

        async def fake_safe_llm_call(*, messages, stream, **kwargs):
            nonlocal call_count
            call_count += 1
            if not stream:
                raise NotImplementedError("This test only covers stream=True path")
            if call_count == 1:
                # First call: stream with one tool call so loop appends to messages and continues
                async def first_stream():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=" ", tool_calls=None),
                            )
                        ]
                    )
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            function=SimpleNamespace(
                                                name="dummy_tool",
                                                arguments='{"value":"x"}',
                                            ),
                                        )
                                    ],
                                ),
                            )
                        ]
                    )

                return first_stream()
            # Second call: must receive only dicts (fix for SimpleNamespace in messages)
            second_call_messages.extend(messages)
            for m in messages:
                assert isinstance(m, dict), f"Expected all messages to be dicts, got {type(m).__name__}"

            async def second_stream():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Done", tool_calls=None),
                        )
                    ]
                )

            return second_stream()

        with patch("gluellm.api._safe_llm_call", side_effect=fake_safe_llm_call):
            chunks = []
            async for ch in stream_complete(
                user_message="Use dummy_tool with value x",
                system_prompt="Use the tool when asked.",
                tools=[dummy_tool],
                execute_tools=True,
            ):
                chunks.append(ch)

        assert call_count == 2, "Expected two LLM calls (first with tool call, second final response)"
        assert all(isinstance(m, dict) for m in second_call_messages), (
            "Second call messages must all be dicts for any_llm CompletionParams validation"
        )
        assert any(ch.done for ch in chunks), "Should receive a final done chunk"


class TestStreamingStructuredOutput:
    """Test stream_complete with response_format (structured output on final chunk)."""

    @pytest.mark.integration
    async def test_stream_complete_with_response_format_final_chunk_has_structured_output(
        self,
    ):
        """When response_format is set, the final chunk may have structured_output."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        class TinyResponse(BaseModel):
            word: Annotated[str, Field(description="A single word")]

        last_chunk = None
        async for chunk in stream_complete(
            user_message="Reply with a single word: hello",
            system_prompt='You must respond with valid JSON only, e.g. {"word": "hello"}.',
            tools=[],
            response_format=TinyResponse,
        ):
            last_chunk = chunk

        assert last_chunk is not None
        assert last_chunk.done is True
        # Parser may or may not succeed depending on model output; if it does, we get structured_output
        if last_chunk.structured_output is not None:
            assert isinstance(last_chunk.structured_output, TinyResponse)
            assert isinstance(last_chunk.structured_output.word, str)


class TestExecutionResultSerialization:
    """Tests that ExecutionResult.model_dump() never emits Pydantic serialization warnings.

    Regression tests for the PydanticSerializationUnexpectedValue warning that occurs
    when the OpenAI SDK returns a ParsedChatCompletion whose message carries a user's
    Pydantic model in the `parsed` field, but the base schema declares `parsed: None`.
    """

    def _make_parsed_chat_completion(self, parsed_value: BaseModel) -> object:
        """Build a minimal ParsedChatCompletion-like object carrying a user Pydantic model."""
        from types import SimpleNamespace

        message = SimpleNamespace(
            role="assistant",
            content='{"items": ["a", "b"], "missing_information_queries": []}',
            tool_calls=None,
            parsed=parsed_value,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return SimpleNamespace(id="chatcmpl-test", model="gpt-4o", choices=[choice], usage=usage)

    def test_model_dump_emits_no_warning_when_parsed_field_holds_user_model(self, recwarn):
        """model_dump() must be silent even when raw_response carries a user Pydantic model in `parsed`."""

        class ContextRelevancyCoverageScore(BaseModel):
            items: list[str]
            missing_information_queries: list[str]

        user_model = ContextRelevancyCoverageScore(items=["a", "b"], missing_information_queries=[])
        raw = self._make_parsed_chat_completion(user_model)

        result = ExecutionResult(
            final_response="test",
            tool_calls_made=0,
            tool_execution_history=[],
            raw_response=raw,
            tokens_used=None,
            estimated_cost_usd=None,
            model="openai:gpt-4o",
            structured_output=user_model,
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # This must not raise - previously it would emit PydanticSerializationUnexpectedValue
            dumped = result.model_dump()

        assert dumped["raw_response"]["id"] == "chatcmpl-test"
        assert dumped["raw_response"]["model"] == "gpt-4o"
        assert len(dumped["raw_response"]["choices"]) == 1
        # `parsed` must NOT appear in the serialized output
        assert "parsed" not in dumped["raw_response"]["choices"][0]["message"]
        assert dumped["raw_response"]["choices"][0]["message"]["role"] == "assistant"

    async def test_model_dump_json_emits_no_warning_when_parsed_field_holds_user_model(self):
        """model_dump_json() must also be silent - covers the JSON serialisation path."""
        import json
        import warnings

        class ScoreModel(BaseModel):
            score: float
            label: str

        user_model = ScoreModel(score=0.95, label="relevant")
        raw = self._make_parsed_chat_completion(user_model)

        result = ExecutionResult(
            final_response="answer",
            tool_calls_made=0,
            tool_execution_history=[],
            raw_response=raw,
            tokens_used={"prompt": 10, "completion": 20, "total": 30},
            estimated_cost_usd=0.001,
            model="openai:gpt-4o-mini",
            structured_output=user_model,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            json_str = result.model_dump_json()

        data = json.loads(json_str)
        assert data["raw_response"]["id"] == "chatcmpl-test"
        assert "parsed" not in data["raw_response"]["choices"][0]["message"]

    async def test_serialize_chat_completion_to_dict_is_warning_free(self):
        """The shared helper itself must produce a clean dict with no `parsed` key."""
        from gluellm.api import _serialize_chat_completion_to_dict

        class MyModel(BaseModel):
            value: int

        raw = self._make_parsed_chat_completion(MyModel(value=42))
        result = _serialize_chat_completion_to_dict(raw)

        assert result["id"] == "chatcmpl-test"
        assert result["usage"]["prompt_tokens"] == 10
        message = result["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert "parsed" not in message


class TestAnyLlmWarningsSuppressed:
    """Regression tests: PydanticSerializationUnexpectedValue from any_llm must not reach callers.

    any_llm calls response.model_dump() on a ParsedChatCompletion whose `message.parsed`
    holds a user Pydantic model at runtime but is typed as None in the base schema.
    _safe_llm_call must swallow that UserWarning before it surfaces to third-party code.
    """

    async def test_safe_llm_call_does_not_emit_pydantic_serialization_warning(self):
        """_safe_llm_call must suppress PydanticSerializationUnexpectedValue from any_llm."""
        import time
        import warnings
        from unittest.mock import AsyncMock, MagicMock, patch

        from any_llm.types.completion import (
            ChatCompletion,
            ChatCompletionMessage,
            Choice,
            CompletionUsage,
        )
        from pydantic import BaseModel

        from gluellm.api import _safe_llm_call

        class MyScore(BaseModel):
            score: float

        async def fake_acompletion(**kwargs):
            warnings.warn(
                "Expected `null` but got `MyScore` - serialized value may not be as expected "
                "[type=PydanticSerializationUnexpectedValue]",
                UserWarning,
                stacklevel=2,
            )
            msg = ChatCompletionMessage(role="assistant", content='{"score": 0.9}')
            choice = Choice(index=0, message=msg, finish_reason="stop")
            usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            return ChatCompletion(
                id="chatcmpl-test",
                choices=[choice],
                created=int(time.time()),
                model="gpt-4o-mini",
                object="chat.completion",
                usage=usage,
            )

        mock_provider = MagicMock()
        mock_provider.acompletion = AsyncMock(side_effect=fake_acompletion)

        with (
            patch("gluellm.api._provider_cache.get_provider", return_value=(mock_provider, "gpt-4o-mini")),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error", UserWarning)
            # Must not raise — the PydanticSerializationUnexpectedValue warning must be suppressed
            result = await _safe_llm_call(
                messages=[{"role": "user", "content": "test"}],
                model="openai:gpt-4o-mini",
                response_format=MyScore,
            )

        assert result is not None


class TestProviderCache:
    """Regression tests for the provider cache that prevents 'Event loop is closed' errors.

    The root cause of the error: every call to any_llm_acompletion() previously
    created a new AsyncOpenAI client. On event loop shutdown those orphaned clients
    were garbage-collected, triggering async cleanup on an already-closed loop.
    The fix caches provider instances so a single httpx client is reused.
    """

    async def test_provider_cache_returns_same_instance_for_same_key(self):
        """Same (model, api_key) must reuse the cached provider — not create a new one."""
        from unittest.mock import MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()
        fake_provider = MagicMock()

        with patch("gluellm.api.AnyLLM.create", return_value=fake_provider) as mock_create:
            p1, m1 = cache.get_provider("openai:gpt-4o-mini", "sk-test")
            p2, m2 = cache.get_provider("openai:gpt-4o-mini", "sk-test")

        # AnyLLM.create must be called exactly once; second call must hit the cache
        mock_create.assert_called_once()
        assert p1 is p2
        assert m1 == "gpt-4o-mini"
        assert m2 == "gpt-4o-mini"

    async def test_provider_cache_creates_separate_instance_for_different_api_keys(self):
        """Different API keys must each get their own provider instance."""
        from unittest.mock import MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()
        provider_a = MagicMock()
        provider_b = MagicMock()

        with patch("gluellm.api.AnyLLM.create", side_effect=[provider_a, provider_b]):
            p1, _ = cache.get_provider("openai:gpt-4o-mini", "sk-key-a")
            p2, _ = cache.get_provider("openai:gpt-4o-mini", "sk-key-b")

        assert p1 is provider_a
        assert p2 is provider_b
        assert p1 is not p2

    async def test_provider_cache_handles_slash_separator_for_embeddings(self):
        """Embedding models use 'provider/model' format — the cache must parse it correctly."""
        from unittest.mock import MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()
        fake_provider = MagicMock()

        with patch("gluellm.api.AnyLLM.create", return_value=fake_provider) as mock_create:
            provider, model_id = cache.get_provider("openai/text-embedding-3-small", None)

        assert provider is fake_provider
        assert model_id == "text-embedding-3-small"
        mock_create.assert_called_once_with("openai", api_key=mock_create.call_args[1]["api_key"])

    async def test_close_all_calls_aclose_on_cached_clients(self):
        """close_all() must call aclose() on every cached provider's HTTP client."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        fake_provider = MagicMock()
        fake_provider.client = mock_client

        with patch("gluellm.api.AnyLLM.create", return_value=fake_provider):
            cache.get_provider("openai:gpt-4o-mini", "sk-test")

        await cache.close_all()

        mock_client.aclose.assert_called_once()
        # Cache must be empty after close so no stale clients linger
        assert len(cache._providers) == 0

    async def test_close_providers_does_not_raise_event_loop_closed_error(self):
        """close_providers() must gracefully handle providers with no client attribute."""
        from unittest.mock import MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()
        fake_provider = MagicMock(spec=[])  # no 'client' attribute

        with patch("gluellm.api.AnyLLM.create", return_value=fake_provider):
            cache.get_provider("openai:gpt-4o-mini", "sk-test")

        # Must not raise even when the provider has no .client
        await cache.close_all()

    async def test_close_all_awaits_async_close_when_aclose_is_absent(self):
        """close_all() must await close() when it is a coroutine (e.g. openai.AsyncAPIClient).

        Regression: the original code called close() synchronously, producing an
        unawaited coroutine that was GC'd with RuntimeWarning.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()

        mock_client = MagicMock(spec=["close"])  # no aclose, only async close
        mock_client.close = AsyncMock()

        fake_provider = MagicMock()
        fake_provider.client = mock_client

        with patch("gluellm.api.AnyLLM.create", return_value=fake_provider):
            cache.get_provider("openai:gpt-4o-mini", "sk-test")

        await cache.close_all()

        mock_client.close.assert_awaited_once()

    async def test_close_all_calls_sync_close_when_not_coroutine(self):
        """close_all() must call close() synchronously when it is a plain function."""
        from unittest.mock import MagicMock, patch

        from gluellm.api import _ProviderCache

        cache = _ProviderCache()

        mock_client = MagicMock(spec=["close"])  # no aclose, sync close
        # MagicMock.close is not a coroutine function by default

        fake_provider = MagicMock()
        fake_provider.client = mock_client

        with patch("gluellm.api.AnyLLM.create", return_value=fake_provider):
            cache.get_provider("openai:gpt-4o-mini", "sk-test")

        await cache.close_all()

        mock_client.close.assert_called_once()

    async def test_safe_llm_call_uses_cached_provider_not_new_instance(self):
        """_safe_llm_call must route through the provider cache, not create a fresh client.

        Regression test: before the fix every call created a new AsyncOpenAI(),
        leaving orphaned clients that raised RuntimeError on GC after loop close.
        """
        import time
        from unittest.mock import AsyncMock, MagicMock, patch

        from any_llm.types.completion import (
            ChatCompletion,
            ChatCompletionMessage,
            Choice,
            CompletionUsage,
        )

        from gluellm.api import _safe_llm_call

        msg = ChatCompletionMessage(role="assistant", content="hello")
        choice = Choice(index=0, message=msg, finish_reason="stop")
        usage = CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        fake_response = ChatCompletion(
            id="chatcmpl-x",
            choices=[choice],
            created=int(time.time()),
            model="gpt-4o-mini",
            object="chat.completion",
            usage=usage,
        )

        mock_provider = MagicMock()
        mock_provider.acompletion = AsyncMock(return_value=fake_response)
        get_provider_calls = []

        def fake_get_provider(model, api_key):
            get_provider_calls.append((model, api_key))
            return mock_provider, "gpt-4o-mini"

        with patch("gluellm.api._provider_cache.get_provider", side_effect=fake_get_provider):
            await _safe_llm_call(
                messages=[{"role": "user", "content": "hi"}],
                model="openai:gpt-4o-mini",
            )
            await _safe_llm_call(
                messages=[{"role": "user", "content": "hi again"}],
                model="openai:gpt-4o-mini",
            )

        # Both calls must have gone through the cache lookup
        assert len(get_provider_calls) == 2
        # The provider's acompletion must have been called twice (not a new client each time)
        assert mock_provider.acompletion.call_count == 2

class TestCondenseToolRound:
    """Unit tests for the _condense_tool_round() utility.

    These tests operate directly on the messages list and require no LLM calls.
    """

    def _make_assistant_with_tool_calls(self, tool_calls: list[dict]) -> dict:
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}

    def _make_tool_response(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    async def test_single_tool_call_condensed(self):
        """One assistant + one tool response becomes a single condensed user message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the weather?"},
            self._make_assistant_with_tool_calls(
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ]
            ),
            self._make_tool_response("call_1", "Sunny, 72°F"),
        ]

        _condense_tool_round(messages)

        assert len(messages) == 3
        condensed = messages[-1]
        assert condensed["role"] == "user"
        assert "[Tool Results]" in condensed["content"]
        assert "get_weather()" in condensed["content"]
        assert "Sunny, 72°F" in condensed["content"]

    async def test_multiple_tool_calls_condensed(self):
        """Multiple tool calls in one round all appear in the single condensed user message."""
        messages = [
            {"role": "user", "content": "Get data"},
            self._make_assistant_with_tool_calls(
                [
                    {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                ]
            ),
            self._make_tool_response("call_1", "Result A"),
            self._make_tool_response("call_2", "Result B"),
        ]

        _condense_tool_round(messages)

        assert len(messages) == 2
        condensed = messages[-1]
        assert condensed["role"] == "user"
        assert "tool_a()" in condensed["content"]
        assert "Result A" in condensed["content"]
        assert "tool_b()" in condensed["content"]
        assert "Result B" in condensed["content"]

    async def test_no_condensation_when_assistant_has_content(self):
        """Rounds where the assistant produced visible text alongside tool calls are left untouched."""
        original_messages = [
            {"role": "user", "content": "Do stuff"},
            {
                "role": "assistant",
                "content": "I will call the tool now.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "some_tool", "arguments": "{}"}}
                ],
            },
            self._make_tool_response("call_1", "Tool result"),
        ]
        messages = list(original_messages)

        _condense_tool_round(messages)

        assert messages == original_messages

    async def test_no_condensation_when_no_tool_messages(self):
        """If the last message is not a tool response, nothing changes."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        _condense_tool_round(messages)

        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Hi there"

    async def test_no_condensation_when_empty_messages(self):
        """Empty messages list is handled gracefully."""
        messages: list = []
        _condense_tool_round(messages)
        assert messages == []

    async def test_error_tool_result_included(self):
        """Error results from tool calls appear in the condensed summary."""
        messages = [
            {"role": "user", "content": "Run the tool"},
            self._make_assistant_with_tool_calls(
                [{"id": "call_err", "type": "function", "function": {"name": "flaky_tool", "arguments": "{}"}}]
            ),
            self._make_tool_response("call_err", "Error executing tool: TimeoutError: timed out"),
        ]

        _condense_tool_round(messages)

        assert len(messages) == 2
        condensed = messages[-1]
        assert "flaky_tool()" in condensed["content"]
        assert "TimeoutError" in condensed["content"]

    async def test_preserves_earlier_messages(self):
        """Messages before the condensed round are left intact."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Turn 1 reply"},
            {"role": "user", "content": "Turn 2"},
            self._make_assistant_with_tool_calls(
                [{"id": "call_x", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
            ),
            self._make_tool_response("call_x", "Lookup result"),
        ]

        _condense_tool_round(messages)

        assert len(messages) == 5
        assert messages[0]["content"] == "System"
        assert messages[1]["content"] == "Turn 1"
        assert messages[2]["content"] == "Turn 1 reply"
        assert messages[3]["content"] == "Turn 2"
        assert messages[4]["role"] == "user"
        assert "lookup()" in messages[4]["content"]

    async def test_condense_tool_messages_disabled_by_default(self):
        """condense_tool_messages defaults to False on GlueLLM and module-level helpers."""
        client = GlueLLM()
        assert client.condense_tool_messages is False

    async def test_condense_tool_messages_can_be_enabled_on_client(self):
        """condense_tool_messages=True can be set on the GlueLLM instance."""
        client = GlueLLM(condense_tool_messages=True)
        assert client.condense_tool_messages is True


def _make_tool_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    """Build a fake non-streaming LLM response with plain text content."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="openai:gpt-4o-mini",
    )


def _make_tool_call_response(tool_name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    """Build a fake non-streaming LLM response that requests a tool call."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(name=tool_name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="openai:gpt-4o-mini",
    )


class TestCondenseToolMessagesIntegration:
    """Integration tests for condense_tool_messages using mocked LLM calls.

    These tests verify the condensation feature end-to-end through the tool loop
    without making real API calls. They patch _llm_call_with_retry (the function
    called inside complete() / structured_complete()) and inspect the messages
    that each subsequent LLM call receives.
    """

    async def test_condensed_message_replaces_tool_round(self):
        """After a tool round with condense=True, the next LLM call receives a single
        condensed user message instead of the raw assistant+tool pair."""
        messages_on_second_call: list[dict] = []
        call_count = 0

        async def fake_llm(*, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_tool_call_response("dummy_tool", '{"value": "hello"}')
            messages_on_second_call.extend(messages)
            return _make_tool_response("All done")

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            result = await complete(
                user_message="Use the tool",
                system_prompt="Use tools.",
                tools=[dummy_tool],
                condense_tool_messages=True,
            )

        assert result.tool_calls_made == 1
        assert call_count == 2

        # The second call should NOT contain raw role:"tool" messages
        roles = [m["role"] for m in messages_on_second_call]
        assert "tool" not in roles, "Raw tool messages should have been condensed away"

        # Condensed summary is a user message (so LLM naturally continues)
        condensed = next(
            (m for m in messages_on_second_call if m.get("role") == "user" and "[Tool Results]" in (m.get("content") or "")),
            None,
        )
        assert condensed is not None, "Condensed summary message not found"
        assert "dummy_tool()" in condensed["content"]
        assert "Tool received: hello" in condensed["content"]

    async def test_no_condensation_when_disabled(self):
        """With condense=False, the next LLM call sees the raw assistant+tool messages."""
        messages_on_second_call: list[dict] = []
        call_count = 0

        async def fake_llm(*, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_tool_call_response("dummy_tool", '{"value": "hello"}')
            messages_on_second_call.extend(messages)
            return _make_tool_response("All done")

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            await complete(
                user_message="Use the tool",
                system_prompt="Use tools.",
                tools=[dummy_tool],
                condense_tool_messages=False,
            )

        roles = [m["role"] for m in messages_on_second_call]
        assert "tool" in roles, "Raw tool messages should be present when condensation is off"
        assert any(m.get("tool_calls") for m in messages_on_second_call if m["role"] == "assistant"), (
            "Raw assistant message with tool_calls should be present"
        )

    async def test_multiple_tool_iterations_each_condensed(self):
        """With two consecutive tool rounds, each round is condensed independently
        so the context grows by one message per round, not N+1."""
        messages_per_call: list[list[dict]] = []
        call_count = 0

        async def fake_llm(*, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            messages_per_call.append(list(messages))
            if call_count == 1:
                return _make_tool_call_response("dummy_tool", '{"value": "round1"}', call_id="call_r1")
            if call_count == 2:
                return _make_tool_call_response("math_tool", '{"a": 3, "b": 4, "operation": "add"}', call_id="call_r2")
            return _make_tool_response("Both rounds done")

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            result = await complete(
                user_message="Do two tool rounds",
                system_prompt="Use tools.",
                tools=[dummy_tool, math_tool],
                condense_tool_messages=True,
            )

        assert call_count == 3
        assert result.tool_calls_made == 2

        # After round 1: messages for call 2 should have one condensed user message
        # After round 2: messages for call 3 should have two condensed user messages
        call3_messages = messages_per_call[2]
        tool_messages = [m for m in call3_messages if m["role"] == "tool"]
        assert len(tool_messages) == 0, "No raw tool messages should remain after two condensed rounds"

        condensed_summaries = [
            m for m in call3_messages if m["role"] == "user" and "[Tool Results]" in (m.get("content") or "")
        ]
        assert len(condensed_summaries) == 2, (
            f"Expected two condensed summaries (one per round), got {len(condensed_summaries)}"
        )
        # First summary references round 1 tool
        assert "dummy_tool()" in condensed_summaries[0]["content"]
        # Second summary references round 2 tool
        assert "math_tool()" in condensed_summaries[1]["content"]

    async def test_multi_turn_conversation_final_response_carries_forward(self):
        """Across two separate client.complete() calls, the final assistant text from turn 1
        is carried into the messages for turn 2 — not the condensed tool summary.

        The condensed tool summaries live only in the ephemeral messages list within a single
        complete() call's tool loop. Only the final text response is persisted in the
        Conversation and replayed on the next turn.
        """
        turn2_messages: list[dict] = []
        call_count = 0

        async def fake_llm(*, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_tool_call_response("dummy_tool", '{"value": "ctx_data"}')
            if call_count == 2:
                return _make_tool_response("Turn 1 final answer")
            # Turn 2: single LLM call - record what history it received
            turn2_messages.extend(messages)
            return _make_tool_response("Turn 2 final answer")

        client = GlueLLM(
            system_prompt="Use tools.",
            tools=[dummy_tool],
            condense_tool_messages=True,
        )

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            result1 = await client.complete("Turn 1: use the tool")
            result2 = await client.complete("Turn 2: follow up")

        assert result1.final_response == "Turn 1 final answer"
        assert result2.final_response == "Turn 2 final answer"

        # Turn 2 gets the final text from turn 1 — NOT the condensed tool summary
        assistant_msgs = [m for m in turn2_messages if m["role"] == "assistant"]
        assert any("Turn 1 final answer" in (m.get("content") or "") for m in assistant_msgs), (
            "Turn 2 should see the final text response from turn 1"
        )
        # Condensed summaries should NOT bleed across turns
        assert not any("[Tool Results]" in (m.get("content") or "") for m in turn2_messages), (
            "Condensed tool summaries should not appear in a subsequent turn's message history"
        )
        # No raw tool messages in turn 2 either
        assert "tool" not in [m["role"] for m in turn2_messages]

    async def test_error_tool_result_appears_in_condensed_summary(self):
        """When a tool raises an error, the error text is included in the condensed message."""
        messages_on_second_call: list[dict] = []
        call_count = 0

        async def fake_llm(*, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_tool_call_response("dummy_tool", "not valid json", call_id="call_bad")
            messages_on_second_call.extend(messages)
            return _make_tool_response("Handled error")

        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            result = await complete(
                user_message="Use the tool",
                system_prompt="Use tools.",
                tools=[dummy_tool],
                condense_tool_messages=True,
            )

        assert call_count == 2
        condensed = next(
            (m for m in messages_on_second_call if "[Tool Results]" in (m.get("content") or "")),
            None,
        )
        assert condensed is not None
        # The error text from malformed JSON should be in the summary
        assert "dummy_tool()" in condensed["content"]

    async def test_stream_complete_condensed_message_on_second_iteration(self):
        """stream_complete with condense=True passes a condensed message to the second
        streaming LLM call instead of raw assistant+tool messages."""
        messages_on_second_call: list[dict] = []
        call_count = 0

        async def fake_safe_llm_call(*, messages, stream, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:

                async def first_stream():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_s1",
                                            function=SimpleNamespace(
                                                name="dummy_tool", arguments='{"value":"streamed"}'
                                            ),
                                        )
                                    ],
                                )
                            )
                        ]
                    )

                return first_stream()
            messages_on_second_call.extend(messages)

            async def second_stream():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Stream done", tool_calls=None))]
                )

            return second_stream()

        with patch("gluellm.api._safe_llm_call", side_effect=fake_safe_llm_call):
            chunks = []
            async for ch in stream_complete(
                user_message="Use the tool",
                system_prompt="Use tools.",
                tools=[dummy_tool],
                execute_tools=True,
                condense_tool_messages=True,
            ):
                chunks.append(ch)

        assert call_count == 2
        roles = [m["role"] for m in messages_on_second_call]
        assert "tool" not in roles, "Raw tool messages should be condensed in stream path"
        condensed = next(
            (m for m in messages_on_second_call if "[Tool Results]" in (m.get("content") or "")),
            None,
        )
        assert condensed is not None, "Condensed summary missing from second streaming call"
        assert "dummy_tool()" in condensed["content"]

    async def test_ten_turns_with_tool_use(self):
        """10 consecutive turns on the same client, each with one tool call.

        Verifies that across a long conversation:
        - Every turn's tool call is condensed within that turn's loop
        - The conversation history seen by each subsequent LLM call contains
          exactly the correct number of prior user+assistant message pairs
        - No raw tool messages or condensed summaries from previous turns
          bleed into later turns (only final text responses persist)
        - The conversation grows linearly (2 messages per turn: user + assistant)
          rather than quadratically (which would happen if tool messages leaked)
        """
        num_turns = 10
        call_count = 0
        # messages seen by the *first* LLM call of each turn (before any tools run)
        first_call_messages_per_turn: list[list[dict]] = []

        async def fake_llm(*, messages, **kwargs):
            nonlocal call_count
            call_count += 1

            # Each even-numbered call (1, 3, 5, …) is the first call of a new turn
            # and should request a tool. Each odd-numbered call (2, 4, 6, …) is the
            # follow-up after the tool ran and returns the final text.
            is_first_call_of_turn = call_count % 2 == 1

            if is_first_call_of_turn:
                first_call_messages_per_turn.append(list(messages))
                turn_number = (call_count + 1) // 2
                return _make_tool_call_response(
                    "dummy_tool",
                    f'{{"value": "turn{turn_number}"}}',
                    call_id=f"call_t{turn_number}",
                )
            turn_number = call_count // 2
            return _make_tool_response(f"Final answer for turn {turn_number}")

        client = GlueLLM(
            system_prompt="Use tools.",
            tools=[dummy_tool],
            condense_tool_messages=True,
        )

        results = []
        with patch("gluellm.api._llm_call_with_retry", side_effect=fake_llm):
            for i in range(1, num_turns + 1):
                result = await client.complete(f"Turn {i}: use the dummy tool")
                results.append(result)

        # Every turn made exactly 2 LLM calls (tool call + final response)
        assert call_count == num_turns * 2

        # Every turn got its tool executed and a final text response
        for i, result in enumerate(results, 1):
            assert result.tool_calls_made == 1, f"Turn {i} should have made 1 tool call"
            assert result.final_response == f"Final answer for turn {i}"

        # Verify conversation growth is linear: each turn adds exactly 2 messages
        # (user + assistant final text). Raw tool messages and condensed summaries
        # must NOT accumulate in the stored conversation.
        stored_messages = client._conversation.messages
        assert len(stored_messages) == num_turns * 2, (
            f"Expected {num_turns * 2} stored messages (user+assistant per turn), got {len(stored_messages)}"
        )
        for i in range(num_turns):
            user_msg = stored_messages[i * 2]
            assistant_msg = stored_messages[i * 2 + 1]
            assert user_msg.role.value == "user"
            assert assistant_msg.role.value == "assistant"
            assert f"Final answer for turn {i + 1}" in assistant_msg.content

        # Verify the messages seen by the first call of each turn:
        # turn N should see exactly (N-1)*2 prior messages (excluding system)
        for turn_idx, msgs in enumerate(first_call_messages_per_turn):
            turn_number = turn_idx + 1
            # Exclude the system message
            non_system = [m for m in msgs if m["role"] != "system"]
            expected_prior_messages = (turn_number - 1) * 2  # user + assistant per prior turn
            assert len(non_system) == expected_prior_messages + 1, (
                f"Turn {turn_number}: expected {expected_prior_messages} prior messages "
                f"+ 1 current user message = {expected_prior_messages + 1} total, "
                f"got {len(non_system)}"
            )
            # No raw tool messages or condensed summaries in stored history
            assert "tool" not in [m["role"] for m in non_system], (
                f"Turn {turn_number}: raw tool messages leaked into conversation history"
            )
            assert not any("[Tool Results]" in (m.get("content") or "") for m in non_system), (
                f"Turn {turn_number}: condensed tool summaries leaked into conversation history"
            )


class TestTrimToolDocstrings:
    """Tests for _trim_tool_docstrings — ensures tool descriptions sent to the LLM are first-line only."""

    async def test_multiline_docstring_trimmed_to_first_line(self):
        from gluellm.api import _trim_tool_docstrings

        def my_tool(x: str) -> str:
            """First line summary.

            Args:
                x: Some value
            Returns:
                A string
            """
            return x

        trimmed = _trim_tool_docstrings([my_tool])
        assert len(trimmed) == 1
        assert trimmed[0].__doc__ == "First line summary."
        assert trimmed[0].__name__ == "my_tool"
        assert trimmed[0](x="hi") == "hi"

    async def test_single_line_docstring_unchanged(self):
        from gluellm.api import _trim_tool_docstrings

        def simple_tool(x: str) -> str:
            """Simple tool."""
            return x

        trimmed = _trim_tool_docstrings([simple_tool])
        assert trimmed[0].__doc__ == "Simple tool."
        assert trimmed[0] is simple_tool

    async def test_no_docstring_handled(self):
        from gluellm.api import _trim_tool_docstrings

        def no_doc(x: str) -> str:
            return x
        no_doc.__doc__ = None

        trimmed = _trim_tool_docstrings([no_doc])
        assert trimmed[0].__doc__ is None or trimmed[0].__doc__ == ""

    async def test_preserves_original_tool_docstring(self):
        from gluellm.api import _trim_tool_docstrings

        def original(x: str) -> str:
            """Line one.

            Extra detail here.
            """
            return x

        _trim_tool_docstrings([original])
        assert "Extra detail" in original.__doc__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
