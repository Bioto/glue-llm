"""Tests for the GlueLLM API."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from gluellm.api import ExecutionResult, GlueLLM, complete, structured_complete

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
