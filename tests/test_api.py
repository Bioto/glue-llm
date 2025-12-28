"""Tests for the GlueLLM API."""

import pytest
from pydantic import BaseModel, Field
from typing import Annotated

from source.api import GlueLLM, complete, structured_complete, ToolExecutionResult


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
    elif operation == "multiply":
        return str(a * b)
    elif operation == "subtract":
        return str(a - b)
    return "Unknown operation"


# Test classes

class TestBasicCompletion:
    """Test basic completion functionality."""
    
    def test_simple_completion_function(self):
        """Test the complete() convenience function."""
        result = complete(
            user_message="Say hello",
            system_prompt="You are a friendly assistant. Always respond with 'Hello!'",
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert isinstance(result.final_response, str)
        assert len(result.final_response) > 0
        assert result.tool_calls_made == 0
        assert len(result.tool_execution_history) == 0
    
    def test_client_completion(self):
        """Test completion using GlueLLM client."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        
        result = client.complete("What is 2+2?")
        
        assert isinstance(result, ToolExecutionResult)
        assert isinstance(result.final_response, str)
        assert result.tool_calls_made == 0
    
    def test_custom_model(self):
        """Test completion with custom model."""
        result = complete(
            user_message="Hello",
            model="openai:gpt-4o-mini",
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert isinstance(result.final_response, str)


class TestStructuredOutput:
    """Test structured output functionality."""
    
    def test_simple_structured_output(self):
        """Test basic structured output."""
        
        class SimpleResponse(BaseModel):
            message: Annotated[str, Field(description="A simple message")]
            number: Annotated[int, Field(description="A number")]
        
        result = structured_complete(
            user_message="Return a message 'test' and the number 42",
            response_format=SimpleResponse,
            system_prompt="Extract the requested information.",
        )
        
        assert isinstance(result, SimpleResponse)
        assert isinstance(result.message, str)
        assert isinstance(result.number, int)
    
    def test_nested_structured_output(self):
        """Test structured output with nested models."""
        
        class Address(BaseModel):
            street: Annotated[str, Field(description="Street address")]
            city: Annotated[str, Field(description="City name")]
        
        class Person(BaseModel):
            name: Annotated[str, Field(description="Person's name")]
            age: Annotated[int, Field(description="Person's age")]
            address: Annotated[Address, Field(description="Person's address")]
        
        result = structured_complete(
            user_message="Extract: John Doe, 30 years old, lives at 123 Main St, Springfield",
            response_format=Person,
            system_prompt="Extract person information from the text.",
        )
        
        assert isinstance(result, Person)
        assert isinstance(result.name, str)
        assert isinstance(result.age, int)
        assert isinstance(result.address, Address)
        assert isinstance(result.address.city, str)
    
    def test_structured_output_with_client(self):
        """Test structured output using GlueLLM client."""
        
        class Color(BaseModel):
            name: Annotated[str, Field(description="Color name")]
            hex_code: Annotated[str, Field(description="Hex color code")]
        
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You extract color information.",
        )
        
        result = client.structured_complete(
            user_message="The color red has hex code #FF0000",
            response_format=Color,
        )
        
        assert isinstance(result, Color)
        assert isinstance(result.name, str)
        assert isinstance(result.hex_code, str)


class TestToolExecution:
    """Test automatic tool execution."""
    
    def test_single_tool_execution(self):
        """Test execution of a single tool."""
        result = complete(
            user_message="Use the dummy tool with value 'test123'",
            system_prompt="You are an assistant that uses tools. Use the dummy_tool when asked.",
            tools=[dummy_tool],
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(result.tool_execution_history) >= 1
        assert result.tool_execution_history[0]['tool_name'] == 'dummy_tool'
    
    def test_multiple_tool_calls(self):
        """Test multiple tool calls in sequence."""
        result = complete(
            user_message="First use dummy_tool with 'first', then use it again with 'second'",
            system_prompt="You are an assistant that uses tools as requested.",
            tools=[dummy_tool],
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert result.tool_calls_made >= 2
        assert len(result.tool_execution_history) >= 2
    
    def test_tool_with_parameters(self):
        """Test tool execution with multiple parameters."""
        result = complete(
            user_message="Use the math tool to add 5 and 3",
            system_prompt="You are a math assistant. Use the math_tool for calculations.",
            tools=[math_tool],
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert result.tool_calls_made >= 1
        
        # Check tool was called with correct params
        history = result.tool_execution_history
        assert len(history) >= 1
        assert history[0]['tool_name'] == 'math_tool'
        assert 'a' in history[0]['arguments']
        assert 'b' in history[0]['arguments']
    
    def test_tool_execution_disabled(self):
        """Test that tool execution can be disabled."""
        result = complete(
            user_message="Use the dummy tool",
            system_prompt="You are an assistant with tools.",
            tools=[dummy_tool],
            execute_tools=False,  # Disable execution
        )
        
        assert isinstance(result, ToolExecutionResult)
        # Tools should not be executed when disabled
        assert result.tool_calls_made == 0
    
    def test_max_iterations(self):
        """Test max iterations limit."""
        # This test ensures we don't get stuck in infinite loops
        result = complete(
            user_message="Test",
            system_prompt="You are an assistant.",
            tools=[dummy_tool],
            max_tool_iterations=2,  # Very low limit
        )
        
        assert isinstance(result, ToolExecutionResult)
        # Should complete without hanging
        assert result.tool_calls_made <= 2


class TestConversationState:
    """Test conversation state management."""
    
    def test_conversation_persists(self):
        """Test that conversation history persists across calls."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant. Remember previous messages.",
        )
        
        # First message
        result1 = client.complete("My name is Alice")
        assert isinstance(result1, ToolExecutionResult)
        
        # Second message referencing first
        result2 = client.complete("What is my name?")
        assert isinstance(result2, ToolExecutionResult)
        # The response should reference Alice (though we can't assert exact text)
        assert len(result2.final_response) > 0
    
    def test_conversation_reset(self):
        """Test conversation reset functionality."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        
        # Add some messages
        client.complete("Remember the number 42")
        
        # Reset conversation
        client.reset_conversation()
        
        # Check conversation was reset
        assert len(client._conversation.messages) == 0
    
    def test_tool_calls_persist_in_conversation(self):
        """Test that tool calls are part of conversation history."""
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            system_prompt="You are a math assistant.",
            tools=[math_tool],
        )
        
        # First calculation
        result1 = client.complete("What is 10 + 5?")
        initial_message_count = len(client._conversation.messages)
        
        # Second calculation (should have context)
        result2 = client.complete("Now multiply that by 2")
        final_message_count = len(client._conversation.messages)
        
        # Conversation should grow
        assert final_message_count > initial_message_count


class TestMultipleTools:
    """Test scenarios with multiple tools."""
    
    def test_multiple_tools_available(self):
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
        
        result = complete(
            user_message="Use tool_a with 'test'",
            system_prompt="You have access to multiple tools. Use them as requested.",
            tools=[tool_a, tool_b],
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert result.tool_calls_made >= 1
        # Should have used tool_a
        assert any(h['tool_name'] == 'tool_a' for h in result.tool_execution_history)
    
    def test_using_different_tools_sequentially(self):
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
        
        result = complete(
            user_message="What's the weather in Tokyo and what time is it there?",
            system_prompt="Use available tools to answer questions.",
            tools=[get_weather, get_time],
        )
        
        assert isinstance(result, ToolExecutionResult)
        # Should use both tools
        tool_names = {h['tool_name'] for h in result.tool_execution_history}
        # At minimum one tool should be called
        assert len(tool_names) >= 1


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_tool_not_found(self):
        """Test handling when LLM tries to call non-existent tool."""
        # This is tricky to test since the LLM won't normally call non-existent tools
        # We'll test the internal _find_tool method instead
        client = GlueLLM(tools=[dummy_tool])
        
        found = client._find_tool("dummy_tool")
        assert found is not None
        
        not_found = client._find_tool("nonexistent_tool")
        assert not_found is None
    
    def test_tool_execution_error(self):
        """Test handling of errors during tool execution."""
        
        def error_tool(x: str) -> str:
            """A tool that raises an error.
            
            Args:
                x: Input that causes error
            """
            raise ValueError("Intentional error for testing")
        
        result = complete(
            user_message="Use error_tool with 'test'",
            system_prompt="Use the error_tool when asked.",
            tools=[error_tool],
        )
        
        assert isinstance(result, ToolExecutionResult)
        # Should handle error gracefully
        if result.tool_execution_history:
            # Error should be captured in result
            assert "Error" in result.tool_execution_history[0]['result'] or "error" in result.tool_execution_history[0]['result']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

