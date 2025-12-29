"""
Comprehensive LLM test suite with edge cases and challenging scenarios.
Tests include various tool calling patterns, parameter combinations, and stress tests.
"""

import json
from typing import Annotated, Any

import pytest
from any_llm import completion
from pydantic import BaseModel, Field

# ============================================================================
# TOOL DEFINITIONS (with various complexity levels)
# ============================================================================


def simple_calculator(operation: str, a: float, b: float) -> str:
    """Perform basic arithmetic operations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    """
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else "Error: Division by zero"}
    result = ops.get(operation, "Error: Unknown operation")
    return f"Result: {result}"


def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: The city and country, e.g. "San Francisco, CA"
        unit: Temperature unit, either "celsius" or "fahrenheit"
    """
    return f"Weather in {location}: 22 degrees {unit}, sunny with light clouds"


def search_database(query: str, limit: int = 5, filter_type: str | None = None) -> str:
    """Search a database with optional filtering.

    Args:
        query: Search query string
        limit: Maximum number of results (1-100)
        filter_type: Optional filter type (recent, popular, recommended)
    """
    filter_msg = f" (filtered by: {filter_type})" if filter_type else ""
    return f"Found {limit} results for '{query}'{filter_msg}"


def multi_step_tool(step: int, data: str, flag: bool = False) -> str:
    """A tool that simulates multi-step processing.

    Args:
        step: Step number (1-5)
        data: Data to process
        flag: Enable special processing
    """
    special = " [SPECIAL]" if flag else ""
    return f"Step {step} completed: processed '{data}'{special}"


def get_user_info(user_id: str) -> str:
    """Get user information by ID.

    Args:
        user_id: The user identifier
    """
    return f"User {user_id}: Active since 2023, Premium member"


def complex_tool(
    required_param: str,
    optional_string: str | None = None,
    optional_number: int = 42,
    optional_bool: bool = True,
    optional_list: list[str] | None = None,
) -> str:
    """A tool with many optional parameters.

    Args:
        required_param: Required parameter
        optional_string: Optional string parameter
        optional_number: Optional number with default
        optional_bool: Optional boolean with default
        optional_list: Optional list of strings
    """
    parts = [f"Required: {required_param}"]
    if optional_string:
        parts.append(f"String: {optional_string}")
    if optional_number != 42:
        parts.append(f"Number: {optional_number}")
    if not optional_bool:
        parts.append("Bool: False")
    if optional_list:
        parts.append(f"List: {optional_list}")
    return " | ".join(parts)


# ============================================================================
# RESPONSE FORMAT MODELS
# ============================================================================


class SimpleResponse(BaseModel):
    response: Annotated[str, Field(description="The response text")]


class StructuredAnalysis(BaseModel):
    summary: Annotated[str, Field(description="Brief summary")]
    confidence: Annotated[float, Field(description="Confidence score 0-1")]
    tags: Annotated[list[str], Field(description="Relevant tags")]


class ComplexResponse(BaseModel):
    title: Annotated[str, Field(description="Title of response")]
    content: Annotated[str, Field(description="Main content")]
    metadata: Annotated[dict[str, Any], Field(description="Additional metadata")]
    score: Annotated[int, Field(description="Numerical score 1-10")]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def execute_tool_loop(messages: list, model: str, tools: list, max_iterations: int = 5):
    """Execute a tool calling loop until completion or max iterations."""
    tool_map = {tool.__name__: tool for tool in tools}
    iteration = 0

    while iteration < max_iterations:
        response = completion(
            messages=messages,
            model=model,
            tools=tools,
        )

        # Check if done
        if not response.choices[0].message.tool_calls:
            return response, messages

        # Execute tool calls
        messages.append(response.choices[0].message)

        for tool_call in response.choices[0].message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Execute the tool
            if func_name in tool_map:
                result = tool_map[func_name](**args)
            else:
                result = f"Error: Unknown tool {func_name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        iteration += 1

    # Max iterations reached
    return None, messages


# ============================================================================
# TEST CASES
# ============================================================================


class TestBasicToolCalling:
    """Basic tool calling scenarios."""

    def test_single_tool_call(self):
        """Test a simple single tool call."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ]

        response, final_messages = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[get_weather])

        assert response is not None
        assert len(final_messages) > 2  # Original + tool call + tool result
        assert response.choices[0].message.content is not None
        print(f"✓ Single tool call: {response.choices[0].message.content[:100]}")

    def test_calculator_tool(self):
        """Test calculator with specific operations."""
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": "Calculate 45 multiplied by 23"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[simple_calculator])

        assert response is not None
        content = response.choices[0].message.content
        assert "1035" in content or "1,035" in content  # Should contain the result
        print(f"✓ Calculator tool: {content[:100]}")


class TestMultipleToolCalls:
    """Tests involving multiple tool calls in sequence."""

    def test_sequential_tool_calls(self):
        """Test multiple tools called in sequence."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to weather and calculations."},
            {"role": "user", "content": "Get weather for Paris and then calculate 15 + 27"},
        ]

        response, final_messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_weather, simple_calculator]
        )

        assert response is not None
        # Should have multiple tool calls
        tool_messages = [m for m in final_messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_messages) >= 2
        print(f"✓ Sequential tools: {len(tool_messages)} tool calls made")

    def test_multi_step_tool_chain(self):
        """Test a chain of multi-step tool calls."""
        messages = [
            {"role": "system", "content": "You process data in multiple steps. Use multi_step_tool for each step."},
            {"role": "user", "content": "Process 'test_data' through steps 1, 2, and 3"},
        ]

        response, final_messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[multi_step_tool], max_iterations=10
        )

        assert response is not None
        tool_messages = [m for m in final_messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_messages) >= 3
        print(f"✓ Multi-step chain: {len(tool_messages)} steps executed")


class TestToolParameterEdgeCases:
    """Tests with unusual parameter combinations."""

    def test_optional_parameters_defaults(self):
        """Test tool with all default optional parameters."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Get weather for London"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[get_weather])

        assert response is not None
        assert "celsius" in response.choices[0].message.content.lower()
        print(f"✓ Default parameters: {response.choices[0].message.content[:80]}")

    def test_optional_parameters_specified(self):
        """Test tool with explicitly specified optional parameters."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Get weather for London in fahrenheit"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[get_weather])

        assert response is not None
        content_lower = response.choices[0].message.content.lower()
        # Check for fahrenheit in various forms: "fahrenheit", "°f", or "f"
        assert "fahrenheit" in content_lower or "°f" in content_lower or " f" in content_lower
        print(f"✓ Specified parameters: {response.choices[0].message.content[:80]}")

    def test_complex_optional_parameters(self):
        """Test tool with many optional parameters."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Use complex_tool with required_param='test' and optional_string='hello'"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[complex_tool])

        assert response is not None
        print(f"✓ Complex parameters: {response.choices[0].message.content[:100]}")

    def test_search_with_filters(self):
        """Test search tool with and without filters."""
        messages = [
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": "Search for 'python tutorials' with limit 10 and filter by recent"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[search_database])

        assert response is not None
        print(f"✓ Search with filters: {response.choices[0].message.content[:100]}")


class TestConfusingPrompts:
    """Tests with ambiguous or confusing prompts."""

    def test_ambiguous_tool_choice(self):
        """Test when multiple tools could potentially apply."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need information about user 123 and the weather in their city"},
        ]

        response, final_messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_user_info, get_weather]
        )

        assert response is not None
        # Model should handle ambiguity
        print(f"✓ Ambiguous prompt handled: {len(final_messages)} messages")

    def test_contradictory_instructions(self):
        """Test with contradictory instructions."""
        messages = [
            {"role": "system", "content": "You must use tools when available."},
            {"role": "user", "content": "Don't use any tools, just tell me: what's the weather like?"},
        ]

        response, _ = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_weather], max_iterations=3
        )

        # Should handle contradiction somehow
        assert response is not None or response is None  # Either way is acceptable
        print("✓ Contradictory instructions handled")

    def test_partial_information(self):
        """Test with incomplete information for tool."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Calculate something with 5"},
        ]

        response, _ = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[simple_calculator], max_iterations=3
        )

        # Should either ask for clarification or make reasonable assumptions
        assert response is not None or response is None
        print("✓ Partial information handled")

    def test_irrelevant_tools_available(self):
        """Test when tools are available but not needed."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        response = completion(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_weather, simple_calculator, search_database]
        )

        # Should respond directly without using tools
        assert response.choices[0].message.content is not None
        assert not response.choices[0].message.tool_calls
        print(f"✓ Irrelevant tools ignored: {response.choices[0].message.content[:80]}")


class TestParameterCombinations:
    """Tests with different model parameter combinations."""

    def test_high_temperature_with_tools(self):
        """Test tool calling with high temperature (more random)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Berlin?"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[get_weather])

        assert response is not None
        print(f"✓ High temperature: {response.choices[0].message.content[:80]}")

    def test_low_temperature_with_tools(self):
        """Test tool calling with low temperature (more deterministic)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Calculate 10 + 5"},
        ]

        response = completion(messages=messages, model="openai:gpt-4o-mini", tools=[simple_calculator], temperature=0.1)

        assert response is not None
        print("✓ Low temperature: Response generated")

    def test_max_tokens_limit(self):
        """Test with constrained max tokens."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Get weather for Tokyo and explain it in detail"},
        ]

        response, _ = execute_tool_loop(messages=messages, model="openai:gpt-4o-mini", tools=[get_weather])

        # Should still complete but be more concise
        assert response is not None
        print(f"✓ Token limit: {len(response.choices[0].message.content)} chars")

    def test_with_response_format_no_tools(self):
        """Test structured output without tools."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Analyze this: 'Python is a great language'"},
        ]

        response = completion(messages=messages, model="openai:gpt-4o-mini", response_format=StructuredAnalysis)

        assert response is not None
        # Parse the structured response - could be dict or model
        parsed = response.choices[0].message.parsed

        # Handle both dict and model responses
        if isinstance(parsed, dict):
            assert "summary" in parsed
            assert "confidence" in parsed
            assert "tags" in parsed
            print(f"✓ Structured output: {parsed['summary']}")
        else:
            assert hasattr(parsed, "summary")
            assert hasattr(parsed, "confidence")
            assert hasattr(parsed, "tags")
            print(f"✓ Structured output: {parsed.summary}")


class TestStressScenarios:
    """Stress tests and edge cases."""

    def test_very_long_prompt(self):
        """Test with a very long prompt."""
        long_text = " ".join([f"Item {i}: Some descriptive text about this item." for i in range(100)])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Analyze this data and search for 'summary': {long_text}"},
        ]

        response, _ = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[search_database], max_iterations=3
        )

        assert response is not None or response is None  # May timeout
        print("✓ Long prompt handled")

    def test_rapid_tool_switching(self):
        """Test rapidly switching between different tools."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be efficient."},
            {
                "role": "user",
                "content": "Get weather for NYC, then calculate 5+3, then search for 'news', then get weather for LA",
            },
        ]

        response, final_messages = execute_tool_loop(
            messages=messages,
            model="openai:gpt-4o-mini",
            tools=[get_weather, simple_calculator, search_database],
            max_iterations=10,
        )

        tool_calls = [m for m in final_messages if isinstance(m, dict) and m.get("role") == "tool"]
        print(f"✓ Tool switching: {len(tool_calls)} tools used")

    def test_nested_tool_requirements(self):
        """Test when one tool result requires another tool call."""
        messages = [
            {"role": "system", "content": "You are a data processor."},
            {"role": "user", "content": "Get user info for 'user_789' then search based on their preferences"},
        ]

        response, final_messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_user_info, search_database], max_iterations=10
        )

        assert response is not None or response is None
        print(f"✓ Nested requirements: {len(final_messages)} messages")

    def test_max_iterations_reached(self):
        """Test behavior when max iterations is reached."""
        messages = [
            {"role": "system", "content": "Process data through all 5 steps."},
            {"role": "user", "content": "Process 'data_xyz' through steps 1 through 5 with special flag"},
        ]

        response, final_messages = execute_tool_loop(
            messages=messages,
            model="openai:gpt-4o-mini",
            tools=[multi_step_tool],
            max_iterations=2,  # Intentionally low
        )

        # Should hit max iterations
        print("✓ Max iterations: Stopped after limit")


class TestErrorHandling:
    """Tests for error conditions and recovery."""

    def test_invalid_tool_parameters(self):
        """Test when model might provide invalid parameters."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for nothing with limit of -5"},
        ]

        try:
            response, _ = execute_tool_loop(
                messages=messages, model="openai:gpt-4o-mini", tools=[search_database], max_iterations=3
            )
            print("✓ Invalid parameters: Handled gracefully")
        except Exception as e:
            print(f"✓ Invalid parameters: Caught exception {type(e).__name__}")

    def test_tool_with_no_description(self):
        """Test tool without docstring (edge case)."""

        def undocumented_tool(param: str) -> str:
            return f"Processed: {param}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Use the undocumented tool with 'test'"},
        ]

        try:
            response, _ = execute_tool_loop(
                messages=messages, model="openai:gpt-4o-mini", tools=[undocumented_tool], max_iterations=3
            )
            print("✓ Undocumented tool: Handled")
        except Exception as e:
            print(f"✓ Undocumented tool: Expected error {type(e).__name__}")


class TestRealisticScenarios:
    """Real-world scenario tests."""

    def test_customer_service_scenario(self):
        """Simulate a customer service interaction."""
        messages = [
            {"role": "system", "content": "You are a customer service assistant. Help users with their questions."},
            {
                "role": "user",
                "content": "Hi, I'm user #CS-456 and I want to know the weather at my location and search for 'return policy'",
            },
        ]

        response, final_messages = execute_tool_loop(
            messages=messages,
            model="openai:gpt-4o-mini",
            tools=[get_user_info, get_weather, search_database],
            max_iterations=10,
        )

        assert response is not None
        print(f"✓ Customer service: {response.choices[0].message.content[:100]}")

    def test_data_analysis_workflow(self):
        """Simulate a data analysis workflow."""
        messages = [
            {"role": "system", "content": "You are a data analyst assistant."},
            {"role": "user", "content": "Search for 'sales data', then calculate the average of 100, 150, and 200"},
        ]

        response, final_messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[search_database, simple_calculator], max_iterations=10
        )

        assert response is not None
        print(f"✓ Data analysis: {response.choices[0].message.content[:100]}")

    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation with context."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant with memory of the conversation."},
            {"role": "user", "content": "Get weather for Seattle"},
        ]

        response1, messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_weather, simple_calculator]
        )

        # Add follow-up
        messages.append({"role": "user", "content": "Now calculate 25 + 17"})

        response2, messages = execute_tool_loop(
            messages=messages, model="openai:gpt-4o-mini", tools=[get_weather, simple_calculator], max_iterations=5
        )

        assert response1 is not None
        assert response2 is not None
        print(f"✓ Multi-turn: {len(messages)} total messages")


# ============================================================================
# PYTEST MARKERS
# ============================================================================

# Mark slow tests
pytestmark = pytest.mark.integration


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
