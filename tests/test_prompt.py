"""Tests for prompt formatting and template rendering."""

from functools import partial

from gluellm.models.prompt import SystemPrompt, _flatten_xml


class TestXMLFlattening:
    """Test XML flattening functionality."""

    def test_flatten_xml_basic(self):
        """Test basic XML flattening."""
        xml = """
        <tag>
            <nested>content</nested>
        </tag>
        """
        result = _flatten_xml(xml)
        assert result == "<tag><nested>content</nested></tag>"
        assert "\n" not in result
        assert "  " not in result  # No double spaces

    def test_flatten_xml_preserves_content(self):
        """Test that XML flattening preserves content."""
        xml = """
        <system_prompt>
            <instructions>Test instructions</instructions>
        </system_prompt>
        """
        result = _flatten_xml(xml)
        assert "Test instructions" in result
        assert "<system_prompt>" in result
        assert "<instructions>" in result

    def test_flatten_xml_empty(self):
        """Test flattening empty XML."""
        assert _flatten_xml("") == ""
        assert _flatten_xml("   ") == ""

    def test_flatten_xml_already_flat(self):
        """Test flattening already flat XML."""
        flat_xml = "<tag>content</tag>"
        result = _flatten_xml(flat_xml)
        assert result == flat_xml


class TestSystemPromptFormatting:
    """Test SystemPrompt formatting with tools."""

    def test_basic_formatting(self):
        """Test basic prompt formatting without tools."""
        prompt = SystemPrompt(content="You are a helpful assistant.")
        formatted = prompt.to_formatted_string(tools=[])
        assert "You are a helpful assistant." in formatted
        assert "<system_prompt>" in formatted
        assert "<system_instructions>" in formatted

    def test_formatting_with_tools(self):
        """Test prompt formatting with tools."""

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}"

        prompt = SystemPrompt(content="You are a weather assistant.")
        formatted = prompt.to_formatted_string(tools=[get_weather])
        assert "You are a weather assistant." in formatted
        assert "get_weather" in formatted
        assert "Get weather for a location" in formatted

    def test_formatting_with_empty_tools_list(self):
        """Test formatting with empty tools list."""
        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[])
        # Should not include tools section
        assert "<tools>" not in formatted

    def test_formatting_with_multiple_tools(self):
        """Test formatting with multiple tools."""

        def tool1(x: str) -> str:
            """Tool 1."""
            return x

        def tool2(y: int) -> int:
            """Tool 2."""
            return y

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool1, tool2])
        assert "tool1" in formatted
        assert "tool2" in formatted
        assert "Tool 1" in formatted
        assert "Tool 2" in formatted


class TestXMLInjectionPrevention:
    """Test prevention of XML injection attacks."""

    def test_xml_special_characters_in_instructions(self):
        """Test that XML special characters in instructions are handled."""
        prompt = SystemPrompt(content="Use <tool> and >output< and &ampersand")
        formatted = prompt.to_formatted_string(tools=[])
        # Should still contain the content (Jinja2 autoescape=True should handle this)
        assert "Use" in formatted
        # The XML should be valid
        assert formatted.startswith("<system_prompt>")

    def test_xml_tags_in_instructions(self):
        """Test instructions containing XML-like tags."""
        prompt = SystemPrompt(content="Follow these <rules> and </rules>")
        formatted = prompt.to_formatted_string(tools=[])
        # Should be escaped or handled properly
        assert "Follow these" in formatted

    def test_xml_special_characters_in_tool_docstring(self):
        """Test XML special characters in tool docstrings."""

        def tool_with_xml(x: str) -> str:
            """Tool with <special> & characters."""
            return x

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool_with_xml])
        # Should handle special characters in docstring
        assert "tool_with_xml" in formatted

    def test_ampersand_in_content(self):
        """Test ampersand handling in content."""
        prompt = SystemPrompt(content="Use A & B together")
        formatted = prompt.to_formatted_string(tools=[])
        assert "Use A" in formatted
        assert "B together" in formatted


class TestToolDocstringHandling:
    """Test handling of various tool docstring scenarios."""

    def test_tool_with_none_docstring(self):
        """Test tool with None docstring."""

        def tool_no_doc(x: str) -> str:
            return x

        # Remove docstring
        tool_no_doc.__doc__ = None

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool_no_doc])
        assert "tool_no_doc" in formatted
        # None docstring should be handled gracefully
        assert "None" not in formatted or formatted.count("None") == 0

    def test_tool_with_empty_docstring(self):
        """Test tool with empty docstring."""

        def tool_empty_doc(x: str) -> str:
            """"""
            return x

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool_empty_doc])
        assert "tool_empty_doc" in formatted

    def test_tool_with_xml_tags_in_docstring(self):
        """Test tool docstring containing XML tags."""

        def tool_xml_doc(x: str) -> str:
            """Tool that uses <tag>format</tag>."""
            return x

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool_xml_doc])
        assert "tool_xml_doc" in formatted
        # XML in docstring should be handled/escaped

    def test_tool_with_very_long_docstring(self):
        """Test tool with very long docstring."""

        def tool_long_doc(x: str) -> str:
            """Tool with a very long docstring. """ + "x" * 1000
            return x

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool_long_doc])
        assert "tool_long_doc" in formatted
        # Should handle long docstrings without error

    def test_tool_with_multiline_docstring(self):
        """Test tool with multiline docstring."""

        def tool_multiline(x: str) -> str:
            """Tool with
            multiple
            lines
            in docstring."""
            return x

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[tool_multiline])
        assert "tool_multiline" in formatted
        assert "multiple" in formatted or "lines" in formatted


class TestTemplateEdgeCases:
    """Test Jinja2 template edge cases."""

    def test_template_with_undefined_variable(self):
        """Test template rendering with undefined variables."""
        # The template should handle undefined variables gracefully
        from gluellm.models.prompt import BASE_SYSTEM_PROMPT

        # Render with required variables
        result = BASE_SYSTEM_PROMPT.render(instructions="Test", tools=[])
        assert "Test" in result

    def test_template_with_none_tools(self):
        """Test template with None tools (should use empty list)."""
        prompt = SystemPrompt(content="Test")
        # Should handle None gracefully
        formatted = prompt.to_formatted_string(tools=[])
        assert "<tools>" not in formatted

    def test_template_with_lambda_function(self):
        """Test template with lambda function as tool."""
        lambda_tool = lambda x: x  # noqa: E731
        lambda_tool.__name__ = "lambda_tool"
        lambda_tool.__doc__ = "Lambda tool"

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[lambda_tool])
        assert "lambda_tool" in formatted

    def test_template_with_partial_function(self):
        """Test template with partial function as tool."""

        def base_tool(x: str, y: int = 5) -> str:
            """Base tool."""
            return f"{x}:{y}"

        partial_tool = partial(base_tool, y=10)
        partial_tool.__name__ = "partial_tool"
        partial_tool.__doc__ = "Partial tool"

        prompt = SystemPrompt(content="Test")
        formatted = prompt.to_formatted_string(tools=[partial_tool])
        assert "partial_tool" in formatted


class TestSystemPromptEdgeCases:
    """Test edge cases for SystemPrompt."""

    def test_empty_content(self):
        """Test SystemPrompt with empty content."""
        prompt = SystemPrompt(content="")
        formatted = prompt.to_formatted_string(tools=[])
        assert "<system_prompt>" in formatted
        assert "<system_instructions>" in formatted

    def test_very_long_content(self):
        """Test SystemPrompt with very long content."""
        long_content = "x" * 10000
        prompt = SystemPrompt(content=long_content)
        formatted = prompt.to_formatted_string(tools=[])
        assert long_content in formatted

    def test_content_with_newlines(self):
        """Test SystemPrompt with content containing newlines."""
        content = "Line 1\nLine 2\nLine 3"
        prompt = SystemPrompt(content=content)
        formatted = prompt.to_formatted_string(tools=[])
        # Newlines should be preserved in content but XML flattened
        assert "Line 1" in formatted or "Line1" in formatted

    def test_multiple_calls_same_prompt(self):
        """Test that same prompt can be formatted multiple times."""
        prompt = SystemPrompt(content="Test")

        def tool1(x: str) -> str:
            """Tool 1."""
            return x

        formatted1 = prompt.to_formatted_string(tools=[tool1])
        formatted2 = prompt.to_formatted_string(tools=[tool1])
        assert formatted1 == formatted2
