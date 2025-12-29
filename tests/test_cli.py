"""Unit tests for CLI commands."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

# Add source directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "gluellm"))

from cli import cli, get_weather


class TestGetWeatherHelper:
    """Tests for the get_weather helper function."""

    def test_get_weather_default_unit(self):
        """Test get_weather with default celsius unit."""
        result = get_weather("Tokyo, Japan")
        assert "Tokyo, Japan" in result
        assert "celsius" in result
        assert "22 degrees" in result
        assert "sunny" in result

    def test_get_weather_fahrenheit(self):
        """Test get_weather with fahrenheit unit."""
        result = get_weather("New York, NY", unit="fahrenheit")
        assert "New York, NY" in result
        assert "fahrenheit" in result
        assert "22 degrees" in result

    def test_get_weather_returns_string(self):
        """Test that get_weather returns a string."""
        result = get_weather("London, UK")
        assert isinstance(result, str)


class TestTestCompletionCommand:
    """Tests for the test_completion CLI command."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_completion_basic_execution(self, mock_completion):
        """Test that test_completion executes without errors."""
        # Mock the completion response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_completion.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["test-completion"])

        assert result.exit_code == 0
        mock_completion.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_completion_calls_with_correct_params(self, mock_completion):
        """Test that test_completion calls completion with correct parameters."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_completion.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["test-completion"])

        # Verify completion was called with expected arguments
        call_kwargs = mock_completion.call_args.kwargs
        assert "messages" in call_kwargs
        assert "model" in call_kwargs
        assert call_kwargs["model"] == "openai:gpt-4o-mini"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_completion_includes_system_message(self, mock_completion):
        """Test that test_completion includes system prompt in messages."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_completion.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["test-completion"])

        call_kwargs = mock_completion.call_args.kwargs
        messages = call_kwargs["messages"]

        # First message should be system message
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"].lower()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_completion_includes_user_message(self, mock_completion):
        """Test that test_completion includes user message about weather."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_completion.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["test-completion"])

        call_kwargs = mock_completion.call_args.kwargs
        messages = call_kwargs["messages"]

        # Should have at least system and user message
        assert len(messages) >= 2
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert "Tokyo" in user_messages[0]["content"]
        assert "Japan" in user_messages[0]["content"]


class TestTestToolCallCommand:
    """Tests for the test_tool_call CLI command."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_tool_call_basic_execution(self, mock_completion):
        """Test that test_tool_call executes without errors when no tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="No tool needed", tool_calls=None))]
        mock_completion.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["test-tool-call"])

        assert result.exit_code == 0
        mock_completion.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_tool_call_with_tool_execution(self, mock_completion):
        """Test that test_tool_call executes tool and makes follow-up call."""
        # Create mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"location": "Tokyo, Japan", "unit": "celsius"})

        # First response with tool call
        mock_first_response = Mock()
        mock_first_message = Mock()
        mock_first_message.tool_calls = [mock_tool_call]
        mock_first_message.content = None
        mock_first_response.choices = [Mock(message=mock_first_message)]

        # Second response after tool execution
        mock_second_response = Mock()
        mock_second_response.choices = [Mock(message=Mock(content="The weather is nice!", tool_calls=None))]

        # Set up mock to return different values on successive calls
        mock_completion.side_effect = [mock_first_response, mock_second_response]

        runner = CliRunner()
        result = runner.invoke(cli, ["test-tool-call"])

        assert result.exit_code == 0
        # Should be called twice: initial and follow-up
        assert mock_completion.call_count == 2

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_tool_call_includes_system_prompt(self, mock_completion):
        """Test that test_tool_call includes appropriate system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_completion.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["test-tool-call"])

        call_kwargs = mock_completion.call_args.kwargs
        messages = call_kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert "get_weather" in messages[0]["content"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_tool_call_follow_up_includes_tool_result(self, mock_completion):
        """Test that follow-up call includes tool result."""
        # Create mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"location": "Paris, France"})

        # First response with tool call
        mock_first_response = Mock()
        mock_first_message = Mock()
        mock_first_message.tool_calls = [mock_tool_call]
        mock_first_response.choices = [Mock(message=mock_first_message)]

        # Second response
        mock_second_response = Mock()
        mock_second_response.choices = [Mock(message=Mock(content="Weather info", tool_calls=None))]

        mock_completion.side_effect = [mock_first_response, mock_second_response]

        runner = CliRunner()
        result = runner.invoke(cli, ["test-tool-call"])

        # Check second call includes tool result
        second_call_kwargs = mock_completion.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # Should have tool role message
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) >= 1
        assert tool_messages[0]["tool_call_id"] == "call_456"
        assert "Paris, France" in tool_messages[0]["content"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("any_llm.completion")
    def test_test_tool_call_parses_json_arguments(self, mock_completion):
        """Test that tool arguments are correctly parsed from JSON."""
        # Create mock tool call with complex arguments
        mock_tool_call = Mock()
        mock_tool_call.id = "call_789"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"location": "London, UK", "unit": "fahrenheit"})

        mock_first_response = Mock()
        mock_first_message = Mock()
        mock_first_message.tool_calls = [mock_tool_call]
        mock_first_response.choices = [Mock(message=mock_first_message)]

        mock_second_response = Mock()
        mock_second_response.choices = [Mock(message=Mock(content="Result", tool_calls=None))]

        mock_completion.side_effect = [mock_first_response, mock_second_response]

        runner = CliRunner()
        result = runner.invoke(cli, ["test-tool-call"])

        # Verify execution succeeded (meaning JSON was parsed correctly)
        assert result.exit_code == 0


class TestRunTestsCommand:
    """Tests for the run_tests CLI command."""

    @patch("subprocess.run")
    def test_run_tests_basic_execution(self, mock_subprocess):
        """Test that run_tests executes pytest."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests"])

        # Command should invoke subprocess
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args.args[0]
        assert "pytest" in call_args

    @patch("subprocess.run")
    def test_run_tests_with_verbose_flag(self, mock_subprocess):
        """Test that run_tests passes verbose flag to pytest."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests", "--verbose"])

        call_args = mock_subprocess.call_args.args[0]
        assert "-v" in call_args

    @patch("subprocess.run")
    def test_run_tests_without_verbose_flag(self, mock_subprocess):
        """Test run_tests without verbose flag."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests"])

        call_args = mock_subprocess.call_args.args[0]
        # When not explicitly verbose, -v shouldn't be there
        # (though -s is always there for output)
        assert "-s" in call_args

    @patch("subprocess.run")
    def test_run_tests_with_specific_test(self, mock_subprocess):
        """Test that run_tests can target a specific test."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests", "--test", "test_single_tool_call"])

        call_args = mock_subprocess.call_args.args[0]
        assert any("test_single_tool_call" in arg for arg in call_args)

    @patch("subprocess.run")
    def test_run_tests_with_class_name(self, mock_subprocess):
        """Test that run_tests can target a specific test class."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests", "--class-name", "TestBasicToolCalling"])

        call_args = mock_subprocess.call_args.args[0]
        assert any("TestBasicToolCalling" in arg for arg in call_args)

    @patch("subprocess.run")
    def test_run_tests_with_no_integration(self, mock_subprocess):
        """Test that run_tests can skip integration tests."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests", "--no-integration"])

        call_args = mock_subprocess.call_args.args[0]
        assert "-m" in call_args
        assert "not integration" in call_args

    @patch("subprocess.run")
    def test_run_tests_sets_correct_working_directory(self, mock_subprocess):
        """Test that run_tests sets the correct working directory."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests"])

        call_kwargs = mock_subprocess.call_args.kwargs
        assert "cwd" in call_kwargs
        assert call_kwargs["cwd"] == "/home/nick/Projects/gluellm"

    @patch("subprocess.run")
    def test_run_tests_exits_with_pytest_returncode(self, mock_subprocess):
        """Test that run_tests propagates pytest's return code."""
        mock_subprocess.return_value = Mock(returncode=1)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests"])

        # CLI should exit with pytest's return code
        assert result.exit_code == 1

    @patch("subprocess.run")
    def test_run_tests_default_runs_all_tests(self, mock_subprocess):
        """Test that run_tests without arguments runs all tests."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests"])

        call_args = mock_subprocess.call_args.args[0]
        assert "tests/" in call_args

    @patch("subprocess.run")
    def test_run_tests_always_shows_output(self, mock_subprocess):
        """Test that run_tests always includes -s flag for output."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(cli, ["run-tests"])

        call_args = mock_subprocess.call_args.args[0]
        assert "-s" in call_args


class TestCLIGroup:
    """Tests for the main CLI group."""

    def test_cli_group_exists(self):
        """Test that CLI group is properly defined."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "GlueLLM CLI" in result.output

    def test_cli_has_test_completion_command(self):
        """Test that test-completion command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert "test-completion" in result.output

    def test_cli_has_test_tool_call_command(self):
        """Test that test-tool-call command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert "test-tool-call" in result.output

    def test_cli_has_run_tests_command(self):
        """Test that run-tests command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert "run-tests" in result.output

    def test_cli_invalid_command_fails(self):
        """Test that invalid command returns error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI commands (require API key)."""

    @pytest.mark.integration
    def test_test_completion_real_api_call(self):
        """Test test_completion with real API call (requires API key)."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        runner = CliRunner()
        result = runner.invoke(cli, ["test-completion"])

        # Should complete successfully with real API
        assert result.exit_code == 0

    @pytest.mark.integration
    def test_test_tool_call_real_api_call(self):
        """Test test_tool_call with real API call (requires API key)."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        runner = CliRunner()
        result = runner.invoke(cli, ["test-tool-call"])

        # Should complete successfully with real API
        assert result.exit_code == 0
