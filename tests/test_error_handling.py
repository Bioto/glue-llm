"""
Test error handling and retry logic for GlueLLM.
Tests the comprehensive error classification and retry mechanisms.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from source.api import (
    GlueLLM,
    complete,
    classify_llm_error,
    TokenLimitError,
    RateLimitError,
    APIConnectionError,
    InvalidRequestError,
    AuthenticationError,
    LLMError,
)


class TestErrorClassification:
    """Test error classification logic."""
    
    def test_token_limit_error_classification(self):
        """Test that token limit errors are correctly classified."""
        errors = [
            Exception("context length exceeded"),
            Exception("maximum context length is 4096"),
            Exception("too many tokens in request"),
            Exception("token limit exceeded"),
        ]
        
        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, TokenLimitError), f"Failed for: {error}"
    
    def test_rate_limit_error_classification(self):
        """Test that rate limit errors are correctly classified."""
        errors = [
            Exception("rate limit exceeded"),
            Exception("too many requests"),
            Exception("429 - rate_limit_exceeded"),
            Exception("quota exceeded"),
            Exception("resource exhausted"),
        ]
        
        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, RateLimitError), f"Failed for: {error}"
    
    def test_connection_error_classification(self):
        """Test that connection errors are correctly classified."""
        errors = [
            Exception("connection timeout"),
            Exception("network error"),
            Exception("503 service unavailable"),
            Exception("502 bad gateway"),
            Exception("unreachable"),
        ]
        
        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, APIConnectionError), f"Failed for: {error}"
    
    def test_auth_error_classification(self):
        """Test that authentication errors are correctly classified."""
        errors = [
            Exception("invalid api key"),
            Exception("401 unauthorized"),
            Exception("authentication failed"),
            Exception("403 forbidden"),
        ]
        
        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, AuthenticationError), f"Failed for: {error}"
    
    def test_invalid_request_error_classification(self):
        """Test that invalid request errors are correctly classified."""
        errors = [
            Exception("invalid request"),
            Exception("400 bad request"),
            Exception("validation error"),
        ]
        
        for error in errors:
            classified = classify_llm_error(error)
            assert isinstance(classified, InvalidRequestError), f"Failed for: {error}"
    
    def test_generic_error_classification(self):
        """Test that unknown errors become generic LLMError."""
        error = Exception("some random error")
        classified = classify_llm_error(error)
        assert isinstance(classified, LLMError)
        assert not isinstance(classified, TokenLimitError)


class TestRetryLogic:
    """Test retry behavior with mocked LLM calls."""
    
    @patch('source.api._safe_llm_call')
    def test_retry_on_rate_limit(self, mock_safe_call):
        """Test that rate limit errors trigger retries."""
        # Create a proper mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        # First two calls raise RateLimitError, third succeeds
        mock_safe_call.side_effect = [
            RateLimitError("Rate limit exceeded"),
            RateLimitError("Rate limit exceeded"),
            mock_response,
        ]
        
        client = GlueLLM(model="openai:gpt-4o-mini")
        result = client.complete("Test message")
        
        # Should have retried and eventually succeeded
        assert mock_safe_call.call_count == 3
        assert result.final_response == "Success"
    
    @patch('source.api._safe_llm_call')
    def test_retry_on_connection_error(self, mock_safe_call):
        """Test that connection errors trigger retries."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Success after retry"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_safe_call.side_effect = [
            APIConnectionError("Connection timeout"),
            mock_response,
        ]
        
        client = GlueLLM(model="openai:gpt-4o-mini")
        result = client.complete("Test message")
        
        assert mock_safe_call.call_count == 2
        assert result.final_response == "Success after retry"
    
    @patch('source.api._safe_llm_call')
    def test_no_retry_on_token_limit(self, mock_safe_call):
        """Test that token limit errors do NOT trigger retries."""
        mock_safe_call.side_effect = TokenLimitError("Context length exceeded")
        
        client = GlueLLM(model="openai:gpt-4o-mini")
        
        with pytest.raises(TokenLimitError):
            client.complete("Test message with too many tokens")
        
        # Should only be called once (no retries)
        assert mock_safe_call.call_count == 1
    
    @patch('source.api._safe_llm_call')
    def test_no_retry_on_auth_error(self, mock_safe_call):
        """Test that authentication errors do NOT trigger retries."""
        mock_safe_call.side_effect = AuthenticationError("Invalid API key")
        
        client = GlueLLM(model="openai:gpt-4o-mini")
        
        with pytest.raises(AuthenticationError):
            client.complete("Test message")
        
        # Should only be called once (no retries)
        assert mock_safe_call.call_count == 1
    
    @patch('source.api._safe_llm_call')
    def test_max_retries_exceeded(self, mock_safe_call):
        """Test that max retries is respected."""
        # Always raise RateLimitError - test the actual retry decorator
        mock_safe_call.side_effect = RateLimitError("Rate limit exceeded")
        
        client = GlueLLM(model="openai:gpt-4o-mini")
        
        with pytest.raises(RateLimitError):
            client.complete("Test message")
        
        # Should retry up to 3 times total
        assert mock_safe_call.call_count == 3


class TestToolExecutionErrorHandling:
    """Test error handling during tool execution."""
    
    @patch('source.api._safe_llm_call')
    def test_tool_execution_exception_handling(self, mock_safe_call):
        """Test that tool execution errors are caught and added to history."""
        
        def error_tool(x: str) -> str:
            """A tool that raises an error."""
            raise ValueError(f"Tool error with input: {x}")
        
        # First call: model wants to use tool
        tool_call_response = Mock()
        tool_call_choice = Mock()
        tool_call_msg = Mock()
        tool_call_msg.content = None
        
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "error_tool"
        mock_tool_call.function.arguments = '{"x": "test"}'
        tool_call_msg.tool_calls = [mock_tool_call]
        tool_call_choice.message = tool_call_msg
        tool_call_response.choices = [tool_call_choice]
        
        # Second call: final response
        final_response = Mock()
        final_choice = Mock()
        final_msg = Mock()
        final_msg.content = "I handled the error"
        final_msg.tool_calls = None
        final_choice.message = final_msg
        final_response.choices = [final_choice]
        
        mock_safe_call.side_effect = [tool_call_response, final_response]
        
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            tools=[error_tool],
        )
        result = client.complete("Use error_tool")
        
        # Check that error was captured in history
        assert len(result.tool_execution_history) == 1
        assert result.tool_execution_history[0]['error'] is True
        assert "ValueError" in result.tool_execution_history[0]['result']
        assert result.final_response == "I handled the error"
    
    @patch('source.api._safe_llm_call')
    def test_malformed_json_in_tool_args(self, mock_safe_call):
        """Test handling of malformed JSON in tool arguments."""
        
        def dummy_tool(x: str) -> str:
            """A simple tool."""
            return f"Result: {x}"
        
        # Mock a response with malformed JSON
        tool_call_response = Mock()
        tool_call_choice = Mock()
        tool_call_msg = Mock()
        tool_call_msg.content = None
        
        mock_tool_call = Mock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "dummy_tool"
        mock_tool_call.function.arguments = '{invalid json}'  # Malformed
        tool_call_msg.tool_calls = [mock_tool_call]
        tool_call_choice.message = tool_call_msg
        tool_call_response.choices = [tool_call_choice]
        
        # Final response
        final_response = Mock()
        final_choice = Mock()
        final_msg = Mock()
        final_msg.content = "Handled invalid JSON"
        final_msg.tool_calls = None
        final_choice.message = final_msg
        final_response.choices = [final_choice]
        
        mock_safe_call.side_effect = [tool_call_response, final_response]
        
        client = GlueLLM(
            model="openai:gpt-4o-mini",
            tools=[dummy_tool],
        )
        result = client.complete("Test")
        
        # Check that JSON error was captured
        assert len(result.tool_execution_history) == 1
        assert result.tool_execution_history[0]['error'] is True
        assert "Invalid JSON" in result.tool_execution_history[0]['result']


class TestStructuredCompleteErrorHandling:
    """Test error handling in structured_complete."""
    
    @patch('source.api._safe_llm_call')
    def test_structured_complete_with_rate_limit_retry(self, mock_safe_call):
        """Test that structured_complete also benefits from retry logic."""
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            name: str
            value: int
        
        # Create proper mock responses
        # First call fails
        first_call_side_effect = RateLimitError("Rate limit exceeded")
        
        # Second call succeeds
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.parsed = TestModel(name="test", value=42)
        mock_message.content = '{"name": "test", "value": 42}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_safe_call.side_effect = [
            first_call_side_effect,
            mock_response,
        ]
        
        client = GlueLLM(model="openai:gpt-4o-mini")
        result = client.structured_complete("Test", TestModel)
        
        # Verify retry happened (2 calls total)
        assert mock_safe_call.call_count == 2
        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

