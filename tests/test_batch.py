"""Tests for batch processing functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from gluellm.api import ExecutionResult
from gluellm.batch import BatchProcessor, batch_complete, batch_complete_simple
from gluellm.models.batch import (
    APIKeyConfig,
    BatchConfig,
    BatchErrorStrategy,
    BatchRequest,
    BatchResult,
)


class TestBatchRequest:
    """Tests for BatchRequest model."""

    def test_batch_request_defaults(self):
        """Test BatchRequest with default values."""
        request = BatchRequest(user_message="Hello")
        assert request.user_message == "Hello"
        assert request.id is None
        assert request.system_prompt is None
        assert request.tools is None
        assert request.execute_tools is True
        assert request.max_tool_iterations is None
        assert request.timeout is None
        assert request.metadata == {}

    def test_batch_request_with_metadata(self):
        """Test BatchRequest with custom metadata."""
        metadata = {"user_id": "123", "session": "abc"}
        request = BatchRequest(
            id="req-1",
            user_message="Hello",
            metadata=metadata,
        )
        assert request.id == "req-1"
        assert request.metadata == metadata


class TestBatchResult:
    """Tests for BatchResult model."""

    def test_batch_result_success(self):
        """Test successful BatchResult."""
        result = BatchResult(
            id="req-1",
            success=True,
            response="Hello, world!",
            tool_calls_made=2,
            elapsed_time=1.5,
        )
        assert result.success is True
        assert result.response == "Hello, world!"
        assert result.tool_calls_made == 2
        assert result.error is None

    def test_batch_result_failure(self):
        """Test failed BatchResult."""
        result = BatchResult(
            id="req-1",
            success=False,
            error="Something went wrong",
            error_type="ValueError",
            elapsed_time=0.5,
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.error_type == "ValueError"
        assert result.response is None


class TestBatchConfig:
    """Tests for BatchConfig model."""

    def test_batch_config_defaults(self):
        """Test BatchConfig with default values."""
        config = BatchConfig()
        assert config.max_concurrent == 5
        assert config.error_strategy == BatchErrorStrategy.CONTINUE
        assert config.show_progress is False
        assert config.retry_failed is False

    def test_batch_config_custom(self):
        """Test BatchConfig with custom values."""
        config = BatchConfig(
            max_concurrent=10,
            error_strategy=BatchErrorStrategy.FAIL_FAST,
            show_progress=True,
            retry_failed=True,
        )
        assert config.max_concurrent == 10
        assert config.error_strategy == BatchErrorStrategy.FAIL_FAST
        assert config.show_progress is True
        assert config.retry_failed is True

    def test_batch_config_with_api_keys(self):
        """Test BatchConfig with API keys."""
        api_keys = [
            APIKeyConfig(key="key1", provider="openai"),
            APIKeyConfig(key="key2", provider="openai"),
        ]
        config = BatchConfig(api_keys=api_keys)
        assert config.api_keys == api_keys
        assert len(config.api_keys) == 2

    def test_batch_config_api_keys_none_by_default(self):
        """Test that api_keys is None by default."""
        config = BatchConfig()
        assert config.api_keys is None


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test processing an empty batch."""
        processor = BatchProcessor()
        response = await processor.process([])
        assert response.total_requests == 0
        assert response.successful_requests == 0
        assert response.failed_requests == 0
        assert len(response.results) == 0

    @pytest.mark.asyncio
    async def test_auto_assign_ids(self):
        """Test that IDs are auto-assigned to requests."""
        processor = BatchProcessor()
        requests = [
            BatchRequest(user_message="Test 1"),
            BatchRequest(user_message="Test 2"),
        ]

        # Mock the complete method
        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Response",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )
            mock_gluellm.return_value = mock_client

            response = await processor.process(requests)

            # Check that all results have IDs
            assert all(result.id is not None for result in response.results)
            assert all(result.id.startswith("batch-") for result in response.results)

    @pytest.mark.asyncio
    async def test_successful_batch(self):
        """Test processing a batch of successful requests."""
        processor = BatchProcessor(config=BatchConfig(max_concurrent=2))
        requests = [
            BatchRequest(id="req-1", user_message="Test 1"),
            BatchRequest(id="req-2", user_message="Test 2"),
        ]

        # Mock the GlueLLM client
        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Response",
                    tool_calls_made=0,
                    tool_execution_history=[],
                    tokens_used={"prompt": 10, "completion": 20, "total": 30},
                )
            )
            mock_gluellm.return_value = mock_client

            response = await processor.process(requests)

            assert response.total_requests == 2
            assert response.successful_requests == 2
            assert response.failed_requests == 0
            assert len(response.results) == 2
            assert all(r.success for r in response.results)
            assert response.total_tokens_used is not None
            assert response.total_tokens_used["total"] == 60  # 30 * 2

    @pytest.mark.asyncio
    async def test_error_strategy_continue(self):
        """Test CONTINUE error strategy."""
        processor = BatchProcessor(
            config=BatchConfig(
                max_concurrent=2,
                error_strategy=BatchErrorStrategy.CONTINUE,
            )
        )
        requests = [
            BatchRequest(id="req-1", user_message="Test 1"),
            BatchRequest(id="req-2", user_message="Test 2"),
        ]

        # Mock one success and one failure
        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client1 = AsyncMock()
            mock_client1.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Success",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )

            mock_client2 = AsyncMock()
            mock_client2.complete = AsyncMock(side_effect=Exception("Test error"))

            # Alternate between success and failure
            call_count = 0

            def get_client(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_client1 if call_count % 2 == 1 else mock_client2

            mock_gluellm.side_effect = get_client

            response = await processor.process(requests)

            assert response.total_requests == 2
            assert response.successful_requests == 1
            assert response.failed_requests == 1
            assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_batch_processor_with_api_key_pool(self):
        """Test BatchProcessor with API key pool."""
        api_keys = [
            APIKeyConfig(key="test-key-1", provider="openai"),
            APIKeyConfig(key="test-key-2", provider="openai"),
        ]
        config = BatchConfig(api_keys=api_keys, max_concurrent=2)
        processor = BatchProcessor(config=config)

        # Should have initialized key pool
        assert processor.key_pool is not None
        assert processor.key_pool.has_keys("openai")

        requests = [
            BatchRequest(id="req-1", user_message="Test 1"),
            BatchRequest(id="req-2", user_message="Test 2"),
        ]

        # Mock successful completions
        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Success",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )
            mock_gluellm.return_value = mock_client

            response = await processor.process(requests)

            assert response.total_requests == 2
            assert response.successful_requests == 2
            # Verify that complete was called
            assert mock_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_processor_without_api_key_pool(self):
        """Test BatchProcessor without API key pool."""
        config = BatchConfig(max_concurrent=2)
        processor = BatchProcessor(config=config)

        # Should not have key pool
        assert processor.key_pool is None

        requests = [BatchRequest(id="req-1", user_message="Test 1")]

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Success",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )
            mock_gluellm.return_value = mock_client

            response = await processor.process(requests)

            assert response.total_requests == 1
            assert response.successful_requests == 1

    @pytest.mark.asyncio
    async def test_error_strategy_skip(self):
        """Test SKIP error strategy."""
        processor = BatchProcessor(
            config=BatchConfig(
                max_concurrent=2,
                error_strategy=BatchErrorStrategy.SKIP,
            )
        )
        requests = [
            BatchRequest(id="req-1", user_message="Test 1"),
            BatchRequest(id="req-2", user_message="Test 2"),
        ]

        # Mock one success and one failure
        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client1 = AsyncMock()
            mock_client1.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Success",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )

            mock_client2 = AsyncMock()
            mock_client2.complete = AsyncMock(side_effect=Exception("Test error"))

            call_count = 0

            def get_client(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_client1 if call_count % 2 == 1 else mock_client2

            mock_gluellm.side_effect = get_client

            response = await processor.process(requests)

            # SKIP means only successful results are returned
            assert len(response.results) == 1
            assert all(r.success for r in response.results)


class TestBatchFunctions:
    """Tests for batch convenience functions."""

    @pytest.mark.asyncio
    async def test_batch_complete(self):
        """Test batch_complete function."""
        requests = [
            BatchRequest(user_message="Test 1"),
            BatchRequest(user_message="Test 2"),
        ]

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Response",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )
            mock_gluellm.return_value = mock_client

            response = await batch_complete(
                requests,
                config=BatchConfig(max_concurrent=2),
            )

            assert response.total_requests == 2
            assert response.successful_requests == 2

    @pytest.mark.asyncio
    async def test_batch_complete_simple(self):
        """Test batch_complete_simple function."""
        messages = ["Test 1", "Test 2", "Test 3"]

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Response",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )
            mock_gluellm.return_value = mock_client

            responses = await batch_complete_simple(messages)

            assert len(responses) == 3
            assert all(r == "Response" for r in responses)

    @pytest.mark.asyncio
    async def test_batch_complete_simple_with_errors(self):
        """Test batch_complete_simple with errors."""
        messages = ["Test 1", "Test 2"]

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client1 = AsyncMock()
            mock_client1.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="Success",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )

            mock_client2 = AsyncMock()
            mock_client2.complete = AsyncMock(side_effect=Exception("Test error"))

            call_count = 0

            def get_client(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_client1 if call_count % 2 == 1 else mock_client2

            mock_gluellm.side_effect = get_client

            responses = await batch_complete_simple(messages)

            assert len(responses) == 2
            assert responses[0] == "Success"
            assert responses[1].startswith("Error:")
