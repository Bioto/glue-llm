"""Tests for batch processing functionality."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from gluellm.api import ExecutionResult
from gluellm.batch import BatchProcessor, batch_complete, batch_complete_simple, batch_structured_complete
from gluellm.models.batch import (
    APIKeyConfig,
    BatchConfig,
    BatchErrorStrategy,
    BatchRequest,
    BatchResult,
)


class CityInfo(BaseModel):
    """Test model for structured output tests."""

    name: str
    country: str


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
        assert request.response_format is None
        assert request.timeout is None
        assert request.metadata == {}

    def test_batch_request_with_response_format(self):
        """Test BatchRequest accepts a Pydantic model class as response_format."""
        request = BatchRequest(
            user_message="Extract: Paris, France",
            response_format=CityInfo,
        )
        assert request.response_format is CityInfo

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
        assert result.structured_output is None
        assert result.error is None

    def test_batch_result_with_structured_output(self):
        """Test BatchResult carries structured_output."""
        city = CityInfo(name="Paris", country="France")
        result = BatchResult(
            id="req-1",
            success=True,
            response='{"name": "Paris", "country": "France"}',
            structured_output=city,
            elapsed_time=0.8,
        )
        assert result.structured_output is city
        assert result.structured_output.name == "Paris"
        assert result.structured_output.country == "France"

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

    @pytest.mark.asyncio
    async def test_batch_structured_complete(self):
        """Test batch_structured_complete convenience function."""
        messages = ["Extract: Paris, France", "Extract: Tokyo, Japan"]
        paris = CityInfo(name="Paris", country="France")
        tokyo = CityInfo(name="Tokyo", country="Japan")

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            call_count = 0

            def get_client(*args, **kwargs):
                nonlocal call_count
                mock_client = AsyncMock()
                city = paris if call_count == 0 else tokyo
                call_count += 1
                mock_client.structured_complete = AsyncMock(
                    return_value=ExecutionResult(
                        final_response=city.model_dump_json(),
                        tool_calls_made=0,
                        tool_execution_history=[],
                        structured_output=city,
                    )
                )
                return mock_client

            mock_gluellm.side_effect = get_client

            response = await batch_structured_complete(
                messages,
                response_format=CityInfo,
                config=BatchConfig(max_concurrent=2),
            )

            assert response.total_requests == 2
            assert response.successful_requests == 2
            outputs = [r.structured_output for r in response.results]
            names = {o.name for o in outputs}
            assert names == {"Paris", "Tokyo"}


class TestBatchStructuredProcessing:
    """Tests for structured_complete support in batch processing."""

    @pytest.mark.asyncio
    async def test_process_single_uses_structured_complete_when_response_format_set(self):
        """When response_format is set, _process_single calls structured_complete."""
        processor = BatchProcessor(config=BatchConfig(max_concurrent=1))
        city = CityInfo(name="Paris", country="France")

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.structured_complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response='{"name": "Paris", "country": "France"}',
                    tool_calls_made=0,
                    tool_execution_history=[],
                    structured_output=city,
                )
            )
            mock_gluellm.return_value = mock_client

            requests = [
                BatchRequest(
                    id="req-structured",
                    user_message="Extract: Paris, France",
                    response_format=CityInfo,
                ),
            ]
            response = await processor.process(requests)

            assert response.successful_requests == 1
            result = response.results[0]
            assert result.structured_output is city
            assert result.structured_output.name == "Paris"
            mock_client.structured_complete.assert_called_once()
            mock_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_single_uses_complete_when_response_format_not_set(self):
        """Without response_format, _process_single calls complete (not structured_complete)."""
        processor = BatchProcessor(config=BatchConfig(max_concurrent=1))

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(
                return_value=ExecutionResult(
                    final_response="plain text response",
                    tool_calls_made=0,
                    tool_execution_history=[],
                )
            )
            mock_gluellm.return_value = mock_client

            requests = [BatchRequest(id="req-plain", user_message="Hello")]
            response = await processor.process(requests)

            assert response.successful_requests == 1
            result = response.results[0]
            assert result.response == "plain text response"
            assert result.structured_output is None
            mock_client.complete.assert_called_once()
            mock_client.structured_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_batch_structured_and_plain(self):
        """A single batch can mix structured and plain requests."""
        processor = BatchProcessor(config=BatchConfig(max_concurrent=2))
        city = CityInfo(name="Tokyo", country="Japan")

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            call_count = 0

            def get_client(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_client = AsyncMock()
                mock_client.complete = AsyncMock(
                    return_value=ExecutionResult(
                        final_response="plain response",
                        tool_calls_made=0,
                        tool_execution_history=[],
                    )
                )
                mock_client.structured_complete = AsyncMock(
                    return_value=ExecutionResult(
                        final_response='{"name": "Tokyo", "country": "Japan"}',
                        tool_calls_made=0,
                        tool_execution_history=[],
                        structured_output=city,
                    )
                )
                return mock_client

            mock_gluellm.side_effect = get_client

            requests = [
                BatchRequest(id="plain", user_message="Hello"),
                BatchRequest(
                    id="structured",
                    user_message="Extract: Tokyo, Japan",
                    response_format=CityInfo,
                ),
            ]
            response = await processor.process(requests)

            assert response.total_requests == 2
            assert response.successful_requests == 2

            plain_result = next(r for r in response.results if r.id == "plain")
            structured_result = next(r for r in response.results if r.id == "structured")

            assert plain_result.response == "plain response"
            assert plain_result.structured_output is None

            assert structured_result.structured_output is city
            assert structured_result.structured_output.country == "Japan"

    @pytest.mark.asyncio
    async def test_structured_batch_error_populates_error_fields(self):
        """A failed structured request populates error fields, not structured_output."""
        processor = BatchProcessor(config=BatchConfig(max_concurrent=1))

        with patch("gluellm.batch.GlueLLM") as mock_gluellm:
            mock_client = AsyncMock()
            mock_client.structured_complete = AsyncMock(
                side_effect=ValueError("bad schema")
            )
            mock_gluellm.return_value = mock_client

            requests = [
                BatchRequest(
                    id="req-fail",
                    user_message="Extract: ???",
                    response_format=CityInfo,
                ),
            ]
            response = await processor.process(requests)

            assert response.failed_requests == 1
            result = response.results[0]
            assert result.success is False
            assert "bad schema" in result.error
            assert result.structured_output is None


class TestBatchStructuredIntegration:
    """Integration tests for batch structured_complete against a real API."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_structured_complete_returns_parsed_models(self):
        """batch_structured_complete returns valid Pydantic instances from the API."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        response = await batch_structured_complete(
            messages=[
                "Extract city info: Paris is in France",
                "Extract city info: Tokyo is in Japan",
            ],
            response_format=CityInfo,
            model="openai:gpt-5.4-2026-03-05",
            system_prompt="Extract the city name and country from the text.",
            config=BatchConfig(max_concurrent=2),
        )

        assert response.total_requests == 2
        assert response.successful_requests == 2
        assert response.failed_requests == 0

        for result in response.results:
            assert result.success is True
            assert isinstance(result.structured_output, CityInfo)
            assert len(result.structured_output.name) > 0
            assert len(result.structured_output.country) > 0

        names = {r.structured_output.name for r in response.results}
        assert "Paris" in names
        assert "Tokyo" in names

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_mixed_structured_and_plain_integration(self):
        """A mixed batch with both plain and structured requests works end-to-end."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        processor = BatchProcessor(
            model="openai:gpt-5.4-2026-03-05",
            config=BatchConfig(max_concurrent=2),
        )

        requests = [
            BatchRequest(
                id="plain-1",
                user_message="Reply with the word OK only.",
                system_prompt="You are a terse assistant. Reply with exactly: OK",
            ),
            BatchRequest(
                id="structured-1",
                user_message="Extract city info: Berlin is in Germany",
                system_prompt="Extract the city name and country from the text.",
                response_format=CityInfo,
            ),
        ]

        response = await processor.process(requests)

        assert response.total_requests == 2
        assert response.successful_requests == 2

        plain = next(r for r in response.results if r.id == "plain-1")
        structured = next(r for r in response.results if r.id == "structured-1")

        assert plain.success is True
        assert plain.response is not None
        assert plain.structured_output is None

        assert structured.success is True
        assert isinstance(structured.structured_output, CityInfo)
        assert structured.structured_output.name == "Berlin"
        assert structured.structured_output.country == "Germany"
