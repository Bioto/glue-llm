"""Tests for evaluation data storage in GlueLLM."""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from gluellm.api import GlueLLM
from gluellm.eval import (
    CallbackStore,
    JSONLFileStore,
    MultiStore,
    enable_callback_recording,
    enable_file_recording,
    get_global_eval_store,
    set_global_eval_store,
)
from gluellm.models.eval import EvalRecord

# Test Fixtures


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file for testing."""
    return tmp_path / "test_records.jsonl"


@pytest.fixture
def mock_callback():
    """Create a mock callback that stores records."""
    records = []

    async def callback(record: EvalRecord):
        records.append(record)

    callback.records = records
    return callback


@pytest.fixture
def sample_eval_record():
    """Create a sample EvalRecord for testing."""
    return EvalRecord(
        user_message="Test message",
        system_prompt="Test prompt",
        model="openai:gpt-4o-mini",
        final_response="Test response",
        tool_calls_made=0,
    )


@pytest.fixture(autouse=True)
def reset_global_eval_store():
    """Reset global eval store before and after each test to prevent pollution."""
    original = get_global_eval_store()
    set_global_eval_store(None)
    yield
    set_global_eval_store(original)


# Test Classes


class TestEvalRecord:
    """Tests for EvalRecord model."""

    def test_model_creation(self):
        """Test creating an EvalRecord with all fields."""
        record = EvalRecord(
            user_message="Hello",
            system_prompt="You are helpful",
            model="openai:gpt-4",
            final_response="Hi there!",
            tool_calls_made=2,
            tool_execution_history=[{"tool_name": "test", "result": "ok"}],
            tools_available=["test_tool"],
            latency_ms=123.45,
            tokens_used={"prompt": 10, "completion": 20, "total": 30},
            estimated_cost_usd=0.001,
            success=True,
        )

        assert record.user_message == "Hello"
        assert record.system_prompt == "You are helpful"
        assert record.model == "openai:gpt-4"
        assert record.final_response == "Hi there!"
        assert record.tool_calls_made == 2
        assert isinstance(record.id, str)
        assert isinstance(record.timestamp, datetime)

    def test_model_defaults(self):
        """Test EvalRecord default values."""
        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        assert record.id is not None
        assert isinstance(record.timestamp, datetime)
        assert record.final_response == ""
        assert record.tool_calls_made == 0
        assert record.tool_execution_history == []
        assert record.tools_available == []
        assert record.latency_ms == 0.0
        assert record.success is True
        assert record.error_type is None
        assert record.error_message is None

    def test_model_serialization_json(self, sample_eval_record):
        """Test JSON serialization."""
        json_str = sample_eval_record.model_dump_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["user_message"] == "Test message"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)  # ISO format

    def test_model_serialization_dict(self, sample_eval_record):
        """Test dictionary serialization."""
        data = sample_eval_record.model_dump_dict()
        assert isinstance(data, dict)
        assert data["user_message"] == "Test message"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)  # ISO format

    def test_datetime_serialization(self):
        """Test that datetime is properly serialized."""
        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        json_str = record.model_dump_json()
        data = json.loads(json_str)

        # Timestamp should be ISO format string
        assert isinstance(data["timestamp"], str)
        # Should be parseable
        parsed_time = datetime.fromisoformat(data["timestamp"])
        assert isinstance(parsed_time, datetime)


class TestJSONLFileStore:
    """Tests for JSONLFileStore."""

    @pytest.mark.asyncio
    async def test_file_creation(self, temp_jsonl_file):
        """Test that file is created on first write."""
        store = JSONLFileStore(temp_jsonl_file)
        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await store.record(record)
        await store.close()

        assert temp_jsonl_file.exists()
        assert temp_jsonl_file.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_directory_auto_creation(self, tmp_path):
        """Test that parent directories are created automatically."""
        nested_file = tmp_path / "nested" / "dir" / "records.jsonl"
        store = JSONLFileStore(nested_file)

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await store.record(record)
        await store.close()

        assert nested_file.exists()
        assert nested_file.parent.exists()

    @pytest.mark.asyncio
    async def test_async_writes(self, temp_jsonl_file):
        """Test async write operations."""
        store = JSONLFileStore(temp_jsonl_file)

        records = [
            EvalRecord(
                user_message=f"Message {i}",
                system_prompt="Test",
                model="test:model",
            )
            for i in range(5)
        ]

        for record in records:
            await store.record(record)

        await store.close()

        # Verify all records were written
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 5

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert "user_message" in data

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, temp_jsonl_file):
        """Test thread-safe concurrent writes."""
        store = JSONLFileStore(temp_jsonl_file)

        async def write_record(i):
            record = EvalRecord(
                user_message=f"Concurrent {i}",
                system_prompt="Test",
                model="test:model",
            )
            await store.record(record)

        # Write 10 records concurrently
        await asyncio.gather(*[write_record(i) for i in range(10)])
        await store.close()

        # Verify all records were written
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 10

    @pytest.mark.asyncio
    async def test_file_content_format(self, temp_jsonl_file):
        """Test that file content is valid JSONL format."""
        store = JSONLFileStore(temp_jsonl_file)

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
            final_response="Response",
        )

        await store.record(record)
        await store.close()

        # Read and verify JSONL format (one JSON object per line)
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

            # Each line should be valid JSON
            data = json.loads(lines[0])
            assert data["user_message"] == "Test"
            assert data["final_response"] == "Response"

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_jsonl_file):
        """Test that store errors don't break recording."""
        store = JSONLFileStore(temp_jsonl_file)

        # Record should succeed
        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await store.record(record)

        # Close the store
        await store.close()

        # Try to record after close (should handle gracefully)
        await store.record(record)  # Should not raise

    @pytest.mark.asyncio
    async def test_close_flush(self, temp_jsonl_file):
        """Test that close() flushes writes."""
        store = JSONLFileStore(temp_jsonl_file)

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await store.record(record)

        # File should exist and have content after close
        await store.close()
        assert temp_jsonl_file.exists()
        assert temp_jsonl_file.stat().st_size > 0

    def test_missing_aiofiles(self, temp_jsonl_file):
        """Test ImportError when aiofiles is missing."""
        with patch("gluellm.eval.jsonl_store.aiofiles", None), pytest.raises(ImportError, match="aiofiles"):
            JSONLFileStore(temp_jsonl_file)


@pytest.mark.asyncio
class TestCallbackStore:
    """Tests for CallbackStore."""

    async def test_async_callback(self, mock_callback):
        """Test async callback execution."""
        store = CallbackStore(mock_callback)

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await store.record(record)

        assert len(mock_callback.records) == 1
        assert mock_callback.records[0].user_message == "Test"

    async def test_sync_callback(self):
        """Test sync callback execution (runs in executor)."""
        records = []

        def sync_callback(record: EvalRecord):
            records.append(record)

        store = CallbackStore(sync_callback)

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await store.record(record)

        assert len(records) == 1
        assert records[0].user_message == "Test"

    async def test_callback_error_handling(self):
        """Test that callback errors don't break recording."""
        error_count = 0

        async def failing_callback(record: EvalRecord):
            nonlocal error_count
            error_count += 1
            raise ValueError("Callback error")

        store = CallbackStore(failing_callback)

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        # Should not raise, error is logged but not propagated
        await store.record(record)

        assert error_count == 1

    async def test_close(self, mock_callback):
        """Test close() method."""
        store = CallbackStore(mock_callback)
        await store.close()  # Should not raise

    async def test_callback_receives_correct_data(self, mock_callback):
        """Test that callback receives correct EvalRecord."""
        store = CallbackStore(mock_callback)

        record = EvalRecord(
            user_message="Specific message",
            system_prompt="Specific prompt",
            model="openai:gpt-4",
            final_response="Specific response",
            tool_calls_made=3,
            latency_ms=456.78,
        )

        await store.record(record)

        assert len(mock_callback.records) == 1
        received = mock_callback.records[0]
        assert received.user_message == "Specific message"
        assert received.system_prompt == "Specific prompt"
        assert received.model == "openai:gpt-4"
        assert received.final_response == "Specific response"
        assert received.tool_calls_made == 3
        assert received.latency_ms == 456.78


@pytest.mark.asyncio
class TestMultiStore:
    """Tests for MultiStore."""

    async def test_fan_out(self, temp_jsonl_file, mock_callback):
        """Test fan-out to multiple stores."""
        file_store = JSONLFileStore(temp_jsonl_file)
        callback_store = CallbackStore(mock_callback)

        multi_store = MultiStore([file_store, callback_store])

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        await multi_store.record(record)
        await multi_store.close()

        # Verify file store received record
        assert temp_jsonl_file.exists()
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

        # Verify callback store received record
        assert len(mock_callback.records) == 1

    async def test_parallel_writes(self, tmp_path):
        """Test parallel writes to multiple stores."""
        file1 = tmp_path / "file1.jsonl"
        file2 = tmp_path / "file2.jsonl"

        store1 = JSONLFileStore(file1)
        store2 = JSONLFileStore(file2)

        multi_store = MultiStore([store1, store2])

        # Write multiple records
        for i in range(5):
            record = EvalRecord(
                user_message=f"Message {i}",
                system_prompt="Test",
                model="test:model",
            )
            await multi_store.record(record)

        await multi_store.close()

        # Verify both files have all records
        for file_path in [file1, file2]:
            with file_path.open() as f:
                lines = [line.strip() for line in f if line.strip()]
                assert len(lines) == 5

    async def test_partial_failure(self, temp_jsonl_file):
        """Test that partial failure doesn't break other stores."""
        records = []

        async def failing_callback(record: EvalRecord):
            raise ValueError("Callback failed")

        file_store = JSONLFileStore(temp_jsonl_file)
        callback_store = CallbackStore(failing_callback)

        multi_store = MultiStore([file_store, callback_store])

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        # Should not raise even though callback fails
        await multi_store.record(record)
        await multi_store.close()

        # File store should still have the record
        assert temp_jsonl_file.exists()
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

    async def test_close_all_stores(self, tmp_path):
        """Test that close() closes all stores."""
        file1 = tmp_path / "file1.jsonl"
        file2 = tmp_path / "file2.jsonl"

        store1 = JSONLFileStore(file1)
        store2 = JSONLFileStore(file2)

        multi_store = MultiStore([store1, store2])

        await multi_store.close()

        # Both stores should be closed
        # Try to record after close (should handle gracefully)
        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )
        await store1.record(record)  # Should not raise
        await store2.record(record)  # Should not raise

    async def test_empty_stores_list(self):
        """Test MultiStore with empty stores list."""
        multi_store = MultiStore([])

        record = EvalRecord(
            user_message="Test",
            system_prompt="Test",
            model="test:model",
        )

        # Should not raise
        await multi_store.record(record)
        await multi_store.close()


class TestGlobalStore:
    """Tests for global store registration."""

    def test_set_get_global_store(self, temp_jsonl_file):
        """Test setting and getting global store."""
        store = JSONLFileStore(temp_jsonl_file)

        set_global_eval_store(store)
        retrieved = get_global_eval_store()

        assert retrieved is store

    def test_get_global_store_none(self):
        """Test getting global store when not set."""
        set_global_eval_store(None)
        assert get_global_eval_store() is None

    def test_enable_file_recording(self, tmp_path):
        """Test enable_file_recording() creates and sets global store."""
        file_path = tmp_path / "global_records.jsonl"

        store = enable_file_recording(str(file_path))

        assert isinstance(store, JSONLFileStore)
        assert get_global_eval_store() is store
        assert store.file_path == file_path

    def test_enable_file_recording_default_path(self):
        """Test enable_file_recording() with default path."""
        store = enable_file_recording()

        assert isinstance(store, JSONLFileStore)
        assert get_global_eval_store() is store
        assert store.file_path is not None

    def test_enable_callback_recording(self, mock_callback):
        """Test enable_callback_recording() creates and sets global store."""
        store = enable_callback_recording(mock_callback)

        assert isinstance(store, CallbackStore)
        assert get_global_eval_store() is store

    @pytest.mark.asyncio
    async def test_glue_llm_uses_global_store(self, mock_callback):
        """Test GlueLLM uses global store when no instance store provided."""
        enable_callback_recording(mock_callback)

        client = GlueLLM()

        # Mock the LLM call to avoid actual API call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test message")

        # Verify callback was called
        assert len(mock_callback.records) == 1
        assert mock_callback.records[0].user_message == "Test message"

    @pytest.mark.asyncio
    async def test_instance_store_overrides_global(self, mock_callback, temp_jsonl_file):
        """Test instance store overrides global store."""
        enable_callback_recording(mock_callback)

        instance_store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=instance_store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test message")

        # Global callback should not be called
        assert len(mock_callback.records) == 0

        # Instance store should have the record
        await instance_store.close()
        assert temp_jsonl_file.exists()


@pytest.mark.asyncio
class TestEvalAPIIntegration:
    """Tests for API integration with evaluation recording."""

    async def test_complete_recording_success(self, temp_jsonl_file):
        """Test that complete() records successful requests."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_call.return_value = mock_response

            result = await client.complete("Test message")

        await store.close()

        # Verify record was written
        assert temp_jsonl_file.exists()
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["user_message"] == "Test message"
            assert data["final_response"] == "Test response"
            assert data["success"] is True
            assert data["tokens_used"]["total"] == 30

    async def test_complete_recording_error(self, temp_jsonl_file):
        """Test that complete() records failed requests."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call to raise an error
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_call.side_effect = ValueError("Test error")

            with pytest.raises(ValueError):
                await client.complete("Test message")

        await store.close()

        # Verify error record was written
        assert temp_jsonl_file.exists()
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["user_message"] == "Test message"
            assert data["success"] is False
            assert data["error_type"] == "ValueError"
            assert "error" in data["error_message"].lower()

    async def test_structured_complete_recording(self, temp_jsonl_file):
        """Test that structured_complete() records with structured output."""
        store = JSONLFileStore(temp_jsonl_file)

        class TestModel(BaseModel):
            value: int

        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"value": 42}'
            mock_response.choices[0].message.parsed = TestModel(value=42)
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            result = await client.structured_complete("Test", response_format=TestModel)

        await store.close()

        # Verify record includes structured output
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["structured_output"] is not None
            assert data["structured_output"]["value"] == 42

    async def test_tool_execution_recording(self, temp_jsonl_file):
        """Test that tool execution history is recorded."""

        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store, tools=[test_tool])

        # Mock the LLM call with tool calls
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            # First call: model wants to call tool
            tool_call_response = MagicMock()
            tool_call_response.choices = [MagicMock()]
            tool_call_response.choices[0].message.content = None
            # Create a proper mock for the function with name attribute
            mock_function = MagicMock()
            mock_function.name = "test_tool"
            mock_function.arguments = '{"x": 5}'
            tool_call_mock = MagicMock()
            tool_call_mock.id = "call_123"
            tool_call_mock.function = mock_function
            tool_call_response.choices[0].message.tool_calls = [tool_call_mock]

            # Second call: final response
            final_response = MagicMock()
            final_response.choices = [MagicMock()]
            final_response.choices[0].message.content = "Result is 10"
            final_response.choices[0].message.tool_calls = None
            final_response.usage = None

            mock_call.side_effect = [tool_call_response, final_response]

            result = await client.complete("Calculate 5 * 2")

        await store.close()

        # Verify tool execution was recorded
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["tool_calls_made"] == 1
            assert len(data["tool_execution_history"]) == 1
            assert data["tool_execution_history"][0]["tool_name"] == "test_tool"
            assert "test_tool" in data["tools_available"]

    async def test_correlation_id_recording(self, temp_jsonl_file):
        """Test that correlation_id is captured."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test", correlation_id="test-correlation-123")

        await store.close()

        # Verify correlation_id was recorded
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert data["correlation_id"] == "test-correlation-123"

    async def test_latency_calculation(self, temp_jsonl_file):
        """Test that latency is calculated correctly."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call with delay
        with patch("gluellm.api._llm_call_with_retry") as mock_call:

            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms delay
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "Test"
                mock_response.choices[0].message.tool_calls = None
                mock_response.usage = None
                return mock_response

            mock_call.side_effect = delayed_response

            await client.complete("Test")

        await store.close()

        # Verify latency was recorded
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert data["latency_ms"] > 0
            assert data["latency_ms"] >= 100  # Should be at least 100ms

    async def test_messages_snapshot(self, temp_jsonl_file):
        """Test that messages snapshot includes full conversation."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("First message")
            await client.complete("Second message")

        await store.close()

        # Verify messages snapshot includes conversation
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 2

            # Second message should include previous conversation
            data = json.loads(lines[1])
            assert len(data["messages_snapshot"]) > 1
            assert any(
                msg.get("role") == "user" and "First message" in msg.get("content", "")
                for msg in data["messages_snapshot"]
            )

    async def test_graceful_degradation(self):
        """Test that recording errors don't break completions."""

        # Create a store that always fails
        class FailingStore:
            async def record(self, record):
                raise RuntimeError("Storage failed")

            async def close(self):
                pass

        store = FailingStore()
        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            # Should complete successfully despite store error
            result = await client.complete("Test")
            assert result.final_response == "Success"

    async def test_no_recording_when_store_none(self):
        """Test that no recording happens when eval_store is None."""
        client = GlueLLM(eval_store=None)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            # Should complete without errors
            result = await client.complete("Test")
            assert result.final_response == "Test"


@pytest.mark.asyncio
class TestEvalEdgeCases:
    """Tests for edge cases and error scenarios."""

    async def test_empty_tools_list(self, temp_jsonl_file):
        """Test recording with empty tools list."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store, tools=[])

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test")

        await store.close()

        # Verify empty tools list is recorded
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert data["tools_available"] == []

    async def test_no_tool_calls(self, temp_jsonl_file):
        """Test recording with no tool calls."""
        store = JSONLFileStore(temp_jsonl_file)

        def test_tool() -> str:
            return "result"

        client = GlueLLM(eval_store=store, tools=[test_tool])

        # Mock the LLM call (no tool calls)
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Direct response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test")

        await store.close()

        # Verify no tool calls recorded
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert data["tool_calls_made"] == 0
            assert len(data["tool_execution_history"]) == 0

    async def test_max_iterations_reached(self, temp_jsonl_file):
        """Test recording when max iterations reached."""
        store = JSONLFileStore(temp_jsonl_file)

        def test_tool() -> str:
            return "result"

        client = GlueLLM(eval_store=store, tools=[test_tool], max_tool_iterations=2)

        # Mock the LLM call to always request tools
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [
                MagicMock(
                    id="call_123",
                    function=MagicMock(name="test_tool", arguments="{}"),
                )
            ]
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test")

        await store.close()

        # Verify max iterations scenario recorded
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert "Maximum tool execution iterations" in data["final_response"]

    async def test_missing_usage_data(self, temp_jsonl_file):
        """Test recording when usage data is None (some providers don't return it)."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call with no usage data
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None  # No usage data from provider
            mock_call.return_value = mock_response

            await client.complete("Test")

        await store.close()

        # Verify record handles missing usage gracefully
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert data["final_response"] == "Test"
            assert data["tokens_used"] is None  # No usage data recorded

    async def test_none_structured_output(self, temp_jsonl_file):
        """Test recording with None structured_output."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            await client.complete("Test")

        await store.close()

        # Verify None structured_output is handled
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            data = json.loads(lines[0])
            assert data.get("structured_output") is None

    async def test_concurrent_requests(self, temp_jsonl_file):
        """Test concurrent requests with same store."""
        store = JSONLFileStore(temp_jsonl_file)
        client = GlueLLM(eval_store=store)

        # Mock the LLM call
        with patch("gluellm.api._llm_call_with_retry") as mock_call:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = None
            mock_call.return_value = mock_response

            # Make concurrent requests
            await asyncio.gather(
                client.complete("Request 1"),
                client.complete("Request 2"),
                client.complete("Request 3"),
            )

        await store.close()

        # Verify all requests were recorded with correct user messages
        with temp_jsonl_file.open() as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 3

            # Collect all user messages and verify each request was recorded
            user_messages = set()
            for line in lines:
                data = json.loads(line)
                assert "user_message" in data
                assert data["success"] is True
                user_messages.add(data["user_message"])

            # Verify all three distinct requests were recorded
            assert user_messages == {"Request 1", "Request 2", "Request 3"}
