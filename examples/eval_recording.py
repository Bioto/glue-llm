"""Examples demonstrating evaluation data recording in GlueLLM.

This module provides comprehensive examples of using the evaluation recording
feature to capture request/response data for LLM evaluations.

Examples:
    - Built-in file recording (JSONL)
    - Custom callback handlers
    - Instance-level vs global stores
    - Multi-store (fan-out to multiple backends)
    - Error recording
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from gluellm import GlueLLM, structured_complete
from gluellm.eval import (
    CallbackStore,
    JSONLFileStore,
    MultiStore,
    enable_callback_recording,
    enable_file_recording,
)
from gluellm.models.eval import EvalRecord


# Example 1: Built-in file recording (global)
async def example_1_file_recording():
    """Example: Enable file-based recording globally."""
    print("\n=== Example 1: File Recording (Global) ===")

    # Enable global file recording
    store = enable_file_recording("./eval_data/records.jsonl")
    print(f"Recording enabled: {store.file_path}")

    # All GlueLLM instances will now record automatically
    client = GlueLLM()
    result = await client.complete("What is 2+2?")
    print(f"Response: {result.final_response}")
    print(f"Recorded to: {store.file_path}")

    # Clean up
    await store.close()


# Example 2: Instance-level file recording
async def example_2_instance_recording():
    """Example: Use instance-specific store."""
    print("\n=== Example 2: Instance-Level Recording ===")

    # Create store for this specific client
    store = JSONLFileStore("./eval_data/instance_records.jsonl")
    client = GlueLLM(eval_store=store)

    result = await client.complete("What is the capital of France?")
    print(f"Response: {result.final_response}")

    # Clean up
    await store.close()


# Example 3: Custom callback handler
async def example_3_callback_recording():
    """Example: Use custom callback for storage."""
    print("\n=== Example 3: Callback Recording ===")

    # In-memory storage for demo (could be database, API, etc.)
    recorded_data = []

    async def save_to_memory(record: EvalRecord):
        """Custom storage handler."""
        recorded_data.append(record.model_dump_dict())
        print(f"Recorded: {record.id} - {record.user_message[:50]}...")

    # Enable callback recording (sets global store)
    enable_callback_recording(save_to_memory)
    client = GlueLLM()

    result = await client.complete("Explain quantum computing in one sentence.")
    print(f"Response: {result.final_response}")
    print(f"Total records: {len(recorded_data)}")


# Example 4: Multi-store (fan-out)
async def example_4_multi_store():
    """Example: Record to multiple stores simultaneously."""
    print("\n=== Example 4: Multi-Store (Fan-Out) ===")

    # In-memory storage
    memory_records = []

    async def save_to_memory(record: EvalRecord):
        memory_records.append(record.id)

    # Create multi-store that writes to both file and memory
    multi_store = MultiStore(
        [
            JSONLFileStore("./eval_data/multi_records.jsonl"),
            CallbackStore(save_to_memory),
        ]
    )

    client = GlueLLM(eval_store=multi_store)
    result = await client.complete("What is Python?")
    print(f"Response: {result.final_response}")
    print(f"Memory records: {len(memory_records)}")

    # Clean up
    await multi_store.close()


# Example 5: Structured output recording
async def example_5_structured_recording():
    """Example: Record structured outputs."""
    print("\n=== Example 5: Structured Output Recording ===")

    class Answer(BaseModel):
        number: int
        reasoning: str

    store = JSONLFileStore("./eval_data/structured_records.jsonl")
    _client = GlueLLM(eval_store=store)  # noqa: F841 - sets up store for structured_complete

    result = await structured_complete(
        "What is 2+2? Provide your reasoning.",
        response_format=Answer,
    )
    print(f"Answer: {result.structured_output.number}")
    print(f"Reasoning: {result.structured_output.reasoning}")

    await store.close()


# Example 6: Tool execution recording
async def example_6_tool_recording():
    """Example: Record tool execution history."""
    print("\n=== Example 6: Tool Execution Recording ===")

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}, 72°F"

    store = JSONLFileStore("./eval_data/tool_records.jsonl")
    client = GlueLLM(
        tools=[get_weather],
        eval_store=store,
    )

    result = await client.complete("What's the weather in Paris?")
    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    print(f"Tool history: {result.tool_execution_history}")

    await store.close()


# Example 7: Error recording
async def example_7_error_recording():
    """Example: Record error cases."""
    print("\n=== Example 7: Error Recording ===")

    store = JSONLFileStore("./eval_data/error_records.jsonl")
    client = GlueLLM(eval_store=store)

    try:
        # This will fail (invalid model)
        await client.complete(
            "Hello",
            model="invalid:model-name",
        )
    except Exception as e:
        print(f"Expected error: {type(e).__name__}")
        # Error is automatically recorded

    await store.close()


# Example 8: Reading recorded data
async def example_8_read_records():
    """Example: Read and analyze recorded data."""
    print("\n=== Example 8: Reading Records ===")

    import json

    # Record some data
    store = JSONLFileStore("./eval_data/read_records.jsonl")
    client = GlueLLM(eval_store=store)

    await client.complete("What is AI?")
    await client.complete("What is ML?")
    await store.close()

    # Read the records
    records = []
    with Path("./eval_data/read_records.jsonl").open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Total records: {len(records)}")
    for record in records:
        print(f"  - {record['id']}: {record['user_message'][:50]}...")
        print(f"    Cost: ${record.get('estimated_cost_usd', 0):.6f}")
        print(f"    Latency: {record['latency_ms']:.2f}ms")


# Example 9: Conditional recording
async def example_9_conditional_recording():
    """Example: Record only specific requests."""
    print("\n=== Example 9: Conditional Recording ===")

    class ConditionalStore:
        """Store that only records expensive requests."""

        def __init__(self, base_store: JSONLFileStore):
            self.base_store = base_store

        async def record(self, record: EvalRecord) -> None:
            # Only record if cost > $0.001
            if record.estimated_cost_usd and record.estimated_cost_usd > 0.001:
                await self.base_store.record(record)
            else:
                print(f"Skipping record {record.id} (cost too low)")

        async def close(self) -> None:
            await self.base_store.close()

    base_store = JSONLFileStore("./eval_data/conditional_records.jsonl")
    conditional_store = ConditionalStore(base_store)
    client = GlueLLM(eval_store=conditional_store)

    await client.complete("Short query")
    await client.complete("This is a much longer query that will likely cost more...")

    await conditional_store.close()


async def main():
    """Run all examples."""
    print("GlueLLM Evaluation Recording Examples")
    print("=" * 60)

    # Create eval_data directory
    Path("./eval_data").mkdir(exist_ok=True)

    await example_1_file_recording()
    await example_2_instance_recording()
    await example_3_callback_recording()
    await example_4_multi_store()
    await example_5_structured_recording()
    await example_6_tool_recording()
    await example_7_error_recording()
    await example_8_read_records()
    await example_9_conditional_recording()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nRecorded files are in ./eval_data/")


if __name__ == "__main__":
    asyncio.run(main())
