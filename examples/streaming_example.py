"""Examples of streaming completion with GlueLLM.

Demonstrates stream_complete for token-by-token output, StreamingChunk model,
and optional structured output on the final chunk.
"""

import asyncio

from gluellm import stream_complete, StreamingChunk


async def example_basic_streaming():
    """Simple streaming: print tokens as they arrive."""
    print("=" * 60)
    print("Example 1: Basic Streaming")
    print("=" * 60)

    print("Response: ", end="", flush=True)
    async for chunk in stream_complete(
        "Name three programming languages in one sentence.",
        system_prompt="You are a concise assistant.",
        execute_tools=False,
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.done:
            print(f"\n[Done. Tool calls: {chunk.tool_calls_made}]")
    print()


async def example_streaming_with_chunk_inspection():
    """Inspect StreamingChunk fields (content, done, tool_calls_made)."""
    print("=" * 60)
    print("Example 2: StreamingChunk Inspection")
    print("=" * 60)

    chunks: list[StreamingChunk] = []
    async for chunk in stream_complete(
        "Say hello in 3 words.",
        system_prompt="Be brief.",
        execute_tools=False,
    ):
        chunks.append(chunk)

    full_text = "".join(c.content for c in chunks)
    done_chunk = next(c for c in chunks if c.done)

    print(f"Full response: {full_text!r}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Final chunk done={done_chunk.done}, tool_calls_made={done_chunk.tool_calls_made}")
    print()


async def example_streaming_structured_output():
    """Stream tokens; final chunk includes structured_output when response_format is set."""
    from pydantic import BaseModel, Field

    class ShortAnswer(BaseModel):
        """A very short answer."""

        answer: str = Field(description="The answer in one short phrase")

    print("=" * 60)
    print("Example 3: Streaming with Structured Output")
    print("=" * 60)

    print("Streamed tokens: ", end="", flush=True)
    final_chunk = None
    async for chunk in stream_complete(
        "What is 2+2? Reply with exactly one short phrase.",
        system_prompt="Answer briefly.",
        response_format=ShortAnswer,
        execute_tools=False,
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        final_chunk = chunk

    if final_chunk and final_chunk.structured_output:
        print(f"\nParsed structured_output: {final_chunk.structured_output}")
    print()


async def main():
    await example_basic_streaming()
    await example_streaming_with_chunk_inspection()
    await example_streaming_structured_output()


if __name__ == "__main__":
    asyncio.run(main())
