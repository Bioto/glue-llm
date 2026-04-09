"""
Example: Streaming with Structured Output

Demonstrates how to combine streaming with a Pydantic response_format.
Most chunks carry only text; the final chunk (chunk.done == True) also
carries chunk.structured_output typed as the requested model.

Example 4 shows eval recording with stream_complete; see examples/eval_recording.py
for complete(), structured_complete(), and other store patterns.
"""

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from gluellm import GlueLLM
from gluellm.api import stream_complete
from gluellm.eval import JSONLFileStore


class ArticleSummary(BaseModel):
    """A concise summary of an article."""

    title: str = Field(description="A short title for the summary")
    key_points: list[str] = Field(description="The 3-5 most important points")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")


class CodeReview(BaseModel):
    """Structured feedback from a code review."""

    verdict: str = Field(description="Overall verdict: approve, request_changes, or comment")
    issues: list[str] = Field(description="List of issues found, empty if none")
    suggestions: list[str] = Field(description="Improvement suggestions")
    score: int = Field(description="Quality score from 1 (poor) to 10 (excellent)")


async def example_stream_with_summary():
    """Stream a response and extract a structured summary from the final chunk."""
    print("=" * 70)
    print("Example 1: Streaming with ArticleSummary structured output")
    print("=" * 70)

    article = """
    Renewable energy adoption hit a record high in 2024, with solar and wind
    power accounting for 35% of global electricity generation. Costs have
    dropped 90% over the last decade, making renewables cheaper than coal in
    most markets. Despite the progress, grid reliability and energy storage
    remain key challenges. Governments are investing heavily in battery
    technology and smart-grid infrastructure to address these gaps. Analysts
    predict renewables will exceed 50% of global generation by 2030.
    """

    print("\nStreaming response (text as it arrives):")
    print("-" * 40)

    # stream_complete infers AsyncIterator[StreamingChunk[ArticleSummary]]
    # because response_format is type[ArticleSummary].
    # The model streams JSON text; we parse it on the final chunk.
    async for chunk in stream_complete(
        user_message=f"Summarise this article as JSON matching the requested schema:\n{article}",
        response_format=ArticleSummary,
        model="openai:gpt-4o-mini",
        system_prompt="You are a summarisation assistant. Always reply with valid JSON matching the requested schema.",
    ):
        print(chunk.content, end="", flush=True)

        if chunk.done:
            print("\n" + "-" * 40)
            if chunk.structured_output is None:
                print("(structured output not available — model did not produce parseable JSON)")
            else:
                # chunk.structured_output is typed as ArticleSummary here — no cast needed
                summary = chunk.structured_output
                print(f"\nTitle:     {summary.title}")
                print(f"Sentiment: {summary.sentiment}")
                print("Key points:")
                for point in summary.key_points:
                    print(f"  - {point}")
            print(f"Tool calls: {chunk.tool_calls_made}")


async def example_stream_with_code_review():
    """Stream a code review with structured verdict and issues."""
    print("\n" + "=" * 70)
    print("Example 2: Streaming with CodeReview structured output")
    print("=" * 70)

    code = """
    def get_user(id):
        db = connect()
        result = db.execute(f"SELECT * FROM users WHERE id = {id}")
        return result[0]
    """

    print("\nStreaming response (text as it arrives):")
    print("-" * 40)

    # stream_complete infers AsyncIterator[StreamingChunk[CodeReview]]
    async for chunk in stream_complete(
        user_message=f"Review this Python function and reply with JSON matching the requested schema:\n```python{code}```",
        response_format=CodeReview,
        model="openai:gpt-4o-mini",
        system_prompt="You are a senior Python engineer. Always reply with valid JSON matching the requested schema.",
    ):
        print(chunk.content, end="", flush=True)

        if chunk.done:
            print("\n" + "-" * 40)
            if chunk.structured_output is None:
                print("(structured output not available — model did not produce parseable JSON)")
            else:
                # chunk.structured_output is typed as CodeReview here
                review = chunk.structured_output
                print(f"\nVerdict: {review.verdict.upper()}")
                print(f"Score:   {review.score}/10")
                if review.issues:
                    print("Issues found:")
                    for issue in review.issues:
                        print(f"  - {issue}")
                if review.suggestions:
                    print("Suggestions:")
                    for suggestion in review.suggestions:
                        print(f"  - {suggestion}")


async def example_stream_without_format():
    """Plain streaming without a response_format — structured_output stays None."""
    print("\n" + "=" * 70)
    print("Example 3: Plain streaming (no response_format)")
    print("=" * 70)
    print("\nStreaming response:")
    print("-" * 40)

    # Without response_format, stream_complete infers AsyncIterator[StreamingChunk[Any]]
    async for chunk in stream_complete(
        user_message="In one sentence, what is Python?",
        model="openai:gpt-4o-mini",
    ):
        print(chunk.content, end="", flush=True)
        if chunk.done:
            print()


async def example_stream_with_eval_recording():
    """Stream a short reply while appending an EvalRecord to JSONL (stream_complete + eval_store)."""
    print("\n" + "=" * 70)
    print("Example 4: Streaming with eval recording (JSONL)")
    print("=" * 70)

    Path("./eval_data").mkdir(exist_ok=True)
    out_path = Path("./eval_data/streaming_records.jsonl")
    store = JSONLFileStore(str(out_path))
    client = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Keep answers brief.",
        tools=[],
        eval_store=store,
    )

    print("\nStreaming response:")
    print("-" * 40)
    try:
        async for chunk in client.stream_complete(
            user_message="In one sentence, what is Python?",
            execute_tools=False,
        ):
            print(chunk.content, end="", flush=True)
            if chunk.done:
                print()
    finally:
        await store.close()

    print(f"\nEval record appended to: {out_path.resolve()}")
    print(
        "Note: streaming eval rows typically omit raw_response, tokens_used, "
        "and estimated_cost_usd (no single ChatCompletion / usage on this path)."
    )

    if out_path.is_file() and out_path.stat().st_size > 0:
        last_line = ""
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line:
            rec = json.loads(last_line)
            preview = (rec.get("final_response") or "")[:120]
            if len(rec.get("final_response") or "") > 120:
                preview += "..."
            print(f"\nLast record id={rec.get('id')} success={rec.get('success')}")
            print(f"final_response (truncated): {preview!r}")


async def main():
    await example_stream_with_summary()
    await example_stream_with_code_review()
    await example_stream_without_format()
    await example_stream_with_eval_recording()

    print("\n" + "=" * 70)
    print("All streaming examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
