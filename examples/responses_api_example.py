"""
Example: OpenResponses API with Built-in Tools

Demonstrates the responses() API for agentic workflows with built-in tools:
- WEB_SEARCH: Web search (requires compatible provider/model)
- CODE_INTERPRETER: Code execution
- FILE_SEARCH: File search over documents
"""

import asyncio

from gluellm import CODE_INTERPRETER, FILE_SEARCH, WEB_SEARCH, responses


async def example_basic_responses():
    """Basic OpenResponses completion."""
    print("=" * 70)
    print("Example 1: Basic OpenResponses API")
    print("=" * 70)

    result = await responses("What is 2+2? Answer with just the number.")

    print(f"Output: {result.output}")
    print(f"Model: {result.model}")
    print(f"Usage: {result.usage}")
    print()


async def example_with_web_search():
    """Use WEB_SEARCH for real-time information."""
    print("=" * 70)
    print("Example 2: OpenResponses with Web Search")
    print("=" * 70)

    result = await responses(
        "What's the latest headline about AI in the news? Summarize in one sentence.",
        tools=[WEB_SEARCH],
        instructions="You are a news summarizer. Be concise.",
    )

    print(f"Output: {result.output}")
    if result.tool_calls:
        print(f"Tool calls made: {len(result.tool_calls)}")
    if result.usage:
        print(f"Usage: {result.usage}")
    print()


async def example_with_instructions():
    """Use instructions (system message) for the model."""
    print("=" * 70)
    print("Example 3: OpenResponses with Instructions")
    print("=" * 70)

    result = await responses(
        "Translate to French: Hello, how are you?",
        instructions="You are a professional translator. Respond with only the translation.",
    )

    print(f"Output: {result.output}")
    print()


async def example_builtin_tools_constants():
    """Show available built-in tool constants."""
    print("=" * 70)
    print("Example 4: Built-in Tool Constants")
    print("=" * 70)

    print("WEB_SEARCH:", WEB_SEARCH)
    print("CODE_INTERPRETER:", CODE_INTERPRETER)
    print("FILE_SEARCH:", FILE_SEARCH)
    print("\nPass tools=[WEB_SEARCH] or tools=[CODE_INTERPRETER, FILE_SEARCH] as needed.")
    print()


async def main():
    """Run all examples."""
    print("\n🧙 OpenResponses API Examples\n")

    await example_basic_responses()
    await example_with_instructions()
    await example_builtin_tools_constants()

    # Web search requires compatible provider - run if desired
    try:
        await example_with_web_search()
    except Exception as e:
        print(f"(Web search example skipped: {e})")

    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
