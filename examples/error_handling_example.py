"""
Example demonstrating error handling and retry logic in GlueLLM.
"""

import logging

from gluellm import (
    APIConnectionError,
    AuthenticationError,
    GlueLLM,
    LLMError,
    RateLimitError,
    TokenLimitError,
    complete,
)

# Enable logging to see retry attempts
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def example_basic_error_handling():
    """Basic error handling example."""
    print("\n=== Basic Error Handling ===\n")

    try:
        result = await complete(
            user_message="What is the capital of France?",
            model="openai:gpt-4o-mini",
        )
        print(f"Success: {result.final_response}")

    except TokenLimitError as e:
        print(f"❌ Input too long: {e}")
        print("💡 Tip: Reduce the input size or use a model with larger context")

    except RateLimitError as e:
        print(f"❌ Rate limit hit (after retries): {e}")
        print("💡 Tip: The library already retried 3 times with exponential backoff")

    except AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Tip: Check your API key environment variable")

    except LLMError as e:
        print(f"❌ General LLM error: {e}")


async def example_tool_error_handling():
    """Demonstrate tool execution error handling."""
    print("\n=== Tool Execution Error Handling ===\n")

    def risky_tool(value: int) -> str:
        """A tool that might raise an error.

        Args:
            value: Input value to process
        """
        if value < 0:
            raise ValueError("Value must be positive!")
        return f"Processed value: {value}"

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        tools=[risky_tool],
    )

    try:
        # This might trigger a tool error
        result = await client.complete("Use risky_tool with value -5")

        # Check tool execution history
        print(f"Tool calls made: {result.tool_calls_made}")

        for i, execution in enumerate(result.tool_execution_history, 1):
            print(f"\nTool Call {i}:")
            print(f"  Tool: {execution['tool_name']}")
            print(f"  Arguments: {execution['arguments']}")
            print(f"  Had Error: {execution.get('error', False)}")
            print(f"  Result: {execution['result']}")

        print(f"\nFinal Response: {result.final_response}")

    except LLMError as e:
        print(f"❌ Error: {e}")


async def example_with_retry_visibility():
    """Show retry attempts with logging enabled."""
    print("\n=== Retry Attempts (check logs) ===\n")

    # Note: This example won't actually fail unless you have rate limit issues
    # The logs will show retry attempts if they occur

    try:
        result = await complete(
            user_message="Tell me a short joke",
            model="openai:gpt-4o-mini",
        )
        print(f"✅ Success: {result.final_response}")
        print("\n💡 If rate limits were hit, you'd see retry attempts in the logs above")

    except RateLimitError as e:
        print(f"❌ Rate limit persisted after 3 retry attempts: {e}")

    except APIConnectionError as e:
        print(f"❌ Connection failed after 3 retry attempts: {e}")


async def example_catching_specific_errors():
    """Demonstrate catching specific error types."""
    print("\n=== Specific Error Handling ===\n")

    def handle_llm_request(prompt: str):
        """Handle LLM request with specific error handling."""
        try:
            result = await complete(
                user_message=prompt,
                model="openai:gpt-4o-mini",
            )
            return result.final_response

        except TokenLimitError:
            return "ERROR: Your input is too long. Please provide a shorter message."

        except RateLimitError:
            return "ERROR: Service is busy. Please try again in a few moments."

        except AuthenticationError:
            return "ERROR: Configuration issue. Please contact support."

        except APIConnectionError:
            return "ERROR: Connection issue. Please check your internet and try again."

        except LLMError as e:
            return f"ERROR: An unexpected error occurred: {type(e).__name__}"

    # Test with various prompts
    prompts = [
        "What is 2 + 2?",
        "Explain quantum mechanics in detail" * 1000,  # Might hit token limit
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt[:50]}...")
        response = handle_llm_request(prompt)
        print(f"Response: {response[:100]}...\n")


async def example_checking_error_details():
    """Check execution history for detailed error information."""
    print("\n=== Detailed Error Inspection ===\n")

    def buggy_tool(x: int, y: int) -> int:
        """A tool with a potential division by zero bug.

        Args:
            x: Numerator
            y: Denominator
        """
        return x / y  # Could raise ZeroDivisionError

    client = GlueLLM(
        model="openai:gpt-4o-mini",
        tools=[buggy_tool],
    )

    try:
        result = await client.complete("Use buggy_tool to divide 10 by 0")

        # Inspect execution history
        errors_found = 0
        for execution in result.tool_execution_history:
            if execution.get("error"):
                errors_found += 1
                print(f"⚠️  Error in {execution['tool_name']}:")
                print(f"   Arguments: {execution['arguments']}")
                print(f"   Error: {execution['result']}\n")

        if errors_found == 0:
            print("✅ No tool execution errors")
        else:
            print(f"Found {errors_found} tool execution error(s)")

        print(f"\nModel's response: {result.final_response}")

    except LLMError as e:
        print(f"❌ LLM Error: {e}")


if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 60)
        print("GlueLLM Error Handling Examples")
        print("=" * 60)

        # Run examples
        try:
            await example_basic_error_handling()
        except Exception as e:
            print(f"Example failed (expected if no API key): {e}")

        try:
            await example_tool_error_handling()
        except Exception as e:
            print(f"Example failed (expected if no API key): {e}")

        try:
            await example_with_retry_visibility()
        except Exception as e:
            print(f"Example failed (expected if no API key): {e}")

        try:
            await example_catching_specific_errors()
        except Exception as e:
            print(f"Example failed (expected if no API key): {e}")

        try:
            await example_checking_error_details()
        except Exception as e:
            print(f"Example failed (expected if no API key): {e}")

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    asyncio.run(main())
