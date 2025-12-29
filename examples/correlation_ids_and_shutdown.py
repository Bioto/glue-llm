"""Example demonstrating correlation IDs, graceful shutdown, and request timeouts.

This example shows how to:
1. Use correlation IDs for request tracking
2. Set up graceful shutdown handlers
3. Configure request timeouts
4. Track in-flight requests
"""

import asyncio
import sys

from gluellm import (
    complete,
    get_correlation_id,
    get_in_flight_count,
    graceful_shutdown,
    is_shutting_down,
    set_correlation_id,
    setup_signal_handlers,
    wait_for_shutdown,
)


async def example_correlation_ids():
    """Example: Using correlation IDs for request tracking."""
    print("=" * 60)
    print("Example 1: Correlation IDs")
    print("=" * 60)

    # Set a custom correlation ID
    set_correlation_id("req-12345")
    print(f"Set correlation ID: {get_correlation_id()}")

    # Make a request - correlation ID will be included in logs
    try:
        result = await complete(
            "What is 2+2?",
            correlation_id="req-12345",  # Can also pass explicitly
        )
        print(f"Response: {result.final_response}")
        print(f"Correlation ID in logs: {get_correlation_id()}")
    except Exception as e:
        print(f"Error: {e}")

    print()


async def example_timeouts():
    """Example: Using request timeouts."""
    print("=" * 60)
    print("Example 2: Request Timeouts")
    print("=" * 60)

    # Request with custom timeout (5 seconds)
    try:
        result = await complete(
            "What is the capital of France?",
            timeout=5.0,  # 5 second timeout
        )
        print(f"Response: {result.final_response}")
    except TimeoutError:
        print("Request timed out!")
    except Exception as e:
        print(f"Error: {e}")

    print()


async def example_graceful_shutdown():
    """Example: Graceful shutdown with signal handling."""
    print("=" * 60)
    print("Example 3: Graceful Shutdown")
    print("=" * 60)

    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    print("Signal handlers registered (Ctrl+C to test)")

    # Simulate some work
    async def process_requests():
        for i in range(5):
            if is_shutting_down():
                print(f"Shutdown detected, stopping at request {i}")
                break

            print(f"Processing request {i + 1}...")
            try:
                result = await complete(
                    f"Count to {i + 1}",
                    timeout=2.0,
                )
                print(f"  Response: {result.final_response[:50]}...")
            except Exception as e:
                print(f"  Error: {e}")

            await asyncio.sleep(0.5)

    # Run requests
    await process_requests()

    # Check in-flight requests
    in_flight = get_in_flight_count()
    print(f"\nIn-flight requests: {in_flight}")

    # Wait for any remaining requests
    if in_flight > 0:
        print("Waiting for in-flight requests to complete...")
        await wait_for_shutdown(max_wait_time=10.0)

    print("Shutdown complete")
    print()


async def example_shutdown_context():
    """Example: Using ShutdownContext to track requests."""
    print("=" * 60)
    print("Example 4: Shutdown Context")
    print("=" * 60)

    from gluellm import ShutdownContext

    # Use ShutdownContext to automatically track requests
    async def process_with_context():
        with ShutdownContext():
            print("Processing request with ShutdownContext...")
            result = await complete("What is Python?", timeout=5.0)
            print(f"Response: {result.final_response[:50]}...")
            print(f"In-flight requests: {get_in_flight_count()}")

    await process_with_context()
    print(f"After context exit, in-flight requests: {get_in_flight_count()}")
    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GlueLLM: Correlation IDs, Timeouts, and Graceful Shutdown")
    print("=" * 60 + "\n")

    try:
        # Example 1: Correlation IDs
        await example_correlation_ids()

        # Example 2: Timeouts
        await example_timeouts()

        # Example 3: Graceful shutdown (comment out if testing manually)
        # await example_graceful_shutdown()

        # Example 4: Shutdown context
        await example_shutdown_context()

        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal, initiating graceful shutdown...")
        await graceful_shutdown(max_wait_time=10.0)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()

    # Run examples
    asyncio.run(main())
