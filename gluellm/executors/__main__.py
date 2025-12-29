"""Main entry point for executor module.

This module provides a simple command-line interface for testing
the SimpleExecutor functionality.
"""

import asyncio

from gluellm.executors import SimpleExecutor


async def main():
    """Run a simple executor demo.

    Creates a SimpleExecutor instance and processes a sample query
    to demonstrate basic functionality.
    """
    executor = SimpleExecutor(
        system_prompt="You are a simple executor that can execute a query",
        tools=[],
    )
    print(await executor.execute("What is the weather in Tokyo?"))


if __name__ == "__main__":
    asyncio.run(main())
