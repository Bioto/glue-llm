"""Example demonstrating GlueLLM logging features.

This example shows:
1. Basic logging configuration
2. Different log levels
3. JSON structured logging
4. Log file location and rotation
5. Using logging utilities and decorators
"""

import asyncio
import logging
import os
from pathlib import Path

from gluellm import complete
from gluellm.config import reload_settings, settings
from gluellm.logging_config import setup_logging
from gluellm.logging_utils import log_async_function_call, log_operation, log_timing


async def example_basic_logging():
    """Demonstrate basic logging with different levels."""
    print("\n=== Basic Logging Example ===")

    logger = logging.getLogger("example.basic")

    logger.debug("This is a DEBUG message - detailed information for debugging")
    logger.info("This is an INFO message - general information about program execution")
    logger.warning("This is a WARNING message - something unexpected but handled")
    logger.error("This is an ERROR message - a serious problem occurred")
    logger.critical("This is a CRITICAL message - a critical error occurred")


async def example_llm_logging():
    """Demonstrate logging during LLM calls."""
    print("\n=== LLM Call Logging Example ===")

    logger = logging.getLogger("example.llm")
    logger.info("Making an LLM call - watch for detailed logging")

    try:
        result = await complete("What is 2+2?")
        logger.info(f"LLM call completed successfully: {result.final_response[:50]}...")
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)


@log_async_function_call(log_args=True, log_result=True)
async def example_decorated_function(x: int, y: int) -> int:
    """Example function with automatic logging decorator."""
    await asyncio.sleep(0.1)  # Simulate some work
    return x + y


async def example_logging_utilities():
    """Demonstrate logging utilities and decorators."""
    print("\n=== Logging Utilities Example ===")

    logger = logging.getLogger("example.utils")

    # Using log_timing context manager
    with log_timing("database_query", logger=logger):
        await asyncio.sleep(0.1)  # Simulate database query

    # Using log_operation context manager
    with log_operation("file_processing", logger=logger):
        await asyncio.sleep(0.1)  # Simulate file processing

    # Using decorated function
    result = await example_decorated_function(5, 3)
    logger.info(f"Function result: {result}")


async def example_json_logging():
    """Demonstrate JSON structured logging."""
    print("\n=== JSON Logging Example ===")

    # Temporarily enable JSON logging
    original_json_format = settings.log_json_format

    try:
        # Reload settings with JSON logging enabled
        os.environ["GLUELLM_LOG_JSON_FORMAT"] = "true"
        reload_settings()

        # Reconfigure logging with JSON format
        setup_logging(
            log_level=settings.log_level,
            log_file_level=settings.log_file_level,
            log_json_format=True,
            force=True,
        )

        logger = logging.getLogger("example.json")
        logger.info("This message will be logged in JSON format in the log file")
        logger.info("JSON format is useful for log aggregation tools like ELK, Datadog, etc.")

        # Check log file location
        log_dir = Path("logs")
        log_file = log_dir / "gluellm.log"
        if log_file.exists():
            print(f"\nLog file location: {log_file.absolute()}")
            print("Check the log file to see JSON formatted logs")

    finally:
        # Restore original setting
        if "GLUELLM_LOG_JSON_FORMAT" in os.environ:
            del os.environ["GLUELLM_LOG_JSON_FORMAT"]
        reload_settings()
        setup_logging(
            log_level=settings.log_level,
            log_file_level=settings.log_file_level,
            log_json_format=original_json_format,
            force=True,
        )


async def example_log_configuration():
    """Demonstrate different logging configurations."""
    print("\n=== Log Configuration Example ===")

    logger = logging.getLogger("example.config")

    print("Current log configuration:")
    print(f"  Console log level: {settings.log_level}")
    print(f"  File log level: {settings.log_file_level}")
    print(f"  Log directory: {settings.log_dir or 'logs'}")
    print(f"  Log file name: {settings.log_file_name}")
    print(f"  JSON format: {settings.log_json_format}")
    print(f"  Max file size: {settings.log_max_bytes / 1024 / 1024:.1f}MB")
    print(f"  Backup count: {settings.log_backup_count}")

    logger.info("You can configure logging via:")
    logger.info("  1. Environment variables (GLUELLM_LOG_LEVEL, etc.)")
    logger.info("  2. Settings in code (settings.log_level = 'DEBUG')")
    logger.info("  3. .env file in project root")


async def main():
    """Run all logging examples."""
    print("=" * 60)
    print("GlueLLM Logging Examples")
    print("=" * 60)

    await example_basic_logging()
    await example_logging_utilities()
    await example_llm_logging()
    await example_log_configuration()

    # JSON logging example (optional, as it modifies global state)
    print("\n" + "=" * 60)
    response = input("Run JSON logging example? (y/n): ")
    if response.lower() == "y":
        await example_json_logging()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print(f"Check the log file at: {Path('logs/gluellm.log').absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
