"""Examples of API key pool and multi-key rotation with GlueLLM.

Demonstrates APIKeyPool, APIKeyConfig, round-robin key rotation,
and using the pool with batch processing.
"""

import asyncio
import os

from gluellm import APIKeyPool, batch_complete_simple
from gluellm.models.batch import APIKeyConfig as BatchAPIKeyConfig
from gluellm.rate_limiting.api_key_pool import APIKeyConfig, get_api_key_env_var


async def example_pool_from_env():
    """APIKeyPool loads keys from environment when created with no args."""
    print("=" * 60)
    print("Example 1: Pool from Environment")
    print("=" * 60)

    pool = APIKeyPool()
    key = pool.get_key("openai")
    if key:
        print(f"Got API key for openai (hash: ...{key[-4:]})")
    else:
        print("No OpenAI key in environment - set OPENAI_API_KEY")
    print()


async def example_multi_key_round_robin():
    """Multiple keys rotate in round-robin order."""
    print("=" * 60)
    print("Example 2: Multi-Key Round Robin")
    print("=" * 60)

    # Use same key twice to demonstrate round-robin (in production, use distinct keys)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set - skipping round-robin demo")
        print()
        return

    keys = [
        APIKeyConfig(key=api_key, provider="openai", requests_per_minute=60),
        APIKeyConfig(key=api_key, provider="openai", requests_per_minute=60),
    ]
    pool = APIKeyPool(keys=keys)

    # Each get_key call rotates to the next key
    for i in range(4):
        _ = pool.get_key("openai")
        print(f"  Call {i + 1}: got key (round-robin)")

    print(f"  Pool has {len(pool._keys_by_provider.get('openai', []))} key(s) for openai")
    print()


async def example_batch_with_key_pool():
    """Use API key pool with batch processing via BatchConfig.api_keys."""
    print("=" * 60)
    print("Example 3: Batch Processing with Key Pool")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set - skipping batch demo")
        print()
        return

    from gluellm import BatchConfig

    # Pass multiple key configs for round-robin during batch
    config = BatchConfig(
        max_concurrent=2,
        api_keys=[
            BatchAPIKeyConfig(key=api_key, provider="openai"),
            BatchAPIKeyConfig(key=api_key, provider="openai"),
        ],
    )

    messages = ["What is 1+1?", "What is 2+2?"]
    responses = await batch_complete_simple(messages, config=config)
    for msg, resp in zip(messages, responses, strict=True):
        print(f"  Q: {msg} -> A: {resp[:50]}...")
    print()


async def example_key_config_with_rate_limits():
    """APIKeyConfig with per-key rate limits."""
    print("=" * 60)
    print("Example 4: Per-Key Rate Limits")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set - skipping")
        print()
        return

    config = APIKeyConfig(
        key=api_key,
        provider="openai",
        requests_per_minute=30,
        burst=5,
    )
    print(f"Key config: provider={config.provider}, rpm=30, burst=5, hash={config.key_hash}")
    print()


async def main():
    print(f"Env var for openai: {get_api_key_env_var('openai')}")
    await example_pool_from_env()
    await example_multi_key_round_robin()
    await example_key_config_with_rate_limits()
    await example_batch_with_key_pool()


if __name__ == "__main__":
    asyncio.run(main())
