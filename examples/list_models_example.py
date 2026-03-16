"""
Example: List Available Models

Demonstrates list_models() to enumerate models for a provider.
Useful for discovering available models and checking API connectivity.
"""

import asyncio

from gluellm import list_models


async def example_list_openai_models():
    """List models available from OpenAI."""
    print("=" * 70)
    print("Example 1: List OpenAI Models")
    print("=" * 70)

    models = await list_models(provider="openai")
    print(f"Found {len(models)} model(s)\n")
    print(f"{'ID':<40} {'Created':<12} {'Owned By':<15}")
    print("-" * 70)
    for m in models[:15]:
        model_id = getattr(m, "id", "?")
        created = getattr(m, "created", "")
        owned_by = getattr(m, "owned_by", "")
        print(f"{str(model_id):<40} {str(created):<12} {str(owned_by):<15}")
    if len(models) > 15:
        print(f"... and {len(models) - 15} more")
    print()


async def example_list_anthropic_models():
    """List models available from Anthropic."""
    print("=" * 70)
    print("Example 2: List Anthropic Models")
    print("=" * 70)

    try:
        models = await list_models(provider="anthropic")
        print(f"Found {len(models)} model(s)\n")
        for m in models[:10]:
            model_id = getattr(m, "id", "?")
            print(f"  - {model_id}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
    except Exception as e:
        print(f"Error (ensure ANTHROPIC_API_KEY is set): {e}")
    print()


async def example_with_api_key_override():
    """List models with optional API key override."""
    print("=" * 70)
    print("Example 3: With API Key Override")
    print("=" * 70)

    # Pass api_key to override env/default
    # models = await list_models(provider="openai", api_key="sk-...")
    # For demo we use default (from OPENAI_API_KEY env)
    models = await list_models(provider="openai")
    print(f"Default: uses OPENAI_API_KEY from environment")
    print(f"Models returned: {len(models)}")
    print()


async def main():
    """Run all examples."""
    print("\n🧙 List Models Examples\n")

    await example_list_openai_models()
    await example_list_anthropic_models()
    await example_with_api_key_override()

    print("=" * 70)
    print("CLI equivalent: gluellm list-models -p openai")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
