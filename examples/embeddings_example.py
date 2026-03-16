"""Examples of embedding generation with GlueLLM.

Demonstrates embed(), EmbeddingResult, batch embedding, and cosine similarity comparison.
"""

import asyncio
import math

from gluellm import EmbeddingResult, embed


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def example_single_embedding():
    """Embed a single text."""
    print("=" * 60)
    print("Example 1: Single Embedding")
    print("=" * 60)

    result = await embed("The quick brown fox jumps over the lazy dog.")
    print(f"Model: {result.model}")
    print(f"Dimension: {result.dimension}")
    print(f"Count: {result.count}")
    print(f"Tokens used: {result.tokens_used}")
    if result.estimated_cost_usd is not None:
        print(f"Estimated cost: ${result.estimated_cost_usd:.6f}")
    vec = result.get_embedding(0)
    print(f"Embedding (first 5 dims): {vec[:5]!r}...")
    print()


async def example_batch_embedding():
    """Embed multiple texts in one call."""
    print("=" * 60)
    print("Example 2: Batch Embedding")
    print("=" * 60)

    texts = [
        "Python is a programming language.",
        "JavaScript runs in the browser.",
        "Machine learning uses data to make predictions.",
    ]
    result = await embed(texts)
    print(f"Input count: {len(texts)}, Embedding count: {result.count}")
    assert result.count == len(texts)
    for i, text in enumerate(texts):
        vec = result.get_embedding(i)
        print(f"  [{i}] {text[:40]!r}... -> dim={len(vec)}")
    print()


async def example_similarity_comparison():
    """Compare embeddings with cosine similarity."""
    print("=" * 60)
    print("Example 3: Similarity Comparison")
    print("=" * 60)

    texts = [
        "A cat sits on a mat.",
        "The feline rests on the rug.",
        "The stock market surged today.",
    ]
    result = await embed(texts)
    a, b, c = result.get_embedding(0), result.get_embedding(1), result.get_embedding(2)

    sim_ab = cosine_similarity(a, b)
    sim_ac = cosine_similarity(a, c)
    sim_bc = cosine_similarity(b, c)

    print(f"Similarity (cat/feline): {sim_ab:.4f}")
    print(f"Similarity (cat/stock):  {sim_ac:.4f}")
    print(f"Similarity (feline/stock): {sim_bc:.4f}")
    print("Semantically similar pairs should have higher similarity.")
    print()


async def main():
    await example_single_embedding()
    await example_batch_embedding()
    await example_similarity_comparison()


if __name__ == "__main__":
    asyncio.run(main())
