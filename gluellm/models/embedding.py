"""Embedding models for GlueLLM.

This module provides Pydantic models for embedding results and related data structures.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """Result of an embedding generation request.

    Attributes:
        embeddings: List of embedding vectors (one per input text)
        model: Model identifier used for embedding generation
        tokens_used: Total number of tokens used
        estimated_cost_usd: Estimated cost in USD, or None if pricing unavailable
    """

    embeddings: Annotated[
        list[list[float]],
        Field(description="List of embedding vectors, one per input text"),
    ]
    model: Annotated[str, Field(description="Model identifier used for embedding generation")]
    tokens_used: Annotated[int, Field(description="Total number of tokens used", ge=0)]
    estimated_cost_usd: Annotated[
        float | None,
        Field(description="Estimated cost in USD, or None if pricing unavailable", default=None),
    ]

    def get_embedding(self, index: int = 0) -> list[float]:
        """Get a single embedding vector by index.

        Args:
            index: Index of the embedding to retrieve (default: 0)

        Returns:
            Embedding vector as a list of floats

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.embeddings):
            raise IndexError(f"Embedding index {index} out of range (0-{len(self.embeddings) - 1})")
        return self.embeddings[index]

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Dimension of the embedding vectors, or 0 if no embeddings
        """
        if not self.embeddings:
            return 0
        return len(self.embeddings[0])

    @property
    def count(self) -> int:
        """Get the number of embeddings.

        Returns:
            Number of embeddings in the result
        """
        return len(self.embeddings)
