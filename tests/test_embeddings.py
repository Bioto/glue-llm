"""Tests for embedding generation functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gluellm.api import GlueLLM
from gluellm.embeddings import embed
from gluellm.models.embedding import EmbeddingResult

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    async def test_simple_embedding_function(self):
        """Test the embed() convenience function."""
        # Mock the any_llm_aembedding call
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        with patch("gluellm.embeddings.any_llm_aembedding", new_callable=AsyncMock) as mock_embedding:
            mock_embedding.return_value = mock_response

            result = await embed("Hello, world!")

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
            assert len(result.embeddings[0]) == 1536
            assert result.model == "openai/text-embedding-3-small"
            assert result.tokens_used == 2
            assert result.dimension == 1536
            assert result.count == 1

    async def test_batch_embedding_function(self):
        """Test batch embedding generation."""
        # Mock the any_llm_aembedding call
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
            MagicMock(embedding=[0.2] * 1536, index=1),
        ]
        mock_response.usage = MagicMock(prompt_tokens=4, total_tokens=4)
        mock_response.model = "openai/text-embedding-3-small"

        with patch("gluellm.embeddings.any_llm_aembedding", new_callable=AsyncMock) as mock_embedding:
            mock_embedding.return_value = mock_response

            result = await embed(["Hello", "World"])

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 2
            assert result.count == 2
            assert result.dimension == 1536
            assert result.tokens_used == 4

    async def test_client_embedding(self):
        """Test embedding using GlueLLM client."""
        # Mock the any_llm_aembedding call
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        with patch("gluellm.embeddings.any_llm_aembedding", new_callable=AsyncMock) as mock_embedding:
            mock_embedding.return_value = mock_response

            client = GlueLLM()
            result = await client.embed("Hello, world!")

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
            assert result.dimension == 1536

    async def test_client_custom_embedding_model(self):
        """Test client with custom embedding model."""
        # Mock the any_llm_aembedding call
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 3072, index=0),  # text-embedding-3-large has 3072 dimensions
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-large"

        with patch("gluellm.embeddings.any_llm_aembedding", new_callable=AsyncMock) as mock_embedding:
            mock_embedding.return_value = mock_response

            client = GlueLLM(embedding_model="openai/text-embedding-3-large")
            result = await client.embed("Hello, world!")

            assert isinstance(result, EmbeddingResult)
            assert result.model == "openai/text-embedding-3-large"
            assert result.dimension == 3072

    async def test_embedding_result_get_embedding(self):
        """Test EmbeddingResult.get_embedding() method."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="openai/text-embedding-3-small",
            tokens_used=4,
            estimated_cost_usd=0.0001,
        )

        assert result.get_embedding(0) == [0.1, 0.2, 0.3]
        assert result.get_embedding(1) == [0.4, 0.5, 0.6]

        with pytest.raises(IndexError):
            result.get_embedding(2)

    async def test_embedding_result_properties(self):
        """Test EmbeddingResult properties."""
        result = EmbeddingResult(
            embeddings=[[0.1] * 1536, [0.2] * 1536],
            model="openai/text-embedding-3-small",
            tokens_used=4,
            estimated_cost_usd=0.0001,
        )

        assert result.dimension == 1536
        assert result.count == 2

        # Empty embeddings
        empty_result = EmbeddingResult(
            embeddings=[],
            model="openai/text-embedding-3-small",
            tokens_used=0,
            estimated_cost_usd=None,
        )
        assert empty_result.dimension == 0
        assert empty_result.count == 0

    async def test_embedding_with_correlation_id(self):
        """Test embedding with correlation ID."""
        # Mock the any_llm_aembedding call
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        with patch("gluellm.embeddings.any_llm_aembedding", new_callable=AsyncMock) as mock_embedding:
            mock_embedding.return_value = mock_response

            result = await embed("Hello", correlation_id="test-correlation-123")

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
