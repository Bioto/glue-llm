"""Tests for embedding generation functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm.types.completion import CreateEmbeddingResponse

from gluellm import api as gluellm_api
from gluellm.api import GlueLLM
from gluellm.embeddings import embed
from gluellm.models.embedding import EmbeddingResult

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    async def test_simple_embedding_function(self):
        """Test the embed() convenience function."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
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
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
            MagicMock(embedding=[0.2] * 1536, index=1),
        ]
        mock_response.usage = MagicMock(prompt_tokens=4, total_tokens=4)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            result = await embed(["Hello", "World"])

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 2
            assert result.count == 2
            assert result.dimension == 1536
            assert result.tokens_used == 4

    async def test_client_embedding(self):
        """Test embedding using GlueLLM client."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            client = GlueLLM()
            result = await client.embed("Hello, world!")

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
            assert result.dimension == 1536

    async def test_client_custom_embedding_model(self):
        """Test client with custom embedding model."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 3072, index=0),  # text-embedding-3-large has 3072 dimensions
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-large"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-large")
        ):
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
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            result = await embed("Hello", correlation_id="test-correlation-123")

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1

    async def test_embedding_with_dimensions(self):
        """Test that dimensions parameter is passed through to the provider."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 512, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            result = await embed("Hello", dimensions=512)

            # Verify _aembedding was called with dimensions in kwargs
            mock_provider._aembedding.assert_called_once()
            call_kwargs = mock_provider._aembedding.call_args.kwargs
            assert call_kwargs.get("dimensions") == 512

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1
            assert result.dimension == 512

    async def test_embedding_uses_default_dimensions_from_config(self, monkeypatch):
        """Test that dimensions defaults to settings.default_embedding_dimensions when not passed."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 256, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            "gluellm.embeddings.settings.default_embedding_dimensions",
            256,
        )
        with patch(
            "gluellm.embeddings._provider_cache.get_provider",
            return_value=(mock_provider, "text-embedding-3-small"),
        ):
            result = await embed("Hello")

            mock_provider._aembedding.assert_called_once()
            call_kwargs = mock_provider._aembedding.call_args.kwargs
            assert call_kwargs.get("dimensions") == 256

            assert isinstance(result, EmbeddingResult)
            assert result.dimension == 256

    async def test_embedding_with_encoding_format(self):
        """Test embedding with encoding_format parameter."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            result = await embed("Hello", encoding_format="float")

            # Verify _aembedding was called with encoding_format in kwargs
            mock_provider._aembedding.assert_called_once()
            call_kwargs = mock_provider._aembedding.call_args.kwargs
            assert call_kwargs.get("encoding_format") == "float"

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1

    async def test_embedding_with_kwargs_pass_through(self):
        """Test that additional kwargs are passed through to the provider."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            result = await embed("Hello", user="test-user-123")

            # Verify _aembedding was called with user in kwargs
            mock_provider._aembedding.assert_called_once()
            call_kwargs = mock_provider._aembedding.call_args.kwargs
            assert call_kwargs.get("user") == "test-user-123"

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 1

    async def test_embedding_with_multiple_options(self):
        """Test embedding with multiple options (encoding_format and kwargs)."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
        ]
        mock_response.usage = MagicMock(prompt_tokens=2, total_tokens=2)
        mock_response.model = "openai/text-embedding-3-small"

        mock_provider = MagicMock()
        mock_provider._aembedding = AsyncMock(return_value=mock_response)
        with patch(
            "gluellm.embeddings._provider_cache.get_provider", return_value=(mock_provider, "text-embedding-3-small")
        ):
            result = await embed("Hello", encoding_format="float", user="test-user")

            # Verify _aembedding was called with all kwargs
            mock_provider._aembedding.assert_called_once()
            call_kwargs = mock_provider._aembedding.call_args.kwargs
            assert call_kwargs.get("encoding_format") == "float"
            assert call_kwargs.get("user") == "test-user"

            assert isinstance(result, EmbeddingResult)
            assert result.dimension == 1536

    async def test_embed_with_dimensions_does_not_pass_dimensions_twice_to_openai(self):
        """Regression: embed(dimensions=N) must not pass dimensions twice to OpenAI embeddings.create().

        any_llm's OpenAI provider previously passed dimensions both as an explicit
        argument and inside embedding_kwargs, causing TypeError. Our patch pops
        dimensions from kwargs before _convert_embedding_params so it's only passed once.
        """
        mock_response = CreateEmbeddingResponse(
            data=[{"embedding": [0.1] * 768, "index": 0, "object": "embedding"}],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 2, "total_tokens": 2},
        )

        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.embeddings.create = mock_create

        with (
            patch.object(gluellm_api._provider_cache, "_providers", {}),
            patch("any_llm.providers.openai.base.AsyncOpenAI", return_value=mock_client),
        ):
            result = await embed(
                "Your document chunk to embed.",
                model="openai:text-embedding-3-small",
                dimensions=768,
                api_key="sk-test-dummy",  # Bypass env check; client is mocked
            )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("dimensions") == 768
        assert call_kwargs.get("model") == "text-embedding-3-small"
        assert "input" in call_kwargs
        assert isinstance(result, EmbeddingResult)
        assert result.dimension == 768
        assert len(result.embeddings) == 1
