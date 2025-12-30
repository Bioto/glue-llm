"""LLM pricing data for cost estimation.

This module contains pricing information for various LLM providers and models.
Prices are in USD per 1 million tokens.

Note: Pricing changes frequently. Last updated: December 2024.
Always verify current pricing with the provider's official documentation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a specific model.

    Attributes:
        input_price_per_million: Cost per 1M input tokens in USD
        output_price_per_million: Cost per 1M output tokens in USD
        cached_input_price_per_million: Cost per 1M cached input tokens (if supported)
    """

    input_price_per_million: float
    output_price_per_million: float
    cached_input_price_per_million: float | None = None


# OpenAI Pricing (as of December 2024)
# https://openai.com/pricing
OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-4o models
    "gpt-4o": ModelPricing(2.50, 10.00, cached_input_price_per_million=1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, cached_input_price_per_million=0.075),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00, cached_input_price_per_million=1.25),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00, cached_input_price_per_million=1.25),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    # GPT-4
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-instruct": ModelPricing(1.50, 2.00),
    # O1 models
    "o1-preview": ModelPricing(15.00, 60.00, cached_input_price_per_million=7.50),
    "o1-mini": ModelPricing(3.00, 12.00, cached_input_price_per_million=1.50),
    "o1": ModelPricing(15.00, 60.00, cached_input_price_per_million=7.50),
}

# Anthropic Pricing (as of December 2024)
# https://www.anthropic.com/pricing
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, cached_input_price_per_million=0.30),
    "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, cached_input_price_per_million=0.30),
    "claude-3-5-sonnet-20240620": ModelPricing(3.00, 15.00),
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, cached_input_price_per_million=0.08),
    "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00, cached_input_price_per_million=0.08),
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, cached_input_price_per_million=1.50),
    "claude-3-opus-latest": ModelPricing(15.00, 75.00, cached_input_price_per_million=1.50),
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00),
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, cached_input_price_per_million=0.03),
}

# xAI Grok Pricing (as of December 2024)
# https://x.ai/api
XAI_PRICING: dict[str, ModelPricing] = {
    "grok-2-1212": ModelPricing(2.00, 10.00),
    "grok-2-vision-1212": ModelPricing(2.00, 10.00),
    "grok-beta": ModelPricing(5.00, 15.00),
    "grok-vision-beta": ModelPricing(5.00, 15.00),
}

# OpenAI Embedding Pricing (as of December 2024)
# https://openai.com/pricing
# Prices are per 1M tokens (input only, embeddings don't have output tokens)
OPENAI_EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# Combined embedding pricing lookup by provider
EMBEDDING_PRICING_BY_PROVIDER: dict[str, dict[str, float]] = {
    "openai": OPENAI_EMBEDDING_PRICING,
}

# Combined pricing lookup by provider
PRICING_BY_PROVIDER: dict[str, dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "xai": XAI_PRICING,
}


def get_model_pricing(provider: str, model_name: str) -> ModelPricing | None:
    """Get pricing for a specific model.

    Args:
        provider: Provider name (openai, anthropic, xai)
        model_name: Model name without provider prefix

    Returns:
        ModelPricing if found, None otherwise

    Example:
        >>> pricing = get_model_pricing("openai", "gpt-4o-mini")
        >>> if pricing:
        ...     print(f"Input: ${pricing.input_price_per_million}/1M tokens")
    """
    provider_pricing = PRICING_BY_PROVIDER.get(provider.lower())
    if not provider_pricing:
        return None

    # Try exact match first
    if model_name in provider_pricing:
        return provider_pricing[model_name]

    # Try partial match for versioned models
    for known_model, pricing in provider_pricing.items():
        if model_name.startswith(known_model) or known_model.startswith(model_name):
            return pricing

    return None


def calculate_cost(
    provider: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Calculate the cost of an LLM call.

    Args:
        provider: Provider name (openai, anthropic, xai)
        model_name: Model name without provider prefix
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens (for providers that support caching)

    Returns:
        Cost in USD, or None if pricing is not available

    Example:
        >>> cost = calculate_cost("openai", "gpt-4o-mini", 1000, 500)
        >>> print(f"Cost: ${cost:.6f}")
    """
    pricing = get_model_pricing(provider, model_name)
    if not pricing:
        return None

    # Calculate costs (convert from per-million to per-token)
    input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_million

    # Add cached input cost if applicable
    cached_cost = 0.0
    if cached_input_tokens > 0 and pricing.cached_input_price_per_million:
        cached_cost = (cached_input_tokens / 1_000_000) * pricing.cached_input_price_per_million

    return input_cost + output_cost + cached_cost


def list_available_models(provider: str | None = None) -> list[str]:
    """List all models with known pricing.

    Args:
        provider: Optional provider to filter by

    Returns:
        List of model names (with provider prefix)

    Example:
        >>> models = list_available_models("openai")
        >>> print(models[:3])
        ['openai:gpt-4o', 'openai:gpt-4o-mini', ...]
    """
    if provider:
        provider_pricing = PRICING_BY_PROVIDER.get(provider.lower(), {})
        return [f"{provider}:{model}" for model in provider_pricing]

    models = []
    for prov, pricing in PRICING_BY_PROVIDER.items():
        models.extend(f"{prov}:{model}" for model in pricing)
    return models


def get_embedding_pricing(provider: str, model_name: str) -> float | None:
    """Get pricing for a specific embedding model.

    Args:
        provider: Provider name (openai, anthropic, xai)
        model_name: Model name without provider prefix

    Returns:
        Price per 1M tokens in USD if found, None otherwise

    Example:
        >>> price = get_embedding_pricing("openai", "text-embedding-3-small")
        >>> if price:
        ...     print(f"Price: ${price}/1M tokens")
    """
    provider_pricing = EMBEDDING_PRICING_BY_PROVIDER.get(provider.lower())
    if not provider_pricing:
        return None

    # Try exact match first
    if model_name in provider_pricing:
        return provider_pricing[model_name]

    # Try partial match for versioned models
    for known_model, price in provider_pricing.items():
        if model_name.startswith(known_model) or known_model.startswith(model_name):
            return price

    return None


def calculate_embedding_cost(
    provider: str,
    model_name: str,
    input_tokens: int,
) -> float | None:
    """Calculate the cost of an embedding call.

    Args:
        provider: Provider name (openai, anthropic, xai)
        model_name: Model name without provider prefix
        input_tokens: Number of input tokens

    Returns:
        Cost in USD, or None if pricing is not available

    Example:
        >>> cost = calculate_embedding_cost("openai", "text-embedding-3-small", 1000)
        >>> print(f"Cost: ${cost:.6f}")
    """
    price_per_million = get_embedding_pricing(provider, model_name)
    if price_per_million is None:
        return None

    # Calculate cost (convert from per-million to per-token)
    return (input_tokens / 1_000_000) * price_per_million
