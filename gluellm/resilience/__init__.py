"""Resilience utilities for GlueLLM (model fallback chains, etc.)."""

from gluellm.resilience.fallback import (
    DEFAULT_FALLBACK_ON,
    ModelFallbackConfig,
    call_with_model_fallback,
    resolve_fallback_chain,
)

__all__ = [
    "DEFAULT_FALLBACK_ON",
    "ModelFallbackConfig",
    "call_with_model_fallback",
    "resolve_fallback_chain",
]
