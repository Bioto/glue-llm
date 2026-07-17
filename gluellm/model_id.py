"""Model identifier helpers for provider routing vs wire format.

GlueLLM supports OpenAI-compatible gateways such as `Otari` — an OpenAI-compatible
LLM gateway built on `any-llm` (including the any-llm server). When
``OPENAI_BASE_URL`` points at a gateway host (anything other than
``api.openai.com``), all requests use the ``openai`` any_llm client and model
ids are sent on the wire unchanged (e.g. ``anthropic:claude-sonnet-4``). The
gateway fans out to upstream providers using its own credentials; the app only
needs ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

_OPENAI_DEFAULT_HOST = "api.openai.com"
_OPENAI_API_BASE_ENV = "OPENAI_BASE_URL"


def _normalize_api_base_host(api_base: str) -> str | None:
    """Return the lowercase hostname from an API base URL, or None if unset/invalid."""
    api_base = api_base.strip()
    if not api_base:
        return None
    if "://" not in api_base:
        api_base = f"//{api_base}"
    host = urlparse(api_base).hostname
    return host.lower() if host else None


def openai_api_base_is_gateway(api_base: str | None = None) -> bool:
    """Return True when the OpenAI provider points at a non-OpenAI gateway.

    Gateways such as Otari (an OpenAI-compatible gateway built on any-llm server)
    are detected when ``OPENAI_BASE_URL`` resolves to a host other than
    ``api.openai.com``. In that mode GlueLLM routes every model through the
    openai client and passes ``provider:model`` ids through unchanged.

    Reads ``OPENAI_BASE_URL`` when *api_base* is not provided. Unset or empty
    values are treated as the default OpenAI host (not a gateway).
    """
    resolved = api_base if api_base is not None else os.environ.get(_OPENAI_API_BASE_ENV, "")
    host = _normalize_api_base_host(resolved or "")
    if host is None:
        return False
    return host != _OPENAI_DEFAULT_HOST


def wire_model_for_provider(
    original_model: str,
    provider_name: str,
    model_id: str,
    *,
    api_base: str | None = None,
) -> str:
    """Return the model string to send on the wire for the given provider.

    Gateways such as Otari require ``provider:model`` (or ``provider/model``) on
    the wire. The default OpenAI API expects only the bare model id.
    """
    if provider_name.lower() == "openai" and openai_api_base_is_gateway(api_base):
        return original_model
    return model_id


def ensure_gateway_wire_model(
    model: str,
    provider_name: str = "openai",
    *,
    api_base: str | None = None,
) -> str:
    """If *api_base* is a gateway and *model* is bare, restore ``provider:model``.

    Idempotent when *model* already contains ``:`` or ``/``.
    """
    if not model or ":" in model or "/" in model:
        return model
    if not openai_api_base_is_gateway(api_base):
        return model
    return f"{provider_name}:{model}"
