"""Pytest configuration for GlueLLM tests."""

import inspect

import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset global settings after each test to ensure test isolation."""
    yield
    # After each test, reload settings to reset to defaults
    import gluellm.api as api
    from gluellm.config import reload_settings

    # reload_settings() replaces config.settings with a new instance; re-bind
    # gluellm.api.settings so modules that imported settings at load time stay in sync.
    api.settings = reload_settings()


@pytest.fixture(autouse=True)
def clear_global_hooks():
    """Clear global hooks before and after each test to ensure test isolation."""
    from gluellm.hooks import clear_global_hooks as clear_hooks

    # Clear before test
    clear_hooks()
    yield
    # Clear after test
    clear_hooks()


@pytest.fixture(autouse=True)
def clear_global_eval_store():
    """Clear global eval store before and after each test to ensure test isolation."""
    from gluellm.eval import set_global_eval_store

    # Clear before test
    set_global_eval_store(None)
    yield
    # Clear after test
    set_global_eval_store(None)


@pytest_asyncio.fixture(autouse=True)
async def close_cached_provider_clients():
    """Close GlueLLM cached provider clients after each test."""
    yield

    from gluellm.api import close_providers

    await close_providers()


# Cache the real AnyLLM.create at first use so we never use a patched reference
# (avoids "MagicMock can't be used in 'await' expression" when tests run in parallel).
_original_any_llm_create = None


def _get_any_llm_original_create():
    global _original_any_llm_create
    if _original_any_llm_create is None:
        from any_llm.any_llm import AnyLLM
        _original_any_llm_create = AnyLLM.create.__func__
    return _original_any_llm_create


@pytest_asyncio.fixture(autouse=True)
async def close_any_llm_created_clients(monkeypatch):
    """Track and close any AnyLLM providers created during a test.

    Some tests call `any_llm.completion()` directly, which instantiates provider
    clients outside GlueLLM's cache. If those AsyncOpenAI clients are not
    explicitly closed, pytest may emit `ResourceWarning: unclosed transport`.
    """
    from any_llm.any_llm import AnyLLM

    created_providers = []
    original_create = _get_any_llm_original_create()

    def create_with_tracking(cls, *args, **kwargs):
        provider = original_create(cls, *args, **kwargs)
        created_providers.append(provider)
        return provider

    monkeypatch.setattr(AnyLLM, "create", classmethod(create_with_tracking))
    yield

    seen_provider_ids = set()
    for provider in created_providers:
        provider_id = id(provider)
        if provider_id in seen_provider_ids:
            continue
        seen_provider_ids.add(provider_id)

        client = getattr(provider, "client", None)
        if client is None:
            continue

        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await result
            continue

        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
