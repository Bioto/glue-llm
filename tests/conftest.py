"""Pytest configuration for GlueLLM tests."""

import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset global settings after each test to ensure test isolation."""
    yield
    # After each test, reload settings to reset to defaults
    from gluellm.config import reload_settings

    reload_settings()


@pytest.fixture(autouse=True)
def clear_global_hooks():
    """Clear global hooks before and after each test to ensure test isolation."""
    from gluellm.hooks import clear_global_hooks as clear_hooks

    # Clear before test
    clear_hooks()
    yield
    # Clear after test
    clear_hooks()
