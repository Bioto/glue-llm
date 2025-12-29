"""Pytest configuration for GlueLLM tests."""

import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset global settings after each test to ensure test isolation."""
    yield
    # After each test, reload settings to reset to defaults
    from gluellm.config import reload_settings

    reload_settings()
