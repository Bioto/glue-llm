import logging
import os
from unittest.mock import patch

from gluellm.observability.logging_config import get_logger, setup_logging


def test_logging_propagation_disabled():
    """Test that gluellm logger has propagation disabled after setup."""
    # Reset logger state for test
    logger = logging.getLogger("gluellm")
    logger.handlers.clear()
    logger.propagate = True

    # Run setup
    setup_logging(console_output=False, force=True)

    # Verify propagation is disabled
    assert logger.propagate is False

    # Verify file handler is added
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)

    # Verify console handler is NOT added
    assert not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)

def test_logging_console_output_enabled():
    """Test that console handler is added when enabled."""
    # Reset logger state for test
    logger = logging.getLogger("gluellm")
    logger.handlers.clear()

    # Run setup with console enabled
    setup_logging(console_output=True, force=True)

    # Verify console handler is added
    assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)

def test_get_logger_auto_configures():
    """Test that get_logger auto-configures if no handlers exist."""
    # Reset logger state
    gluellm_logger = logging.getLogger("gluellm")
    gluellm_logger.handlers.clear()

    with patch("gluellm.observability.logging_config.setup_logging") as mock_setup:
        get_logger("gluellm.test")
        mock_setup.assert_called_once()

def test_get_logger_respects_disable_env():
    """Test that get_logger respects GLUELLM_DISABLE_LOGGING."""
    # Reset logger state
    gluellm_logger = logging.getLogger("gluellm")
    gluellm_logger.handlers.clear()

    with patch.dict(os.environ, {"GLUELLM_DISABLE_LOGGING": "true"}), patch(
        "gluellm.observability.logging_config.setup_logging"
    ) as mock_setup:
        get_logger("gluellm.test")
        mock_setup.assert_not_called()
