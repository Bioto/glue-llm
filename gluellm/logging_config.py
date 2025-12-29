"""Logging configuration for GlueLLM.

This module provides production-grade logging setup with:
- Colored console output using colorlog
- File logging with rotation
- Optional JSON structured logging
- Environment variable configuration support
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
from pythonjsonlogger import json


def setup_logging(
    log_level: str = "INFO",
    log_file_level: str = "DEBUG",
    log_dir: Path | str | None = None,
    log_file_name: str = "gluellm.log",
    log_json_format: bool = False,
    log_max_bytes: int = 10485760,  # 10MB
    log_backup_count: int = 5,
    force: bool = False,
) -> None:
    """Configure production-grade logging for GlueLLM.

    Sets up:
    - Colored console handler (INFO level by default)
    - Rotating file handler (DEBUG level by default)
    - Optional JSON formatter for structured logging

    Args:
        log_level: Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file_level: File log level (typically DEBUG for full details)
        log_dir: Directory for log files (defaults to 'logs' in project root)
        log_file_name: Name of the log file
        log_json_format: Enable JSON structured logging format
        log_max_bytes: Maximum size of log file before rotation (default: 10MB)
        log_backup_count: Number of backup log files to keep (default: 5)
        force: Force reconfiguration even if logging is already configured

    Environment Variables:
        GLUELLM_LOG_LEVEL: Override console log level
        GLUELLM_LOG_FILE_LEVEL: Override file log level
        GLUELLM_LOG_DIR: Override log directory
        GLUELLM_LOG_FILE_NAME: Override log file name
        GLUELLM_LOG_JSON_FORMAT: Enable JSON logging (set to 'true' or '1')
        GLUELLM_LOG_MAX_BYTES: Override max file size
        GLUELLM_LOG_BACKUP_COUNT: Override backup count
    """
    # Check if logging is already configured
    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        # Logging already configured, skip
        return

    # Get environment variable overrides
    log_level = os.getenv("GLUELLM_LOG_LEVEL", log_level).upper()
    log_file_level = os.getenv("GLUELLM_LOG_FILE_LEVEL", log_file_level).upper()
    log_dir = os.getenv("GLUELLM_LOG_DIR", log_dir)
    log_file_name = os.getenv("GLUELLM_LOG_FILE_NAME", log_file_name)
    log_json_format = os.getenv("GLUELLM_LOG_JSON_FORMAT", str(log_json_format)).lower() in ("true", "1", "yes")
    try:
        log_max_bytes = int(os.getenv("GLUELLM_LOG_MAX_BYTES", log_max_bytes))
    except ValueError:
        log_max_bytes = 10485760
    try:
        log_backup_count = int(os.getenv("GLUELLM_LOG_BACKUP_COUNT", log_backup_count))
    except ValueError:
        log_backup_count = 5

    # Determine log directory
    if log_dir is None:
        # Default to 'logs' directory in project root
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Convert log levels to logging constants
    numeric_level = getattr(logging, log_level, logging.INFO)
    numeric_file_level = getattr(logging, log_file_level, logging.DEBUG)

    # Configure root logger
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Remove existing handlers if forcing reconfiguration
    if force:
        root_logger.handlers.clear()

    # Console handler with colorlog
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Color scheme for different log levels
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    console_handler.setFormatter(color_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    log_file_path = log_dir / log_file_name
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_file_level)

    # Choose formatter based on JSON setting
    if log_json_format:
        # JSON structured logging
        json_formatter = json.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(json_formatter)
    else:
        # Standard text formatter
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

    root_logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: console={log_level}, file={log_file_level}, "
        f"file_path={log_file_path}, json_format={log_json_format}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    This is a convenience function that ensures logging is configured
    before returning a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure logging is configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()

    return logging.getLogger(name)
