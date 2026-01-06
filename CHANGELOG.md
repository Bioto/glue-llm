# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core API with automatic tool execution
- Multi-agent workflows (iterative, pipeline, debate)
- Comprehensive error handling with retry logic
- CLI interface for testing and demos
- New `console_output` parameter for `setup_logging()` to control console handler
- New `log_console_output` configuration setting and `GLUELLM_LOG_CONSOLE_OUTPUT` environment variable
- **New `GLUELLM_DISABLE_LOGGING` environment variable** to completely disable GlueLLM's logging setup for full application control
- **Tool support for structured completions**: `structured_complete()` and `GlueLLM.structured_complete()` now accept `tools`, `execute_tools`, and `max_tool_iterations` parameters, allowing the LLM to call tools to gather information before returning structured output

### Changed
- **BREAKING**: Console logging is now disabled by default in `setup_logging()` to avoid conflicts when GlueLLM is used as a library. Set `console_output=True` or `GLUELLM_LOG_CONSOLE_OUTPUT=true` to enable console output.
- Logging now respects parent application's logging configuration by default
- `get_logger()` now checks `GLUELLM_DISABLE_LOGGING` before auto-configuring logging

### Added
- Initial release of GlueLLM
- Support for multiple LLM providers via any-llm-sdk
- Automatic tool execution loop
- Structured output with Pydantic models
- Multi-turn conversation management
- Three workflow patterns: iterative refinement, pipeline, and debate
- Comprehensive CLI with demo commands
- Error classification and automatic retry with exponential backoff
- Configuration management with pydantic-settings
