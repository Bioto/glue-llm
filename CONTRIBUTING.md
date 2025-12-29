# Contributing to GlueLLM

Thank you for your interest in contributing to GlueLLM! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Set up your development environment**:
   ```bash
   uv pip install -e ".[dev]"
   uv run pre-commit install
   ```

3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

- Follow the existing code style (enforced by `ruff`)
- Run the formatter before committing:
  ```bash
  uv run ruff format .
  uv run ruff check .
  ```

### Testing

- Add tests for new features
- Ensure all tests pass:
  ```bash
  uv run pytest tests/
  ```
- For integration tests that require API keys, mark them with `@pytest.mark.integration`
- Skip integration tests when running locally:
  ```bash
  uv run pytest -m "not integration"
  ```

### Documentation

- Update the README.md if adding new features
- Add docstrings to all new functions and classes
- Update examples if adding new functionality
- Follow the existing documentation style

## Pull Request Process

1. **Before submitting**:
   - Ensure all tests pass
   - Run linters and formatters
   - Update documentation as needed
   - Add yourself to contributors (if applicable)

2. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Describe what changes you made and why
   - Reference any related issues
   - Include examples if adding new features

3. **Code Review**:
   - Address review comments promptly
   - Keep discussions focused and constructive
   - Be open to feedback and suggestions

## Issue Reporting

When reporting issues, please include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: Detailed steps to reproduce the behavior
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: Python version, OS, GlueLLM version
- **Error Messages**: Full error traceback if applicable

## Feature Requests

For feature requests, please include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions you've considered

## Commit Messages

Use conventional commit messages:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

Example:
```
feat: add streaming support for real-time responses
```

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Check existing issues and PRs
- Review the README.md for usage examples

Thank you for contributing to GlueLLM! 🎉
