# Development Guide

This document provides information for developers contributing to GlueLLM.

## Setup

### Installing Dependencies

GlueLLM uses `uv` for package management:

```bash
# Install all dependencies including dev dependencies
uv sync

# Install only production dependencies
uv sync --no-dev
```

### Pre-commit Hooks

Pre-commit hooks are configured to automatically check and format code before commits:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files manually
uv run pre-commit run --all-files

# Update pre-commit hook versions
uv run pre-commit autoupdate
```

## Code Quality Tools

### Ruff

GlueLLM uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

#### Linting

```bash
# Check all source files
uv run ruff check source/

# Auto-fix issues where possible
uv run ruff check source/ --fix

# Show unsafe fixes
uv run ruff check source/ --unsafe-fixes
```

#### Formatting

```bash
# Format all source files
uv run ruff format source/

# Check formatting without making changes
uv run ruff format source/ --check

# Format specific files
uv run ruff format source/api.py source/config.py
```

### Configuration

Ruff is configured in `pyproject.toml`:

- **Line Length**: 120 characters
- **Target Python**: 3.12+
- **Selected Rules**: Includes pycodestyle, pyflakes, isort, pyupgrade, bugbear, and more
- **Ignored Rules**: See `pyproject.toml` for specific exclusions

#### Per-File Ignores

- `__init__.py`: Allows unused imports (F401)
- Test files: Relaxed rules for test-specific patterns
- CLI/Examples: Allows print statements

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_api.py

# Run specific test
uv run pytest tests/test_api.py::test_complete_basic

# Skip integration tests
uv run pytest -m "not integration"
```

### Using the CLI Test Runner

```bash
# Run all tests via CLI
uv run python -m source.cli run-tests

# Run specific test class
uv run python -m source.cli run-tests -c TestBasicToolCalling

# Run with verbose output
uv run python -m source.cli run-tests -v

# Skip integration tests
uv run python -m source.cli run-tests --no-integration
```

## Documentation

### Docstring Style

GlueLLM uses Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    More detailed description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this is raised
        TypeError: When and why this is raised

    Example:
        >>> example_function("test", 42)
        True
    """
    pass
```

### Building Documentation

Documentation can be generated using tools like:
- `pdoc`: `uv run pdoc source/`
- `sphinx`: Configure with autodoc extension

## Code Style Guidelines

### Imports

- Group imports using isort (handled by Ruff)
- Order: stdlib, third-party, first-party
- Use absolute imports for source modules

### Type Hints

- Use type hints for all function parameters and return values
- Use `Optional[T]` for nullable values
- Use `list[T]`, `dict[K, V]` (Python 3.9+ style)
- Use `Type[T]` for class types

### Naming Conventions

- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

### Error Handling

- Use custom exception classes from `source.api`
- Always provide descriptive error messages
- Log errors with appropriate levels

## Pre-commit Hook Details

The pre-commit configuration includes:

1. **Standard Hooks**:
   - Trailing whitespace removal
   - End-of-file fixer
   - YAML/TOML validation
   - Large file detection
   - Private key detection

2. **Ruff Hooks**:
   - Linting with auto-fix
   - Code formatting

These run automatically on `git commit` and can be bypassed with `--no-verify` if absolutely necessary (not recommended).

## Development Workflow

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and test**:
   ```bash
   # Make your changes
   uv run pytest
   uv run ruff check source/ --fix
   uv run ruff format source/
   ```

3. **Commit** (pre-commit hooks run automatically):
   ```bash
   git add .
   git commit -m "Add feature X"
   ```

4. **Push and create PR**:
   ```bash
   git push origin feature/my-feature
   ```

## Common Commands

```bash
# Full quality check
uv run ruff check source/ --fix && uv run ruff format source/ && uv run pytest

# Quick format and lint
uv run ruff check source/ --fix && uv run ruff format source/

# Run demos
uv run python -m source.cli demo

# Run examples
uv run python examples/basic_usage.py
```

## Troubleshooting

### Pre-commit Failures

If pre-commit fails:
1. Review the error messages
2. Fix the issues or let auto-fix handle them
3. Re-stage the files: `git add -u`
4. Commit again

### Ruff Errors

For unfixable ruff errors:
1. Read the error message and rule documentation
2. Fix manually or add `# noqa: RULE_CODE` if justified
3. Consider if the rule should be disabled project-wide in `pyproject.toml`

### Test Failures

1. Run the specific failing test with `-v` flag
2. Check logs and error messages
3. Ensure all dependencies are up to date: `uv sync`
4. Clear caches: `rm -rf .pytest_cache __pycache__`

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [uv Documentation](https://github.com/astral-sh/uv)
