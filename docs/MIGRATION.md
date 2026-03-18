# GlueLLM Migration Guide

Guidance for upgrading between GlueLLM versions.

## Checking Versions

```bash
gluellm --version
# or
python -c "import gluellm; print(gluellm.__version__)"
```

## Breaking Changes

When a release introduces breaking changes, they will be documented in [CHANGELOG.md](../CHANGELOG.md) with migration instructions.

### Common Migration Patterns

#### API Parameter Renames

If a parameter is renamed, update your call sites:

```python
# Old
result = await complete("Hi", old_param=value)

# New
result = await complete("Hi", new_param=value)
```

#### Deprecated Functions

Deprecated functions will emit warnings. Replace with the recommended alternative before removal.

#### Config Key Changes

If `GlueLLMSettings` fields change:

```python
# Update .env or environment variables
# OLD: GLUELLM_OLD_SETTING
# NEW: GLUELLM_NEW_SETTING
```

#### Import Path Changes

If modules are reorganized:

```python
# Old
from gluellm.old_module import Foo

# New
from gluellm.new_module import Foo
```

## Dependency Updates

After upgrading GlueLLM, check `pyproject.toml` and run:

```bash
uv sync
# or
pip install -e ".[dev]"
```

## Testing After Migration

Run the test suite:

```bash
cd gluellm
uv run pytest tests/ -v
```

Use the CLI to verify key paths:

```bash
gluellm test-completion
gluellm test-streaming
gluellm test-iterative-workflow
```

## See Also

- [CHANGELOG.md](../CHANGELOG.md) - Version history and changes
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
