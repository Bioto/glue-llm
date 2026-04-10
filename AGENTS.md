# GlueLLM Agent Instructions

## Cursor Cloud specific instructions

GlueLLM is a Python SDK/library (not a web app). There is no server to start — the "application" is the importable package and its CLI.

### Quick reference

| Task | Command |
|---|---|
| Install deps | `uv sync --all-extras --dev` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Unit tests | `uv run pytest tests/ -m "not integration"` |
| Build | `uv build` |
| CLI | `uv run gluellm --help` |

### Caveats

- **Python 3.12+** is required (`.python-version` specifies 3.12).
- **`uv`** is the package manager. The lockfile is `uv.lock`. Use `uv sync --all-extras --dev` to install, and prefix commands with `uv run`.
- **Unit tests are fully mocked** — no API keys are needed. Integration tests (marked `@pytest.mark.integration`) require `OPENAI_API_KEY`.
- **Rate limiting tests** are ignored by default via `pytest.ini` (`--ignore=tests/test_rate_limiting.py`).
- Tests run in parallel via `pytest-xdist` (`-n auto` in `pytest.ini`).
- The `gluellm config` CLI command has a pre-existing bug (`rate_limit_requests_per_second` vs `rate_limit_requests_per_minute` attribute mismatch). Don't be alarmed by a traceback there.
- `ruff format --check .` reports pre-existing unformatted files in the repo. `ruff check .` (linting) passes clean.
