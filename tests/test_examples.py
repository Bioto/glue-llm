"""Integration tests that run example scripts as-is.

Each example is executed in a subprocess so event loops, sys.exit() calls, and
stdin behaviour are fully isolated from the test runner.

Skipped when OPENAI_API_KEY is not set (or is a placeholder).
Run with:  uv run pytest tests/test_examples.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _openai_key_ok() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key and key != "sk-test")


def _example_files():
    for f in sorted(_EXAMPLES_DIR.glob("*.py")):
        yield f.name


@pytest.mark.integration
@pytest.mark.parametrize("example_name", list(_example_files()))
def test_example_runs(example_name):
    """Run each example script as a subprocess; skips when no API key."""
    if not _openai_key_ok():
        pytest.skip("OPENAI_API_KEY not set; examples require live API access")

    script_path = _EXAMPLES_DIR / example_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"{example_name} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
