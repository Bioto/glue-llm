"""Single source of truth for the GlueLLM version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path


def get_version() -> str:
    """Return the installed package version (or fallback to pyproject.toml)."""
    try:
        return package_version("gluellm")
    except PackageNotFoundError:
        # Fallback for running from a source checkout without installation.
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if not pyproject_path.exists():
            return "0.0.0"

        import tomllib

        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        return str(data.get("project", {}).get("version", "0.0.0"))
