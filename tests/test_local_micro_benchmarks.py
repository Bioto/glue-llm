"""CPU-only micro-benchmarks (no API keys). Runs in default pytest (not ``benchmark`` marker)."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_benchmark_module(name: str, filename: str):
    path = Path(__file__).resolve().parent.parent / "benchmarks" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aaak_local_micro_benchmark = _load_benchmark_module(
    "aaak_local_micro_benchmark",
    "aaak_local_micro_benchmark.py",
)
workflow_benchmark = _load_benchmark_module(
    "workflow_benchmark",
    "workflow_benchmark.py",
)

pytestmark = pytest.mark.asyncio


async def test_aaak_local_micro_benchmark() -> None:
    args = argparse.Namespace(iterations=15, turns=20)
    aaak_local_micro_benchmark.main_sync(args)


async def test_workflow_benchmark() -> None:
    args = argparse.Namespace(iterations=25)
    await workflow_benchmark.main_async(args)
