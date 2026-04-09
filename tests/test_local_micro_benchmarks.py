"""CPU-only micro-benchmarks (no API keys). Runs in default pytest (not ``benchmark`` marker)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import aaak_local_micro_benchmark  # noqa: E402
import workflow_benchmark  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_aaak_local_micro_benchmark() -> None:
    args = argparse.Namespace(iterations=15, turns=20)
    await aaak_local_micro_benchmark.main_async(args)


async def test_workflow_benchmark() -> None:
    args = argparse.Namespace(iterations=25)
    await workflow_benchmark.main_async(args)
