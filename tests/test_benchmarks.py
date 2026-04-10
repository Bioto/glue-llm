"""Pytest wrappers for live benchmarks (appear in test / CI history)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

# benchmarks/ is not a package — add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import aaak_live_benchmark  # noqa: E402
import glue_orchestration_benchmark  # noqa: E402
import standard_benchmark  # noqa: E402
import streaming_benchmark  # noqa: E402

pytestmark = pytest.mark.benchmark


def _skip_if_no_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Missing OPENAI_API_KEY")


def _skip_if_aaak_live_keys_missing() -> None:
    """Match ``aaak_live_benchmark`` compression + judge model API key requirements."""
    errs = aaak_live_benchmark.missing_benchmark_key_errors(
        aaak_live_benchmark.MODEL,
        aaak_live_benchmark.JUDGE_MODEL,
    )
    if errs:
        pytest.skip("; ".join(errs))


def _standard_samples() -> int:
    explicit = os.environ.get("BENCHMARK_STANDARD_SAMPLES")
    if explicit:
        return int(explicit)
    if os.environ.get("BENCHMARK_FULL"):
        return 50
    return 5


@pytest.mark.asyncio
async def test_aaak_live_benchmark() -> None:
    _skip_if_aaak_live_keys_missing()
    args = argparse.Namespace(
        trials=1,
        only_section_a=False,
        only_section_b=False,
        only_section_c=False,
        only_section_d=False,
        only_section_e=False,
        verbose_section_a=False,
        no_deterministic_sampling=False,
        concurrency=10,
        judge_model=None,
    )
    await aaak_live_benchmark.main_async(args)


@pytest.mark.asyncio
async def test_standard_benchmark() -> None:
    _skip_if_no_key()
    args = argparse.Namespace(
        tasks=None,
        samples=_standard_samples(),
        concurrency=10,
        no_deterministic_sampling=False,
        model=None,
    )
    await standard_benchmark.main_async(args)


@pytest.mark.asyncio
async def test_glue_orchestration_benchmark() -> None:
    _skip_if_no_key()
    args = argparse.Namespace(
        model=None,
        skip_structured=True,
        no_deterministic_sampling=False,
    )
    await glue_orchestration_benchmark.main_async(args)


@pytest.mark.asyncio
async def test_streaming_benchmark() -> None:
    _skip_if_no_key()
    args = argparse.Namespace(
        model=None,
        prompt="Reply with exactly the word: STREAM-OK",
        no_deterministic_sampling=False,
    )
    await streaming_benchmark.main_async(args)
