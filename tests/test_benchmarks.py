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


def test_aaak_live_api_key_env_for_model_uses_provider_specific_env() -> None:
    assert aaak_live_benchmark.api_key_env_for_model("openai:gpt-5.4-nano") == "OPENAI_API_KEY"
    assert aaak_live_benchmark.api_key_env_for_model("groq:qwen/qwen3-32b") == "GROQ_API_KEY"
    assert aaak_live_benchmark.api_key_env_for_model("mystery:model-x") == "MYSTERY_API_KEY"


@pytest.mark.asyncio
async def test_eval_one_question_skips_judge_when_deterministic_match(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _fake_call(messages: list[dict], *, model: str = aaak_live_benchmark.MODEL) -> tuple[str, aaak_live_benchmark.TokenUsage]:
        calls.append(model)
        if model == aaak_live_benchmark.JUDGE_MODEL:
            raise AssertionError("judge should not be called when deterministic check succeeds")
        return "JWT access tokens expire in 15 minutes.", aaak_live_benchmark.TokenUsage(prompt=3, completion=2)

    monkeypatch.setattr(aaak_live_benchmark, "_call", _fake_call)

    ok, tok, row = await aaak_live_benchmark._eval_one_question(
        [{"role": "system", "content": "You are helpful."}],
        {"question": "What is the expiry?", "key_facts": ["15 minutes"]},
        collect_debug=True,
    )

    assert ok == 1
    assert tok == aaak_live_benchmark.TokenUsage(prompt=3, completion=2)
    assert calls == [aaak_live_benchmark.MODEL]
    assert row is not None
    assert row["deterministic_ok"] is True
    assert row["judge_ok"] is None
    assert row["verdict_raw"] is None
    assert row["judge_user"] is None


@pytest.mark.asyncio
async def test_standard_benchmark_main_async_restores_deterministic_completion_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_task(task_name: str, samples: int, *, model: str = standard_benchmark.MODEL):
        return [], []

    monkeypatch.setattr(standard_benchmark, "run_task", _fake_run_task)
    monkeypatch.setattr(standard_benchmark, "print_full_report", lambda task_data: None)

    await standard_benchmark.main_async(argparse.Namespace(
        tasks=["gsm8k"],
        samples=1,
        concurrency=1,
        no_deterministic_sampling=True,
        model=None,
    ))
    assert standard_benchmark._benchmark_completion_extra == {}

    await standard_benchmark.main_async(argparse.Namespace(
        tasks=["gsm8k"],
        samples=1,
        concurrency=1,
        no_deterministic_sampling=False,
        model=None,
    ))
    assert standard_benchmark._benchmark_completion_extra == {"temperature": 0, "top_p": 1}


@pytest.mark.asyncio
async def test_run_task_raises_clear_error_when_gsm8k_samples_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(standard_benchmark._TASK_LOADERS, "gsm8k", lambda n: [])

    with pytest.raises(ValueError, match="gsm8k samples empty"):
        await standard_benchmark.run_task("gsm8k", n_samples=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("samples", "concurrency", "expected_error"),
    [
        (0, 1, "ERROR: --samples must be > 0"),
        (1, 0, "ERROR: --concurrency must be > 0"),
    ],
)
async def test_standard_benchmark_main_async_validates_positive_args_before_dispatch(
    samples: int,
    concurrency: int,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    async def _fake_run_task(task_name: str, sample_count: int, *, model: str = standard_benchmark.MODEL):
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(standard_benchmark, "run_task", _fake_run_task)
    monkeypatch.setattr(standard_benchmark, "print_full_report", lambda task_data: None)

    await standard_benchmark.main_async(argparse.Namespace(
        tasks=["gsm8k"],
        samples=samples,
        concurrency=concurrency,
        no_deterministic_sampling=False,
        model=None,
    ))

    out = capsys.readouterr().out
    assert expected_error in out
    assert called is False
