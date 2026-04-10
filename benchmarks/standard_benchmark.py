"""AAAK vs Raw — benchmark for AAAK's intended use case.
=========================================================
Evaluates AAAK compression against a raw (no-compression) baseline on tasks
that match AAAK's intended domain: structured, factual context passed between
turns in an agent conversation (not prose document QA).

  gsm8k     Math reasoning; compresses 8-shot CoT context from train set.
            Context is formatted as fake user/assistant conversation turns,
            matching how AAAK would encounter few-shot examples in a real
            agent session.  Metric: exact match on final numeric answer (#### N).

  databench Real-world table QA (cardiffnlp/databench).
            Each table is serialised as a JSON array-of-objects (20-row sample),
            mirroring the format AAAK encounters from database query tool results.
            Every sample has a unique table — tests per-sample compression.
            Answer types: boolean, number, category, list[category/number].
            Metric: type-aware answer match.

AAAK is designed for agent conversation history — tool call results, config
key=value pairs, JSON API responses, schemas, and structured factual text.
It is NOT a document compressor; prose narrative tasks (NarrativeQA, SQuAD,
QuALITY) are excluded because they fall outside the intended domain.

Requires:
  pip install datasets pandas    (HuggingFace datasets + pandas for parquet loading)
  OPENAI_API_KEY (or equivalent provider key) in environment.

Run:
  set -a && source ../.env && set +a
  uv run python benchmarks/standard_benchmark.py
  uv run python benchmarks/standard_benchmark.py --tasks gsm8k --samples 20
  uv run python benchmarks/standard_benchmark.py --tasks databench --samples 20
  uv run python benchmarks/standard_benchmark.py --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

import tiktoken

from gluellm.compression.aaak import AAAKCompressor

_ENC = tiktoken.encoding_for_model("gpt-4o")

# ── tunables ──────────────────────────────────────────────────────────────────
MODEL = "openai:gpt-5.4-2026-03-05"
GSM8K_FEW_SHOT_N = 8    # CoT examples from train used as compressible context
# AAAK has a fixed overhead (preamble + markers + compression call) that only pays
# off when the context is long enough.  Below this threshold the context is passed
# through raw in AAAK mode and the sample is flagged as "passthrough".
AAAK_MIN_CONTEXT_TOKENS = 400

_benchmark_completion_extra: dict[str, Any] = {"temperature": 0, "top_p": 1}
_benchmark_semaphore: asyncio.Semaphore | None = None

AVAILABLE_TASKS = [
    "gsm8k",
    "databench",
]


# ─────────────────────────────────────────────────────────────────────────────
# Token utilities
# ─────────────────────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(count_tokens(str(m.get("content") or "")) for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────


def _extract_gsm8k_number(text: str) -> str | None:
    """Return the final numeric answer from a GSM8K-style '#### N' response."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def score_gsm8k(prediction: str, gold_answer: str) -> float:
    pred = _extract_gsm8k_number(prediction)
    gold = _extract_gsm8k_number(gold_answer)
    if pred is None or gold is None:
        return 0.0
    try:
        return float(abs(float(pred) - float(gold)) < 1e-6)
    except ValueError:
        return float(pred == gold)


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and articles — standard QA normalization."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score_databench(prediction: str, answer: str, answer_type: str) -> float:
    """Type-aware scoring for DataBench answers.

    boolean      → check if 'true'/'yes' or 'false'/'no' appears in prediction.
    number       → extract number from prediction; compare with 1e-3 tolerance.
    category     → denotation match (normalised gold appears in normalised pred).
    list[*]      → each gold item must appear in normalised prediction.
    """
    if answer is None:
        return 0.0
    pred = _normalize_text(prediction)

    if answer_type == "boolean":
        gold = answer.strip().lower()
        has_word_no = bool(re.search(r"\bno\b", pred))
        has_word_yes = bool(re.search(r"\byes\b", pred))
        if gold == "true":
            has_true = "true" in pred or has_word_yes
            has_false = "false" in pred or has_word_no
            return float(has_true and not has_false)
        if gold == "false":
            has_false = "false" in pred or has_word_no
            has_true = "true" in pred or has_word_yes
            return float(has_false and not has_true)
        return 0.0

    if answer_type == "number":
        nums = re.findall(r"-?[\d,]+\.?\d*", prediction.replace(",", ""))
        try:
            gold_f = float(answer.replace(",", ""))
            for n in nums:
                try:
                    if abs(float(n) - gold_f) < max(1e-3, abs(gold_f) * 1e-3):
                        return 1.0
                except ValueError:
                    pass
        except ValueError:
            pass
        return 0.0

    if answer_type.startswith("list"):
        try:
            import ast
            gold_items: list[str] = ast.literal_eval(answer)
        except Exception:
            gold_items = [answer]
        gold_norms = [_normalize_text(str(g)) for g in gold_items if g]
        return float(all(g in pred for g in gold_norms)) if gold_norms else 0.0

    # category (and fallback)
    gold_norm = _normalize_text(answer)
    return float(bool(gold_norm) and gold_norm in pred)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt += other.prompt
        self.completion += other.completion
        return self


@dataclass
class Sample:
    task: str
    context: str       # long context to compress (document or CoT examples)
    question: str
    answers: list[str]
    meta: dict = field(default_factory=dict)  # task-specific metadata (e.g. answer_type)


@dataclass
class SampleResult:
    mode: str
    task: str
    original_tokens: int    # tokens in raw (uncompressed) messages
    compressed_tokens: int  # tokens in messages after mode applied
    score: float            # 0.0–1.0
    compress_tok: TokenUsage = field(default_factory=TokenUsage)
    answer_tok: TokenUsage = field(default_factory=TokenUsage)

    @property
    def compression_ratio(self) -> float:
        return self.original_tokens / self.compressed_tokens if self.compressed_tokens else 1.0

    @property
    def token_reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * (self.original_tokens - self.compressed_tokens) / self.original_tokens

    @property
    def total_api_tokens(self) -> int:
        return self.compress_tok.total + self.answer_tok.total


# ─────────────────────────────────────────────────────────────────────────────
# LLM call helper
# ─────────────────────────────────────────────────────────────────────────────


async def _call(messages: list[dict], *, model: str = MODEL) -> tuple[str, TokenUsage]:
    from gluellm.api import _provider_cache

    sem = _benchmark_semaphore
    async with sem if sem is not None else asyncio.Lock():
        provider, model_id = _provider_cache.get_provider(model, api_key=None)
        resp = await provider.acompletion(model=model_id, messages=messages, **dict(_benchmark_completion_extra))
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return content, TokenUsage(
        prompt=getattr(usage, "prompt_tokens", 0) or 0,
        completion=getattr(usage, "completion_tokens", 0) or 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AAAK compression helper (wraps provider to capture token usage)
# ─────────────────────────────────────────────────────────────────────────────


async def _aaak_compress(old_messages: list[dict], *, model: str = MODEL) -> tuple[str, TokenUsage]:
    """Compress ``old_messages`` with AAAK; return (encoded_text, token_usage).

    Patches ``_provider_cache.get_provider`` like ``aaak_live_benchmark.prepare_ctx_aaak``; may break if GlueLLM internals change.
    """
    from gluellm.api import _provider_cache

    captured: list[TokenUsage] = []
    orig_get = _provider_cache.get_provider

    def tracking_get(model: str, api_key: str | None = None) -> tuple[Any, str]:
        provider, model_id = orig_get(model, api_key)

        class _Tracked:
            async def acompletion(_, **kwargs: Any) -> Any:
                resp = await provider.acompletion(**kwargs)
                u = getattr(resp, "usage", None)
                captured.append(TokenUsage(
                    getattr(u, "prompt_tokens", 0) or 0,
                    getattr(u, "completion_tokens", 0) or 0,
                ))
                return resp

        return _Tracked(), model_id

    _provider_cache.get_provider = tracking_get
    try:
        encoded = await AAAKCompressor.compress_messages(
            old_messages,
            model=model,
            completion_extra=_benchmark_completion_extra,
        )
    finally:
        _provider_cache.get_provider = orig_get

    tok = TokenUsage(sum(u.prompt for u in captured), sum(u.completion for u in captured))
    return encoded, tok


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_GSM8K_SYSTEM = (
    "You are a math tutor. Solve step by step. End your answer with '#### N' where N is "
    "the final numeric answer (digits only, no commas)."
)
_TABLE_SYSTEM = (
    "You are a data analyst. Answer questions using only the values in the provided table. "
    "Be concise and direct. For boolean questions reply with True or False. "
    "For lists, reply with a Python-style list like ['a', 'b']. For numbers, give just the number."
)


def _gsm8k_system_with_aaak() -> dict[str, str]:
    msg = {"role": "system", "content": _GSM8K_SYSTEM}
    AAAKCompressor.ensure_preamble_in_system(msg)
    return msg


def _table_system_with_aaak() -> dict[str, str]:
    msg = {"role": "system", "content": _TABLE_SYSTEM}
    AAAKCompressor.ensure_preamble_in_system(msg)
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_raw_messages(sample: Sample) -> list[dict[str, str]]:
    """Build full-context (uncompressed) messages for a sample."""
    if sample.task == "gsm8k":
        return [
            {"role": "system", "content": _GSM8K_SYSTEM},
            {"role": "user", "content": (
                f"Here are example problems with solutions:\n\n{sample.context}\n\n"
                f"Now solve this problem:\n{sample.question}"
            )},
        ]
    # databench (and any future table task)
    return [
        {"role": "system", "content": _TABLE_SYSTEM},
        {"role": "user", "content": f"Table data:\n{sample.context}\n\nQuestion: {sample.question}"},
    ]


def _gsm8k_aaak_messages(question: str, encoded: str) -> list[dict[str, str]]:
    """Build AAAK-compressed messages for a GSM8K sample given a pre-encoded context."""
    return [
        _gsm8k_system_with_aaak(),
        {"role": "user", "content": f"[AAAK CTX]\n{encoded}\n[/AAAK CTX]"},
        {"role": "user", "content": f"Now solve this problem:\n{question}"},
    ]


def _table_aaak_messages(question: str, encoded: str) -> list[dict[str, str]]:
    """Build AAAK-compressed messages for a table QA sample given a pre-encoded table context."""
    return [
        _table_system_with_aaak(),
        {"role": "user", "content": f"[AAAK CTX]\n{encoded}\n[/AAAK CTX]"},
        {"role": "user", "content": f"Question: {question}"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample scoring
# ─────────────────────────────────────────────────────────────────────────────


def _score_sample(sample: Sample, answer: str) -> float:
    """Dispatch to the right metric based on task."""
    if sample.task == "gsm8k":
        return score_gsm8k(answer, sample.answers[0])
    if sample.task == "databench":
        return score_databench(answer, sample.answers[0], sample.meta.get("answer_type", "category"))
    raise ValueError(f"Unknown task: {sample.task!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample runners
# ─────────────────────────────────────────────────────────────────────────────


async def _run_raw_sample(sample: Sample, *, model: str = MODEL) -> SampleResult:
    msgs = _build_raw_messages(sample)
    original_tokens = messages_tokens(msgs)
    answer, answer_tok = await _call(msgs, model=model)
    score = _score_sample(sample, answer)
    return SampleResult(
        mode="raw",
        task=sample.task,
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        score=score,
        answer_tok=answer_tok,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────


def _require_datasets() -> Any:
    try:
        import datasets  # type: ignore[import]
        return datasets
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for standard benchmarks.\n"
            "Install it with: pip install datasets"
        ) from exc


_gsm8k_few_shot_cache: list[dict[str, str]] | None = None


def _gsm8k_few_shot_context() -> str:
    """Build the 8-shot CoT context from the GSM8K train split."""
    global _gsm8k_few_shot_cache
    if _gsm8k_few_shot_cache is None:
        datasets = _require_datasets()
        ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
        _gsm8k_few_shot_cache = [
            {"question": row["question"], "answer": row["answer"]}
            for row in list(ds)[:GSM8K_FEW_SHOT_N]
        ]
    return "\n\n".join(
        f"Q: {ex['question']}\nA: {ex['answer']}"
        for ex in _gsm8k_few_shot_cache
    )


def load_gsm8k(n: int) -> list[Sample]:
    datasets = _require_datasets()
    few_shot_ctx = _gsm8k_few_shot_context()
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    return [
        Sample(task="gsm8k", context=few_shot_ctx, question=row["question"], answers=[row["answer"]])
        for row in list(ds)[:n]
    ]


def load_databench(n: int) -> list[Sample]:
    """cardiffnlp/databench — real-world table QA (parquet-native, no loading scripts).

    Each QA pair references a named dataset (e.g. '001_Forbes').  The 20-row
    sample table is loaded from parquet via the hf:// protocol and serialised as a
    JSON array-of-objects, mirroring a DB query tool result.  Samples whose JSON
    table falls below AAAK_MIN_CONTEXT_TOKENS are skipped so every sample actually
    exercises the compression path.
    """
    import pandas as pd

    hf_ds = _require_datasets()
    qa = hf_ds.load_dataset("cardiffnlp/databench", "qa", split="train")
    samples: list[Sample] = []
    for row in qa:
        if len(samples) >= n:
            break
        ds_id = row["dataset"]
        answer = row["answer"]
        if answer is None:
            continue
        try:
            df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{ds_id}/sample.parquet")
        except Exception as e:
            logger.warning("Skipping databench dataset %s: %s", ds_id, e)
            continue
        context = df.to_json(orient="records", indent=None)
        if count_tokens(context) < AAAK_MIN_CONTEXT_TOKENS:
            continue
        samples.append(Sample(
            task="databench",
            context=context,
            question=row["question"],
            answers=[str(answer)],
            meta={"answer_type": row["type"]},
        ))
    return samples


_TASK_LOADERS: dict[str, Any] = {
    "gsm8k": load_gsm8k,
    "databench": load_databench,
}


# ─────────────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────────────


async def run_task(task_name: str, n_samples: int, *, model: str = MODEL) -> tuple[list[SampleResult], list[SampleResult]]:
    """Run raw and aaak modes on ``n_samples`` from ``task_name``.

    Returns (raw_results, aaak_results).

    GSM8K: the few-shot context is shared across all samples, so the context is
    compressed once and reused — answer calls run concurrently.

    databench: each sample has a unique table, so compression runs per-sample
    sequentially (``_aaak_compress`` patches global provider state and cannot
    be called concurrently); answer calls follow compression immediately.
    """
    loader = _TASK_LOADERS[task_name]
    samples = loader(n_samples)

    # ── raw mode (all samples in parallel) ───────────────────────────────────
    raw_results: list[SampleResult] = list(
        await asyncio.gather(*[_run_raw_sample(s, model=model) for s in samples])
    )

    aaak_results: list[SampleResult] = []

    if task_name == "gsm8k":
        # ── GSM8K: compress shared few-shot context once, answer in parallel ─
        few_shot_messages = [
            {"role": "user", "content": "Here are example math problems with solutions:"},
            {"role": "assistant", "content": samples[0].context},
        ]
        precompressed, shared_comp_tok = await _aaak_compress(few_shot_messages, model=model)

        answer_results: list[tuple[str, TokenUsage]] = list(
            await asyncio.gather(*[
                _call(_gsm8k_aaak_messages(s.question, precompressed), model=model) for s in samples
            ])
        )
        for i, (sample, (answer, answer_tok), raw_r) in enumerate(
            zip(samples, answer_results, raw_results)
        ):
            compressed_msgs = _gsm8k_aaak_messages(sample.question, precompressed)
            # Only the first sample "pays" for the compression call; subsequent
            # samples reuse the cached encoding — reflecting real amortised cost.
            c_tok = shared_comp_tok if i == 0 else TokenUsage()
            aaak_results.append(SampleResult(
                mode="aaak",
                task=task_name,
                original_tokens=raw_r.original_tokens,
                compressed_tokens=messages_tokens(compressed_msgs),
                score=_score_sample(sample, answer),
                compress_tok=c_tok,
                answer_tok=answer_tok,
            ))

    else:
        # ── databench: each sample has a unique table — compress then answer sequentially ──
        for sample, raw_r in zip(samples, raw_results):
            table_messages = [
                {"role": "user", "content": "Query results:"},
                {"role": "assistant", "content": sample.context},
            ]
            encoded, comp_tok = await _aaak_compress(table_messages, model=model)
            compressed_msgs = _table_aaak_messages(sample.question, encoded)
            answer, answer_tok = await _call(compressed_msgs, model=model)
            aaak_results.append(SampleResult(
                mode="aaak",
                task=task_name,
                original_tokens=raw_r.original_tokens,
                compressed_tokens=messages_tokens(compressed_msgs),
                score=_score_sample(sample, answer),
                compress_tok=comp_tok,
                answer_tok=answer_tok,
            ))

    return raw_results, aaak_results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _task_report(
    task_name: str,
    raw_results: list[SampleResult],
    aaak_results: list[SampleResult],
    elapsed_s: float,
) -> None:
    print(f"\n{'═' * 110}")
    metric_label = {
        "gsm8k": "exact-match (#### N)",
        "databench": "type-aware answer match",
    }.get(task_name, "accuracy")
    print(f"  {task_name.upper()}  ({len(raw_results)} samples, {elapsed_s:.1f}s)  metric: {metric_label}")
    print(f"{'═' * 110}")

    header = (
        f"  {'mode':<8}  {'orig tok':>9}  {'cmprs tok':>10}  {'ratio':>6}  "
        f"{'tok Δ%':>8}  {'accuracy':>9}  {'compress API tok':>17}  {'answer API tok':>15}"
    )
    print(header)
    print(f"  {'':─<8}  {'':─<9}  {'':─<10}  {'':─<6}  {'':─<8}  {'':─<9}  {'':─<17}  {'':─<15}")

    n_passthrough = sum(1 for r in aaak_results if "passthrough" in r.mode)
    if n_passthrough:
        print(f"  note: {n_passthrough}/{len(aaak_results)} samples below {AAAK_MIN_CONTEXT_TOKENS}-token threshold → passthrough (no compression)")

    for results in (raw_results, aaak_results):
        mode = results[0].mode
        mean_orig = _mean([r.original_tokens for r in results])
        mean_comp = _mean([r.compressed_tokens for r in results])
        mean_ratio = _mean([r.compression_ratio for r in results])
        mean_reduc = _mean([r.token_reduction_pct for r in results])
        mean_score = _mean([r.score for r in results]) * 100
        total_comp_tok = sum(r.compress_tok.total for r in results)
        total_ans_tok = sum(r.answer_tok.total for r in results)
        print(
            f"  {mode:<8}  {mean_orig:>9.0f}  {mean_comp:>10.0f}  {mean_ratio:>6.2f}x  "
            f"{mean_reduc:>+7.1f}%  {mean_score:>8.1f}%  {total_comp_tok:>17}  {total_ans_tok:>15}"
        )

    # Per-sample score delta summary
    score_deltas = [a.score - r.score for r, a in zip(raw_results, aaak_results)]
    mean_delta = _mean(score_deltas) * 100
    n_improved = sum(1 for d in score_deltas if d > 0)
    n_same = sum(1 for d in score_deltas if d == 0)
    n_degraded = sum(1 for d in score_deltas if d < 0)
    print(
        f"\n  aaak vs raw  Δ accuracy: {mean_delta:+.1f}%  "
        f"(improved={n_improved}  same={n_same}  degraded={n_degraded})"
    )


def print_full_report(
    task_data: list[tuple[str, list[SampleResult], list[SampleResult], float]],
) -> None:
    """Print per-task reports then a cross-task summary table."""
    for task_name, raw_results, aaak_results, elapsed in task_data:
        _task_report(task_name, raw_results, aaak_results, elapsed)

    if len(task_data) < 2:
        return

    print(f"\n{'═' * 110}")
    print("  CROSS-TASK SUMMARY")
    print(f"{'═' * 110}")
    hdr = f"  {'task':<14}  {'mode':<6}  {'ratio':>6}  {'tok Δ%':>8}  {'accuracy':>9}"
    print(hdr)
    print(f"  {'':─<14}  {'':─<6}  {'':─<6}  {'':─<8}  {'':─<9}")
    for task_name, raw_results, aaak_results, _ in task_data:
        for label, results in (("raw", raw_results), ("aaak", aaak_results)):
            mean_ratio = _mean([r.compression_ratio for r in results])
            mean_reduc = _mean([r.token_reduction_pct for r in results])
            mean_score = _mean([r.score for r in results]) * 100
            print(
                f"  {task_name:<14}  {label:<6}  {mean_ratio:>6.2f}x  "
                f"{mean_reduc:>+7.1f}%  {mean_score:>8.1f}%"
            )
    print(f"\n{'═' * 110}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    global _benchmark_completion_extra, _benchmark_semaphore

    bench_model = args.model if getattr(args, "model", None) else MODEL

    if args.no_deterministic_sampling:
        _benchmark_completion_extra = {}
        print("  completion_extra={} (no deterministic sampling kwargs)")
    else:
        print(f"  completion_extra={_benchmark_completion_extra}")

    _benchmark_semaphore = asyncio.Semaphore(args.concurrency)

    task_names: list[str] = args.tasks or AVAILABLE_TASKS
    for t in task_names:
        if t not in AVAILABLE_TASKS:
            print(f"ERROR: Unknown task {t!r}. Available: {AVAILABLE_TASKS}")
            return
    print(f"  tasks={task_names}  samples_per_task={args.samples}  model={bench_model}")
    print(f"  gsm8k_few_shot_n={GSM8K_FEW_SHOT_N}  aaak_min_context_tokens={AAAK_MIN_CONTEXT_TOKENS}\n")

    task_data: list[tuple[str, list[SampleResult], list[SampleResult], float]] = []

    for task_name in task_names:
        print(f"  Loading '{task_name}'...", flush=True)
        t0 = time.perf_counter()
        raw_results, aaak_results = await run_task(task_name, args.samples, model=bench_model)
        elapsed = time.perf_counter() - t0

        raw_acc = _mean([r.score for r in raw_results]) * 100
        aaak_acc = _mean([r.score for r in aaak_results]) * 100
        mean_cr = _mean([r.compression_ratio for r in aaak_results])
        print(
            f"    done  raw={raw_acc:.1f}%  aaak={aaak_acc:.1f}%  "
            f"compression_ratio={mean_cr:.2f}x  ({elapsed:.1f}s)"
        )
        task_data.append((task_name, raw_results, aaak_results, elapsed))

    print_full_report(task_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AAAK vs Raw benchmark on structured-data tasks (real API calls)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help=f"Tasks to run (default: all). Choices: {AVAILABLE_TASKS}",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per task (default: 50).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max simultaneous in-flight API calls (default: 10).",
    )
    parser.add_argument(
        "--no-deterministic-sampling",
        action="store_true",
        help="Omit temperature/top_p kwargs (use if the model rejects them).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help=f"LLM model id for compression and answering (default: {MODEL}).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
