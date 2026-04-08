"""AAAK vs Raw — standard NLP dataset benchmark.
=================================================
Evaluates AAAK compression against a raw (no-compression) baseline on the
publicly-available datasets used by LLMLingua, LLMLingua-2, and Cmprsr:

  gsm8k          Math reasoning; compresses 8-shot CoT context from train set.
                 Metric: exact match on final numeric answer (#### N).

  narrativeqa    Long-document single-hop QA (deepmind/narrativeqa).
  hotpotqa       Multi-hop multi-document QA (hotpotqa/hotpot_qa).
  squad          Standard reading comprehension (rajpurkar/squad).
                 Metric for all QA tasks: max F1 over gold answers.

For every task two modes are measured:
  raw     Full context, no compression (baseline).
  aaak    Context replaced by an AAAK-compressed block.

Documents are truncated to MAX_DOC_TOKENS (2 000 by default) before
compression — matching the "2 000-token constraint" used in LLMLingua-2 §4.

Requires:
  pip install datasets          (HuggingFace datasets library)
  OPENAI_API_KEY (or equivalent provider key) in environment.

Run:
  set -a && source ../.env && set +a
  uv run python benchmarks/standard_benchmark.py
  uv run python benchmarks/standard_benchmark.py --tasks gsm8k --samples 20
  uv run python benchmarks/standard_benchmark.py --tasks narrativeqa hotpotqa
  uv run python benchmarks/standard_benchmark.py --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from gluellm.compression.aaak import AAAK_PREAMBLE_MARKER, AAAKCompressor

_ENC = tiktoken.encoding_for_model("gpt-4o")

# ── tunables ──────────────────────────────────────────────────────────────────
MODEL = "openai:gpt-5.4-2026-03-05"
MAX_DOC_TOKENS = 2_000   # document budget, matches LLMLingua-2 §4 constraint
GSM8K_FEW_SHOT_N = 8    # CoT examples from train used as compressible context
# AAAK has a fixed overhead (preamble + markers + compression call) that only pays
# off when the context is long enough.  Below this threshold the context is passed
# through raw in AAAK mode and the sample is flagged as "passthrough".
AAAK_MIN_CONTEXT_TOKENS = 400

_benchmark_completion_extra: dict[str, Any] = {"temperature": 0, "top_p": 1}
_benchmark_semaphore: asyncio.Semaphore | None = None

AVAILABLE_TASKS = ["gsm8k", "narrativeqa", "hotpotqa", "squad"]


# ─────────────────────────────────────────────────────────────────────────────
# Token utilities
# ─────────────────────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENC.decode(tokens[:max_tokens])


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(count_tokens(str(m.get("content") or "")) for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and articles — standard QA normalization."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_f1(prediction: str, gold: str) -> float:
    pred_toks = _normalize_text(prediction).split()
    gold_toks = _normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction: str, answers: list[str]) -> float:
    return max((compute_f1(prediction, a) for a in answers), default=0.0)


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


async def _aaak_compress(old_messages: list[dict]) -> tuple[str, TokenUsage]:
    """Compress ``old_messages`` with AAAK; return (encoded_text, token_usage)."""
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
            model=MODEL,
            completion_extra=_benchmark_completion_extra,
        )
    finally:
        _provider_cache.get_provider = orig_get

    tok = TokenUsage(sum(u.prompt for u in captured), sum(u.completion for u in captured))
    return encoded, tok


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Give a concise factual answer."
)
_GSM8K_SYSTEM = (
    "You are a math tutor. Solve step by step. End your answer with '#### N' where N is "
    "the final numeric answer (digits only, no commas)."
)


def _qa_system_with_aaak() -> dict[str, str]:
    msg = {"role": "system", "content": _QA_SYSTEM}
    AAAKCompressor.ensure_preamble_in_system(msg)
    return msg


def _gsm8k_system_with_aaak() -> dict[str, str]:
    msg = {"role": "system", "content": _GSM8K_SYSTEM}
    AAAKCompressor.ensure_preamble_in_system(msg)
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_raw_messages(sample: Sample) -> list[dict[str, str]]:
    """Build full-context (uncompressed) messages for a sample."""
    system_content = _GSM8K_SYSTEM if sample.task == "gsm8k" else _QA_SYSTEM
    if sample.task == "gsm8k":
        user_content = (
            f"Here are example problems with solutions:\n\n{sample.context}\n\n"
            f"Now solve this problem:\n{sample.question}"
        )
    else:
        user_content = f"Context:\n{sample.context}\n\nQuestion: {sample.question}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def _build_aaak_messages(
    sample: Sample,
    precompressed: str | None = None,
) -> tuple[list[dict[str, str]], TokenUsage, bool]:
    """Build AAAK-compressed messages for a sample.

    Returns (messages, compression_token_usage, was_passthrough).

    If the context is shorter than AAAK_MIN_CONTEXT_TOKENS the compression is
    skipped and the raw context is used instead (passthrough=True).  This avoids
    the fixed AAAK overhead dominating very short documents (e.g. SQuAD paragraphs).

    If ``precompressed`` is provided (GSM8K reuses a single compressed context),
    the compression LLM call is skipped and zero compression tokens are charged.
    """
    system_msg = _gsm8k_system_with_aaak() if sample.task == "gsm8k" else _qa_system_with_aaak()
    question_text = (
        f"Now solve this problem:\n{sample.question}"
        if sample.task == "gsm8k"
        else f"Question: {sample.question}"
    )

    ctx_tokens = count_tokens(sample.context)
    if precompressed is None and ctx_tokens < AAAK_MIN_CONTEXT_TOKENS:
        # Passthrough: context is too short to benefit from compression.
        raw_msgs = _build_raw_messages(sample)
        return raw_msgs, TokenUsage(), True

    if precompressed is not None:
        encoded, comp_tok = precompressed, TokenUsage()
    else:
        old_messages = [
            {"role": "user", "content": "Reference document:"},
            {"role": "assistant", "content": sample.context},
        ]
        encoded, comp_tok = await _aaak_compress(old_messages)

    aaak_block = f"[AAAK CTX]\n{encoded}\n[/AAAK CTX]"
    messages = [
        system_msg,
        {"role": "user", "content": aaak_block},
        {"role": "user", "content": question_text},
    ]
    return messages, comp_tok, False


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample runners
# ─────────────────────────────────────────────────────────────────────────────


async def _run_raw_sample(sample: Sample) -> SampleResult:
    msgs = _build_raw_messages(sample)
    original_tokens = messages_tokens(msgs)
    answer, answer_tok = await _call(msgs)
    score = score_gsm8k(answer, sample.answers[0]) if sample.task == "gsm8k" else max_f1(answer, sample.answers)
    return SampleResult(
        mode="raw",
        task=sample.task,
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        score=score,
        answer_tok=answer_tok,
    )


async def _run_aaak_sample(
    sample: Sample,
    raw_original_tokens: int,
    precompressed: str | None = None,
) -> SampleResult:
    msgs, comp_tok, passthrough = await _build_aaak_messages(sample, precompressed=precompressed)
    answer, answer_tok = await _call(msgs)
    score = score_gsm8k(answer, sample.answers[0]) if sample.task == "gsm8k" else max_f1(answer, sample.answers)
    return SampleResult(
        mode="aaak (passthrough)" if passthrough else "aaak",
        task=sample.task,
        original_tokens=raw_original_tokens,
        compressed_tokens=messages_tokens(msgs),
        score=score,
        compress_tok=comp_tok,
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


def load_narrativeqa(n: int) -> list[Sample]:
    """deepmind/narrativeqa — book/movie-length document QA."""
    datasets = _require_datasets()
    ds = datasets.load_dataset("deepmind/narrativeqa", split="test")
    samples: list[Sample] = []
    for row in list(ds)[:n]:
        context = truncate_to_tokens(row["document"]["text"], MAX_DOC_TOKENS)
        answers = [a["text"] for a in row["answers"]]
        samples.append(Sample(
            task="narrativeqa",
            context=context,
            question=row["question"]["text"],
            answers=answers or [""],
        ))
    return samples


def load_hotpotqa(n: int) -> list[Sample]:
    """hotpotqa/hotpot_qa (distractor config) — multi-hop Wikipedia QA."""
    datasets = _require_datasets()
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    samples: list[Sample] = []
    for row in list(ds)[:n]:
        # Flatten the multi-document context: "Title: sent1 sent2 ..."
        parts = [
            f"{title}: {''.join(sents)}"
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        context = truncate_to_tokens("\n\n".join(parts), MAX_DOC_TOKENS)
        samples.append(Sample(
            task="hotpotqa",
            context=context,
            question=row["question"],
            answers=[row["answer"]],
        ))
    return samples


def load_squad(n: int) -> list[Sample]:
    """rajpurkar/squad — standard single-paragraph reading comprehension."""
    datasets = _require_datasets()
    ds = datasets.load_dataset("rajpurkar/squad", split="validation")
    samples: list[Sample] = []
    for row in list(ds)[:n]:
        answers = row["answers"]["text"]
        samples.append(Sample(
            task="squad",
            context=truncate_to_tokens(row["context"], MAX_DOC_TOKENS),
            question=row["question"],
            answers=answers or [""],
        ))
    return samples


_TASK_LOADERS: dict[str, Any] = {
    "gsm8k": load_gsm8k,
    "narrativeqa": load_narrativeqa,
    "hotpotqa": load_hotpotqa,
    "squad": load_squad,
}


# ─────────────────────────────────────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────────────────────────────────────


async def run_task(task_name: str, n_samples: int) -> tuple[list[SampleResult], list[SampleResult]]:
    """Run raw and aaak modes on ``n_samples`` from ``task_name``.

    Returns (raw_results, aaak_results). AAAK compression is run sequentially
    to avoid patching global provider state from multiple coroutines at once;
    answer calls run concurrently inside each mode.
    """
    loader = _TASK_LOADERS[task_name]
    samples = loader(n_samples)

    # ── raw mode (all samples in parallel) ───────────────────────────────────
    raw_results: list[SampleResult] = list(
        await asyncio.gather(*[_run_raw_sample(s) for s in samples])
    )

    # ── aaak mode ────────────────────────────────────────────────────────────
    # For GSM8K the few-shot context is identical across all samples, so we
    # compress it exactly once and reuse the encoded string.
    aaak_results: list[SampleResult] = []

    if task_name == "gsm8k":
        # Compress shared few-shot context once (sequential — patches global).
        few_shot_messages = [
            {"role": "user", "content": "Here are example math problems with solutions:"},
            {"role": "assistant", "content": samples[0].context},
        ]
        precompressed, shared_comp_tok = await _aaak_compress(few_shot_messages)

        # Answer all test samples in parallel (no global patching).
        def _gsm8k_aaak_msgs(question: str) -> list[dict[str, str]]:
            return [
                _gsm8k_system_with_aaak(),
                {"role": "user", "content": f"[AAAK CTX]\n{precompressed}\n[/AAAK CTX]"},
                {"role": "user", "content": f"Now solve this problem:\n{question}"},
            ]

        answer_results: list[tuple[str, TokenUsage]] = list(
            await asyncio.gather(*[_call(_gsm8k_aaak_msgs(s.question)) for s in samples])
        )
        for i, (sample, (answer, answer_tok), raw_r) in enumerate(
            zip(samples, answer_results, raw_results)
        ):
            compressed_msgs = _gsm8k_aaak_msgs(sample.question)
            # Only the first sample "pays" for the compression call; subsequent
            # samples reuse the cached encoding — reflecting real amortised cost.
            c_tok = shared_comp_tok if i == 0 else TokenUsage()
            aaak_results.append(SampleResult(
                mode="aaak",
                task=task_name,
                original_tokens=raw_r.original_tokens,
                compressed_tokens=messages_tokens(compressed_msgs),
                score=score_gsm8k(answer, sample.answers[0]),
                compress_tok=c_tok,
                answer_tok=answer_tok,
            ))
    else:
        # LongBench: each sample has a different document → compress per sample.
        # Must run sequentially because _aaak_compress patches global state.
        for sample, raw_r in zip(samples, raw_results):
            result = await _run_aaak_sample(
                sample, raw_original_tokens=raw_r.original_tokens
            )
            aaak_results.append(result)

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
    metric_label = "exact-match" if task_name == "gsm8k" else "max-F1"
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

    if args.no_deterministic_sampling:
        _benchmark_completion_extra = {}
        print("  completion_extra={} (no deterministic sampling kwargs)")
    else:
        print(f"  completion_extra={_benchmark_completion_extra}")

    _benchmark_semaphore = asyncio.Semaphore(args.concurrency)

    task_names: list[str] = args.tasks or AVAILABLE_TASKS
    print(f"  tasks={task_names}  samples_per_task={args.samples}  model={MODEL}")
    print(f"  max_doc_tokens={MAX_DOC_TOKENS}  gsm8k_few_shot_n={GSM8K_FEW_SHOT_N}\n")

    task_data: list[tuple[str, list[SampleResult], list[SampleResult], float]] = []

    for task_name in task_names:
        if task_name not in _TASK_LOADERS:
            print(f"  [WARN] Unknown task '{task_name}' — skipping. Choose from: {AVAILABLE_TASKS}")
            continue

        print(f"  Loading '{task_name}'...", flush=True)
        t0 = time.perf_counter()
        raw_results, aaak_results = await run_task(task_name, args.samples)
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
        description="AAAK vs Raw benchmark on standard NLP datasets (real API calls)."
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
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
