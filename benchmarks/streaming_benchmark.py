"""Streaming vs one-shot complete — latency and text parity (real API calls).

Compares ``stream_complete`` (time to first content chunk, total time) with
``complete`` on the same prompt.

Requires API keys for the chosen model.

Run:
  uv run python benchmarks/streaming_benchmark.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
from typing import Any

from gluellm.api import complete, stream_complete

DEFAULT_MODEL = "openai:gpt-4o-mini"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


async def main_async(args: argparse.Namespace) -> None:
    model = args.model or os.environ.get("GLUELLM_BENCH_MODEL", DEFAULT_MODEL)
    extra: dict[str, Any] = {}
    if not args.no_deterministic_sampling:
        extra = {"temperature": 0, "top_p": 1}

    prompt = args.prompt
    sys_prompt = "Answer briefly and literally. No preamble."

    print(f"  model={model}\n  prompt={prompt!r}\n")

    t0 = time.perf_counter()
    one = await complete(
        user_message=prompt,
        model=model,
        system_prompt=sys_prompt,
        execute_tools=False,
        **extra,
    )
    complete_s = time.perf_counter() - t0
    complete_text = one.final_response or ""

    ttft: float | None = None
    stream_parts: list[str] = []
    t_stream_start = time.perf_counter()
    async for chunk in stream_complete(
        user_message=prompt,
        model=model,
        system_prompt=sys_prompt,
        execute_tools=False,
        **extra,
    ):
        if chunk.content:
            if ttft is None:
                ttft = time.perf_counter() - t_stream_start
            stream_parts.append(chunk.content)
    stream_s = time.perf_counter() - t_stream_start
    stream_text = "".join(stream_parts)

    n_complete = _normalize(complete_text)
    n_stream = _normalize(stream_text)
    parity = n_complete == n_stream
    prefix_match = bool(n_complete and n_stream and (
        n_complete.startswith(n_stream[: min(20, len(n_stream))])
        or n_stream.startswith(n_complete[: min(20, len(n_complete))])
    ))

    print(f"  complete total:     {complete_s*1000:.0f} ms  len={len(complete_text)}")
    print(f"  stream total:       {stream_s*1000:.0f} ms  len={len(stream_text)}")
    print(f"  stream TTFT:        {(ttft or 0)*1000:.0f} ms")
    print(f"  normalized equal:   {parity}")
    if not parity:
        print(f"  prefix-ish match: {prefix_match}")
        print(f"  complete (norm): {n_complete[:200]!r}")
        print(f"  stream (norm):   {n_stream[:200]!r}")
    print("\n  done.\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Streaming vs complete benchmark (live API).")
    p.add_argument("--model", default=None, help=f"Model id (default: {DEFAULT_MODEL} or GLUELLM_BENCH_MODEL).")
    p.add_argument(
        "--prompt",
        default="Reply with exactly the word: STREAM-OK",
        help="User message for both paths.",
    )
    p.add_argument(
        "--no-deterministic-sampling",
        action="store_true",
        help="Do not pass temperature/top_p.",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
