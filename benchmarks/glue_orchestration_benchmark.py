"""GlueLLM orchestration benchmark — end-to-end SDK path (real API calls).

Measures wall time and tool usage for ``GlueLLM.complete`` and
``structured_complete`` with a small tool, versus raw ``complete`` without tools.

Requires OPENAI_API_KEY or another configured provider for the chosen model.

Run:
  uv run python benchmarks/glue_orchestration_benchmark.py
  uv run python benchmarks/glue_orchestration_benchmark.py --model openai:gpt-5.4-2026-03-05 --skip-structured
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Annotated, Any

from pydantic import BaseModel, Field

from gluellm.api import GlueLLM, complete, structured_complete

DEFAULT_MODEL = "openai:gpt-5.4-2026-03-05"


class _TinyFact(BaseModel):
    """Structured output for orchestration check."""

    answer: Annotated[str, Field(description="Short factual answer, one phrase")]
    confidence: Annotated[str, Field(description="'high' or 'low'")]


def _math_tool(a: int, b: int) -> str:
    """Return the sum of two integers (used to force one tool round)."""
    return str(a + b)


async def _timed(coro):
    t0 = time.perf_counter()
    result = await coro
    return result, time.perf_counter() - t0


async def main_async(args: argparse.Namespace) -> None:
    model = args.model or os.environ.get("GLUELLM_BENCH_MODEL", DEFAULT_MODEL)
    extra: dict[str, Any] = {}
    if not args.no_deterministic_sampling:
        extra = {"temperature": 0, "top_p": 1}

    print(f"  model={model}  deterministic_sampling={not args.no_deterministic_sampling}\n")

    msg = "Reply with exactly: OK-BENCH"
    r1, e1 = await _timed(
        complete(
            user_message=msg,
            model=model,
            system_prompt="Follow the user instruction literally.",
            tools=None,
            execute_tools=True,
            **extra,
        )
    )
    print(f"  [complete no-tools]  {e1*1000:.0f} ms  response_len={len(r1.final_response)}  tools={r1.tool_calls_made}")

    r2, e2 = await _timed(
        complete(
            user_message="Use the math tool to compute 19 + 23. Reply with only the numeric result.",
            model=model,
            system_prompt="You must call math_tool when asked to compute.",
            tools=[_math_tool],
            execute_tools=True,
            max_tool_iterations=4,
            **extra,
        )
    )
    print(
        f"  [complete + tool]    {e2*1000:.0f} ms  tools={r2.tool_calls_made}  "
        f"history={len(r2.tool_execution_history)}  tail={r2.final_response[:80]!r}..."
    )

    client = GlueLLM(
        model=model,
        system_prompt="You are concise.",
        tools=[_math_tool],
        max_tool_iterations=4,
        model_kwargs=extra,
    )
    r3, e3 = await _timed(client.complete("What is 40 + 2 using the math tool? One line."))
    print(
        f"  [GlueLLM.complete]   {e3*1000:.0f} ms  tools={r3.tool_calls_made}  "
        f"tail={r3.final_response[:80]!r}..."
    )

    if not args.skip_structured:
        r4, e4 = await _timed(
            structured_complete(
                user_message="Structured: answer='France capital' as city name only, confidence=high",
                response_format=_TinyFact,
                model=model,
                system_prompt="Fill the schema from the user message.",
                **extra,
            )
        )
        so = r4.structured_output
        print(
            f"  [structured_complete] {e4*1000:.0f} ms  tools={r4.tool_calls_made}  "
            f"parsed={so is not None}  answer={getattr(so, 'answer', None)!r}"
        )

    print("\n  done.\n")


def main() -> None:
    p = argparse.ArgumentParser(description="GlueLLM orchestration latency benchmark (live API).")
    p.add_argument("--model", default=None, help=f"Model id (default: {DEFAULT_MODEL} or GLUELLM_BENCH_MODEL).")
    p.add_argument("--skip-structured", action="store_true", help="Skip structured_complete scenario.")
    p.add_argument(
        "--no-deterministic-sampling",
        action="store_true",
        help="Do not pass temperature/top_p (some models reject them).",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
