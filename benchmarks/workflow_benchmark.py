"""Workflow orchestration micro-benchmark — mocked executors (no API).

Run:
  uv run python benchmarks/workflow_benchmark.py --iterations 200
"""

from __future__ import annotations

import argparse
import asyncio
import time

from gluellm.api import ExecutionResult
from gluellm.executors._base import Executor
from gluellm.models.workflow import ExpertConfig, MoEConfig
from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow


class _MockExec(Executor):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0

    async def _execute_internal(self, query: str) -> ExecutionResult:
        r = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return ExecutionResult(final_response=r, tool_calls_made=0, tool_execution_history=[])


async def main_async(args: argparse.Namespace) -> None:
    e1 = _MockExec(["alpha"])
    e2 = _MockExec(["beta"])
    combiner = _MockExec(["combined"])

    workflow = MixtureOfExpertsWorkflow(
        experts=[
            ExpertConfig(
                executor=e1,
                specialty="one",
                description="first",
                activation_keywords=["x"],
            ),
            ExpertConfig(
                executor=e2,
                specialty="two",
                description="second",
                activation_keywords=["y"],
            ),
        ],
        combiner=combiner,
        config=MoEConfig(routing_strategy="all", combine_strategy="synthesize"),
    )

    n = args.iterations
    t0 = time.perf_counter()
    for _ in range(n):
        await workflow.execute("benchmark query x y")
    elapsed = time.perf_counter() - t0

    print("  MixtureOfExpertsWorkflow (routing=all, 2 experts + combiner)")
    print(f"  iterations={n}  total={elapsed*1000:.2f} ms  mean={elapsed/n*1e6:.2f} µs/iter")
    print(f"  mock call counts: e1={e1.call_count} e2={e2.call_count} combiner={combiner.call_count}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Workflow micro-benchmark (no API).")
    p.add_argument("--iterations", type=int, default=200)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
