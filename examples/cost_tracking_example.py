"""Examples of cost tracking with GlueLLM.

Demonstrates CostTracker, get_global_tracker, budget limits, estimate_cost,
CostSummary, and record export.
"""

import asyncio
import os

from gluellm import complete
from gluellm.costing.cost_tracker import (
    CostTracker,
    estimate_cost,
    get_global_tracker,
    reset_global_tracker,
)
from gluellm.costing.pricing_data import list_available_models


async def example_estimate_cost():
    """Estimate cost without recording usage."""
    print("=" * 60)
    print("Example 1: Estimate Cost")
    print("=" * 60)

    cost = estimate_cost("openai:gpt-5.4-2026-03-05", input_tokens=1000, output_tokens=500)
    print(f"Estimated cost for 1000 in + 500 out tokens (gpt-5.4-2026-03-05): ${cost:.6f}")

    cost2 = estimate_cost("anthropic:claude-3-5-sonnet-20241022", input_tokens=5000, output_tokens=1000)
    print(f"Estimated cost for 5000 in + 1000 out tokens (claude-3-5-sonnet): ${cost2:.6f}")

    models = list_available_models("openai")[:3]
    print(f"Sample models with pricing: {models}")
    print()


async def example_cost_tracker_basic():
    """Basic CostTracker: record usage, get summary."""
    print("=" * 60)
    print("Example 2: CostTracker Basics")
    print("=" * 60)

    tracker = CostTracker()
    tracker.record_usage("openai:gpt-5.4-2026-03-05", input_tokens=100, output_tokens=50)
    tracker.record_usage("openai:gpt-5.4-2026-03-05", input_tokens=200, output_tokens=100)

    print(f"Daily cost: ${tracker.get_daily_cost():.6f}")
    print(f"Session cost: ${tracker.get_session_cost():.6f}")
    summary = tracker.get_summary()
    print(f"Request count: {summary.request_count}")
    print(f"Cost by model: {summary.cost_by_model}")
    print()


async def example_budget_limits():
    """Set budget limits and check remaining."""
    print("=" * 60)
    print("Example 3: Budget Limits")
    print("=" * 60)

    tracker = CostTracker(daily_limit=10.0, session_limit=1.0)
    tracker.record_usage("openai:gpt-5.4-2026-03-05", input_tokens=500, output_tokens=200)

    remaining = tracker.get_remaining_daily_budget()
    print(f"Session cost: ${tracker.get_session_cost():.6f}")
    print(f"Remaining daily budget: ${remaining:.6f}" if remaining is not None else "No budget set")
    print()


async def example_global_tracker_with_complete():
    """Use global tracker alongside complete() - track costs from real calls."""
    print("=" * 60)
    print("Example 4: Global Tracker with complete()")
    print("=" * 60)

    reset_global_tracker()
    tracker = get_global_tracker()
    tracker.set_budget(daily_limit=1.0)

    result = await complete(
        "Say hello in one word.",
        system_prompt="Be brief.",
    )
    print(f"Response: {result.final_response}")

    if result.tokens_used and result.model:
        tracker.record_usage(
            model=result.model,
            input_tokens=result.tokens_used.get("prompt", 0),
            output_tokens=result.tokens_used.get("completion", 0),
        )

    print(f"Session cost after call: ${tracker.get_session_cost():.6f}")
    print(f"Remaining daily budget: ${tracker.get_remaining_daily_budget():.6f}")
    print()


async def example_export_records():
    """Export records in dict, JSON, and CSV formats."""
    print("=" * 60)
    print("Example 5: Export Records")
    print("=" * 60)

    tracker = CostTracker()
    tracker.record_usage("openai:gpt-5.4-2026-03-05", input_tokens=50, output_tokens=20, request_id="req-1")
    tracker.record_usage("anthropic:claude-3-5-haiku-20241022", input_tokens=80, output_tokens=30)

    exported_dict = tracker.export_records("dict")
    print(f"Dict export: {len(exported_dict)} records")
    print(f"First record keys: {list(exported_dict[0].keys())}")

    csv_export = tracker.export_records("csv")
    lines = csv_export.strip().split("\n")
    print(f"CSV export: {len(lines)} lines (header + records)")
    print()


async def main():
    await example_estimate_cost()
    await example_cost_tracker_basic()
    await example_budget_limits()
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY") != "sk-test":
        await example_global_tracker_with_complete()
    else:
        print("(Skipping example 4: OPENAI_API_KEY not set)")
    await example_export_records()


if __name__ == "__main__":
    asyncio.run(main())
