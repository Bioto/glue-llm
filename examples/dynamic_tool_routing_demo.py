#!/usr/bin/env python3
"""
Dynamic Tool Routing + Condensing Demo

Demonstrates tool_mode="dynamic" vs "standard" and condense_tool_messages, with
multiple rounds of tool calls. Compares context sizes across the interaction.

- Standard mode: all tool schemas in every call
- Dynamic mode: router discovers tools on demand
- Condensing: each tool round replaced with a short summary (reduces history bloat)

Usage:
    uv run python examples/dynamic_tool_routing_demo.py              # Dry-run estimates
    uv run python examples/dynamic_tool_routing_demo.py --dry-run    # Same
    uv run python examples/dynamic_tool_routing_demo.py --live       # Live API (needs OPENAI_API_KEY)
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from gluellm.api import GlueLLM
from gluellm.events import ProcessEvent
from gluellm.models.prompt import BASE_SYSTEM_PROMPT
from gluellm.tool_router import build_router_tool

# -----------------------------------------------------------------------------
# Demo tools (enough to show context bloat)
# -----------------------------------------------------------------------------

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get the current weather for a city. Supports celsius or fahrenheit."""
    return {"city": city, "temp": 22, "conditions": "sunny", "unit": unit}


def get_forecast(city: str, days: int = 5) -> list:
    """Get a multi-day weather forecast for a city."""
    return [{"day": i, "temp": 20 + i, "conditions": "partly cloudy"} for i in range(days)]


def search_flights(origin: str, destination: str, date: str) -> list:
    """Search for available flights between two cities on a given date."""
    return [{"flight": "AA123", "departure": "08:00", "price": 299}]


def book_hotel(city: str, check_in: str, check_out: str) -> dict:
    """Book a hotel in a city for given check-in and check-out dates."""
    return {"hotel": "Grand Hotel", "confirmation": "CONF-12345"}


def calculate(expression: str) -> float:
    """Evaluate a mathematical expression. Supports +, -, *, /, parentheses."""
    return eval(expression)


def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get the current exchange rate between two currencies."""
    return 1.08 if "USD" in from_currency else 0.92


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to a target language. Supports common language codes."""
    return f"[Translated to {target_lang}]: {text}"


def search_restaurants(city: str, cuisine: str | None = None) -> list:
    """Search for restaurants in a city, optionally filtered by cuisine type."""
    return [{"name": "Bistro One", "cuisine": cuisine or "international", "rating": 4.5}]


def get_stock_price(symbol: str) -> dict:
    """Get the current stock price and change for a ticker symbol."""
    return {"symbol": symbol, "price": 150.25, "change_pct": 1.2}


ALL_TOOLS = [
    get_weather,
    get_forecast,
    search_flights,
    book_hotel,
    calculate,
    get_exchange_rate,
    translate_text,
    search_restaurants,
    get_stock_price,
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return max(1, len(text) // 4)


# -----------------------------------------------------------------------------
# Dry-run: compute context sizes without API calls
# -----------------------------------------------------------------------------

@dataclass
class ContextEstimate:
    mode: str
    phase: str
    chars: int
    est_tokens: int
    tool_count: int
    condensed: bool = False
    round_num: int | None = None


def run_dry_run(system_prompt: str, user_message: str) -> list[ContextEstimate]:
    """Estimate context sizes for standard vs dynamic, with multi-round condensing."""
    estimates: list[ContextEstimate] = []
    user_msg_str = json.dumps({"role": "user", "content": user_message})

    # Approximate message sizes
    raw_assistant_tool = json.dumps({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
    })
    raw_tool_result = json.dumps({"role": "tool", "tool_call_id": "call_1", "content": '{"city": "Paris", "temp": 22}'})
    condensed_msg = json.dumps({"role": "assistant", "content": "[Tool Results]\n- get_weather() -> 22°C in Paris"})

    # Standard mode - 3 rounds, NO condensing (raw assistant + tool per round)
    standard_system = BASE_SYSTEM_PROMPT.render(
        instructions=system_prompt,
        tools=ALL_TOOLS,
    ).strip()
    for r in range(1, 4):
        ctx = standard_system + "\n\n" + user_msg_str
        for _ in range(r):
            ctx += "\n\n" + raw_assistant_tool + "\n\n" + raw_tool_result
        estimates.append(ContextEstimate(
            mode="standard",
            phase=f"after round {r} (no condense)",
            chars=len(ctx),
            est_tokens=estimate_tokens(ctx),
            tool_count=len(ALL_TOOLS),
            condensed=False,
            round_num=r,
        ))

    # Standard mode - 3 rounds, WITH condensing
    for r in range(1, 4):
        ctx = standard_system + "\n\n" + user_msg_str
        for _ in range(r):
            ctx += "\n\n" + condensed_msg
        estimates.append(ContextEstimate(
            mode="standard",
            phase=f"after round {r} (condensed)",
            chars=len(ctx),
            est_tokens=estimate_tokens(ctx),
            tool_count=len(ALL_TOOLS),
            condensed=True,
            round_num=r,
        ))

    # Dynamic mode - call 1 (router)
    router_tool = build_router_tool(ALL_TOOLS)
    dynamic1_system = BASE_SYSTEM_PROMPT.render(
        instructions=system_prompt,
        tools=[router_tool],
    ).strip()
    ctx1 = dynamic1_system + "\n\n" + user_msg_str
    estimates.append(ContextEstimate(
        mode="dynamic",
        phase="call 1 (router)",
        chars=len(ctx1),
        est_tokens=estimate_tokens(ctx1),
        tool_count=1,
        condensed=False,
    ))

    # Dynamic mode - rounds 2..4 (matched tools, with condensing)
    matched = [get_weather, get_forecast, calculate]
    dynamic_system = BASE_SYSTEM_PROMPT.render(
        instructions=system_prompt,
        tools=matched,
    ).strip()
    for r in range(1, 4):
        ctx = dynamic_system + "\n\n" + user_msg_str
        for _ in range(r):
            ctx += "\n\n" + condensed_msg
        estimates.append(ContextEstimate(
            mode="dynamic",
            phase=f"round {r+1} (matched, condensed)",
            chars=len(ctx),
            est_tokens=estimate_tokens(ctx),
            tool_count=len(matched),
            condensed=True,
            round_num=r + 1,
        ))

    return estimates


def print_dry_run(estimates: list[ContextEstimate]) -> None:
    """Print dry-run comparison."""
    print("\n" + "=" * 78)
    print("DYNAMIC TOOL ROUTING + CONDENSING - CONTEXT SIZE COMPARISON (3 tool rounds)")
    print("=" * 78)
    print(f"\nTotal tools available: {len(ALL_TOOLS)}")
    print("Condensing: each tool round replaced with short summary before next LLM call\n")

    max_tokens = max(e.est_tokens for e in estimates)
    for e in estimates:
        bar_len = min(50, int(50 * e.est_tokens / max_tokens)) if max_tokens else 0
        bar = "█" * bar_len + "░" * (50 - bar_len)
        cond = " [condensed]" if e.condensed else ""
        print(f"  {e.mode:8} | {e.phase:28} | {e.tool_count} tools | "
              f"{e.chars:6} chars | ~{e.est_tokens:5} tokens{cond}")
        print(f"             | {bar}")

    std_no_cond = next(e for e in estimates if e.mode == "standard" and e.round_num == 3 and not e.condensed)
    std_cond = next(e for e in estimates if e.mode == "standard" and e.round_num == 3 and e.condensed)
    dyn_cond = next(e for e in estimates if e.mode == "dynamic" and e.round_num == 4)
    print(f"\n  Condensing savings (standard, 3 rounds):  "
          f"~{100 * (1 - std_cond.est_tokens / std_no_cond.est_tokens):.0f}% fewer tokens")
    print(f"  Dynamic + condensing vs standard (raw):  "
          f"~{100 * (1 - dyn_cond.est_tokens / std_no_cond.est_tokens):.0f}% fewer tokens")
    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# Live demo: real API calls with token tracking
# -----------------------------------------------------------------------------

@dataclass
class DemoStats:
    events: list[ProcessEvent] = field(default_factory=list)
    tool_route_query: str | None = None
    matched_tools: list[str] = field(default_factory=list)
    round_message_counts: list[int] = field(default_factory=list)  # message_count per LLM call


async def run_live_demo(
    system_prompt: str,
    user_message: str,
    tool_mode: str,
    condense_tool_messages: bool = True,
) -> dict[str, Any]:
    """Run a live completion and capture stats."""
    stats = DemoStats()

    def on_status(event: ProcessEvent) -> None:
        stats.events.append(event)
        if event.kind == "tool_route":
            stats.tool_route_query = getattr(event, "route_query", None)
            stats.matched_tools = getattr(event, "matched_tools", []) or []
        if event.kind == "llm_call_start" and getattr(event, "message_count", None) is not None:
            stats.round_message_counts.append(event.message_count)

    client = GlueLLM(
        model="openai:gpt-5.4-2026-03-05",
        system_prompt=system_prompt,
        tools=ALL_TOOLS,
        tool_mode=tool_mode,
        condense_tool_messages=condense_tool_messages,
    )
    result = await client.complete(
        user_message,
        tool_mode=tool_mode,
        condense_tool_messages=condense_tool_messages,
        on_status=on_status,
        max_tokens=50,
    )
    return {
        "result": result,
        "stats": stats,
    }


def _get_total_tokens(data: dict[str, Any]) -> int:
    t = data["result"].tokens_used
    if not t:
        return 0
    return t.get("total", t.get("total_tokens", 0))


def _verify_completion(
    tool_execution_history: list[dict[str, Any]],
    expected_tools: list[str],
) -> tuple[list[str], list[str], bool]:
    """Check which expected tools were called.

    Returns:
        (hit, missed, complete) — hit and missed tool names, and whether all were called.
    """
    called = {h.get("tool_name") for h in (tool_execution_history or [])}
    hit = [t for t in expected_tools if t in called]
    missed = [t for t in expected_tools if t not in called]
    return hit, missed, len(missed) == 0


def print_live_results(
    results: dict[str, dict[str, Any]],
    user_message: str,
    expected_tools: list[str] | None = None,
) -> None:
    """Print comparison of all live run results."""

    def token_str(t: dict | None) -> str:
        if not t:
            return "N/A"
        p = t.get("prompt", t.get("prompt_tokens", 0))
        c = t.get("completion", t.get("completion_tokens", 0))
        tot = t.get("total", t.get("total_tokens", 0))
        return f"prompt={p}, completion={c}, total={tot}"

    print("\n" + "=" * 78)
    print("LIVE DEMO RESULTS — 4-WAY COMPARISON")
    print("=" * 78)
    print(f"\nQuery: {user_message[:70]}{'...' if len(user_message) > 70 else ''}")
    print(f"Tools available: {len(ALL_TOOLS)}")
    if expected_tools:
        print(f"Expected tools:  {expected_tools}\n")
    else:
        print()

    labels = {
        "std_raw":  "STANDARD (no condensing)",
        "std_cond": "STANDARD (condensed)",
        "dyn_raw":  "DYNAMIC  (no condensing)",
        "dyn_cond": "DYNAMIC  (condensed)",
    }

    for key, label in labels.items():
        data = results[key]
        result = data["result"]
        stats = data["stats"]

        print(f"  {label}:")
        if stats.tool_route_query:
            print(f'    Router query:  "{stats.tool_route_query}"')
            print(f"    Matched tools: {stats.matched_tools}")
        print(f"    Tool calls:    {result.tool_calls_made}")
        if stats.round_message_counts:
            print(f"    Context sizes: {stats.round_message_counts}")
        print(f"    Tokens:        {token_str(result.tokens_used)}")
        print(f"    Cost:          ${result.estimated_cost_usd or 0:.6f}")
        if expected_tools:
            hit, missed, complete = _verify_completion(
                result.tool_execution_history or [],
                expected_tools,
            )
            if complete:
                print(f"    Completed:     YES ({len(hit)}/{len(expected_tools)} expected tools called)")
            else:
                print(f"    Completed:     NO — missing: {', '.join(missed)} ({len(hit)}/{len(expected_tools)})")
        print()

    # Pairwise comparisons
    std_raw_tot = _get_total_tokens(results["std_raw"])
    std_cond_tot = _get_total_tokens(results["std_cond"])
    dyn_raw_tot = _get_total_tokens(results["dyn_raw"])
    dyn_cond_tot = _get_total_tokens(results["dyn_cond"])

    print("  Comparisons:")
    if std_raw_tot:
        pct = 100 * (1 - std_cond_tot / std_raw_tot)
        print(f"    Condensing alone:         {pct:+.0f}% tokens (standard raw → standard condensed)")
    if std_raw_tot:
        pct = 100 * (1 - dyn_raw_tot / std_raw_tot)
        print(f"    Dynamic routing alone:    {pct:+.0f}% tokens (standard raw → dynamic raw)")
    if std_raw_tot:
        pct = 100 * (1 - dyn_cond_tot / std_raw_tot)
        print(f"    Both optimizations:       {pct:+.0f}% tokens (standard raw → dynamic condensed)")
    print("=" * 78 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic tool routing demo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate context sizes without API calls",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live API demo (requires OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    system_prompt = (
        "You are a helpful assistant with access to various tools. "
        "Use them step by step when needed. Make multiple tool calls if the user asks for several things."
    )
    # Query designed to trigger multiple tool rounds: weather, then forecast, then calculate
    user_message = (
        "First get the current weather in Paris. "
        "Then get the 5-day forecast for Paris. "
        "Finally use the calculator to convert 22 celsius to fahrenheit (formula: 22 * 9/5 + 32). "
        "Give me a brief summary at the end."
    )

    if args.dry_run or (not args.live and not args.dry_run):
        estimates = run_dry_run(system_prompt, user_message)
        print_dry_run(estimates)
        if not args.dry_run:
            print("Run with --live to execute real API calls and see token usage.\n")

    if args.live:
        # Scenario 1: short/batchable (3 tools, can be called in one round)
        expected_scenario1 = ["get_weather", "get_forecast", "calculate"]
        print("Running Scenario 1: SHORT CHAIN (batchable tools)...\n")
        results1 = {
            "std_raw":  await run_live_demo(system_prompt, user_message, "standard", condense_tool_messages=False),
            "std_cond": await run_live_demo(system_prompt, user_message, "standard", condense_tool_messages=True),
            "dyn_raw":  await run_live_demo(system_prompt, user_message, "dynamic", condense_tool_messages=False),
            "dyn_cond": await run_live_demo(system_prompt, user_message, "dynamic", condense_tool_messages=True),
        }
        print_live_results(results1, user_message, expected_tools=expected_scenario1)

        # Scenario 2: long sequential chain (each step depends on prior results)
        expected_scenario2 = [
            "get_weather",
            "get_forecast",
            "search_flights",
            "calculate",
            "get_exchange_rate",
            "translate_text",
        ]
        long_message = (
            "I need help planning a trip. Do each step one at a time, waiting for results before proceeding:\n"
            "1. Get the current weather in Paris\n"
            "2. Get the current weather in London\n"
            "3. Get the current weather in Tokyo\n"
            "4. Get the 5-day forecast for whichever city is warmest\n"
            "5. Search for flights from New York to that warmest city for 2025-06-15\n"
            "6. Use the calculator to compute the total trip cost: flight price * 2 passengers + 200 hotel\n"
            "7. Get the exchange rate from USD to EUR\n"
            "8. Use the calculator to convert the total cost to EUR using the exchange rate\n"
            "9. Translate a one-sentence trip summary into French\n"
            "Give me the final summary."
        )
        print("\nRunning Scenario 2: LONG CHAIN (sequential, 9 dependent steps)...\n")
        results2 = {
            "std_raw":  await run_live_demo(system_prompt, long_message, "standard", condense_tool_messages=False),
            "std_cond": await run_live_demo(system_prompt, long_message, "standard", condense_tool_messages=True),
            "dyn_raw":  await run_live_demo(system_prompt, long_message, "dynamic", condense_tool_messages=False),
            "dyn_cond": await run_live_demo(system_prompt, long_message, "dynamic", condense_tool_messages=True),
        }
        print_live_results(results2, long_message, expected_tools=expected_scenario2)


if __name__ == "__main__":
    asyncio.run(main())
