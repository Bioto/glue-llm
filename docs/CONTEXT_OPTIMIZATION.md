# Context Optimization

GlueLLM includes two opt-in features for reducing token usage in tool-heavy workloads: **context condensing** and **dynamic tool routing**. Both are disabled by default and can be enabled independently or together.

## Context Condensing

### What it does

In a standard tool execution loop, every completed tool round appends two messages to the conversation context:

```
assistant  → { tool_calls: [get_weather(Paris), get_forecast(Paris)] }
tool       → { result: "22°C, sunny" }
tool       → { result: [{ day: 1, temp: 21 }, ...] }
```

With `condense_tool_messages=True`, GlueLLM replaces that entire group with a single condensed `user` message after the round completes — before the next LLM call:

```
user → [Tool Results]
       - get_weather() -> {'city': 'Paris', 'temp': 22, ...}
       - get_forecast() -> [{'day': 1, 'temp': 21}, ...]
```

The condensed message contains the full tool result (no truncation). Only the raw message format changes, not the information content.

### Why it helps

Without condensing, context grows by `1 + N` messages per round (1 assistant + N tool results). With condensing, every round collapses to exactly 1 message regardless of how many tools ran in parallel. For a 9-step sequential chain this is the difference between 20 messages and 11 messages at the final call.

The condensed message uses `role: "user"` (not `assistant`) so the LLM naturally continues from it rather than treating the turn as finished.

### How to enable

Off by default. Enable per-call:

```python
from gluellm.api import complete

result = await complete(
    user_message="Do 9 sequential steps...",
    tools=[...],
    condense_tool_messages=True,  # opt-in
)
```

Or set it as the client default:

```python
from gluellm import GlueLLM

client = GlueLLM(tools=[...], condense_tool_messages=True)
result = await client.complete("Do 9 sequential steps...")
```

A per-call value always overrides the client default:

```python
client = GlueLLM(tools=[...], condense_tool_messages=True)
result = await client.complete("Quick one-shot.", condense_tool_messages=False)
```

Works identically on `complete()`, `structured_complete()`, and `stream_complete()`.

### When to use it

- Long sequential chains (5+ tool rounds) where context would otherwise grow large
- Cases where you want predictable, bounded context growth
- Combined with `max_tokens` to keep completions short during tool iteration

### When to skip it

- Single-tool-round tasks — the overhead of rewriting the message format isn't worth it
- Debugging: raw tool messages are easier to inspect than condensed summaries
- Workflows where you need the exact raw tool response format preserved in conversation history (note: condensed summaries are ephemeral within a single `complete()` call; they are not persisted to the conversation history between turns)

---

## Dynamic Tool Routing

### What it does

In standard mode (`tool_mode="standard"`), the full schema for every tool is included in the system prompt on every LLM call. With a large toolset this can add hundreds of tokens per call — most of which describe tools the model won't use.

With `tool_mode="dynamic"`, GlueLLM replaces the upfront schema dump with a two-phase approach:

1. **Router call**: A fast, cheap LLM call with a single `route_to_tools(query)` tool. The model returns the names of tools it needs.
2. **Execution call(s)**: Only the matched tool schemas are injected into the system prompt for the actual work.

### Why it helps

For a 9-tool set where a task only needs 3 tools, the router call adds 1 cheap call but saves the schema tokens for 6 unused tools on every subsequent call. The savings compound over multi-round chains.

```
Standard (9 tools, 5 rounds):   9 schemas × 5 calls = 45 schema-call units
Dynamic  (9 tools, 3 matched):  1 router call + 3 schemas × 5 calls = 16 units
```

### How to enable

Off by default (`tool_mode="standard"`). Enable per-call:

```python
from gluellm.api import complete

result = await complete(
    user_message="Check the weather and search flights...",
    tools=[get_weather, search_flights, book_hotel, calculate, translate, ...],
    tool_mode="dynamic",  # opt-in
)
```

Or set it as the client default:

```python
from gluellm import GlueLLM

client = GlueLLM(
    tools=[...],
    tool_mode="dynamic",
    tool_route_model="openai:gpt-5.4-2026-03-05",  # optional: fast cheap model for the router
)
result = await client.complete("Check the weather and search flights...")
```

`tool_route_model` defaults to the global `settings.tool_route_model`. You can set a faster/cheaper model here since the router only needs to output tool names, not full reasoning.

### Observing routing decisions

When `on_status` is set, a `tool_route` event fires after the router call with the matched tool names:

```python
def on_status(event):
    if event.kind == "tool_route":
        print(f"Router matched: {event.matched_tools}")

result = await complete(
    "Check weather and book a hotel.",
    tools=[...],
    tool_mode="dynamic",
    on_status=on_status,
)
```

### When to use it

- 6 or more tools available
- Most tasks only use a subset of available tools
- Minimizing prompt tokens per call is a priority

### When to skip it

- Small toolsets (2-4 tools) — the extra router call cost exceeds the schema savings
- Tasks that reliably use most or all tools — routing adds a call with little savings
- Latency-sensitive paths where the extra round-trip matters

---

## Conversation Summarization

### What it does

In a long multi-turn conversation the full message history is sent to the LLM on every call. As turns accumulate the context window fills, costs rise, and older content competes for the model's attention against the current task.

With `summarize_context=True`, GlueLLM monitors the number of messages before each LLM call. When the count exceeds `summarize_context_threshold` (default: 20), it automatically:

1. Keeps the system prompt and the most recent `summarize_context_keep_recent` messages (default: 6) verbatim.
2. Sends everything in between to a summarizer LLM call.
3. Replaces the old messages with a single `[Conversation Summary]` message containing the condensed history.

The summarization call is transparent — if it fails for any reason, GlueLLM logs a warning and continues with the original messages unchanged.

### Why it helps

Without summarization, a 40-turn conversation sends 40 messages (plus system prompt) on every call. With summarization triggered at 20 messages and keeping the last 6, the context stays bounded at roughly 8 messages regardless of how long the conversation runs — `[system] + [summary] + 6 recent`.

This is particularly valuable for:

- Long user-facing chat sessions where history accumulates across many turns.
- Agentic loops where the model iterates on a task over many iterations.
- Any scenario where the conversation grows longer than the model's effective context window.

### How to enable

Off by default. Enable on the `GlueLLM` client:

```python
from gluellm import GlueLLM

client = GlueLLM(summarize_context=True)
result = await client.complete("Continue from where we left off...")
```

Or per-call:

```python
result = await client.complete(
    "Summarize what we discussed.",
    summarize_context=True,
    summarize_context_threshold=15,   # trigger earlier
    summarize_context_keep_recent=4,  # keep fewer recent messages
)
```

Or via the stateless helper:

```python
from gluellm import complete

result = await complete(
    "Continue our discussion.",
    summarize_context=True,
)
```

### Using a cheaper model for summarization

The summarization call is cheap and doesn't need the full capability of your primary model. Set `summarize_context_model` to a fast, low-cost model:

```python
client = GlueLLM(
    model="openai:gpt-4o",
    summarize_context=True,
    summarize_context_model="openai:gpt-5.4-2026-03-05",  # cheaper summarizer
)
```

If `summarize_context_model` is not set, the primary model is used.

### Global defaults via configure()

Set summarization defaults globally so all clients pick them up without per-call configuration:

```python
import gluellm

gluellm.configure(
    default_summarize_context=True,
    default_summarize_context_threshold=25,
    default_summarize_context_keep_recent=8,
)
```

Or via environment variables:

```
GLUELLM_DEFAULT_SUMMARIZE_CONTEXT=true
GLUELLM_DEFAULT_SUMMARIZE_CONTEXT_THRESHOLD=25
GLUELLM_DEFAULT_SUMMARIZE_CONTEXT_KEEP_RECENT=8
```

### When to use it

- Chat sessions expected to run longer than ~15–20 turns.
- Agentic workflows that iterate many times on a complex task.
- When you want predictable, bounded context cost regardless of conversation length.

### When to skip it

- Short sessions (fewer turns than the threshold) — no benefit, no overhead.
- Tasks where verbatim recall of early messages is critical (e.g. the user gave a long spec at turn 1 that the model must re-read in full at every step). Consider raising `summarize_context_keep_recent` in this case instead.
- Debugging conversations — the full raw history is easier to inspect.

### Interaction with condense_tool_messages

Both features reduce context size but target different sources of growth:

- `condense_tool_messages` compresses **tool-call rounds** within a single `complete()` call.
- `summarize_context` compresses **conversation turns** that have accumulated across multiple calls.

They are independent and compose cleanly when both are enabled. In a long agentic session with heavy tool use, combining them provides the maximum context reduction.

---

## Combining Both (condense + dynamic routing)

```python
from gluellm import GlueLLM

client = GlueLLM(
    tools=[...],
    condense_tool_messages=True,   # compress history
    tool_mode="dynamic",           # route to relevant tools
)
result = await client.complete("Plan a 9-step trip...")
```

In a 9-step sequential chain, this combination reduces context to roughly:

- Router call: system + user (2 messages)
- Per round: 1 condensed message instead of 1 + N raw messages
- Final context: ~10 messages vs ~20 for standard raw

Both overrides also work at the per-call level, so you can mix and match within a multi-turn session:

```python
client = GlueLLM(tools=[...])  # both off by default

# Turn 1: standard
await client.complete("Quick lookup.", tools=[lookup])

# Turn 2: with both optimizations
await client.complete(
    "Long multi-step task.",
    condense_tool_messages=True,
    tool_mode="dynamic",
)
```

---

## Benchmarks

The [`examples/dynamic_tool_routing_demo.py`](../examples/dynamic_tool_routing_demo.py) script runs a 4-way live comparison across two realistic scenarios. Run it yourself:

```bash
uv run python examples/dynamic_tool_routing_demo.py --live
```

Full output:

```
Running Scenario 1: SHORT CHAIN (batchable tools)...

==============================================================================
LIVE DEMO RESULTS — 4-WAY COMPARISON
==============================================================================

Query: First get the current weather in Paris. Then get the 5-day forecast fo...
Tools available: 9
Expected tools:  ['get_weather', 'get_forecast', 'calculate']

  STANDARD (no condensing):
    Tool calls:    20
    Context sizes: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    Tokens:        prompt=1539, completion=50, total=1589
    Cost:          $0.000261
    Completed:     NO — missing: calculate (2/3)

  STANDARD (condensed):
    Tool calls:    3
    Context sizes: [2, 3, 4]
    Tokens:        prompt=693, completion=50, total=743
    Cost:          $0.000134
    Completed:     YES (3/3 expected tools called)

  DYNAMIC  (no condensing):
    Router query:  "current weather in Paris"
    Matched tools: ['get_weather', 'get_forecast', 'calculate']
    Tool calls:    14
    Context sizes: [2, 2, 5, 8, 11, 14, 17, 20, 22, 24]
    Tokens:        prompt=1077, completion=50, total=1127
    Cost:          $0.000192
    Completed:     YES (3/3 expected tools called)

  DYNAMIC  (condensed):
    Router query:  "current weather Paris"
    Matched tools: ['get_weather', 'get_forecast', 'calculate']
    Tool calls:    5
    Context sizes: [2, 2, 3, 4, 5]
    Tokens:        prompt=458, completion=50, total=508
    Cost:          $0.000099
    Completed:     YES (3/3 expected tools called)

  Comparisons:
    Condensing alone:         +53% tokens (standard raw → standard condensed)
    Dynamic routing alone:    +29% tokens (standard raw → dynamic raw)
    Both optimizations:       +68% tokens (standard raw → dynamic condensed)
==============================================================================


Running Scenario 2: LONG CHAIN (sequential, 9 dependent steps)...

==============================================================================
LIVE DEMO RESULTS — 4-WAY COMPARISON
==============================================================================

Query: I need help planning a trip. Do each step one at a time, waiting for r...
Tools available: 9
Expected tools:  ['get_weather', 'get_forecast', 'search_flights', 'calculate', 'get_exchange_rate', 'translate_text']

  STANDARD (no condensing):
    Tool calls:    9
    Context sizes: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    Tokens:        prompt=1103, completion=50, total=1153
    Cost:          $0.000195
    Completed:     YES (6/6 expected tools called)

  STANDARD (condensed):
    Tool calls:    9
    Context sizes: [2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    Tokens:        prompt=1016, completion=50, total=1066
    Cost:          $0.000182
    Completed:     YES (6/6 expected tools called)

  DYNAMIC  (no condensing):
    Router query:  "current weather in Paris"
    Matched tools: ['get_weather', 'get_forecast', 'search_flights', 'calculate', 'get_exchange_rate', 'translate_text']
    Tool calls:    9
    Context sizes: [2, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    Tokens:        prompt=914, completion=28, total=942
    Cost:          $0.000154
    Completed:     YES (6/6 expected tools called)

  DYNAMIC  (condensed):
    Router query:  "current weather in Paris"
    Matched tools: ['get_weather', 'get_forecast', 'search_flights', 'calculate', 'get_exchange_rate', 'translate_text']
    Tool calls:    9
    Context sizes: [2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Tokens:        prompt=803, completion=28, total=831
    Cost:          $0.000137
    Completed:     YES (6/6 expected tools called)

  Comparisons:
    Condensing alone:         +8% tokens (standard raw → standard condensed)
    Dynamic routing alone:    +18% tokens (standard raw → dynamic raw)
    Both optimizations:       +28% tokens (standard raw → dynamic condensed)
==============================================================================
```

### Reading the results

**Scenario 1 (short batchable chain)** is the more striking case. Standard raw looped 20 times and still missed `calculate`. The context kept growing (29 messages by the 10th observed call) until the model lost track of what was still outstanding. Condensing alone fixed this — 3 calls, all tools hit. Both optimizations together: 5 calls, 508 total tokens, a **68% reduction** vs standard raw, at a third of the cost.

The `context sizes` column shows messages visible to each LLM call. Standard raw grows by 3 per round (assistant + 2 tool results). Condensed stays flat (+1 per round). Dynamic's first call always shows `2` (system + user, router only).

**Scenario 2 (long sequential chain)** is the clean apples-to-apples case — all modes complete all 6 expected tools in 9 calls. Savings are purely structural:
- Condensing alone: **−8% total tokens** (prompt shrinks, completions equalized)
- Dynamic routing alone: **−18% total tokens** (fewer schema tokens per call)
- Both together: **−28% total tokens** (savings compound independently)

The context size progression tells the story: standard raw ends at 20 messages, dynamic condensed ends at 10.

---

## Reference

| Parameter | Type | Default | Scope |
|---|---|---|---|
| `condense_tool_messages` | `bool` | `False` | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
| `tool_mode` | `"standard" \| "dynamic"` | `"standard"` | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
| `tool_route_model` | `str \| None` | `settings.tool_route_model` | `GlueLLM.__init__` only |
| `max_tokens` | `int \| None` | `None` (provider default) | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
| `summarize_context` | `bool` | `False` | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
| `summarize_context_threshold` | `int` | `20` | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
| `summarize_context_model` | `str \| None` | `None` (uses primary model) | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
| `summarize_context_keep_recent` | `int` | `6` | `GlueLLM.__init__`, `complete()`, `structured_complete()`, `stream_complete()` |
