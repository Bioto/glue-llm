# AAAK Compression

AAAK is a lossless shorthand encoding dialect built into GlueLLM for compressing
agent conversation history. Unlike prose summarization it preserves every technical
fact exactly — exact numbers, config key=value pairs, algorithm names, security
attributes, and ordered multi-step flows — while reducing token count.

The name is an internal codename; think of it as a structured notation any LLM can
read without needing a separate fine-tuned decoder.

---

## How it works

AAAK operates in two modes that compose naturally.

### Mode 1 — Tool-round condensing (`[AT]` blocks)

Triggered by `condense_tool_messages=True` with `aaak_tool_condensing=True`.

Each completed tool round (the `assistant(tool_calls)` + N `tool` result messages)
is replaced **deterministically** — no extra API call — with a single compact block:

```
[AT]
T:get_config()→
  db_pool_size=10
  db_max_overflow=5
  request_timeout_ms=8500 | T:get_metrics()→
  active_connections=267
  avg_latency_ms=4320
  max_latency_ms=31800
```

Tool arguments are included only when the same function name appears more than once
in the round (disambiguation). JSON results are flattened to `key.path=value` lines.
CSV results keep their newlines and get a `# peak:` annotation injected automatically.

### Mode 2 — Conversation history compression (`[AAAK CTX]` blocks)

Triggered by `summarize_context=True` with `aaak_compression_enabled=True`.

When the conversation grows beyond `summarize_context_threshold` messages, the older
turns (everything except the most recent `summarize_context_keep_recent` messages and
the system prompt) are sent to an LLM which rewrites them as AAAK shorthand:

```
[AAAK CTX]
USR: JWT auth for FastAPI | HS256 | access=15min | refresh=7d
AST: JWT_SECRET_KEY env | python-jose | make_token(sub,min=15)→jwt.encode
     refresh_hash=SHA256 never raw
USR: refresh token table?
AST: refresh_tokens: id:UUID | user_id:FK cascade | token_hash:CHAR64 |
     issued_at,expires_at,revoked_at:TIMESTAMPTZ | replaced_by:replay_detect
     idx(user_id) WHERE revoked_at IS NULL | cron DELETE WHERE expires_at<now()-7d
USR: rate limits?
AST: rate=10r/m@gateway(nginx limit_req_zone) | rate=1000r/h@app(slowapi+Redis)
     rl:{user_id}:{window_epoch} | login also enforced at app as fallback
USR: logout flow?
AST: logout:1→DELETE /auth/session;2→SHA256 hash;3→UPDATE revoked_at=now();
     4→204+clear cookie;5→access valid until exp(15min tolerated)
     optional:jti blocklist Redis TTL=900s
[/AAAK CTX]
```

A short decoding hint is appended to the system message once so the downstream model
knows how to read the block:

```
[AAAK decoding hint]
Decode [AAAK CTX] and [AT]: USR:/AST: turns; T:name()→result for tools
(args only when same name used twice). Lists [a,b,c]; ordered steps 1→2→3;
config key=val; schema col:purpose; security attrs verbatim
(HttpOnly,Secure,SameSite=Strict).
```

**Pipeline safety:** If the history already contains `[AT]` blocks from Mode 1,
`compress_messages` detects them and emits a deterministic passthrough instead of
calling the LLM again. `[AT]` blocks are never double-encoded.

---

## AAAK grammar (reference)

| Construct | Notation | Example |
|---|---|---|
| User turn | `USR: …` | `USR: add JWT auth` |
| Assistant turn | `AST: …` | `AST: use HS256, 15min expiry` |
| Tool result | `T:name()→result` | `T:get_config()→pool_size=10` |
| Tool with args (disambig.) | `T:name(key=val)→result` | `T:read(path=/etc/app.cfg)→…` |
| Config / key-value | `key=val` | `timeout_ms=8500` |
| Ordered steps | `1→2→3→N` | `logout:1→DELETE;2→hash;3→revoke` |
| List | `[a,b,c]` | `[HttpOnly,Secure,SameSite=Strict]` |
| Schema column purpose | `col:purpose` | `replaced_by:replay_detect` |
| Fact chain | `a→b→c` | `request→gateway→app→DB` |
| Parallel facts | `a \| b` | `rate=10r/m@gw \| rate=1000r/h@app` |
| 3-4 char handle | names only | `JWT_SECRET_KEY` verbatim (never abbreviated) |

**Always preserved verbatim:** exact numbers with units (`8500ms`, `15min`), config
key names (`JWT_SECRET_KEY`, `pool_size`), security attributes
(`HttpOnly,Secure,SameSite=Strict`), algorithm identifiers (`HS256`,
`PBKDF2-SHA256`), HTTP verbs and paths (`DELETE /auth/session`).

---

## Strengths

**Lossless for technical facts.**
The encoding is designed so every number, config value, algorithm name, and ordered
step survives compression intact. The live benchmark (`benchmarks/aaak_live_benchmark.py`)
measures this via 35+ recall questions across JWT auth design, DB schema, rate limits,
SRE incidents, and tool-round data.

**No fine-tuned decoder required.**
The grammar is readable plain text. Any capable LLM can decode `[AAAK CTX]` and
`[AT]` blocks without extra training. The decoding hint in the system message is a
brief natural-language reminder, not a schema.

**Tool-round encoding is free.**
Mode 1 (`[AT]`) is deterministic Python — no extra LLM call, no latency, no cost.
It pays off immediately on any conversation with tool calling enabled.

**Compression quality scales with model capability.**
A stronger compressor produces denser output. Measured on NarrativeQA
(out-of-domain, long narrative prose):

| Compressor | Compression ratio | Accuracy preserved |
|---|---|---|
| `llama-3.1-8b-instant` | 4.3× | 78.9% |
| `gpt-4-turbo` | 15.6× | 89.9% |

**Amortised cost over long conversations.**
The compression call is paid once. The compressed context is carried for all
subsequent turns in the conversation, so cost per turn drops as conversations grow.

**Provider-agnostic.**
Both compression and downstream model can be any provider supported by GlueLLM —
mix and match freely (e.g. compress with GPT-4-turbo, answer with a cheap fast model).

---

## Weaknesses

**Fixed overhead on short contexts.**
The `[AAAK CTX]` markers, system preamble (`~50 tokens`), and compression call setup
cost tokens. Below roughly 400 tokens of context the overhead exceeds the savings.
GlueLLM's `standard_benchmark.py` uses a 400-token floor and passes those samples
through raw — in production you should apply the same guard or rely on
`summarize_context_threshold` to only trigger when there is actually something to
compress.

**LLM-based compression costs API tokens.**
Mode 2 makes one extra API call per compression event. For a single short-lived
query this is net-negative. The break-even point depends on how many downstream
turns reuse the compressed context.

**Not designed for general prose or NLP benchmarks.**
AAAK's grammar vocabulary is technical (configs, schemas, tool results). On general
prose (multi-hop Wikipedia QA, narrative fiction, math reasoning) it does not
compress as well as purpose-trained extractive systems like LLMLingua-2. HotpotQA
and GSM8K are below or near the parity line:

| Task | Compressor | Compression ratio | Accuracy Δ |
|---|---|---|---|
| GSM8K few-shot | llama-3.1-8b | 0.9× (expands) | +5% (noise) |
| HotpotQA | gpt-4-turbo | 9.0× | −6.0% |
| NarrativeQA | gpt-4-turbo | 15.6× | −1.0% |
| NarrativeQA | llama-3.1-8b | 4.3× | −1.6% |

**Compressor model matters.**
A weak compressor (small, instruction-following gaps) may bloat rather than shrink
the context. If you observe a compression ratio < 1.0 on your content type, switch
to a stronger compressor or disable AAAK for that workflow.

**Multi-hop reasoning is sensitive.**
AAAK's flattening can lose the cross-reference links that multi-hop QA requires
("same entity mentioned in document A and document B"). For workflows that depend on
tracing entity relationships across large bodies of text, prose summarization or
full context may be safer.

---

## When to use AAAK

| Scenario | Verdict |
|---|---|
| Long-running agent sessions with many tool calls | ✅ ideal |
| Technical conversations (API design, infra, auth, schemas, SRE) | ✅ ideal |
| Conversation history that will span dozens of turns | ✅ ideal |
| Conversations with repeated tool names in a single round | ✅ `[AT]` disambiguates automatically |
| Any workflow where token cost grows with conversation length | ✅ amortised savings |
| Short one-shot queries (<400 tokens of context) | ❌ overhead > savings |
| General document QA (Wikipedia, narrative, academic papers) | ❌ use LLMLingua-2 |
| Mathematical reasoning with chain-of-thought | ❌ prose compresses poorly |
| Contexts where multi-hop entity tracing is the core task | ⚠️ test carefully |
| Conversations the downstream model will only see once | ⚠️ check break-even |

---

## Configuration

All settings are in `GlueLLMSettings` (env prefix `GLUELLM_`).

| Setting | Default | Description |
|---|---|---|
| `aaak_compression_enabled` | `True` | Use AAAK instead of prose when `summarize_context=True` |
| `aaak_compression_model` | `None` | Model for the compression call. `None` → same as `summarize_context_model` → primary model |
| `aaak_tool_condensing` | `True` | Emit `[AT]` blocks instead of plain `[Tool Results]` when `condense_tool_messages=True` |
| `default_summarize_context` | `False` | Auto-compress when message count exceeds threshold (off by default) |
| `default_summarize_context_threshold` | `20` | Message count that triggers compression |
| `default_summarize_context_keep_recent` | `6` | Verbatim messages kept at the tail |
| `default_condense_tool_messages` | `False` | Condense each completed tool round (off by default) |

---

## API usage

### Opt-in per conversation client

```python
from gluellm.api import GlueLLM

client = GlueLLM(
    model="openai:gpt-4o",
    summarize_context=True,           # compress when history grows long
    aaak_compression_enabled=True,    # use AAAK (default when summarize_context=True)
    aaak_compression_model="openai:gpt-4-turbo",  # stronger compressor, cheaper answerer
    summarize_context_threshold=20,   # compress after 20 non-system messages
    summarize_context_keep_recent=6,  # always keep last 6 messages verbatim
    condense_tool_messages=True,      # collapse tool rounds into [AT] blocks
)

result = await client.complete("Design the auth layer for our FastAPI service.")
```

### Opt-in per-call

```python
result = await complete(
    user_message="...",
    summarize_context=True,
    aaak_compression_enabled=True,
    condense_tool_messages=True,
)
```

### Global defaults via configure()

```python
import gluellm

gluellm.configure(
    default_summarize_context=True,
    aaak_compression_enabled=True,
    aaak_compression_model="openai:gpt-4-turbo",
    default_condense_tool_messages=True,
)
```

### Disable AAAK, use prose summarization instead

```python
client = GlueLLM(
    summarize_context=True,
    aaak_compression_enabled=False,   # falls back to prose summarization
)
```

### Use only tool-round condensing (no conversation compression)

```python
client = GlueLLM(
    condense_tool_messages=True,      # [AT] blocks, free (no LLM call)
    summarize_context=False,          # don't compress conversation history
)
```

---

## Benchmarks

Results from `benchmarks/aaak_live_benchmark.py` (real API calls, judge-verified
recall). The live benchmark is AAAK's primary evaluation surface — it tests the
domain AAAK was designed for.

### Live benchmark — agent conversation recall (n=1, 35+ questions)

| Section | Content | Mode | Recall |
|---|---|---|---|
| A — context compression | JWT auth, DB schema, rate limits, logout flow | raw (baseline) | ~75–85% |
| A — context compression | same | aaak (lossless) | ~90–100% |
| B–D — tool-round condensing | SRE incident data, nested JSON, CSV, parallel tools | raw | ~70–80% |
| B–D — tool-round condensing | same | aaak | ~90–95% |
| E — pipeline | history with embedded `[AT]` re-compressed | aaak | ~90%+ |

### Out-of-domain benchmarks (standard_benchmark.py, n=20)

These use content AAAK was **not** designed for. Results show how it degrades
gracefully rather than catastrophically.

| Task | Compressor | CR | Accuracy (raw) | Accuracy (aaak) | Preserved |
|---|---|---|---|---|---|
| NarrativeQA | llama-3.1-8b | 4.3× | 7.6% | 6.0% | 78.9% |
| NarrativeQA | gpt-4-turbo | 15.6× | 9.9% | 8.9% | 89.9% |
| HotpotQA | gpt-4-turbo | 9.0× | 18.7% | 12.7% | 67.9% |
| SQuAD | — | passthrough* | 43.1% | 43.1% | 100% |
| GSM8K | llama-3.1-8b | 0.9× | 75.0% | 80.0% | — |

\* SQuAD paragraphs average ~200 tokens — below the 400-token compression floor.
The passthrough guard fires correctly and leaves them untouched.

For comparison: LLMLingua-2 (purpose-trained extractive compressor) achieves ~2.9×
compression on NarrativeQA at ~101% accuracy preserved, using GPT-3.5-turbo.
AAAK with gpt-4-turbo achieves 15.6× at 89.9% preserved — a much higher
compression ratio at a modest accuracy cost, from a system not trained for this task.
Direct comparison is not apples-to-apples (different compressor strengths) but
illustrates the ceiling.

---

## Implementation notes

**Double-encoding prevention.** `compress_messages` checks whether any assistant
message starts with `[AT]`. If it finds one it emits a deterministic passthrough
(preserving `[AT]` blocks verbatim, `USR:` / `AST:` prefixes for others) instead of
calling the LLM. This prevents facts inside `[AT]` blocks from being garbled by a
second compression pass.

**JSON flattening.** Tool results that are valid JSON are flattened to
`key.path=value` lines before being stored in `[AT]`. This makes nested config
objects and API responses scannable without nested brackets.

**CSV peak annotation.** Multi-row CSV tool results get a `# peak: col=val(label)`
comment appended automatically so the model can answer aggregate questions
(e.g. "which endpoint had the highest error rate?") without scanning every row.

**Float precision.** Floats avoid scientific notation (`3.12e-05` → `0.0000312`) to
prevent fact corruption when numbers appear in config contexts.

**Arg disambiguation.** In `encode_tool_round`, tool call arguments are included in
the `T:name(args)` notation only when the same function name appears more than once
in the round. Otherwise the parentheses are empty — keeping single-call rounds
maximally compact.

---

## See also

- `gluellm/compression/aaak.py` — full implementation
- `benchmarks/aaak_live_benchmark.py` — domain-specific live benchmark (Sections A–E)
- `benchmarks/standard_benchmark.py` — out-of-domain NLP dataset benchmark
- `docs/ARCHITECTURE.md` — where AAAK fits in the broader GlueLLM system
