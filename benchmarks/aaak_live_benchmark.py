"""AAAK Live Benchmark — real LLM calls, no mocking.
=====================================================
Five benchmark sections, same judge infrastructure.

  SECTION A (primary stress test for LLM context compression):
    raw          Full conversation — no compression (baseline)
    prose        _summarize_old_messages, prose summary (lossy)
    aaak_ctx     _summarize_old_messages, AAAK encoding (lossless; prompts tuned for
                 rate limits + layer, schema columns, cookie attrs, ordered steps)

  SECTION B: Tool-round condensing
    raw_tools    Full tool messages — no condensing (baseline)
    plain        _condense_tool_round, [Tool Results] text (lossy)
    aaak_tools   _condense_tool_round, [AT] encoding (lossless)

  SECTION C: Hard tool-round condensing (nested JSON, arrays, markdown, CSV, disambiguation)
    hard_raw     Full tool messages — no condensing (baseline)
    hard_plain   Plain condensed
    hard_aaak    AAAK condensed

  SECTION D: Edge tool rounds (many parallel tools, truncation, stack trace, numeric precision)
    edge_raw     Full tool messages — no condensing (baseline)
    edge_plain   Plain condensed
    edge_aaak    AAAK condensed

  SECTION E: Pipeline (context compression over history that already contains [AT] blocks)
    pipe_raw     No context compression (baseline)
    pipe_aaak    Old turns (including embedded [AT]) compressed into [AAAK CTX]

Requires OPENAI_API_KEY in environment.

Run:
    set -a && source ../.env && set +a
    uv run python benchmarks/aaak_live_benchmark.py

Options:
    --trials N              Run the full benchmark N times; print mean/min/max recall and
                            per-question pass rates (default: 1).
    --only-section-a/b/c/d/e
                            Run only one section.
    --concurrency N         Max in-flight API calls across all parallel recall/judge/compress
                            tasks (default: 10). Sections B-E run concurrently; within B/C/D all
                            modes run in parallel; Section A and E modes run sequentially (they
                            patch global provider state). All recall questions within a mode run
                            in parallel, bounded by this semaphore.
    --verbose-section-a     After aaak_ctx, print [AAAK CTX] body and recall failures
                            (answer + judge input) for debugging compression.
    --no-deterministic-sampling
                            Do not pass temperature=0, top_p=1 to the API (some models reject it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import tiktoken

from gluellm.api import _condense_tool_round, _summarize_old_messages
from gluellm.compression.aaak import (
    AAAK_PREAMBLE_MARKER,
    AAAKCompressor,
    _COMPRESS_SYSTEM,
    _COMPRESS_USER_PREFIX,
    transcript_from_messages,
)

RecallQuestion = dict[str, str | list[str]]

_ENC = tiktoken.encoding_for_model("gpt-4o")

MODEL = "groq:llama-3.1-8b-instant"

# Benchmark-only: merged into provider.acompletion for judge, recall answers, prose summarize,
# and AAAK compress. Overridable via --no-deterministic-sampling.
_benchmark_completion_extra: dict[str, Any] = {"temperature": 0, "top_p": 1}
_benchmark_verbose_section_a: bool = False

# Semaphore caps in-flight provider calls across all concurrent recall/judge/compress tasks.
# Initialized in main_async() from --concurrency N (default 10).
_benchmark_semaphore: asyncio.Semaphore | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            total += count_tokens(" ".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            ))
        else:
            total += count_tokens(str(content))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += count_tokens(str(fn.get("name", "")))
            total += count_tokens(str(fn.get("arguments", "")))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Section A fixture — multi-turn technical conversation
# ─────────────────────────────────────────────────────────────────────────────

CTX_CONVERSATION: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": "You are a senior engineer helping design a Python microservice.",
    },
    {
        "role": "user",
        "content": "We need to add JWT auth to our FastAPI service. Our team decided to use HS256 with a 15-minute access token expiry and 7-day refresh tokens.",
    },
    {
        "role": "assistant",
        "content": (
            "Good choices. For HS256 you'll need a strong secret key — at least 32 bytes, stored in "
            "an environment variable called JWT_SECRET_KEY. Use python-jose for encoding/decoding. "
            "Here's the core:\n\n"
            "```python\nfrom jose import jwt\nfrom datetime import datetime, timedelta\n\n"
            "SECRET = os.getenv('JWT_SECRET_KEY')\nALGO = 'HS256'\n\n"
            "def make_token(sub: str, minutes: int = 15) -> str:\n"
            "    return jwt.encode({'sub': sub, 'exp': datetime.utcnow() + timedelta(minutes=minutes)},\n"
            "                      SECRET, algorithm=ALGO)\n```\n\n"
            "For refresh tokens, store a SHA-256 hash of the raw token in your DB, never the raw value."
        ),
    },
    {
        "role": "user",
        "content": "We'll store refresh tokens in a PostgreSQL table. What columns does it need?",
    },
    {
        "role": "assistant",
        "content": (
            "Minimum viable schema:\n\n"
            "```sql\nCREATE TABLE refresh_tokens (\n"
            "    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n"
            "    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,\n"
            "    token_hash  CHAR(64) NOT NULL UNIQUE,  -- SHA-256 hex\n"
            "    issued_at   TIMESTAMPTZ NOT NULL DEFAULT now(),\n"
            "    expires_at  TIMESTAMPTZ NOT NULL,\n"
            "    revoked_at  TIMESTAMPTZ,\n"
            "    replaced_by UUID REFERENCES refresh_tokens(id)\n"
            ");\nCREATE INDEX ON refresh_tokens(user_id) WHERE revoked_at IS NULL;\n```\n\n"
            "The `replaced_by` column enables token-family tracking: if a client uses a revoked "
            "refresh token you can revoke the entire family (detect replay attacks). "
            "Add a cron job to DELETE rows WHERE expires_at < now() - interval '7 days'."
        ),
    },
    {
        "role": "user",
        "content": "Our rate limits: 10 req/min on /auth/login, 1000 req/hour per authenticated user. Where do we enforce these?",
    },
    {
        "role": "assistant",
        "content": (
            "Two-layer approach:\n\n"
            "1. **Gateway layer** (nginx/Traefik): IP-based limit on /auth/login — 10 req/min. "
            "This blocks bots before they hit your app. nginx config: "
            "`limit_req_zone $binary_remote_addr zone=login:10m rate=10r/m;`\n\n"
            "2. **App layer** (slowapi + Redis): per-user 1000 req/hour for authenticated endpoints. "
            "Use the authenticated user ID as the key, not the IP. "
            "Redis key pattern: `rl:{user_id}:{window_start_epoch}`.\n\n"
            "Important: the 10 req/min login limit should also be enforced at the app layer as a "
            "fallback in case someone bypasses the gateway."
        ),
    },
    {
        "role": "user",
        "content": "For the /auth/login endpoint itself, what should the response body look like on success?",
    },
    {
        "role": "assistant",
        "content": (
            "Standard OAuth2 response:\n\n"
            "```json\n{\n"
            '  "access_token": "<jwt>",\n'
            '  "token_type": "bearer",\n'
            '  "expires_in": 900\n'
            "}\n```\n\n"
            "Send the refresh token as an HttpOnly, Secure, SameSite=Strict cookie — "
            "never in the JSON body. Cookie name: `refresh_token`. "
            "Set Max-Age to 604800 (7 days in seconds). "
            "The `expires_in` field is 900 (15 minutes × 60 seconds)."
        ),
    },
    {
        "role": "user",
        "content": "We need to handle token revocation when a user logs out. Walk me through the exact flow.",
    },
    {
        "role": "assistant",
        "content": (
            "Logout flow (5 steps):\n\n"
            "1. Client sends DELETE /auth/session with the HttpOnly refresh_token cookie.\n"
            "2. App extracts the cookie value, computes SHA-256 hash.\n"
            "3. App UPDATE refresh_tokens SET revoked_at = now() WHERE token_hash = $1 AND revoked_at IS NULL.\n"
            "4. App responds 204 No Content and clears the cookie (Set-Cookie: refresh_token=; Max-Age=0).\n"
            "5. Access tokens remain valid until their 15-minute expiry — tolerated because the window is short.\n\n"
            "If you need immediate access token invalidation, add a Redis blocklist keyed on the "
            "JWT `jti` claim (a UUID you include at token creation). Check the blocklist on every "
            "authenticated request. TTL = access token expiry = 900 seconds."
        ),
    },
    # kept verbatim (recent 2)
    {
        "role": "user",
        "content": "Last thing: our CI/CD pipeline. We use GitHub Actions. What secrets do we need configured?",
    },
    {
        "role": "assistant",
        "content": (
            "For this service you need these GitHub Actions secrets:\n\n"
            "- JWT_SECRET_KEY: your production signing key (generate with `openssl rand -hex 32`)\n"
            "- DATABASE_URL: postgres://user:pass@host:5432/dbname\n"
            "- REDIS_URL: redis://host:6379\n\n"
            "Never commit these. For staging vs production, use GitHub Environments with separate "
            "secret scopes. Reference them in your workflow: `${{ secrets.JWT_SECRET_KEY }}`."
        ),
    },
]

CTX_RECALL_QUESTIONS: list[RecallQuestion] = [
    {
        "question": "What hashing algorithm did we decide to use for JWTs, and what is the access token expiry in minutes?",
        "key_facts": ["HS256", "15 minutes"],
    },
    {
        "question": "What is the exact name of the environment variable for the JWT secret key?",
        "key_facts": ["JWT_SECRET_KEY"],
    },
    {
        "question": "What column in the refresh_tokens table enables token-family replay attack detection?",
        "key_facts": ["replaced_by"],
    },
    {
        "question": "What is the rate limit on the /auth/login endpoint, and at which layer is it primarily enforced?",
        "key_facts": ["10 req/min", "gateway/nginx"],
    },
    {
        "question": "What cookie attributes should the refresh token cookie have?",
        "key_facts": ["HttpOnly", "Secure", "SameSite=Strict"],
    },
    {
        "question": "What are the exact steps for the logout flow?",
        "key_facts": ["DELETE /auth/session", "SHA-256 hash", "UPDATE revoked_at", "204", "clear cookie"],
    },
    {
        "question": "What is the expires_in value in the /auth/login success response, and what does it represent?",
        "key_facts": ["900", "15 minutes in seconds"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Section B fixture — multi-round tool execution (production incident debug)
# Every fact embedded in args and results is probed by a recall question.
# ─────────────────────────────────────────────────────────────────────────────

# Four tool rounds an agent ran while debugging a production incident.
# Each round: (tool_calls list, tool_responses list)
TOOL_ROUNDS: list[tuple[list[dict], list[dict]]] = [
    # Round 1 — search logs
    (
        [
            {
                "id": "c1a",
                "type": "function",
                "function": {
                    "name": "search_logs",
                    "arguments": json.dumps({"service": "auth-service", "level": "ERROR", "last_minutes": 60}),
                },
            },
            {
                "id": "c1b",
                "type": "function",
                "function": {
                    "name": "search_logs",
                    "arguments": json.dumps({"service": "payment-service", "level": "ERROR", "last_minutes": 60}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "c1a",
                "content": json.dumps({
                    "count": 847,
                    "top_error": "ConnectionPoolExhausted",
                    "first_seen": "2024-01-15T14:23:11Z",
                }),
            },
            {
                "role": "tool",
                "tool_call_id": "c1b",
                "content": json.dumps({
                    "count": 12,
                    "top_error": "Timeout waiting for auth-service",
                    "first_seen": "2024-01-15T14:23:45Z",
                }),
            },
        ],
    ),
    # Round 2 — read config
    (
        [
            {
                "id": "c2a",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "/etc/config/auth-service.yaml"}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "c2a",
                "content": (
                    "db_pool_size: 10\n"
                    "db_max_overflow: 5\n"
                    "db_pool_timeout: 30\n"
                    "max_connections: 250\n"
                    "request_timeout_ms: 8500\n"
                    "db_host: pg-primary.internal\n"
                    "redis_host: redis-cluster.internal\n"
                ),
            },
        ],
    ),
    # Round 3 — query database
    (
        [
            {
                "id": "c3a",
                "type": "function",
                "function": {
                    "name": "run_query",
                    "arguments": json.dumps({
                        "sql": "SELECT COUNT(*) as cnt FROM active_connections WHERE service='auth-service'",
                        "database": "monitoring",
                    }),
                },
            },
            {
                "id": "c3b",
                "type": "function",
                "function": {
                    "name": "run_query",
                    "arguments": json.dumps({
                        "sql": "SELECT AVG(duration_ms) as avg_ms, MAX(duration_ms) as max_ms FROM request_log WHERE service='auth-service' AND created_at > NOW() - INTERVAL '10 minutes'",
                        "database": "monitoring",
                    }),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "c3a",
                "content": json.dumps({"cnt": 267}),
            },
            {
                "role": "tool",
                "tool_call_id": "c3b",
                "content": json.dumps({"avg_ms": 4320, "max_ms": 31800}),
            },
        ],
    ),
    # Round 4 — check env vars and metrics
    (
        [
            {
                "id": "c4a",
                "type": "function",
                "function": {
                    "name": "get_env",
                    "arguments": json.dumps({"names": ["DB_POOL_SIZE", "DB_MAX_OVERFLOW", "WORKER_THREADS"]}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "c4a",
                "content": json.dumps({
                    "DB_POOL_SIZE": "10",
                    "DB_MAX_OVERFLOW": "5",
                    "WORKER_THREADS": "32",
                }),
            },
        ],
    ),
]

# The surrounding conversation: system + user prompt + assistant summary + recent
TOOL_SYSTEM = {"role": "system", "content": "You are an SRE agent diagnosing a production incident."}
TOOL_USER_PROMPT = {"role": "user", "content": "Auth service is throwing 503s. Investigate and summarise your findings."}
TOOL_ASSISTANT_SUMMARY = {
    "role": "assistant",
    "content": (
        "Investigation complete. Root cause: auth-service DB connection pool exhaustion. "
        "Pool size is 10 (config) / 10 (env), max_overflow 5, but 267 active connections "
        "observed — well above the pool limit. Average request latency is 4320ms, max 31800ms, "
        "causing cascading timeouts into payment-service (12 errors downstream). "
        "Recommendation: raise DB_POOL_SIZE to 50 and DB_MAX_OVERFLOW to 20 immediately."
    ),
}
TOOL_RECENT = [
    {"role": "user", "content": "What should our immediate remediation steps be?"},
    {
        "role": "assistant",
        "content": (
            "Immediate steps (in order):\n"
            "1. Set DB_POOL_SIZE=50 and DB_MAX_OVERFLOW=20 and rolling-restart auth-service.\n"
            "2. Monitor active_connections — should drop below 55 within 2 minutes.\n"
            "3. Verify 503 rate drops in payment-service logs.\n"
            "4. Schedule post-incident review to add connection pool metrics to alerting."
        ),
    },
]

TOOL_RECALL_QUESTIONS: list[RecallQuestion] = [
    {
        "question": "How many ERROR logs did the auth-service produce in the last 60 minutes, and what was the top error?",
        "key_facts": ["847", "ConnectionPoolExhausted"],
    },
    {
        "question": "How many ERROR logs did the payment-service produce, and what was its top error message?",
        "key_facts": ["12", "Timeout waiting for auth-service"],
    },
    {
        "question": "What are the db_pool_size and db_max_overflow values from the auth-service config file?",
        "key_facts": ["db_pool_size: 10", "db_max_overflow: 5"],
    },
    {
        "question": "What was the request_timeout_ms value in the auth-service config?",
        "key_facts": ["8500"],
    },
    {
        "question": "How many active database connections were observed for auth-service, and what database was queried?",
        "key_facts": ["267", "monitoring"],
    },
    {
        "question": "What were the average and maximum request durations in milliseconds for auth-service over the last 10 minutes?",
        "key_facts": ["avg 4320ms", "max 31800ms"],
    },
    {
        "question": "What were the actual environment variable values for DB_POOL_SIZE, DB_MAX_OVERFLOW, and WORKER_THREADS?",
        "key_facts": ["DB_POOL_SIZE=10", "DB_MAX_OVERFLOW=5", "WORKER_THREADS=32"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Section C fixture — harder tool rounds (nested JSON, tables, CSV, disambiguation)
# ─────────────────────────────────────────────────────────────────────────────

_HARD_GATEWAY_CONFIG = {
    "service": "api-gateway",
    "routes": {
        "/api/v2/search": {"burst": 120, "window_ms": 1000, "strategy": "token_bucket"},
        "/api/v1/login": {"burst": 20, "window_ms": 60000, "strategy": "sliding_window"},
        "/health": {"burst": 500, "window_ms": 1000, "strategy": "token_bucket"},
    },
    "tls": {
        "cert_path": "/etc/ssl/gateway.crt",
        "key_path": "/etc/ssl/gateway.key",
        "cipher_suites": [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
        ],
    },
}

_HARD_DEPLOYMENTS = [
    {
        "version": "2.4.1",
        "sha": "a1b2c3d4e5f6",
        "deployed_by": "alice",
        "timestamp": "2025-03-01T10:00:00Z",
        "status": "success",
        "rollback_of": None,
    },
    {
        "version": "2.4.2",
        "sha": "e5f6a7b8c9d0",
        "deployed_by": "bob",
        "timestamp": "2025-03-02T14:30:00Z",
        "status": "success",
        "rollback_of": None,
    },
    {
        "version": "2.4.0",
        "sha": "f0e1d2c3b4a5",
        "deployed_by": "carol",
        "timestamp": "2025-03-03T09:15:00Z",
        "status": "rollback",
        "rollback_of": "2.4.1",
    },
    {
        "version": "2.4.3",
        "sha": "9z8y7x6w5v4u",
        "deployed_by": "alice",
        "timestamp": "2025-03-04T11:00:00Z",
        "status": "success",
        "rollback_of": None,
    },
    {
        "version": "2.4.4",
        "sha": "1q2w3e4r5t6y",
        "deployed_by": "dave",
        "timestamp": "2025-03-05T16:45:00Z",
        "status": "success",
        "rollback_of": None,
    },
]

_HARD_COMPARE_MD = (
    "Service comparison (last 1h window):\n\n"
    "| service | p50_ms | p99_ms | error_rate | pod_count | cpu_request | memory_limit |\n"
    "|---------|--------|--------|------------|-----------|-------------|-------------|\n"
    "| auth | 12 | 89 | 0.001 | 8 | 500m | 1Gi |\n"
    "| gateway | 5 | 45 | 0.008 | 12 | 250m | 512Mi |\n"
    "| billing | 25 | 210 | 0.012 | 4 | 1000m | 2Gi |\n"
)

_HARD_CSV_METRICS = (
    "timestamp,service,rps,error_pct,latency_p50,latency_p99\n"
    "2025-04-01T10:00:00Z,auth,1200,0.2,11,85\n"
    "2025-04-01T10:00:00Z,gateway,3400,0.8,4,38\n"
    "2025-04-01T10:00:00Z,billing,890,1.2,22,195\n"
    "2025-04-01T11:00:00Z,auth,1180,0.3,12,88\n"
    "2025-04-01T11:00:00Z,gateway,3600,1.8,5,42\n"
    "2025-04-01T11:00:00Z,billing,920,0.9,24,205\n"
    "2025-04-01T12:00:00Z,auth,1250,0.1,10,82\n"
    "2025-04-01T12:00:00Z,gateway,3300,0.5,4,40\n"
    "2025-04-01T12:00:00Z,billing,900,0.7,23,198\n"
    "2025-04-01T13:00:00Z,auth,1190,0.4,11,90\n"
)

HARD_TOOL_ROUNDS: list[tuple[list[dict], list[dict]]] = [
    # Round 1 — nested JSON (API gateway)
    (
        [
            {
                "id": "h1",
                "type": "function",
                "function": {
                    "name": "get_config",
                    "arguments": json.dumps({"component": "api-gateway", "region": "us-east-1"}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "h1",
                "content": json.dumps(_HARD_GATEWAY_CONFIG, ensure_ascii=False),
            },
        ],
    ),
    # Round 2 — JSON array of deployment objects
    (
        [
            {
                "id": "h2",
                "type": "function",
                "function": {
                    "name": "list_deployments",
                    "arguments": json.dumps({"service": "auth", "limit": 5}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "h2",
                "content": json.dumps(_HARD_DEPLOYMENTS, ensure_ascii=False),
            },
        ],
    ),
    # Round 3 — markdown table
    (
        [
            {
                "id": "h3",
                "type": "function",
                "function": {
                    "name": "compare_services",
                    "arguments": json.dumps({"services": ["auth", "gateway", "billing"]}),
                },
            },
        ],
        [
            {"role": "tool", "tool_call_id": "h3", "content": _HARD_COMPARE_MD},
        ],
    ),
    # Round 4 — CSV export
    (
        [
            {
                "id": "h4",
                "type": "function",
                "function": {
                    "name": "export_metrics",
                    "arguments": json.dumps({"format": "csv", "hours": 6}),
                },
            },
        ],
        [
            {"role": "tool", "tool_call_id": "h4", "content": _HARD_CSV_METRICS},
        ],
    ),
    # Round 5 — parallel configs (staging vs production disambiguation)
    (
        [
            {
                "id": "h5a",
                "type": "function",
                "function": {
                    "name": "get_config",
                    "arguments": json.dumps({"service": "auth-service", "env": "staging"}),
                },
            },
            {
                "id": "h5b",
                "type": "function",
                "function": {
                    "name": "get_config",
                    "arguments": json.dumps({"service": "auth-service", "env": "production"}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "h5a",
                "content": (
                    "max_connections: 50\n"
                    "pool_timeout: 10\n"
                    "replicas: 2\n"
                ),
            },
            {
                "role": "tool",
                "tool_call_id": "h5b",
                "content": (
                    "max_connections: 500\n"
                    "pool_timeout: 30\n"
                    "replicas: 12\n"
                ),
            },
        ],
    ),
    # Round 6 — prose log analysis
    (
        [
            {
                "id": "h6",
                "type": "function",
                "function": {
                    "name": "analyze_logs",
                    "arguments": json.dumps({"query": "OOM", "hours": 24}),
                },
            },
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "h6",
                "content": (
                    "Found 23 OOM kills across 4 pods. Pod auth-7f8a peaked at 2.1Gi RSS at "
                    "03:47 UTC (limit 2Gi). Most frequent caller: `/api/v2/users/bulk` "
                    "(18 of 23 kills). Heap dump SHA: a3f8c91d."
                ),
            },
        ],
    ),
]

HARD_TOOL_SYSTEM = {
    "role": "system",
    "content": "You are a platform engineer auditing infrastructure and deployment data.",
}
HARD_TOOL_USER_PROMPT = {
    "role": "user",
    "content": "Pull gateway config, deployment history, service comparison, metrics export, env configs, and OOM analysis. Summarise risks.",
}
# Summary must not repeat judge key_facts — otherwise Section C measures summary reading, not tool encoding.
HARD_TOOL_ASSISTANT_SUMMARY = {
    "role": "assistant",
    "content": (
        "Audit summary: Reviewed api-gateway JSON config, auth deployment history (JSON array), "
        "markdown service comparison, CSV metrics export, parallel staging vs production YAML "
        "configs for auth-service, and OOM analysis prose. Overall: rate limits and TLS on the "
        "gateway, rollout/rollback history, SLO table vs CSV trends, env skew between stages, "
        "and memory pressure themes — details are only in the tool outputs above."
    ),
}
HARD_TOOL_RECENT = [
    {"role": "user", "content": "Which single change should we prioritize before tomorrow's release?"},
    {
        "role": "assistant",
        "content": (
            "Prioritize whatever the comparison table and CSV show as the worst reliability hotspot, "
            "then validate gateway limits under load and reconcile pool sizing across environments — "
            "use the tool results for exact numbers."
        ),
    },
]

HARD_TOOL_RECALL_QUESTIONS: list[RecallQuestion] = [
    {
        "question": (
            "For the api-gateway config tool result, what is the burst limit and strategy "
            "for the /api/v2/search route?"
        ),
        "key_facts": ["120", "token_bucket"],
    },
    {
        "question": "Name one TLS cipher suite listed in the api-gateway TLS configuration.",
        "key_facts": ["TLS_AES_256_GCM_SHA384"],
    },
    {
        "question": (
            "Which deployment version was a rollback, and which prior version did it roll back from?"
        ),
        "key_facts": ["2.4.0", "2.4.1"],
    },
    {
        "question": "What is the git SHA for deployment version 2.4.2?",
        "key_facts": ["e5f6a7b8c9d0"],
    },
    {
        "question": "In the markdown service comparison table, what is billing's p99 latency in ms?",
        "key_facts": ["210"],
    },
    {
        "question": (
            "In the same comparison table, which service has the highest error_rate and what is that rate?"
        ),
        "key_facts": ["billing", "0.012"],
    },
    {
        "question": (
            "In the CSV metrics export at 2025-04-01T10:00:00Z, what are rps and error_pct for billing?"
        ),
        "key_facts": ["890", "1.2"],
    },
    {
        "question": (
            "Across all rows in the CSV export, which service and hour (timestamp hour only, e.g. 11:00) "
            "had the peak error_pct, and what was that percentage?"
        ),
        "key_facts": ["gateway", "11:00", "1.8"],
    },
    {
        "question": (
            "What are max_connections for auth-service in staging versus production from the parallel "
            "get_config tool results?"
        ),
        "key_facts": ["50", "500"],
    },
    {
        "question": (
            "From the OOM log analysis: what pod name peaked in RSS, what was the peak RSS value, "
            "and what is the heap dump SHA prefix?"
        ),
        "key_facts": ["auth-7f8a", "2.1Gi", "a3f8c91d"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Section D — edge tool rounds (many tools, truncation, plain text, numerics)
# ─────────────────────────────────────────────────────────────────────────────

_EDGE_LATENCY = {"p50": 45, "p95": 210, "p99": 890, "max": 3400}
_EDGE_ERRORS = {"service": "billing", "rate": 0.0128, "total": 42390, "errors": 542}
_EDGE_DB = {"connections": 47, "slow_queries": 12, "deadlocks": 0, "cache_hit_pct": 94.7}
_EDGE_QUEUE = {"queue": "payment-events", "depth": 18923, "lag_ms": 4500, "consumers": 8}
_EDGE_CACHE = {"hit_rate": 0.9831, "evictions": 7412, "memory_mb": 2048, "keys": 1240983}

_EDGE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/app/migrate.py", line 84, in run\n'
    "    conn.execute(ALTER_TABLE_SQL)\n"
    "psycopg2.errors.LockNotAvailable: could not obtain lock on relation \"users\"\n"
    "Process exited with code 1"
)
_EDGE_SYSTEMD = (
    "● auth-service.service - Auth API\n"
    "   Active: failed (Result: exit-code) since 2025-04-01 14:32:11 UTC\n"
    "   Process: ExecStart=/usr/bin/python3 -m auth.app (code=exited, status=1)"
)

_EDGE_FINANCIALS = {
    "revenue_usd": 1234567.89,
    "fee_rate": 0.00285,
    "chargeback_rate": 0.0000312,
    "avg_transaction_usd": 47.32,
    "yoy_growth_pct": 23.7,
}


def _build_edge_audit_log_json() -> str:
    """JSON array >2000 chars; critical facts in first two entries (within truncation window)."""
    entries: list[dict[str, Any]] = [
        {"ts": "2025-04-01T09:00:00Z", "user": "svc-deploy", "action": "deploy", "version": "2.4.2"},
        {"ts": "2025-04-01T09:01:15Z", "user": "admin-rsk", "action": "approve", "target": "payment"},
    ]
    for i in range(18):
        entries.append({
            "ts": f"2025-04-01T{10 + i // 6:02d}:{(i * 3) % 60:02d}:00Z",
            "user": f"batch-reader-{i}",
            "action": "read",
            "resource": f"/api/v2/ledger/segment-{i}/" + "x" * 80,
        })
    return json.dumps(entries, separators=(",", ":"))


_EDGE_AUDIT_LOG_JSON = _build_edge_audit_log_json()

EDGE_TOOL_ROUNDS: list[tuple[list[dict], list[dict]]] = [
    # Round 1 — five parallel tools (unique names)
    (
        [
            {
                "id": "e1",
                "type": "function",
                "function": {"name": "get_latency_percentiles", "arguments": "{}"},
            },
            {
                "id": "e2",
                "type": "function",
                "function": {"name": "get_error_summary", "arguments": "{}"},
            },
            {
                "id": "e3",
                "type": "function",
                "function": {"name": "get_db_health", "arguments": "{}"},
            },
            {
                "id": "e4",
                "type": "function",
                "function": {"name": "get_queue_stats", "arguments": "{}"},
            },
            {
                "id": "e5",
                "type": "function",
                "function": {"name": "get_cache_stats", "arguments": "{}"},
            },
        ],
        [
            {"role": "tool", "tool_call_id": "e1", "content": json.dumps(_EDGE_LATENCY, separators=(",", ":"))},
            {"role": "tool", "tool_call_id": "e2", "content": json.dumps(_EDGE_ERRORS, separators=(",", ":"))},
            {"role": "tool", "tool_call_id": "e3", "content": json.dumps(_EDGE_DB, separators=(",", ":"))},
            {"role": "tool", "tool_call_id": "e4", "content": json.dumps(_EDGE_QUEUE, separators=(",", ":"))},
            {"role": "tool", "tool_call_id": "e5", "content": json.dumps(_EDGE_CACHE, separators=(",", ":"))},
        ],
    ),
    # Round 2 — large JSON (truncation path in _format_tool_result)
    (
        [
            {
                "id": "e6",
                "type": "function",
                "function": {"name": "export_audit_log", "arguments": json.dumps({"hours": 24})},
            },
        ],
        [
            {"role": "tool", "tool_call_id": "e6", "content": _EDGE_AUDIT_LOG_JSON},
        ],
    ),
    # Round 3 — plain text (traceback + systemd)
    (
        [
            {
                "id": "e7",
                "type": "function",
                "function": {"name": "run_migration", "arguments": "{}"},
            },
            {
                "id": "e8",
                "type": "function",
                "function": {"name": "systemctl_status", "arguments": json.dumps({"unit": "auth-service"})},
            },
        ],
        [
            {"role": "tool", "tool_call_id": "e7", "content": _EDGE_TRACEBACK},
            {"role": "tool", "tool_call_id": "e8", "content": _EDGE_SYSTEMD},
        ],
    ),
    # Round 4 — numeric precision (JSON)
    (
        [
            {
                "id": "e9",
                "type": "function",
                "function": {"name": "get_financial_metrics", "arguments": json.dumps({"quarter": "Q1"})},
            },
        ],
        [
            {"role": "tool", "tool_call_id": "e9", "content": json.dumps(_EDGE_FINANCIALS, separators=(",", ":"))},
        ],
    ),
]

EDGE_TOOL_SYSTEM = {
    "role": "system",
    "content": "You are an SRE reviewing production telemetry, migrations, and finance exports.",
}
EDGE_TOOL_USER_PROMPT = {
    "role": "user",
    "content": (
        "Pull latency percentiles, error summary, DB health, queue and cache stats, audit export, "
        "migration failure output, unit status, and quarterly financial metrics."
    ),
}
EDGE_TOOL_ASSISTANT_SUMMARY = {
    "role": "assistant",
    "content": (
        "Gathered five parallel metric tools, a large audit JSON export, migration traceback plus "
        "systemd status, and financial JSON. Exact numbers live only in the tool outputs above."
    ),
}
EDGE_TOOL_RECENT = [
    {"role": "user", "content": "Which area needs the deepest dive before the exec review?"},
    {
        "role": "assistant",
        "content": (
            "Prioritise whichever slice shows the worst tail latency or error concentration versus "
            "finance sensitivity — cite exact values from the tool results."
        ),
    },
]

EDGE_TOOL_RECALL_QUESTIONS: list[RecallQuestion] = [
    {
        "question": "From get_latency_percentiles, what is the p99 latency in milliseconds?",
        "key_facts": ["890"],
    },
    {
        "question": (
            "From get_error_summary for billing: what is the error rate (decimal) and how many errors out of total?"
        ),
        "key_facts": ["0.0128", "542", "42390"],
    },
    {
        "question": "From get_db_health: how many slow_queries and what is cache_hit_pct?",
        "key_facts": ["12", "94.7"],
    },
    {
        "question": "From get_queue_stats: what is queue depth and lag_ms for payment-events?",
        "key_facts": ["18923", "4500"],
    },
    {
        "question": "From get_cache_stats: how many keys and what is the hit_rate (decimal)?",
        "key_facts": ["1240983", "0.9831"],
    },
    {
        "question": (
            "From the audit log export: what action and version appear in the earliest entry (09:00:00Z), "
            "and which user approved the payment target?"
        ),
        "key_facts": ["deploy", "2.4.2", "admin-rsk"],
    },
    {
        "question": (
            "From run_migration: what exception type occurred, on which source line number, "
            "and what process exit code?"
        ),
        "key_facts": ["LockNotAvailable", "84", "1"],
    },
    {
        "question": "From systemctl_status: at what UTC timestamp did auth-service fail?",
        "key_facts": ["2025-04-01 14:32:11"],
    },
    {
        "question": "From get_financial_metrics: what is revenue_usd (exact) and fee_rate (exact decimal)?",
        "key_facts": ["1234567.89", "0.00285"],
    },
    {
        "question": "From get_financial_metrics: what are chargeback_rate and yoy_growth_pct?",
        "key_facts": ["0.0000312", "23.7"],
    },
    {
        "question": "From get_financial_metrics: what is avg_transaction_usd?",
        "key_facts": ["47.32"],
    },
    {
        "question": "From run_migration: what relation name could not be locked?",
        "key_facts": ["users"],
    },
]


def _build_tool_messages_raw() -> list[dict]:
    """Full message list with every tool round as proper assistant+tool messages."""
    msgs: list[dict] = [TOOL_SYSTEM, TOOL_USER_PROMPT]
    for tool_calls, tool_responses in TOOL_ROUNDS:
        msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        msgs.extend(tool_responses)
    msgs.append(TOOL_ASSISTANT_SUMMARY)
    msgs.extend(TOOL_RECENT)
    return msgs


def _build_tool_messages_condensed(*, aaak: bool) -> list[dict]:
    """Message list with each tool round condensed (plain or AAAK)."""
    msgs: list[dict] = [TOOL_SYSTEM, TOOL_USER_PROMPT]
    for tool_calls, tool_responses in TOOL_ROUNDS:
        assistant_msg: dict = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        round_msgs = [*msgs, assistant_msg, *tool_responses]
        _condense_tool_round(round_msgs, aaak_tool_condensing=aaak)
        msgs = round_msgs
    msgs.append(TOOL_ASSISTANT_SUMMARY)
    msgs.extend(TOOL_RECENT)
    return msgs


def _build_hard_tool_messages_raw() -> list[dict]:
    """Full message list for Section C hard tool fixture."""
    msgs: list[dict] = [HARD_TOOL_SYSTEM, HARD_TOOL_USER_PROMPT]
    for tool_calls, tool_responses in HARD_TOOL_ROUNDS:
        msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        msgs.extend(tool_responses)
    msgs.append(HARD_TOOL_ASSISTANT_SUMMARY)
    msgs.extend(HARD_TOOL_RECENT)
    return msgs


def _build_hard_tool_messages_condensed(*, aaak: bool) -> list[dict]:
    """Section C with each tool round condensed (plain or AAAK)."""
    msgs: list[dict] = [HARD_TOOL_SYSTEM, HARD_TOOL_USER_PROMPT]
    for tool_calls, tool_responses in HARD_TOOL_ROUNDS:
        assistant_msg: dict = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        round_msgs = [*msgs, assistant_msg, *tool_responses]
        _condense_tool_round(round_msgs, aaak_tool_condensing=aaak)
        msgs = round_msgs
    msgs.append(HARD_TOOL_ASSISTANT_SUMMARY)
    msgs.extend(HARD_TOOL_RECENT)
    return msgs


def _build_edge_tool_messages_raw() -> list[dict]:
    """Full message list for Section D edge tool fixture."""
    msgs: list[dict] = [EDGE_TOOL_SYSTEM, EDGE_TOOL_USER_PROMPT]
    for tool_calls, tool_responses in EDGE_TOOL_ROUNDS:
        msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        msgs.extend(tool_responses)
    msgs.append(EDGE_TOOL_ASSISTANT_SUMMARY)
    msgs.extend(EDGE_TOOL_RECENT)
    return msgs


def _build_edge_tool_messages_condensed(*, aaak: bool) -> list[dict]:
    """Section D with each tool round condensed (plain or AAAK)."""
    msgs: list[dict] = [EDGE_TOOL_SYSTEM, EDGE_TOOL_USER_PROMPT]
    for tool_calls, tool_responses in EDGE_TOOL_ROUNDS:
        assistant_msg: dict = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        round_msgs = [*msgs, assistant_msg, *tool_responses]
        _condense_tool_round(round_msgs, aaak_tool_condensing=aaak)
        msgs = round_msgs
    msgs.append(EDGE_TOOL_ASSISTANT_SUMMARY)
    msgs.extend(EDGE_TOOL_RECENT)
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Section E — pipeline: [AT] blocks in history, then context compression
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_SYSTEM = {
    "role": "system",
    "content": "You are a staff engineer reviewing prior diagnostics embedded in the thread.",
}
PIPELINE_RECENT = [
    {"role": "user", "content": "Anything blocking tonight's release from those earlier pulls?"},
    {
        "role": "assistant",
        "content": (
            "Check the embedded diagnostics for timeouts, pool sizing, and live gateway load — "
            "use only what appears in the earlier assistant tool summaries, not this reply."
        ),
    },
]


def _build_pipeline_conversation() -> list[dict[str, Any]]:
    """History where prior tool rounds were already condensed to [AT] assistant text."""
    tc1 = [
        {
            "id": "p1",
            "type": "function",
            "function": {"name": "get_service_config", "arguments": "{}"},
        },
    ]
    tm1 = [
        {
            "role": "tool",
            "tool_call_id": "p1",
            "content": json.dumps({"request_timeout_ms": 8500, "pool_size": 10}, separators=(",", ":")),
        },
    ]
    at1 = AAAKCompressor.encode_tool_round(tc1, tm1, {"p1": "get_service_config"})

    tc2 = [
        {
            "id": "p2",
            "type": "function",
            "function": {"name": "get_live_metrics", "arguments": json.dumps({"svc": "gateway"})},
        },
    ]
    tm2 = [
        {
            "role": "tool",
            "tool_call_id": "p2",
            "content": json.dumps(
                {"rps": 3400, "error_pct": 0.4, "shard": "us-east-1a"},
                separators=(",", ":"),
            ),
        },
    ]
    at2 = AAAKCompressor.encode_tool_round(tc2, tm2, {"p2": "get_live_metrics"})

    return [
        PIPELINE_SYSTEM,
        {"role": "user", "content": "Set up the service config review."},
        {"role": "assistant", "content": at1},
        {"role": "user", "content": "Now pull the live gateway metrics."},
        {"role": "assistant", "content": at2},
        *PIPELINE_RECENT,
    ]


PIPELINE_RECALL_QUESTIONS: list[RecallQuestion] = [
    {
        "question": "From the first embedded tool summary, what is request_timeout_ms?",
        "key_facts": ["8500"],
    },
    {
        "question": "From the first embedded tool summary, what is pool_size?",
        "key_facts": ["10"],
    },
    {
        "question": "From the second embedded tool summary, what is rps?",
        "key_facts": ["3400"],
    },
    {
        "question": "From the second embedded tool summary, what is error_pct?",
        "key_facts": ["0.4"],
    },
    {
        "question": "From the second embedded tool summary, what shard value is reported?",
        "key_facts": ["us-east-1a"],
    },
    {
        "question": "From the first [AT] block, what are the exact config values for request_timeout_ms and pool_size?",
        "key_facts": ["8500", "10"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
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
class ModeResult:
    mode: str
    lossless: bool
    section: str  # "ctx", "tool", "hard", "edge", or "pipeline"
    compression_tokens: TokenUsage = field(default_factory=TokenUsage)
    recall_tokens: TokenUsage = field(default_factory=TokenUsage)
    recall_scores: list[int] = field(default_factory=list)
    compressed_size: int = 0
    original_size: int = 0
    elapsed_s: float = 0.0

    @property
    def total_api_tokens(self) -> int:
        return self.compression_tokens.total + self.recall_tokens.total

    @property
    def recall_pct(self) -> float:
        if not self.recall_scores:
            return 0.0
        return 100.0 * sum(self.recall_scores) / len(self.recall_scores)

    @property
    def msg_reduction_pct(self) -> float:
        if self.original_size == 0:
            return 0.0
        return 100.0 * (self.original_size - self.compressed_size) / self.original_size


# ─────────────────────────────────────────────────────────────────────────────
# LLM call helper
# ─────────────────────────────────────────────────────────────────────────────


async def _call(messages: list[dict], *, model: str = MODEL) -> tuple[str, TokenUsage]:
    from gluellm.api import _provider_cache

    sem = _benchmark_semaphore
    async with sem if sem is not None else asyncio.Lock():
        provider, model_id = _provider_cache.get_provider(model, api_key=None)
        extra = dict(_benchmark_completion_extra)
        resp = await provider.acompletion(model=model_id, messages=messages, **extra)
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return content, TokenUsage(
        prompt=getattr(usage, "prompt_tokens", 0) or 0,
        completion=getattr(usage, "completion_tokens", 0) or 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section A: context compression modes
# ─────────────────────────────────────────────────────────────────────────────


async def prepare_ctx_raw(conversation: list[dict], keep_recent: int) -> tuple[list[dict], TokenUsage]:
    return list(conversation), TokenUsage()


async def prepare_ctx_prose(conversation: list[dict], keep_recent: int) -> tuple[list[dict], TokenUsage]:
    from gluellm.api import _ProviderCache

    captured: list[TokenUsage] = []
    original_get_provider = _ProviderCache.get_provider

    class _TrackingCache(_ProviderCache):
        def get_provider(self, model, api_key):
            provider, model_id = original_get_provider(self, model, api_key)

            class _Tracked:
                async def acompletion(_, **kwargs):
                    resp = await provider.acompletion(**kwargs)
                    u = getattr(resp, "usage", None)
                    captured.append(TokenUsage(
                        getattr(u, "prompt_tokens", 0) or 0,
                        getattr(u, "completion_tokens", 0) or 0,
                    ))
                    return resp

            return _Tracked(), model_id

    import gluellm.api as _api
    _orig = _api._ProviderCache
    _api._ProviderCache = _TrackingCache
    try:
        result = await _summarize_old_messages(
            list(conversation),
            keep_recent=keep_recent,
            model=MODEL,
            use_aaak=False,
            completion_extra=_benchmark_completion_extra,
        )
    finally:
        _api._ProviderCache = _orig

    total = TokenUsage(sum(u.prompt for u in captured), sum(u.completion for u in captured))
    return result, total


async def prepare_ctx_aaak(conversation: list[dict], keep_recent: int) -> tuple[list[dict], TokenUsage]:
    from gluellm.api import _provider_cache

    old = conversation[1 : len(conversation) - keep_recent]
    recent = conversation[len(conversation) - keep_recent :]
    system_msg = dict(conversation[0])

    # Wrap provider to capture token usage (passthrough path makes no LLM call → 0 tokens)
    captured: list[TokenUsage] = []
    orig_get = _provider_cache.get_provider

    def tracking_get(model, api_key=None):
        provider, model_id = orig_get(model, api_key)

        class _Tracked:
            async def acompletion(_, **kwargs):
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
            old,
            model=MODEL,
            completion_extra=_benchmark_completion_extra,
        )
    finally:
        _provider_cache.get_provider = orig_get

    tok = TokenUsage(sum(u.prompt for u in captured), sum(u.completion for u in captured))
    summary = {"role": "user", "content": f"[AAAK CTX]\n{encoded}\n[/AAAK CTX]"}
    AAAKCompressor.ensure_preamble_in_system(system_msg)
    return [system_msg, summary, *recent], tok


# ─────────────────────────────────────────────────────────────────────────────
# Section B: tool-round condensing modes
# (condensing itself is pure Python — no LLM call; only recall uses LLM)
# ─────────────────────────────────────────────────────────────────────────────


async def prepare_tool_raw() -> tuple[list[dict], TokenUsage]:
    return _build_tool_messages_raw(), TokenUsage()


async def prepare_tool_plain() -> tuple[list[dict], TokenUsage]:
    return _build_tool_messages_condensed(aaak=False), TokenUsage()


async def prepare_tool_aaak() -> tuple[list[dict], TokenUsage]:
    return _build_tool_messages_condensed(aaak=True), TokenUsage()


# ─────────────────────────────────────────────────────────────────────────────
# Section C: hard tool-round condensing
# ─────────────────────────────────────────────────────────────────────────────


async def prepare_hard_tool_raw() -> tuple[list[dict], TokenUsage]:
    return _build_hard_tool_messages_raw(), TokenUsage()


async def prepare_hard_tool_plain() -> tuple[list[dict], TokenUsage]:
    return _build_hard_tool_messages_condensed(aaak=False), TokenUsage()


async def prepare_hard_tool_aaak() -> tuple[list[dict], TokenUsage]:
    return _build_hard_tool_messages_condensed(aaak=True), TokenUsage()


async def prepare_edge_raw() -> tuple[list[dict], TokenUsage]:
    return _build_edge_tool_messages_raw(), TokenUsage()


async def prepare_edge_plain() -> tuple[list[dict], TokenUsage]:
    return _build_edge_tool_messages_condensed(aaak=False), TokenUsage()


async def prepare_edge_aaak() -> tuple[list[dict], TokenUsage]:
    return _build_edge_tool_messages_condensed(aaak=True), TokenUsage()


# ─────────────────────────────────────────────────────────────────────────────
# Recall evaluation (shared)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict recall judge. You will be shown a question, a JSON array of required facts, "
    "and a response.\n\n"
    "Treat every array item as an independent checklist item. "
    "Score the response: reply with ONLY '1' if ALL required facts are present (even if paraphrased), "
    "or '0' if any key fact is missing or incorrect. No explanation.\n\n"
    "Important: Ignore formatting differences. Bullet lists, backticks, bold markers (**), markdown, "
    "and punctuation do not matter — only whether the fact itself is present. "
    "The order of facts does not matter unless the question explicitly asks for a specific ordering. "
    "Exact wording is not required; paraphrasing or equivalent expressions count as present. "
    "For identifiers and constants (e.g. environment variable names, algorithm names, header names), "
    "treat comparisons as case-insensitive — JWT_SECRET_KEY and jwt_secret_key are the same fact."
)


def _coerce_key_facts(raw: str | list[str]) -> list[str]:
    """Return key facts as atomic checklist items.

    New fixtures should pass lists directly. Older comma-delimited strings are still
    supported so benchmark questions remain easy to write and migrate incrementally.
    """
    if isinstance(raw, list):
        return [fact.strip() for fact in raw if fact.strip()]
    return [fact.strip() for fact in raw.split(",") if fact.strip()]


def _normalize_answer(answer: str) -> str:
    """Strip markdown formatting so the judge focuses on content, not presentation.

    Does NOT lowercase — key facts like JWT_SECRET_KEY are case-sensitive identifiers
    and must survive normalization intact. Only removes backticks, bold markers (**),
    leading bullets (-, *, +), and collapses extra whitespace.
    The original answer is always logged separately.
    """
    import re
    s = answer.replace("`", "")
    s = s.replace("**", "")
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_for_fact_match(text: str) -> str:
    """Normalize text for deterministic atomic-fact containment checks."""
    import re

    s = _normalize_answer(text).casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _contains_atomic_fact(answer: str, fact: str) -> bool:
    needle = _normalize_for_fact_match(fact)
    if not needle:
        return False
    haystack = f" {_normalize_for_fact_match(answer)} "
    return f" {needle} " in haystack


async def _eval_one_question(
    messages: list[dict],
    qa: RecallQuestion,
    *,
    collect_debug: bool,
) -> tuple[int, TokenUsage, dict[str, Any] | None]:
    """Evaluate a single recall question. Safe to run concurrently."""
    answer, tok_a = await _call([*messages, {"role": "user", "content": qa["question"]}])

    normalized = _normalize_answer(answer)
    key_facts = _coerce_key_facts(qa["key_facts"])
    deterministic_ok = all(_contains_atomic_fact(normalized, fact) for fact in key_facts)
    judge_user = (
        f"Question: {qa['question']}\n"
        f"Required facts JSON: {json.dumps(key_facts, ensure_ascii=True)}\n"
        f"Model answer: {normalized}"
    )
    verdict, tok_j = await _call([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_user},
    ])
    judge_ok = verdict.strip().startswith("1")
    ok = 1 if deterministic_ok or judge_ok else 0
    tok = TokenUsage(tok_a.prompt + tok_j.prompt, tok_a.completion + tok_j.completion)
    row = None
    if collect_debug:
        row = {
            "question": qa["question"],
            "key_facts": key_facts,
            "answer": answer,
            "normalized_answer": normalized,
            "deterministic_ok": deterministic_ok,
            "judge_ok": judge_ok,
            "verdict_raw": verdict.strip(),
            "judge_user": judge_user,
            "score": ok,
        }
    return ok, tok, row


async def run_recall(
    messages: list[dict],
    questions: list[RecallQuestion],
    *,
    collect_debug: bool = False,
) -> tuple[list[int], TokenUsage, list[dict[str, Any]] | None]:
    # All questions are independent — run them concurrently and preserve order.
    rows = await asyncio.gather(*[
        _eval_one_question(messages, qa, collect_debug=collect_debug)
        for qa in questions
    ])
    scores = [ok for ok, _, _ in rows]
    total = TokenUsage()
    for _, tok, _ in rows:
        total += tok
    debug_rows = [dr for _, _, dr in rows] if collect_debug else None
    return scores, total, debug_rows


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────


def _extract_aaak_ctx_body(msgs: list[dict[str, Any]]) -> str:
    """Return the AAAK-encoded conversation body from the actual summary message.

    Only inspects role='user' messages whose content begins with '[AAAK CTX]\n'
    and contains the closing '[/AAAK CTX]' tag. This avoids accidentally matching
    the system decode hint, which merely mentions the tag as inline text.
    """
    for m in msgs:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if not c.startswith("[AAAK CTX]\n") and not c.startswith("[AAAK CTX] "):
            continue
        start = len("[AAAK CTX]")
        end = c.find("[/AAAK CTX]")
        if end == -1:
            return c[start:].strip()
        return c[start:end].strip()
    return ""


async def run_ctx_mode(name: str, lossless: bool, prepare_fn, conversation: list[dict], keep_recent: int) -> ModeResult:
    r = ModeResult(mode=name, lossless=lossless, section="ctx", original_size=messages_tokens(conversation))
    t0 = time.perf_counter()
    msgs, comp_tok = await prepare_fn(conversation, keep_recent)
    r.elapsed_s = time.perf_counter() - t0
    r.compressed_size = messages_tokens(msgs)
    r.compression_tokens = comp_tok
    collect = name == "aaak_ctx" and _benchmark_verbose_section_a
    r.recall_scores, r.recall_tokens, dbg = await run_recall(
        msgs, CTX_RECALL_QUESTIONS, collect_debug=collect
    )
    if collect and dbg:
        body = _extract_aaak_ctx_body(msgs)
        print("\n  --- verbose Section A (aaak_ctx) ---")
        print(f"  [AAAK CTX] body ({len(body)} chars):\n{body[:4000]}{'...' if len(body) > 4000 else ''}\n")
        for row in dbg:
            if row["score"] == 0:
                print(f"  MISS: {row['question'][:100]}...")
                print(f"    key_facts : {row['key_facts']}")
                orig = row["answer"]
                norm = row.get("normalized_answer", orig)
                print(f"    orig answer: {orig[:500]}{'...' if len(orig) > 500 else ''}")
                print(f"    norm answer: {norm[:500]}{'...' if len(norm) > 500 else ''}")
                print(f"    deterministic_ok: {row.get('deterministic_ok')!r}")
                print(f"    judge_ok       : {row.get('judge_ok')!r}")
                print(f"    judge payload:\n      {row['judge_user'][:800]}")
                print(f"    judge verdict: {row['verdict_raw']!r}")
        print("  --- end verbose ---\n")
    return r


async def run_tool_mode(name: str, lossless: bool, prepare_fn) -> ModeResult:
    raw_size = messages_tokens(_build_tool_messages_raw())
    r = ModeResult(mode=name, lossless=lossless, section="tool", original_size=raw_size)
    t0 = time.perf_counter()
    msgs, comp_tok = await prepare_fn()
    r.elapsed_s = time.perf_counter() - t0
    r.compressed_size = messages_tokens(msgs)
    r.compression_tokens = comp_tok
    r.recall_scores, r.recall_tokens, _ = await run_recall(msgs, TOOL_RECALL_QUESTIONS)
    return r


async def run_hard_tool_mode(name: str, lossless: bool, prepare_fn) -> ModeResult:
    raw_size = messages_tokens(_build_hard_tool_messages_raw())
    r = ModeResult(mode=name, lossless=lossless, section="hard", original_size=raw_size)
    t0 = time.perf_counter()
    msgs, comp_tok = await prepare_fn()
    r.elapsed_s = time.perf_counter() - t0
    r.compressed_size = messages_tokens(msgs)
    r.compression_tokens = comp_tok
    r.recall_scores, r.recall_tokens, _ = await run_recall(msgs, HARD_TOOL_RECALL_QUESTIONS)
    return r


async def run_edge_tool_mode(name: str, lossless: bool, prepare_fn) -> ModeResult:
    raw_size = messages_tokens(_build_edge_tool_messages_raw())
    r = ModeResult(mode=name, lossless=lossless, section="edge", original_size=raw_size)
    t0 = time.perf_counter()
    msgs, comp_tok = await prepare_fn()
    r.elapsed_s = time.perf_counter() - t0
    r.compressed_size = messages_tokens(msgs)
    r.compression_tokens = comp_tok
    r.recall_scores, r.recall_tokens, _ = await run_recall(msgs, EDGE_TOOL_RECALL_QUESTIONS)
    return r


async def run_pipeline_mode(
    name: str,
    lossless: bool,
    prepare_fn,
    conversation: list[dict[str, Any]],
    keep_recent: int,
) -> ModeResult:
    r = ModeResult(mode=name, lossless=lossless, section="pipeline", original_size=messages_tokens(conversation))
    t0 = time.perf_counter()
    msgs, comp_tok = await prepare_fn(conversation, keep_recent)
    r.elapsed_s = time.perf_counter() - t0
    r.compressed_size = messages_tokens(msgs)
    r.compression_tokens = comp_tok
    r.recall_scores, r.recall_tokens, _ = await run_recall(msgs, PIPELINE_RECALL_QUESTIONS)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Multi-trial aggregation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkSnapshot:
    ctx_results: list[ModeResult]
    tool_results: list[ModeResult]
    hard_tool_results: list[ModeResult]
    edge_tool_results: list[ModeResult]
    pipeline_results: list[ModeResult]


def _print_aggregate_section(
    title: str,
    trials: list[BenchmarkSnapshot],
    get_modes: Callable[[BenchmarkSnapshot], list[ModeResult]],
    questions: list[RecallQuestion],
) -> None:
    n = len(trials)
    if n < 2:
        return
    template = get_modes(trials[0])
    if not template:
        return
    mode_names = [r.mode for r in template]
    print(f"\n{'═' * 128}")
    print(f"  AGGREGATE ({n} trials): {title}")
    print(f"{'═' * 128}\n")
    hdr = f"  {'Mode':<14}  {'mean%':>7}  {'min%':>7}  {'max%':>7}  " + " ".join(f"Q{i + 1:>3}%" for i in range(len(questions)))
    print(hdr)
    print(f"  {'':─<14}  {'':─<7}  {'':─<7}  {'':─<7}  " + " ".join("────" for _ in questions))
    for mode in mode_names:
        recalls: list[float] = []
        per_q_passes = [0] * len(questions)
        for snap in trials:
            mr = next(r for r in get_modes(snap) if r.mode == mode)
            recalls.append(mr.recall_pct)
            for qi, sc in enumerate(mr.recall_scores):
                per_q_passes[qi] += sc
        mean_r = sum(recalls) / n
        min_r = min(recalls)
        max_r = max(recalls)
        q_cells = " ".join(f"{100.0 * per_q_passes[qi] / n:>4.0f}" for qi in range(len(questions)))
        print(f"  {mode:<14}  {mean_r:>6.1f}%  {min_r:>6.1f}%  {max_r:>6.1f}%  {q_cells}")


def print_aggregate_report(trials: list[BenchmarkSnapshot]) -> None:
    if len(trials) < 2:
        return
    print("\n" + "═" * 128)
    print(f"  AGGREGATED RECALL OVER {len(trials)} TRIALS (per-question cells = % of trials passed)")
    print("═" * 128)
    _print_aggregate_section(
        "SECTION A — CONTEXT COMPRESSION",
        trials,
        lambda s: s.ctx_results,
        CTX_RECALL_QUESTIONS,
    )
    _print_aggregate_section(
        "SECTION B — TOOL-ROUND CONDENSING",
        trials,
        lambda s: s.tool_results,
        TOOL_RECALL_QUESTIONS,
    )
    _print_aggregate_section(
        "SECTION C — HARD TOOL-ROUND CONDENSING",
        trials,
        lambda s: s.hard_tool_results,
        HARD_TOOL_RECALL_QUESTIONS,
    )
    _print_aggregate_section(
        "SECTION D — EDGE TOOL ROUNDS",
        trials,
        lambda s: s.edge_tool_results,
        EDGE_TOOL_RECALL_QUESTIONS,
    )
    _print_aggregate_section(
        "SECTION E — PIPELINE",
        trials,
        lambda s: s.pipeline_results,
        PIPELINE_RECALL_QUESTIONS,
    )
    print(f"\n{'═' * 128}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _bar(pct: float, width: int = 22, neg_char: str = "▓") -> str:
    clamped = max(-100.0, min(100.0, pct))
    if clamped < 0:
        filled = int(width * (-clamped) / 100)
        return neg_char * filled + "░" * (width - filled)
    filled = int(width * clamped / 100)
    return "█" * filled + "░" * (width - filled)


def _section_report(title: str, results: list[ModeResult], questions: list[RecallQuestion]) -> None:
    if not results:
        return
    SEP = "─" * 128
    raw = next((r for r in results if "raw" in r.mode), results[0])

    print(f"\n{'═' * 128}")
    print(f"  {title}")
    print(f"{'═' * 128}")

    # Token usage
    print("\n  TOKEN USAGE\n")
    hdr = (
        f"  {'Mode':<14}  {'loss?':>5}  {'msg tok':>8}  {'msg Δ%':>7}  "
        f"{'compress API':>13}  {'recall API':>11}  {'total API':>10}  {'time':>7}  progress"
    )
    print(hdr)
    print(f"  {'':─<14}  {'':─<5}  {'':─<8}  {'':─<7}  {'':─<13}  {'':─<11}  {'':─<10}  {'':─<7}")
    for r in results:
        lossy = "lossy" if not r.lossless else "  ✓  "
        reduc = 100.0 * (raw.compressed_size - r.compressed_size) / raw.compressed_size if raw.compressed_size else 0
        bar = _bar(reduc)
        print(
            f"  {r.mode:<14}  {lossy:>5}  {r.compressed_size:>8}  {reduc:>+6.1f}%  "
            f"  {r.compression_tokens.total:>11}  {r.recall_tokens.total:>11}  "
            f"{r.total_api_tokens:>10}  {r.elapsed_s:>6.1f}s  {bar}"
        )

    # Recall table
    print(f"\n{SEP}")
    print("\n  RECALL ACCURACY  (LLM judge per question: ✓ all facts present  ✗ facts missing)\n")
    col_q = 76
    print(f"  {'Question':<{col_q}}" + "".join(f"  {r.mode:<10}" for r in results))
    print(f"  {'':─<{col_q}}" + "".join("  ──────────" for _ in results))
    for i, qa in enumerate(questions):
        q_short = qa["question"][:col_q - 3] + "..." if len(qa["question"]) > col_q else qa["question"]
        scores_str = "".join(
            f"  {'✓' if r.recall_scores[i] else '✗':^10}" for r in results
        )
        print(f"  {q_short:<{col_q}}{scores_str}")
    print(f"\n  {'TOTAL RECALL %':<{col_q}}" + "".join(f"  {r.recall_pct:>8.0f}%" for r in results))

    # Summary bars
    print(f"\n{SEP}")
    print("\n  SUMMARY\n")
    for r in results:
        reduc = 100.0 * (raw.compressed_size - r.compressed_size) / raw.compressed_size if raw.compressed_size else 0
        lossy_tag = "(lossless)" if r.lossless else "(lossy)   "
        print(
            f"  {r.mode:<14} {lossy_tag}  recall {r.recall_pct:>5.1f}%  {_bar(r.recall_pct)}  "
            f"|  msg tok {reduc:>+5.1f}%  {_bar(reduc, neg_char='▓')}"
        )
    print()


def print_full_report(
    ctx_results: list[ModeResult],
    tool_results: list[ModeResult],
    hard_tool_results: list[ModeResult],
    edge_tool_results: list[ModeResult],
    pipeline_results: list[ModeResult],
) -> None:
    _section_report(
        "SECTION A — CONTEXT COMPRESSION  (old conversation turns → single compressed message)",
        ctx_results,
        CTX_RECALL_QUESTIONS,
    )
    _section_report(
        "SECTION B — TOOL-ROUND CONDENSING  (assistant+tool messages → single condensed message per round)",
        tool_results,
        TOOL_RECALL_QUESTIONS,
    )
    _section_report(
        "SECTION C — HARD TOOL-ROUND CONDENSING  (nested JSON, arrays, markdown, CSV, disambiguation)",
        hard_tool_results,
        HARD_TOOL_RECALL_QUESTIONS,
    )
    _section_report(
        "SECTION D — EDGE TOOL ROUNDS  (many parallel tools, truncation, stack trace, numeric precision)",
        edge_tool_results,
        EDGE_TOOL_RECALL_QUESTIONS,
    )
    _section_report(
        "SECTION E — PIPELINE  (context compression over history that already contains [AT] blocks)",
        pipeline_results,
        PIPELINE_RECALL_QUESTIONS,
    )

    all_results = ctx_results + tool_results + hard_tool_results + edge_tool_results + pipeline_results
    if not all_results:
        return

    # Cross-section summary
    print(f"\n{'═' * 128}")
    print("  CROSS-SECTION COMPARISON\n")
    print(f"  {'Mode':<18}  {'Section':<8}  {'lossless':>9}  {'recall%':>8}  {'msg tok Δ':>10}  {'total API tok':>14}")
    print(f"  {'':─<18}  {'':─<8}  {'':─<9}  {'':─<8}  {'':─<10}  {'':─<14}")
    for r in all_results:
        ref = next((x for x in all_results if x.section == r.section and "raw" in x.mode), r)
        reduc = 100.0 * (ref.compressed_size - r.compressed_size) / ref.compressed_size if ref.compressed_size else 0
        print(
            f"  {r.mode:<18}  {r.section:<8}  {'yes' if r.lossless else 'no':>9}  "
            f"{r.recall_pct:>7.1f}%  {reduc:>+9.1f}%  {r.total_api_tokens:>14}"
        )
    print(f"\n{'═' * 128}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


async def _run_section_b_parallel(tool_modes: list) -> list[ModeResult]:
    """Run all Section B modes concurrently (pure-Python prep, no monkeypatching)."""
    return list(await asyncio.gather(*[
        run_tool_mode(name, lossless, fn)
        for name, lossless, fn in tool_modes
    ]))


async def _run_section_c_parallel(hard_tool_modes: list) -> list[ModeResult]:
    """Run all Section C modes concurrently (pure-Python prep, no monkeypatching)."""
    return list(await asyncio.gather(*[
        run_hard_tool_mode(name, lossless, fn)
        for name, lossless, fn in hard_tool_modes
    ]))


async def _run_section_d_parallel(edge_tool_modes: list) -> list[ModeResult]:
    """Run all Section D modes concurrently (pure-Python prep, no monkeypatching)."""
    return list(await asyncio.gather(*[
        run_edge_tool_mode(name, lossless, fn)
        for name, lossless, fn in edge_tool_modes
    ]))


async def _run_section_e_sequential(
    pipeline_modes: list,
    pipeline_conv: list[dict[str, Any]],
    keep_recent: int,
) -> list[ModeResult]:
    """Run Section E modes sequentially — pipe_aaak monkeypatches a global."""
    results = []
    for name, lossless, fn in pipeline_modes:
        results.append(await run_pipeline_mode(name, lossless, fn, pipeline_conv, keep_recent))
    return results


def _print_section_results(header: str, results: list[ModeResult]) -> None:
    """Print section progress lines in the canonical mode order after parallel gather."""
    print(header)
    for r in results:
        print(f"    [{r.mode}]... recall={r.recall_pct:.0f}%  API={r.total_api_tokens} tok  ({r.elapsed_s:.1f}s)")


async def run_full_benchmark_once(
    keep_recent: int,
    pipeline_conv: list[dict[str, Any]],
    *,
    only_section: str | None = None,
) -> BenchmarkSnapshot:
    ctx_modes = [
        ("raw", False, prepare_ctx_raw),
        ("prose", False, prepare_ctx_prose),
        ("aaak_ctx", True, prepare_ctx_aaak),
    ]
    tool_modes = [
        ("raw_tools", False, prepare_tool_raw),
        ("plain", False, prepare_tool_plain),
        ("aaak_tools", True, prepare_tool_aaak),
    ]
    hard_tool_modes = [
        ("hard_raw", False, prepare_hard_tool_raw),
        ("hard_plain", False, prepare_hard_tool_plain),
        ("hard_aaak", True, prepare_hard_tool_aaak),
    ]
    edge_tool_modes = [
        ("edge_raw", False, prepare_edge_raw),
        ("edge_plain", False, prepare_edge_plain),
        ("edge_aaak", True, prepare_edge_aaak),
    ]
    pipeline_modes = [
        ("pipe_raw", False, prepare_ctx_raw),
        ("pipe_aaak", True, prepare_ctx_aaak),
    ]

    ctx_results: list[ModeResult] = []
    tool_results: list[ModeResult] = []
    hard_tool_results: list[ModeResult] = []
    edge_tool_results: list[ModeResult] = []
    pipeline_results: list[ModeResult] = []

    # Section A modes monkeypatch global provider state — must run sequentially.
    if only_section in (None, "a"):
        print("  SECTION A: context compression")
        for name, lossless, fn in ctx_modes:
            print(f"    [{name}]...", end=" ", flush=True)
            r = await run_ctx_mode(name, lossless, fn, CTX_CONVERSATION, keep_recent)
            ctx_results.append(r)
            print(f"recall={r.recall_pct:.0f}%  API={r.total_api_tokens} tok  ({r.elapsed_s:.1f}s)")

    # Sections B-E are independent of A (A is already done) and of each other.
    # B/C/D run all their modes in parallel; E runs modes sequentially (pipe_aaak monkeypatches).
    parallel_sections: list[tuple[str, str, Any]] = []
    if only_section in (None, "b"):
        parallel_sections.append(("b", "\n  SECTION B: tool-round condensing", _run_section_b_parallel(tool_modes)))
    if only_section in (None, "c"):
        parallel_sections.append(("c", "\n  SECTION C: hard tool-round condensing", _run_section_c_parallel(hard_tool_modes)))
    if only_section in (None, "d"):
        parallel_sections.append(("d", "\n  SECTION D: edge tool-round condensing", _run_section_d_parallel(edge_tool_modes)))
    if only_section in (None, "e"):
        parallel_sections.append(("e", "\n  SECTION E: pipeline (compress history with embedded [AT])", _run_section_e_sequential(pipeline_modes, pipeline_conv, keep_recent)))

    if parallel_sections:
        if len(parallel_sections) > 1:
            labels = ", ".join(sec.upper() for sec, _, _ in parallel_sections)
            print(f"\n  Running sections {labels} in parallel...")
        gathered = await asyncio.gather(*[coro for _, _, coro in parallel_sections])
        for (sec, header, _), results in zip(parallel_sections, gathered):
            _print_section_results(header, results)
            if sec == "b":
                tool_results = results
            elif sec == "c":
                hard_tool_results = results
            elif sec == "d":
                edge_tool_results = results
            elif sec == "e":
                pipeline_results = results

    return BenchmarkSnapshot(
        ctx_results=ctx_results,
        tool_results=tool_results,
        hard_tool_results=hard_tool_results,
        edge_tool_results=edge_tool_results,
        pipeline_results=pipeline_results,
    )


async def main_async(args: argparse.Namespace) -> None:
    global _benchmark_completion_extra, _benchmark_verbose_section_a, _benchmark_semaphore

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        return

    if args.no_deterministic_sampling:
        _benchmark_completion_extra = {}
    else:
        _benchmark_completion_extra = {"temperature": 0, "top_p": 1}

    _benchmark_verbose_section_a = bool(args.verbose_section_a)
    _benchmark_semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))

    keep_recent = 2
    trials_n = max(1, int(args.trials))
    only_section = None
    for sec in ("a", "b", "c", "d", "e"):
        if getattr(args, f"only_section_{sec}", False):
            only_section = sec
            break

    print("AAAK Live Benchmark")
    print(f"Model: {MODEL}  |  keep_recent={keep_recent}  |  trials={trials_n}  |  concurrency={args.concurrency}")
    if _benchmark_completion_extra:
        print(f"  completion_extra={_benchmark_completion_extra}")
    else:
        print("  completion_extra={} (no deterministic sampling kwargs)")
    pipeline_conv = _build_pipeline_conversation()
    if only_section is None:
        print(f"Context: {len(CTX_CONVERSATION)} turns, {len(CTX_RECALL_QUESTIONS)} recall questions")
        print(f"Tool rounds (B): {len(TOOL_ROUNDS)} rounds, {len(TOOL_RECALL_QUESTIONS)} recall questions")
        print(
            f"Hard tool rounds (C): {len(HARD_TOOL_ROUNDS)} rounds, "
            f"{len(HARD_TOOL_RECALL_QUESTIONS)} recall questions"
        )
        print(
            f"Edge tool rounds (D): {len(EDGE_TOOL_ROUNDS)} rounds, "
            f"{len(EDGE_TOOL_RECALL_QUESTIONS)} recall questions"
        )
        print(
            f"Pipeline (E): {len(pipeline_conv)} turns, {len(PIPELINE_RECALL_QUESTIONS)} recall questions\n"
        )
    else:
        section_labels = {
            "a": f"Section A only: {len(CTX_CONVERSATION)} turns, {len(CTX_RECALL_QUESTIONS)} recall questions",
            "b": f"Section B only: {len(TOOL_ROUNDS)} rounds, {len(TOOL_RECALL_QUESTIONS)} recall questions",
            "c": f"Section C only: {len(HARD_TOOL_ROUNDS)} rounds, {len(HARD_TOOL_RECALL_QUESTIONS)} recall questions",
            "d": f"Section D only: {len(EDGE_TOOL_ROUNDS)} rounds, {len(EDGE_TOOL_RECALL_QUESTIONS)} recall questions",
            "e": f"Section E only: {len(pipeline_conv)} turns, {len(PIPELINE_RECALL_QUESTIONS)} recall questions",
        }
        print(section_labels[only_section] + "\n")

    trial_snapshots: list[BenchmarkSnapshot] = []
    for t in range(trials_n):
        if trials_n > 1:
            print(f"\n{'─' * 64}  TRIAL {t + 1}/{trials_n}  {'─' * 64}")
        snap = await run_full_benchmark_once(
            keep_recent,
            pipeline_conv,
            only_section=only_section,
        )
        trial_snapshots.append(snap)

    print_full_report(
        trial_snapshots[-1].ctx_results,
        trial_snapshots[-1].tool_results,
        trial_snapshots[-1].hard_tool_results,
        trial_snapshots[-1].edge_tool_results,
        trial_snapshots[-1].pipeline_results,
    )
    print_aggregate_report(trial_snapshots)


def main() -> None:
    parser = argparse.ArgumentParser(description="AAAK live benchmark (real API calls).")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of full benchmark runs; >1 prints aggregated mean/min/max and per-question pass rates.",
    )
    only_section_group = parser.add_mutually_exclusive_group()
    only_section_group.add_argument(
        "--only-section-a",
        action="store_true",
        help="Run only Section A (context compression).",
    )
    only_section_group.add_argument(
        "--only-section-b",
        action="store_true",
        help="Run only Section B (tool-round condensing).",
    )
    only_section_group.add_argument(
        "--only-section-c",
        action="store_true",
        help="Run only Section C (hard tool-round condensing).",
    )
    only_section_group.add_argument(
        "--only-section-d",
        action="store_true",
        help="Run only Section D (edge tool rounds).",
    )
    only_section_group.add_argument(
        "--only-section-e",
        action="store_true",
        help="Run only Section E (pipeline).",
    )
    parser.add_argument(
        "--verbose-section-a",
        action="store_true",
        help="Print [AAAK CTX] body and failed recall rows for aaak_ctx (Section A).",
    )
    parser.add_argument(
        "--no-deterministic-sampling",
        action="store_true",
        help="Omit temperature/top_p kwargs on API completions (use if the model errors on them).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max simultaneous in-flight API calls across all sections/modes/recall questions (default: 10).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
