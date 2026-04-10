---
name: cloud-agent-starter
description: Quick-start workflow for Cloud agents to install deps, authenticate providers, run smoke checks, configure runtime flags, and execute targeted tests by codebase area.
---
# Cloud Agent Starter Skill (GlueLLM)

Use this skill when a Cloud agent is dropped into this repo and needs to become productive immediately.

## 1) Fast bootstrap (first 2-3 minutes)

### Install dependencies
```bash
uv pip install -e ".[dev]"
```

### "Log in" to providers (API key setup)
GlueLLM reads provider keys from env vars.

```bash
export OPENAI_API_KEY="sk-..."
# Optional providers
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
```

Quick auth sanity check:
```bash
uv run gluellm list-models -p openai
```

### Start the app / smoke-run the codebase
This repo is primarily a library + CLI (no long-running web server required for normal dev). Start with CLI smoke tests:

```bash
uv run gluellm test-completion
uv run gluellm test-streaming --message "Say hello in one short sentence."
```

If you want an interactive path:
```bash
uv run gluellm demo
```

## 2) Feature flags and "mock mode" setup

GlueLLM uses environment variables as runtime flags.

### Common flags to toggle quickly
```bash
export GLUELLM_DEFAULT_TOOL_MODE="dynamic"
export GLUELLM_DEFAULT_CONDENSE_TOOL_MESSAGES="true"
export GLUELLM_DEFAULT_TOOL_EXECUTION_ORDER="parallel"
export GLUELLM_ENABLE_TRACING="false"
export GLUELLM_LOG_CONSOLE_OUTPUT="true"
export GLUELLM_LOG_LEVEL="DEBUG"
```

### Mock/no-live-API workflow
For local-only loops that should not hit provider APIs:

```bash
export OPENAI_API_KEY="sk-test"
uv run pytest tests/ -m "not integration"
```

Notes:
- Integration tests in this repo intentionally skip when `OPENAI_API_KEY` is unset or `sk-test`.
- `tests/test_rate_limiting.py` is ignored by default in `pytest.ini`; run it explicitly when needed.

## 3) Testing workflows by codebase area

Run these from repo root.

### A) Core API + conversation models (`gluellm/api.py`, config/conversation primitives)
```bash
uv run pytest tests/test_api.py tests/test_conversation.py tests/test_prompt.py tests/test_schema.py -m "not integration"
```

Live-provider pass for API event/streaming paths:
```bash
uv run pytest tests/test_api.py -m integration -k "ProcessStatusEvents or StreamingStructuredOutput"
```

### B) Tools, routing, and batching (`gluellm/tool_*`, `batch`, router logic)
```bash
uv run pytest tests/test_tool_router.py tests/test_batch.py tests/test_pydantic_tool_params.py tests/test_custom_retry.py -m "not integration"
```

CLI-level tool smoke:
```bash
uv run gluellm test-tool-call
uv run gluellm test-batch-processing --count 5
```

### C) Workflows + executors (`gluellm/workflows/*`, `executors`)
```bash
uv run pytest tests/test_workflows.py tests/test_executors.py tests/test_hooks.py tests/test_guardrails.py -m "not integration"
```

Workflow CLI smoke:
```bash
uv run gluellm test-iterative-workflow --input "Topic: async Python" --iterations 2
uv run gluellm test-pipeline-workflow --input "Write a short note on retries"
```

### D) CLI and examples (`gluellm/cli/**`, `examples/`)
```bash
uv run pytest tests/test_cli.py tests/test_examples.py -m "not integration"
```

Live examples sweep:
```bash
uv run pytest tests/test_examples.py -m integration -v
```

### E) Infrastructure, observability, and runtime safety
```bash
uv run pytest tests/test_config.py tests/test_logging.py tests/test_runtime.py tests/test_telemetry.py tests/test_observability_utils.py -m "not integration"
```

Rate limiting tests (explicit target, because default pytest args ignore this file):
```bash
uv run pytest tests/test_rate_limiting.py
```

### F) OpenResponses and embeddings
```bash
uv run pytest tests/test_responses.py tests/test_embeddings.py -m "not integration"
uv run gluellm test-responses --prompt "What is 2+2?"
uv run gluellm test-embedding --text "hello world"
```

## 4) Practical agent workflow defaults

When making changes:
1. Run focused area tests first (from section 3).
2. Then run broader safety net:
   ```bash
   uv run pytest tests/ -m "not integration"
   ```
3. For API-facing changes, add at least one targeted integration check if a real key is available.

Debugging tips:
- Add `-n 0` to disable parallel xdist when debugging flaky tests.
- Add `-s` to see print/log output during runs.

## 5) Updating this skill when new runbook knowledge appears

Keep this skill minimal and operationally useful.

When you discover a new "testing trick" or runbook step:
1. Add it to the closest codebase-area section above (A-F), not a random new section.
2. Include one copy/paste command and one sentence explaining when to use it.
3. If the trick depends on flags, list exact env vars needed.
4. Prefer replacing outdated commands over appending duplicates.
5. Sanity-check all new commands once before committing docs updates.
