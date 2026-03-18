# GlueLLM Documentation

Documentation for the GlueLLM LLM orchestration framework.

## Getting Started

- [Main README](../README.md) - Project overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [API.md](API.md) - Main API reference
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration system

## Core API

| Document | Description |
|----------|-------------|
| [API.md](API.md) | `complete`, `structured_complete`, `stream_complete`, `embed`, `GlueLLM` |
| [CONFIGURATION.md](CONFIGURATION.md) | `GlueLLMSettings`, `configure()`, environment variables |
| [ERROR_HANDLING.md](ERROR_HANDLING.md) | Exception hierarchy, retry logic |
| [TOOL_EXECUTION.md](TOOL_EXECUTION.md) | Tool modes, `@static_tool`, execution flow |

## Data Models

| Document | Description |
|----------|-------------|
| [MODELS.md](MODELS.md) | `Conversation`, `Message`, `BatchRequest`, `EvalRecord`, etc. |
| [BATCH_PROCESSING.md](BATCH_PROCESSING.md) | Batch API and models |

## Workflows & Agents

| Document | Description |
|----------|-------------|
| [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md) | Workflow selection and patterns |
| [WORKFLOWS_API.md](WORKFLOWS_API.md) | Workflow implementations API |
| [AGENTS.md](AGENTS.md) | Agent system |

## Extensibility

| Document | Description |
|----------|-------------|
| [HOOKS.md](HOOKS.md) | Hook system |
| [GUARDRAILS.md](GUARDRAILS.md) | Safety and validation |
| [EXTENDING.md](EXTENDING.md) | Custom workflows, hooks, providers |

## Advanced Features

| Document | Description |
|----------|-------------|
| [RATE_LIMITING.md](RATE_LIMITING.md) | Rate limiting and API key pools |
| [OBSERVABILITY.md](OBSERVABILITY.md) | Logging, tracing, metrics |
| [EVALUATION.md](EVALUATION.md) | Evaluation recording |
| [RUNTIME.md](RUNTIME.md) | Shutdown, correlation IDs, context |
| [COST_TRACKING.md](COST_TRACKING.md) | Cost tracking |
| [EMBEDDINGS.md](EMBEDDINGS.md) | Embedding generation |
| [CLI.md](CLI.md) | Command-line interface |
| [CONNECTION_POOLING.md](CONNECTION_POOLING.md) | HTTP connection pooling |

## Operations

| Document | Description |
|----------|-------------|
| [MIGRATION.md](MIGRATION.md) | Version upgrades |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues |
| [PERFORMANCE.md](PERFORMANCE.md) | Optimization tips |

## Examples

See the [examples/](../examples/) directory for runnable code:
- `basic_usage.py` - Simple completion
- `batch_processing.py` - Batch processing
- `workflow_example.py` - Workflows
- `hooks_example.py` - Hooks
- And more.
