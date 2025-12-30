# GlueLLM Architecture

This document provides a comprehensive overview of GlueLLM's architecture, including component diagrams, data flow, and design decisions.

## High-Level Overview

GlueLLM is an LLM orchestration framework that provides:
- **Unified API** for multiple LLM providers
- **Automatic tool execution** with retry logic
- **Multi-agent workflows** for complex tasks
- **Hooks system** for cross-cutting concerns
- **Observability** through OpenTelemetry and MLflow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Application                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GlueLLM API Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │    complete()    │  │structured_complete│  │     stream_complete()     │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬───────────────┘  │
│           │                    │                         │                   │
│           └────────────────────┼─────────────────────────┘                   │
│                                ▼                                             │
│                    ┌───────────────────────┐                                 │
│                    │      GlueLLM Class     │                                │
│                    │   (Conversation State) │                                │
│                    └───────────┬───────────┘                                 │
└────────────────────────────────┼─────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
          ┌─────────────┐ ┌───────────┐ ┌───────────┐
          │ Tool Exec   │ │  Hooks    │ │ Telemetry │
          │    Loop     │ │  System   │ │   Layer   │
          └──────┬──────┘ └─────┬─────┘ └─────┬─────┘
                 │              │             │
                 └──────────────┼─────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Infrastructure                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Rate Limiter │  │  API Key Pool │  │   Retry      │  │    Error         │ │
│  │               │  │              │  │   Logic      │  │  Classification  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          any-llm SDK Layer                                   │
│                    (Multi-provider abstraction)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
    ┌───────────┐          ┌───────────┐          ┌───────────┐
    │  OpenAI   │          │ Anthropic │          │    xAI    │
    │    API    │          │    API    │          │    API    │
    └───────────┘          └───────────┘          └───────────┘
```

## Core Components

### 1. API Layer (`gluellm/api.py`)

The main entry point for all LLM interactions.

```
┌──────────────────────────────────────────────────────────────┐
│                        GlueLLM Class                          │
├──────────────────────────────────────────────────────────────┤
│ Attributes:                                                   │
│   - model: str                                               │
│   - system_prompt: str                                       │
│   - tools: list[Callable]                                    │
│   - max_tool_iterations: int                                 │
│   - _conversation: Conversation                              │
├──────────────────────────────────────────────────────────────┤
│ Methods:                                                      │
│   + complete() → ToolExecutionResult                         │
│   + structured_complete[T]() → T                             │
│   + stream_complete() → AsyncIterator[StreamingChunk]        │
│   + reset_conversation()                                     │
│   - _format_system_prompt()                                  │
│   - _find_tool()                                             │
└──────────────────────────────────────────────────────────────┘
```

**Key Responsibilities:**
- Manage conversation state
- Execute tool calling loops
- Parse structured outputs
- Handle streaming responses

### 2. Tool Execution Loop

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       ▼
┌─────────────────────┐
│  Build Messages     │
│  (System + History) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Call LLM with      │◄───────────────────────┐
│  Retry Logic        │                        │
└──────────┬──────────┘                        │
           ▼                                   │
    ┌─────────────┐     Yes    ┌─────────────┐ │
    │ Tool Calls? ├───────────►│ Execute     │ │
    │             │            │ Tools       │ │
    └──────┬──────┘            └──────┬──────┘ │
           │ No                       │        │
           ▼                          │        │
    ┌─────────────┐            ┌──────┴──────┐ │
    │   Return    │            │ Append Tool │ │
    │   Result    │            │ Results     ├─┘
    └─────────────┘            └─────────────┘
```

### 3. Workflow System

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Workflow Base Class                            │
├───────────────────────────────────────────────────────────────────────┤
│  + execute(input) → WorkflowResult                                    │
│  # _execute_internal(input, context) → WorkflowResult  [abstract]     │
│  # validate_config() → bool                                           │
│  - _run_pre_hooks()                                                   │
│  - _run_post_hooks()                                                  │
└───────────────────────────────────────────────────────────────────────┘
                                    △
                                    │ extends
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
┌──────┴──────┐            ┌───────┴───────┐            ┌───────┴───────┐
│  Pipeline   │            │   Iterative   │            │    Debate     │
│  Workflow   │            │   Workflow    │            │   Workflow    │
└─────────────┘            └───────────────┘            └───────────────┘
       │                            │                            │
       ▼                            ▼                            ▼
   Sequential              Refinement with             Adversarial
   Execution               Critics                     Discussion
```

### 4. Executor System

```
┌─────────────────────────────────────────────────────────────┐
│                      Executor (Abstract)                     │
├─────────────────────────────────────────────────────────────┤
│  + execute(prompt: str) → str                               │
│  + execute_with_context(prompt, context) → str              │
└─────────────────────────────────────────────────────────────┘
                              △
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────┴───────┐    ┌───────┴───────┐    ┌───────┴───────┐
│ SimpleExecutor│    │ AgentExecutor │    │ Custom        │
│               │    │  (with tools) │    │ Executor      │
└───────────────┘    └───────────────┘    └───────────────┘
```

### 5. Hook System

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Hook Registry                              │
├─────────────────────────────────────────────────────────────────────┤
│  hooks: dict[HookStage, list[HookConfig]]                           │
│                                                                      │
│  + add_hook(stage, config)                                          │
│  + remove_hook(stage, name)                                         │
│  + get_hooks(stage) → list[HookConfig]                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                         Uses   │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Hook Manager                               │
├─────────────────────────────────────────────────────────────────────┤
│  + run_hooks(stage, context) → HookContext                          │
│  - _execute_hook(hook, context, timeout)                            │
│  - _handle_hook_error(hook, error, strategy)                        │
└─────────────────────────────────────────────────────────────────────┘

Hook Stages:
  PRE_EXECUTOR  ──► Before LLM call
  POST_EXECUTOR ──► After LLM response
  PRE_WORKFLOW  ──► Before workflow starts
  POST_WORKFLOW ──► After workflow completes
```

### 6. Telemetry Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Telemetry Module                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐           ┌─────────────────┐                  │
│  │  OpenTelemetry  │           │     MLflow      │                  │
│  │     Tracing     │           │    Tracking     │                  │
│  ├─────────────────┤           ├─────────────────┤                  │
│  │ • Span creation │           │ • Metrics       │                  │
│  │ • Context       │           │ • Parameters    │                  │
│  │   propagation   │           │ • Experiments   │                  │
│  │ • OTLP export   │           │ • Run tracking  │                  │
│  └────────┬────────┘           └────────┬────────┘                  │
│           │                             │                            │
│           └─────────────┬───────────────┘                            │
│                         ▼                                            │
│              ┌─────────────────────┐                                 │
│              │    trace_llm_call   │                                 │
│              │  (context manager)  │                                 │
│              └─────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Request Flow

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. VALIDATION & SETUP                                                │
│    • Set/generate correlation ID                                     │
│    • Check shutdown state                                            │
│    • Initialize shutdown context                                     │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. MESSAGE BUILDING                                                  │
│    • Format system prompt with tools                                 │
│    • Add user message to conversation                                │
│    • Build message list for API call                                 │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. LLM CALL (with retry)                                            │
│    • Acquire rate limit                                              │
│    • Set temporary API key (if pool used)                            │
│    • Start tracing span                                              │
│    • Call any-llm SDK                                                │
│    • Classify any errors                                             │
│    • Retry on transient failures                                     │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TOOL EXECUTION (if needed)                                       │
│    • Parse tool calls from response                                  │
│    • Execute each tool (sync or async)                               │
│    • Append tool results to messages                                 │
│    • Loop back to LLM call                                           │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. RESPONSE PROCESSING                                               │
│    • Extract token usage                                             │
│    • Record metrics to MLflow                                        │
│    • Update conversation history                                     │
│    • Clear correlation ID                                            │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
ToolExecutionResult
```

### Workflow Execution Flow

```
Workflow.execute(input)
         │
         ▼
┌────────────────────┐
│ Run PRE_WORKFLOW   │
│ Hooks              │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ _execute_internal  │ ◄─── Workflow-specific logic
│ (abstract method)  │
└─────────┬──────────┘
          │
          ▼
   ┌──────────────────────────────────────────┐
   │         Executor Calls                    │
   │  ┌──────────────┐  ┌──────────────┐      │
   │  │PRE_EXECUTOR  │  │POST_EXECUTOR │      │
   │  │   hooks      │  │    hooks     │      │
   │  └──────┬───────┘  └───────▲──────┘      │
   │         │                  │             │
   │         ▼                  │             │
   │  ┌──────────────────────────┐            │
   │  │   Executor.execute()     │            │
   │  │   (LLM call with tools)  │            │
   │  └──────────────────────────┘            │
   └──────────────────────────────────────────┘
          │
          ▼
┌────────────────────┐
│ Run POST_WORKFLOW  │
│ Hooks              │
└─────────┬──────────┘
          │
          ▼
    WorkflowResult
```

## Error Handling

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Error Classification                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Raw Exception from any-llm                                         │
│          │                                                           │
│          ▼                                                           │
│   classify_llm_error()                                               │
│          │                                                           │
│          ├──► TokenLimitError     (context length exceeded)          │
│          ├──► RateLimitError      (429, quota exceeded)              │
│          ├──► APIConnectionError  (network, timeout)                 │
│          ├──► AuthenticationError (401, 403)                         │
│          ├──► InvalidRequestError (400, validation)                  │
│          └──► LLMError            (catch-all)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Retry Strategy:
  ┌────────────────────┐
  │ Retryable Errors:  │  ──► Automatic retry with exponential backoff
  │ • RateLimitError   │
  │ • APIConnectionError│
  └────────────────────┘

  ┌────────────────────┐
  │ Non-Retryable:     │  ──► Immediate failure
  │ • TokenLimitError  │
  │ • AuthenticationError│
  │ • InvalidRequestError│
  └────────────────────┘
```

## Configuration

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GlueLLMSettings (Pydantic)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model Settings                     API Keys                         │
│  ├── default_model                  ├── openai_api_key              │
│  ├── default_system_prompt          ├── anthropic_api_key           │
│  └── max_tool_iterations            └── xai_api_key                 │
│                                                                      │
│  Retry Settings                     Timeout Settings                 │
│  ├── retry_max_attempts             ├── default_request_timeout     │
│  ├── retry_min_wait                 └── max_request_timeout         │
│  ├── retry_max_wait                                                  │
│  └── retry_multiplier               Telemetry Settings               │
│                                     ├── enable_tracing              │
│  Rate Limiting                      ├── mlflow_tracking_uri         │
│  ├── rate_limit_requests_per_second ├── mlflow_experiment_name      │
│  └── rate_limit_burst_size          └── otel_exporter_endpoint      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Configuration Sources (priority order):
  1. Constructor arguments
  2. Environment variables (GLUELLM_*)
  3. .env file
  4. Default values
```

## Shutdown Handling

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Graceful Shutdown                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Signal (SIGTERM/SIGINT)                                            │
│          │                                                           │
│          ▼                                                           │
│   Set shutdown event ─────────────────────────────────────────────── │
│          │                                                           │
│          │    New requests check is_shutting_down()                  │
│          │    └──► Rejected with RuntimeError                        │
│          │                                                           │
│          ▼                                                           │
│   Wait for in-flight requests ──────────────────────────────────────│
│          │    (with configurable timeout)                            │
│          │                                                           │
│          ▼                                                           │
│   Execute shutdown callbacks ───────────────────────────────────────│
│          │    • Telemetry cleanup                                    │
│          │    • MLflow run closing                                   │
│          │    • Custom callbacks                                     │
│          │                                                           │
│          ▼                                                           │
│   Shutdown complete                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Extension Points

### Adding New Workflows

1. Create a new file in `gluellm/workflows/`
2. Extend `Workflow` base class
3. Implement `_execute_internal()` method
4. Create config model in `gluellm/models/workflow.py`

### Adding New Hooks

1. Create hook function: `def my_hook(context: HookContext) -> HookContext`
2. Register with `HookRegistry.add_hook()`
3. Handle errors via `HookErrorStrategy`

### Adding New Providers

1. Provider support is handled by `any-llm` SDK
2. Add API key to settings/environment
3. Use model format: `provider:model_name`

## Performance Considerations

- **Rate Limiting:** Token bucket algorithm with configurable burst
- **Connection Pooling:** Handled by any-llm SDK
- **Retry Logic:** Exponential backoff with jitter
- **Parallel Execution:** Workflows use `asyncio.gather()` for parallel agent execution
- **Token Tracking:** Minimal overhead, optional MLflow integration

## Security Considerations

- **API Key Management:** Keys stored in environment or secure settings
- **PII Handling:** Use hooks for content filtering before sending to LLMs
- **Audit Logging:** Hook system supports comprehensive audit trails
- **Rate Limiting:** Prevents accidental API abuse
