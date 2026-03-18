# GlueLLM Data Models Reference

This document describes the Pydantic models used throughout GlueLLM.

## Conversation Models

### Role

Enumeration of message roles in chat interactions.

```python
from gluellm.models.conversation import Role

Role.SYSTEM    # System/instruction messages
Role.USER      # User input messages
Role.ASSISTANT  # LLM response messages
Role.TOOL      # Tool execution result messages
```

### Message

A single message in a conversation.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier (UUID) |
| `role` | `Role` | Message role |
| `content` | `str` | Text content |

### Conversation

Manages conversation history with ordered messages.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Conversation UUID |
| `messages` | `list[Message]` | Chronological messages |

**Methods:**
- `add_message(role, content)` - Add a message
- `messages_dict` - Property: list of `{role, content}` dicts for API calls

---

## Prompt Models

### Prompt

Basic prompt container.

| Field | Type |
|-------|------|
| `system_prompt` | `str` |

### SystemPrompt

System prompt with tool integration via Jinja2 template.

| Field | Type |
|-------|------|
| `content` | `str` |

**Methods:**
- `to_formatted_string(tools: list[Callable])` - Format with tool definitions

---

## Request Configuration

### RequestConfig

Configuration for a single LLM request.

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | `provider:model_name` |
| `system_prompt` | `SystemPrompt` | Model behavior |
| `response_format` | `type[BaseModel] \| None` | Structured output schema |
| `tools` | `list[Callable]` | Available tools |

**Methods:**
- `add_message_to_conversation(role, content)`
- `get_conversation()` - Full conversation including system message

---

## Embedding

### EmbeddingResult

Result of embedding generation.

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `list[list[float]]` | One vector per input text |
| `model` | `str` | Model identifier |
| `tokens_used` | `int` | Total tokens |
| `estimated_cost_usd` | `float \| None` | Estimated cost |

**Methods:**
- `get_embedding(index=0)` - Get single vector by index
- `dimension` - Property: vector dimension
- `count` - Property: number of embeddings

---

## Batch Processing

See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for usage details.

### BatchErrorStrategy

- `FAIL_FAST` - Stop on first error
- `CONTINUE` - Process all, collect errors
- `SKIP` - Return only successful results

### BatchRequest

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Auto-generated if not provided |
| `user_message` | `str` | User request |
| `system_prompt` | `str \| None` | Override |
| `tools` | `list[Callable] \| None` | Override |
| `execute_tools` | `bool` | Whether to execute tools |
| `response_format` | `type[BaseModel] \| None` | Structured output |
| `metadata` | `dict` | Custom metadata |

### BatchResult

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Request ID |
| `success` | `bool` | Whether succeeded |
| `response` | `str \| None` | Response text |
| `structured_output` | `Any \| None` | Parsed model (structured) |
| `tool_calls_made` | `int` | Tool call count |
| `tool_execution_history` | `list[dict]` | Tool history |
| `tokens_used` | `dict \| None` | Token usage |
| `error` | `str \| None` | Error message |
| `error_type` | `str \| None` | Exception type |
| `elapsed_time` | `float` | Processing time (seconds) |

### BatchConfig

| Field | Type | Default |
|-------|------|---------|
| `max_concurrent` | `int` | `5` |
| `error_strategy` | `BatchErrorStrategy` | `CONTINUE` |
| `show_progress` | `bool` | `False` |
| `retry_failed` | `bool` | `False` |
| `api_keys` | `list[APIKeyConfig] \| None` | `None` |

---

## Evaluation

### EvalRecord

Complete request/response lifecycle for evaluation and debugging.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Record UUID |
| `correlation_id` | `str \| None` | Request tracking ID |
| `timestamp` | `datetime` | Request time |
| `user_message` | `str` | User input |
| `system_prompt` | `str` | System prompt |
| `model` | `str` | Model identifier |
| `messages_snapshot` | `list[dict]` | Full conversation state |
| `final_response` | `str` | Final text response |
| `tool_calls_made` | `int` | Tool call count |
| `tool_execution_history` | `list[dict]` | Tool history |
| `tokens_used` | `dict \| None` | Token usage |
| `estimated_cost_usd` | `float \| None` | Estimated cost |
| `success` | `bool` | Whether succeeded |
| `error_type` | `str \| None` | Error type if failed |
| `error_message` | `str \| None` | Error message if failed |

Agent fields (when using AgentExecutor): `agent_name`, `agent_description`, `agent_model`, `agent_system_prompt`, `agent_tools`, `agent_max_tool_iterations`.

---

## Workflow

### WorkflowResult

| Field | Type | Description |
|-------|------|-------------|
| `final_output` | `str` | Final output |
| `iterations` | `int` | Iteration count |
| `agent_interactions` | `list[dict]` | Interaction history |
| `metadata` | `dict` | Additional metadata |
| `hooks_executed` | `int` | Hook count |
| `hook_errors` | `list[dict]` | Hook errors |

Workflow config models (e.g., `IterativeConfig`, `MapReduceConfig`, `ReActConfig`) are documented in [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md).

---

## See Also

- [API.md](API.md) - ExecutionResult, StreamingChunk
- [BATCH_PROCESSING.md](BATCH_PROCESSING.md) - Batch models usage
