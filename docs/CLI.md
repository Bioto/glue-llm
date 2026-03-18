# GlueLLM CLI

Command-line interface for testing and demonstrations.

## Usage

```bash
gluellm --help
gluellm <command> --help
```

## Command Groups

### Completion

- `test-completion` - Basic completion
- `test-streaming` - Streaming completion
- `test-structured` - Structured output

### Tools

- Tool calling and execution tests

### Infrastructure

- `test-error-handling` - Error classification
- `test-hooks` - Hook system
- `test-correlation-ids` - Correlation ID propagation
- `test-telemetry` - Tracing
- `test-rate-limiting` - Rate limiter
- `test-timeout` - Timeout behavior
- `test-api-key-pool` - API key pool
- `test-different-models` - Multi-provider

### Workflows

- `test-iterative-workflow` - Iterative refinement
- `test-pipeline-workflow` - Pipeline
- `test-debate-workflow` - Debate
- `test-consensus-workflow` - Consensus
- `test-round-robin-workflow` - Round robin
- Additional workflow tests

### Utilities

- Demo and example runners

## Examples

```bash
# Test completion
gluellm test-completion

# Test streaming with custom message
gluellm test-streaming --message "Write a haiku"

# Test iterative workflow
gluellm test-iterative-workflow --input "Topic: async Python" --iterations 3

# Test pipeline workflow
gluellm test-pipeline-workflow --input "Write about machine learning"

# Test infrastructure
gluellm test-rate-limiting --requests 10
gluellm test-timeout --timeout 5
```

## Version

```bash
gluellm --version
```

## See Also

- [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md) - Workflow usage
- [examples/](../examples/) - Full examples
