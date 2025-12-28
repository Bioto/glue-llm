# GlueLLM Test Suite

Comprehensive test suite for LLM interactions with edge cases and stress tests.

## Installation

Install test dependencies:

```bash
uv pip install -e ".[dev]"
```

## Running Tests

Run all tests:
```bash
uv run pytest
```

Run specific test class:
```bash
uv run pytest tests/test_llm_edge_cases.py::TestBasicToolCalling
```

Run with output:
```bash
uv run pytest -s
```

Skip integration tests (which call real LLM APIs):
```bash
uv run pytest -m "not integration"
```

## Test Categories

### TestBasicToolCalling
- Single tool calls
- Calculator operations
- Basic weather queries

### TestMultipleToolCalls
- Sequential tool execution
- Multi-step tool chains
- Complex workflows

### TestToolParameterEdgeCases
- Default vs specified parameters
- Complex optional parameters
- Filter combinations

### TestConfusingPrompts
- Ambiguous instructions
- Contradictory requirements
- Partial information
- Irrelevant tools available

### TestParameterCombinations
- High/low temperature
- Token limits
- Structured outputs
- Various model parameters

### TestStressScenarios
- Very long prompts
- Rapid tool switching
- Nested tool requirements
- Max iteration limits

### TestErrorHandling
- Invalid parameters
- Missing documentation
- Error recovery

### TestRealisticScenarios
- Customer service workflows
- Data analysis tasks
- Multi-turn conversations

## Notes

- These tests make **real API calls** to LLM providers
- Set appropriate API keys (XAI_API_KEY or OPENAI_API_KEY)
- Tests are designed to expose edge cases and potential failures
- Some tests may timeout or fail intentionally to test error handling
