# Tool Support for Structured Completions - Implementation Summary

## Overview

The `structured_complete` methods now support tools! The LLM can call tools to gather information, perform calculations, or validate data before returning the final structured output.

## Changes Made

### 1. API Updates (`gluellm/api.py`)

#### `GlueLLM.structured_complete()` - Instance Method
**New Parameters:**
- `tools: list[Callable] | None = None` - List of tools to use (defaults to instance tools)
- `execute_tools: bool = True` - Whether to automatically execute tools

**Implementation:**
- Added full tool execution loop similar to `complete()`
- Tools are called iteratively until LLM returns structured output
- Tracks tool calls and execution history
- Aggregates token usage and costs across all iterations
- On final iteration or when no tools needed, requests structured output
- Returns `ExecutionResult` with both structured output and tool metadata

#### `structured_complete()` - Standalone Function
**New Parameters:**
- `tools: list[Callable] | None = None` - List of tools to use
- `execute_tools: bool = True` - Whether to automatically execute tools
- `max_tool_iterations: int | None = None` - Maximum tool call iterations

#### `_find_tool()` - Helper Method
**Updated Signature:**
- `tools: list[Callable] | None = None` - Optional tools list to search (defaults to `self.tools`)

### 2. Tests (`tests/test_api.py`)

Added two new test cases:
- `test_structured_output_with_tools()` - Tests basic tool usage with structured output
- `test_structured_output_with_multiple_tool_calls()` - Tests multiple tool calls

### 3. Documentation

#### README.md
- Updated convenience function example to show tool parameters
- Added comprehensive "Structured Output with Tools" section with:
  - Weather report example showing single tool call
  - Data analysis example showing multiple tool calls
  - Explanation of how it works
  - Common use cases

#### CHANGELOG.md
- Added entry documenting the new feature

### 4. Examples

Created `examples/structured_completion_with_tools.py` with 4 comprehensive examples:
1. Weather report using weather API tool
2. Mathematical calculation using calculator tool
3. Direct structured response without tools
4. Weather comparison using multiple tool calls

## How It Works

### Execution Flow

1. **Initialize**: User calls `structured_complete()` with a Pydantic model and optional tools
2. **Tool Loop**:
   - LLM evaluates if tools are needed
   - If yes: calls tools, receives results, loops back
   - If no: proceeds to step 3
3. **Structured Output**: LLM returns data in the specified Pydantic format
4. **Return**: User receives typed, validated data plus execution metadata

### Key Features

- **Automatic Tool Orchestration**: LLM decides when and how to use tools
- **Multi-iteration Support**: Can make multiple tool calls before returning
- **Cost Tracking**: Aggregates costs across all iterations
- **Token Tracking**: Sums token usage from all LLM calls
- **Error Handling**: Tool execution errors are captured and reported to LLM
- **Type Safety**: Final output is always validated against the Pydantic model

## Example Usage

```python
from pydantic import BaseModel, Field
from typing import Annotated
from gluellm.api import structured_complete

class WeatherReport(BaseModel):
    location: Annotated[str, Field(description="City name")]
    temperature: Annotated[float, Field(description="Temperature in Celsius")]
    conditions: Annotated[str, Field(description="Weather conditions")]

def get_weather(city: str) -> dict:
    """Fetch weather data for a city."""
    # Call external weather API
    return {"temp": 18, "conditions": "sunny"}

# LLM will call get_weather, then return structured output
result = await structured_complete(
    user_message="What's the weather in San Francisco?",
    response_format=WeatherReport,
    tools=[get_weather],
)

print(f"Temperature: {result.structured_output.temperature}°C")
print(f"Tool calls made: {result.tool_calls_made}")
print(f"Cost: ${result.estimated_cost_usd:.6f}")
```

## Use Cases

1. **Data Fetching**: Query databases/APIs before structuring results
2. **Validation**: Verify information using external sources
3. **Calculations**: Perform computations before generating reports
4. **Multi-step Reasoning**: Gather evidence before drawing conclusions

## Backward Compatibility

✅ **Fully backward compatible**
- Existing code continues to work without changes
- Tools are optional (defaults to no tools)
- Same return type (`ExecutionResult`)
- Same error handling

## Testing

Run the new tests:
```bash
pytest tests/test_api.py::TestStructuredOutput::test_structured_output_with_tools -v
pytest tests/test_api.py::TestStructuredOutput::test_structured_output_with_multiple_tool_calls -v
```

Run the examples:
```bash
python examples/structured_completion_with_tools.py
```

## Files Modified

1. `gluellm/api.py` - Core implementation
2. `tests/test_api.py` - New tests
3. `README.md` - Updated documentation
4. `CHANGELOG.md` - Added changelog entry
5. `examples/structured_completion_with_tools.py` - New comprehensive example

## Notes

- No linter errors introduced
- All existing tests should continue to pass
- The implementation follows the same patterns as `complete()` for consistency
- Tool execution is traced in OpenTelemetry spans when tracing is enabled
