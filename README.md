# GlueLLM

A high-level Python SDK for Large Language Models with automatic tool execution and structured output support.

## Features

- 🔄 **Automatic Tool Execution Loop** - No manual tool call handling required
- 📊 **Structured Output** - Type-safe responses with Pydantic models
- 💬 **Conversation Management** - Built-in multi-turn conversation support
- 🛠️ **Multiple Tools** - Easy integration of multiple tools
- 🎯 **Simple API** - Clean, intuitive interface for common LLM tasks
- 🔌 **Provider Agnostic** - Built on `any-llm-sdk` for multi-provider support
- ⚡ **Automatic Retry with Exponential Backoff** - Smart retry logic for rate limits and connection issues
- 🛡️ **Comprehensive Error Handling** - Catch and classify errors from any LLM provider
- 📝 **Enhanced Logging** - Track retry attempts and tool execution errors

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"
```

### Simple Example

```python
from source.api import complete

result = complete(
    user_message="What is the capital of France?",
    system_prompt="You are a helpful geography assistant.",
)

print(result.final_response)
```

### With Tool Execution

```python
from source.api import complete

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.
    
    Args:
        location: City and country, e.g. "Tokyo, Japan"
        unit: Temperature unit ("celsius" or "fahrenheit")
    """
    # Your weather API call here
    return f"Weather in {location}: 22°{unit[0].upper()}, sunny"

result = complete(
    user_message="What's the weather in Tokyo and Paris?",
    system_prompt="You are a weather assistant. Use get_weather for queries.",
    tools=[get_weather],
)

print(result.final_response)
print(f"Tool calls made: {result.tool_calls_made}")
```

### Structured Output

```python
from source.api import structured_complete
from pydantic import BaseModel, Field
from typing import Annotated

class PersonInfo(BaseModel):
    name: Annotated[str, Field(description="Full name")]
    age: Annotated[int, Field(description="Age in years")]
    city: Annotated[str, Field(description="City of residence")]

person = structured_complete(
    user_message="Extract info: John Smith, 35, lives in Seattle",
    response_format=PersonInfo,
)

print(f"{person.name} is {person.age} years old and lives in {person.city}")
```

### Multi-turn Conversations

```python
from source.api import GlueLLM

client = GlueLLM(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)

# Turn 1
result1 = client.complete("My favorite color is blue")

# Turn 2 (has context from turn 1)
result2 = client.complete("What's my favorite color?")
print(result2.final_response)  # Will reference blue
```

## Documentation

- **[Error Handling & Retry Logic](docs/error-handling.md)** - Comprehensive error handling guide
- **[API Documentation](docs/API.md)** - Complete API reference and examples
- **[Examples](examples/)** - More usage examples
- **[Tests](tests/)** - Test suite with usage patterns

## Core Concepts

### Tool Execution Loop

GlueLLM automatically handles the tool execution loop:

1. Send user message to LLM
2. If LLM wants to call a tool, execute it
3. Send tool result back to LLM
4. Repeat until LLM provides final answer

No more manual tool call management!

### Structured Output

Define your expected output structure with Pydantic:

```python
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

product = structured_complete(
    user_message="Extract: iPhone 15 Pro, $999, available",
    response_format=Product,
)
# Returns type-safe Product instance
```

### Conversation State

Use the `GlueLLM` client to maintain conversation history:

```python
client = GlueLLM(tools=[my_tool])

client.complete("First message")
client.complete("Second message with context")
client.reset_conversation()  # Start fresh
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_api.py

# Run with verbose output
uv run pytest tests/ -v

# Using the CLI
uv run python source/cli.py run-tests
```

### Running Examples

```bash
# Run basic usage examples
uv run python examples/basic_usage.py

# Run CLI test commands
uv run python source/cli.py test-completion
uv run python source/cli.py test-tool-call
```

## Architecture

```
source/
├── api.py              # High-level API (GlueLLM, complete, structured_complete)
├── cli.py              # Command-line interface
└── models/
    ├── config.py       # Request configuration
    ├── conversation.py # Conversation management
    └── prompt.py       # Prompt templates and formatting

examples/
└── basic_usage.py      # Usage examples

tests/
├── test_api.py         # API tests
└── test_llm_edge_cases.py  # LLM integration tests

docs/
└── API.md             # Comprehensive API documentation
```

## Requirements

- Python 3.12+
- `any-llm-sdk[openai]>=1.5.0`
- `pydantic>=2.12.5`
- See [pyproject.toml](pyproject.toml) for full dependencies

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Run tests before submitting

## License

[Add your license here]

## Credits

Built on top of [any-llm-sdk](https://github.com/yourusername/any-llm-sdk) for multi-provider LLM support.

