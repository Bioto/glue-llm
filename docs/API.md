# GlueLLM API Documentation

GlueLLM provides a high-level Python API for working with Large Language Models, with automatic tool execution and structured output support.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

### Simple Completion

```python
from source.api import complete

result = complete(
    user_message="What is the capital of France?",
    system_prompt="You are a helpful geography assistant.",
)

print(result.final_response)
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

print(f"Name: {person.name}, Age: {person.age}, City: {person.city}")
```

### Automatic Tool Execution

```python
from source.api import complete

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.
    
    Args:
        location: City and country, e.g. "Tokyo, Japan"
        unit: Temperature unit ("celsius" or "fahrenheit")
    """
    return f"Weather in {location}: 22°{unit[0].upper()}, sunny"

result = complete(
    user_message="What's the weather in Tokyo?",
    system_prompt="Use get_weather tool for weather queries.",
    tools=[get_weather],
)

print(result.final_response)
print(f"Tool calls made: {result.tool_calls_made}")
```

## Core Features

### 1. Automatic Tool Execution Loop

GlueLLM automatically handles the tool execution loop for you:

- LLM requests to call a tool
- Tool is executed
- Result is sent back to LLM
- Process repeats until LLM provides final answer

No more manual tool call handling!

### 2. Structured Output with Pydantic

Define your expected output structure using Pydantic models, and GlueLLM returns typed, validated data:

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    address: Address

# Get structured output
person = structured_complete(
    user_message="Extract: John, 30, 123 Main St, Springfield, 12345",
    response_format=Person,
)

# Access with type safety
print(person.address.city)  # "Springfield"
```

### 3. Conversation State Management

Use the `GlueLLM` client for multi-turn conversations:

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

# Reset conversation
client.reset_conversation()
```

### 4. Multiple Tools Support

Provide multiple tools and let the LLM choose which to use:

```python
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    return f"Time in {timezone}: 12:00 PM"

result = complete(
    user_message="What's the weather and time in Tokyo?",
    tools=[get_weather, get_time],
)
```

## API Reference

### Functions

#### `complete()`

Quick completion with automatic tool execution.

```python
def complete(
    user_message: str,
    model: str = "openai:gpt-4o-mini",
    system_prompt: Optional[str] = None,
    tools: Optional[list[Callable]] = None,
    execute_tools: bool = True,
    max_tool_iterations: int = 10,
) -> ToolExecutionResult
```

**Parameters:**
- `user_message` (str): The user's message/request
- `model` (str): Model identifier in format "provider:model_name"
- `system_prompt` (Optional[str]): System prompt content
- `tools` (Optional[list[Callable]]): List of callable functions to use as tools
- `execute_tools` (bool): Whether to automatically execute tools
- `max_tool_iterations` (int): Maximum number of tool call iterations

**Returns:**
- `ToolExecutionResult`: Contains final response, tool call history, and metadata

#### `structured_complete()`

Quick completion with structured output.

```python
def structured_complete(
    user_message: str,
    response_format: Type[T],
    model: str = "openai:gpt-4o-mini",
    system_prompt: Optional[str] = None,
) -> T
```

**Parameters:**
- `user_message` (str): The user's message/request
- `response_format` (Type[T]): Pydantic model class for structured output
- `model` (str): Model identifier in format "provider:model_name"
- `system_prompt` (Optional[str]): System prompt content

**Returns:**
- Instance of `response_format` with parsed data

### Classes

#### `GlueLLM`

Main client for LLM interactions with conversation state.

```python
class GlueLLM:
    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        system_prompt: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
        max_tool_iterations: int = 10,
    )
```

**Methods:**

##### `complete()`

```python
def complete(
    self,
    user_message: str,
    execute_tools: bool = True,
) -> ToolExecutionResult
```

Complete a request with automatic tool execution loop.

##### `structured_complete()`

```python
def structured_complete(
    self,
    user_message: str,
    response_format: Type[T],
) -> T
```

Complete a request and return structured output.

##### `reset_conversation()`

```python
def reset_conversation(self) -> None
```

Reset the conversation history.

#### `ToolExecutionResult`

Result object containing execution details.

```python
class ToolExecutionResult(BaseModel):
    final_response: str  # The final text response from the model
    tool_calls_made: int  # Number of tool calls made
    tool_execution_history: list[dict[str, Any]]  # History of tool calls
    raw_response: ChatCompletion  # The raw final response
```

**Tool Execution History Format:**

```python
{
    "tool_name": str,      # Name of the tool called
    "arguments": dict,     # Arguments passed to the tool
    "result": str,         # Result returned by the tool
}
```

## Examples

### Example 1: Weather Assistant

```python
from source.api import GlueLLM

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast."""
    return f"{days}-day forecast for {location}: Mostly sunny"

client = GlueLLM(
    system_prompt="You are a weather assistant. Use tools to answer weather queries.",
    tools=[get_weather, get_forecast],
)

result = client.complete("What's the weather in Paris and give me a 5-day forecast?")
print(result.final_response)
```

### Example 2: Data Extraction Pipeline

```python
from source.api import structured_complete
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool

products_text = """
- iPhone 15 Pro: $999, Electronics, Available
- Nike Shoes: $120, Clothing, Out of stock
"""

product = structured_complete(
    user_message=f"Extract first product: {products_text}",
    response_format=ProductInfo,
)

print(f"Product: {product.name}, Price: ${product.price}")
```

### Example 3: Calculator with Memory

```python
from source.api import GlueLLM

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calc = GlueLLM(
    system_prompt="You are a calculator. Use the calculate tool for all math.",
    tools=[calculate],
)

# Multi-turn with context
result1 = calc.complete("What is 25 * 4?")
print(result1.final_response)

result2 = calc.complete("Add 50 to that")
print(result2.final_response)

result3 = calc.complete("Divide the result by 2")
print(result3.final_response)
```

### Example 4: Research Assistant with Multiple Tools

```python
from source.api import complete

def search_papers(query: str) -> str:
    """Search academic papers."""
    return f"Found 5 papers about '{query}'"

def summarize_paper(paper_id: str) -> str:
    """Get paper summary."""
    return f"Summary of paper {paper_id}: Important findings..."

def get_citations(paper_id: str) -> str:
    """Get citation count."""
    return f"Paper {paper_id} has 42 citations"

result = complete(
    user_message="Find papers about neural networks, summarize the top one, and tell me its citations",
    system_prompt="You are a research assistant. Use tools to help with academic research.",
    tools=[search_papers, summarize_paper, get_citations],
)

print(result.final_response)
print(f"\nTools used: {result.tool_calls_made}")
for exec_info in result.tool_execution_history:
    print(f"  - {exec_info['tool_name']}: {exec_info['result'][:50]}...")
```

## Best Practices

### 1. Tool Design

**DO:**
- Write clear docstrings (LLM uses them to understand the tool)
- Use type hints for all parameters
- Return strings or simple types
- Handle errors gracefully within tools

```python
def good_tool(city: str, unit: str = "celsius") -> str:
    """Get weather for a city.
    
    Args:
        city: The city name (e.g., "Tokyo")
        unit: Temperature unit ("celsius" or "fahrenheit")
    
    Returns:
        Weather description string
    """
    try:
        # Implementation
        return f"Weather in {city}: 22°{unit[0].upper()}"
    except Exception as e:
        return f"Error: {e}"
```

**DON'T:**
- Use complex return types (dicts, objects)
- Omit docstrings or parameter descriptions
- Let exceptions propagate

### 2. System Prompts

**DO:**
- Be specific about tool usage
- Provide context about the assistant's role
- Include formatting instructions if needed

```python
system_prompt = """
You are a weather assistant. 
Use the get_weather tool for current conditions.
Use the get_forecast tool for future predictions.
Always include temperature and conditions in your responses.
"""
```

**DON'T:**
- Use overly long system prompts
- Contradict tool capabilities
- Leave system prompt empty when using tools

### 3. Structured Output

**DO:**
- Use descriptive field names
- Add Field descriptions
- Use appropriate types (int, float, bool, str)
- Use nested models for complex structures

```python
class GoodModel(BaseModel):
    full_name: Annotated[str, Field(description="Person's full name")]
    age_years: Annotated[int, Field(description="Age in years")]
    is_active: Annotated[bool, Field(description="Whether person is active")]
```

**DON'T:**
- Use vague field names like `data` or `value`
- Omit Field descriptions
- Over-nest models unnecessarily

### 4. Error Handling

Always handle potential errors:

```python
try:
    result = complete(
        user_message="Process this",
        tools=[my_tool],
    )
    print(result.final_response)
except Exception as e:
    print(f"Error: {e}")
```

### 5. Token Management

For long conversations, reset periodically:

```python
client = GlueLLM(tools=[tool1, tool2])

for i, user_input in enumerate(user_inputs):
    result = client.complete(user_input)
    
    # Reset every 10 turns to manage context
    if i % 10 == 0:
        client.reset_conversation()
```

### 6. Model Selection

Choose the right model for your task:

- **Fast/Cheap**: `openai:gpt-4o-mini` - Good for simple tasks
- **Balanced**: `openai:gpt-4o` - Good for most tasks
- **Advanced**: `openai:gpt-4` - Complex reasoning

```python
# For simple extraction
result = complete(
    user_message="Extract name from: John Smith",
    model="openai:gpt-4o-mini",
)

# For complex tool orchestration
result = complete(
    user_message="Complex multi-tool task...",
    model="openai:gpt-4o",
    tools=[tool1, tool2, tool3],
)
```

## Troubleshooting

### Tools Not Being Called

- Check that tool docstrings are clear
- Verify system prompt mentions tool usage
- Ensure `execute_tools=True` (default)

### Infinite Tool Loops

- Adjust `max_tool_iterations` (default: 10)
- Check that tools return useful information
- Verify tools don't have contradictory outputs

### Structured Output Validation Errors

- Ensure model output matches Pydantic schema
- Add more specific Field descriptions
- Use simpler models for better reliability

### Conversation Context Issues

- Check that you're using `GlueLLM` client (not `complete()` function)
- Verify conversation isn't being reset unintentionally
- Monitor conversation length for token limits

## Contributing

See the main [README.md](../README.md) for contribution guidelines.

## License

See [LICENSE](../LICENSE) for details.

