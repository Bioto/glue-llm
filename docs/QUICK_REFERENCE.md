# GlueLLM Quick Reference

Quick reference for common GlueLLM operations.

## Imports

```python
from source.api import (
    complete,              # One-off completion with tools
    structured_complete,   # One-off structured output
    GlueLLM,              # Client for multi-turn conversations
    ToolExecutionResult,  # Result type
)
from pydantic import BaseModel, Field
from typing import Annotated
```

## Basic Operations

### Simple Completion

```python
result = complete(user_message="Your question here")
print(result.final_response)
```

### With Custom Model

```python
result = complete(
    user_message="Your question",
    model="openai:gpt-4o",
)
```

### With System Prompt

```python
result = complete(
    user_message="Your question",
    system_prompt="You are an expert in...",
)
```

## Tool Execution

### Single Tool

```python
def my_tool(param: str) -> str:
    """Tool description.

    Args:
        param: Parameter description
    """
    return f"Result for {param}"

result = complete(
    user_message="Use my_tool with 'test'",
    tools=[my_tool],
)
```

### Multiple Tools

```python
def tool1(x: str) -> str:
    """Tool 1 description."""
    return "Result 1"

def tool2(y: int) -> str:
    """Tool 2 description."""
    return "Result 2"

result = complete(
    user_message="Your request",
    tools=[tool1, tool2],
)
```

### Check Tool Usage

```python
result = complete(user_message="...", tools=[...])

print(f"Tools called: {result.tool_calls_made}")

for call in result.tool_execution_history:
    print(f"  {call['tool_name']}({call['arguments']}) -> {call['result']}")
```

### Disable Tool Execution

```python
result = complete(
    user_message="...",
    tools=[my_tool],
    execute_tools=False,  # LLM can see tools but won't execute
)
```

## Structured Output

### Simple Model

```python
class Response(BaseModel):
    answer: str
    confidence: float

result = structured_complete(
    user_message="Extract data from...",
    response_format=Response,
)

print(result.answer)
print(result.confidence)
```

### With Field Descriptions

```python
class Person(BaseModel):
    name: Annotated[str, Field(description="Full name")]
    age: Annotated[int, Field(description="Age in years")]
    email: Annotated[str, Field(description="Email address")]

person = structured_complete(
    user_message="Extract: John Doe, 30, john@example.com",
    response_format=Person,
)
```

### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    address: Address

person = structured_complete(
    user_message="Extract person info...",
    response_format=Person,
)

print(person.address.city)
```

### With Lists

```python
class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: list[Item]
    total: float

order = structured_complete(
    user_message="Extract order with multiple items...",
    response_format=Order,
)

for item in order.items:
    print(f"{item.name}: ${item.price}")
```

## Multi-turn Conversations

### Basic Conversation

```python
client = GlueLLM()

result1 = client.complete("My name is Alice")
result2 = client.complete("What's my name?")  # Remembers context
```

### With Tools

```python
client = GlueLLM(
    tools=[tool1, tool2],
    system_prompt="Use tools as needed",
)

result = client.complete("Use tools to answer...")
```

### Structured Completion in Conversation

```python
client = GlueLLM()

# Regular turn
client.complete("Tell me about Paris")

# Structured turn
info = client.structured_complete(
    user_message="Extract key facts from what you just said",
    response_format=CityInfo,
)
```

### Reset Conversation

```python
client = GlueLLM()

client.complete("Message 1")
client.complete("Message 2")

# Start fresh
client.reset_conversation()

client.complete("New conversation")
```

### Conversation with Periodic Reset

```python
client = GlueLLM()

for i, message in enumerate(messages):
    result = client.complete(message)

    # Reset every 10 messages to manage context
    if i % 10 == 0:
        client.reset_conversation()
```

## Configuration

### Custom Settings

```python
client = GlueLLM(
    model="openai:gpt-4o-mini",
    system_prompt="Your instructions",
    tools=[tool1, tool2],
    max_tool_iterations=5,  # Limit tool loops
)
```

### Available Models

```python
# Fast and cheap
model="openai:gpt-4o-mini"

# Balanced
model="openai:gpt-4o"

# Most capable
model="openai:gpt-4"
```

## Error Handling

### Basic Try-Catch

```python
try:
    result = complete(user_message="...")
    print(result.final_response)
except Exception as e:
    print(f"Error: {e}")
```

### Tool Errors

```python
def risky_tool(x: str) -> str:
    """Tool that might fail."""
    try:
        # Risky operation
        return "Success"
    except Exception as e:
        return f"Error: {e}"  # Return error as string

result = complete(
    user_message="Use risky_tool",
    tools=[risky_tool],
)
```

### Validation Errors (Structured Output)

```python
try:
    result = structured_complete(
        user_message="...",
        response_format=MyModel,
    )
except ValidationError as e:
    print(f"Failed to parse: {e}")
```

## Common Patterns

### Data Extraction Pipeline

```python
class ExtractedData(BaseModel):
    field1: str
    field2: int

results = []
for text in texts:
    data = structured_complete(
        user_message=f"Extract from: {text}",
        response_format=ExtractedData,
    )
    results.append(data)
```

### Interactive Chat Loop

```python
client = GlueLLM(system_prompt="You are a helpful assistant.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    result = client.complete(user_input)
    print(f"Bot: {result.final_response}")
```

### Agent with Context

```python
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"

def summarize_tool(text: str) -> str:
    """Summarize text."""
    return f"Summary of {text[:50]}..."

agent = GlueLLM(
    system_prompt="You are a research assistant. Use tools to help users.",
    tools=[search_tool, summarize_tool],
)

result = agent.complete("Research topic X and summarize findings")
```

### Batch Processing with Progress

```python
from tqdm import tqdm

class Result(BaseModel):
    category: str
    sentiment: str

results = []
for item in tqdm(items):
    result = structured_complete(
        user_message=f"Analyze: {item}",
        response_format=Result,
    )
    results.append(result)
```

### Classification Task

```python
class Classification(BaseModel):
    category: Annotated[str, Field(description="One of: spam, important, newsletter")]
    confidence: Annotated[float, Field(description="Confidence 0-1")]

def classify_email(email_text: str) -> Classification:
    return structured_complete(
        user_message=f"Classify this email: {email_text}",
        response_format=Classification,
    )

# Use it
result = classify_email("Your email text...")
print(f"Category: {result.category} ({result.confidence:.2%} confident)")
```

### Named Entity Extraction

```python
class Entity(BaseModel):
    text: str
    type: str  # PERSON, ORG, LOCATION, etc.

class Entities(BaseModel):
    entities: list[Entity]

entities = structured_complete(
    user_message="Extract entities from: Apple CEO Tim Cook visited Paris",
    response_format=Entities,
)

for entity in entities.entities:
    print(f"{entity.text} ({entity.type})")
```

## Tips & Tricks

### Better Tool Descriptions

```python
# ❌ Bad
def tool(x):
    return x

# ✅ Good
def tool(location: str, unit: str = "celsius") -> str:
    """Get weather for a location.

    Args:
        location: City and country, e.g. "Tokyo, Japan"
        unit: Temperature unit, "celsius" or "fahrenheit"

    Returns:
        Weather description string
    """
    return f"Weather in {location}: 22°{unit[0].upper()}"
```

### Better System Prompts

```python
# ❌ Too vague
system_prompt = "You help users"

# ✅ Specific and clear
system_prompt = """
You are a customer service assistant for TechCorp.
Use get_order_status to check orders.
Use process_refund to handle refunds.
Always be polite and verify order numbers.
"""
```

### Better Field Descriptions

```python
# ❌ Unclear
class Data(BaseModel):
    value: str

# ✅ Descriptive
class Data(BaseModel):
    full_name: Annotated[str, Field(
        description="Person's complete name in format 'First Last'"
    )]
    age_years: Annotated[int, Field(
        description="Age in complete years (not months or days)"
    )]
```

### Access Raw Response

```python
result = complete(user_message="...")

# Get raw LLM response object
raw = result.raw_response

# Access details
print(raw.usage.prompt_tokens)
print(raw.usage.completion_tokens)
print(raw.model)
```

### Conversation Message Count

```python
client = GlueLLM()

client.complete("Message 1")
client.complete("Message 2")

# Check conversation length
msg_count = len(client._conversation.messages)
print(f"Messages in conversation: {msg_count}")
```

## Debugging

### Print Tool Execution

```python
result = complete(
    user_message="...",
    tools=[my_tool],
)

print(f"Tool calls: {result.tool_calls_made}")
for i, call in enumerate(result.tool_execution_history, 1):
    print(f"\n{i}. {call['tool_name']}")
    print(f"   Args: {call['arguments']}")
    print(f"   Result: {call['result']}")
```

### Inspect Conversation

```python
client = GlueLLM()
client.complete("Hello")
client.complete("How are you?")

# See all messages
for msg in client._conversation.messages:
    print(f"{msg.role}: {msg.content}")
```

### Check Model Response

```python
result = complete(user_message="...")

# Check if response exists
if result.final_response:
    print(result.final_response)
else:
    print("No response received")

# Check raw response
print(result.raw_response.choices[0].message)
```

## See Also

- [Full API Documentation](API.md)
- [Getting Started Guide](GETTING_STARTED.md)
- [Examples](../examples/)
- [Tests](../tests/)
