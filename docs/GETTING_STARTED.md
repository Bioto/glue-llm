# Getting Started with GlueLLM

This guide will help you get up and running with GlueLLM in 5 minutes.

## Prerequisites

- Python 3.12 or higher
- API key for OpenAI (or another LLM provider)
- `uv` package manager (recommended) or `pip`

## Installation

### Step 1: Clone/Install

```bash
# If you have the project locally
cd /path/to/gluellm

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Step 2: Set Environment Variables

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Or add to your ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

## Your First GlueLLM Program

### Example 1: Simple Chat

Create a file called `hello.py`:

```python
from source.api import complete

# Simple question-answering
result = complete(
    user_message="What is the capital of France?",
    system_prompt="You are a helpful geography assistant.",
)

print(result.final_response)
```

Run it:
```bash
python hello.py
```

### Example 2: Using Tools

Create a file called `weather.py`:

```python
from source.api import complete

# Define a tool (function the LLM can call)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and country, e.g. "Tokyo, Japan"
        unit: Temperature unit, either "celsius" or "fahrenheit"
    """
    # In real life, you'd call a weather API here
    return f"The weather in {location} is 22 degrees {unit} and sunny."

# Ask a question that requires the tool
result = complete(
    user_message="What's the weather like in Tokyo, Japan?",
    system_prompt="You are a weather assistant. Use get_weather to answer questions.",
    tools=[get_weather],
)

print(result.final_response)
print(f"\nTool was called {result.tool_calls_made} times")
```

Run it:
```bash
python weather.py
```

GlueLLM will automatically:
1. Send your message to the LLM
2. Detect that the LLM wants to call `get_weather`
3. Execute the function with the right parameters
4. Send the result back to the LLM
5. Get the final natural language response

### Example 3: Structured Output

Create a file called `extract.py`:

```python
from source.api import structured_complete
from pydantic import BaseModel, Field
from typing import Annotated

# Define the structure you want
class Person(BaseModel):
    name: Annotated[str, Field(description="Person's full name")]
    age: Annotated[int, Field(description="Age in years")]
    occupation: Annotated[str, Field(description="Job or profession")]

# Extract structured data from text
person = structured_complete(
    user_message="Extract info: John Smith is a 35 year old software engineer.",
    response_format=Person,
)

print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")

# It's a real Pydantic object!
print(f"\nType: {type(person)}")
```

Run it:
```bash
python extract.py
```

### Example 4: Multi-turn Conversation

Create a file called `conversation.py`:

```python
from source.api import GlueLLM

# Create a client to maintain conversation state
client = GlueLLM(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant with perfect memory.",
)

# First message
result1 = client.complete("My name is Alice and I'm 25 years old")
print("Bot:", result1.final_response)

# Second message - bot remembers the context
result2 = client.complete("How old am I?")
print("Bot:", result2.final_response)

# Third message
result3 = client.complete("What's my name?")
print("Bot:", result3.final_response)
```

Run it:
```bash
python conversation.py
```

## Using the CLI

GlueLLM comes with a CLI for testing and demos:

```bash
# Run quick demos
uv run python source/cli.py demo

# Run all examples
uv run python source/cli.py examples

# Run tests
uv run python source/cli.py run-tests

# Test tool calling
uv run python source/cli.py test-tool-call
```

## Next Steps

### Explore Examples

Check out more complex examples:

```bash
python examples/basic_usage.py
```

### Read the Documentation

- [API Documentation](API.md) - Complete API reference
- [README](../README.md) - Project overview

### Run Tests

See how features work:

```bash
# Run all tests
uv run pytest tests/ -v

# Run just API tests
uv run pytest tests/test_api.py -v

# Run a specific test
uv run pytest tests/test_api.py::TestToolExecution::test_single_tool_execution -v
```

### Build Your Own

Common patterns:

#### Pattern 1: Data Extraction Pipeline

```python
from source.api import structured_complete
from pydantic import BaseModel

class ExtractedData(BaseModel):
    # Define your fields...
    pass

# Process a batch
for item in data:
    extracted = structured_complete(
        user_message=f"Extract from: {item}",
        response_format=ExtractedData,
    )
    # Use extracted data...
```

#### Pattern 2: Agent with Multiple Tools

```python
from source.api import GlueLLM

def tool1(...): ...
def tool2(...): ...
def tool3(...): ...

agent = GlueLLM(
    system_prompt="You are an agent that can do X, Y, and Z.",
    tools=[tool1, tool2, tool3],
)

result = agent.complete("Complex task requiring multiple tools")
```

#### Pattern 3: Conversational Interface

```python
from source.api import GlueLLM

chat = GlueLLM()

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    result = chat.complete(user_input)
    print(f"Bot: {result.final_response}")
```

## Common Issues

### "API key not found"

Make sure you've set your environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### "Module not found"

Make sure you're in the right directory and installed dependencies:
```bash
cd /path/to/gluellm
uv pip install -e .
```

### Tools not being called

- Check that your tool has a clear docstring
- Mention the tool in your system prompt
- Verify the tool name matches the function name

### Import errors

Use the correct import path:
```python
from source.api import complete, GlueLLM, structured_complete
```

## Get Help

- Check the [API Documentation](API.md)
- Look at [examples/](../examples/)
- Read test cases in [tests/](../tests/)
- Review the source code in [source/api.py](../source/api.py)

## What's Next?

Once you're comfortable with the basics, you can:

1. **Integrate with real APIs** - Replace mock tool functions with real API calls
2. **Build agents** - Create complex multi-tool agents for specific domains
3. **Create pipelines** - Chain multiple LLM calls together
4. **Add streaming** - Implement streaming for real-time responses (coming soon)
5. **Optimize costs** - Use appropriate models and manage conversation length

Happy building! 🚀

