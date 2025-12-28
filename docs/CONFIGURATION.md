# Configuration Guide

GlueLLM uses `pydantic-settings` for configuration management. This allows you to configure the library through environment variables, `.env` files, or programmatically.

## Configuration Methods

### 1. Environment Variables

Set environment variables with the `GLUELLM_` prefix:

```bash
export GLUELLM_DEFAULT_MODEL="anthropic:claude-3-5-sonnet-20241022"
export GLUELLM_DEFAULT_SYSTEM_PROMPT="You are a helpful coding assistant."
export GLUELLM_MAX_TOOL_ITERATIONS=15
```

### 2. .env File

Create a `.env` file in your project root:

```bash
# .env
GLUELLM_DEFAULT_MODEL=openai:gpt-4o-mini
GLUELLM_DEFAULT_SYSTEM_PROMPT=You are a helpful assistant.
GLUELLM_MAX_TOOL_ITERATIONS=10
GLUELLM_RETRY_MAX_ATTEMPTS=3
```

See `env.example` for a complete list of available settings.

### 3. Programmatic Configuration

Import and use the settings object:

```python
from gluellm import settings

# Read settings
print(settings.default_model)  # "openai:gpt-4o-mini"
print(settings.max_tool_iterations)  # 10

# Modify settings at runtime
settings.default_model = "anthropic:claude-3-5-sonnet-20241022"
settings.max_tool_iterations = 15
```

Or create a new settings instance:

```python
from gluellm import GlueLLMSettings

my_settings = GlueLLMSettings(
    default_model="openai:gpt-4o",
    max_tool_iterations=20,
)
```

## Available Settings

### Model Settings

- **`default_model`** (str): Default LLM model to use
  - Default: `"openai:gpt-4o-mini"`
  - Format: `"provider:model_name"`

- **`default_system_prompt`** (str): Default system prompt for all completions
  - Default: `"You are a helpful assistant."`

### Tool Execution Settings

- **`max_tool_iterations`** (int): Maximum number of tool execution loops
  - Default: `10`
  - Prevents infinite loops in tool execution

### Retry Settings

- **`retry_max_attempts`** (int): Maximum number of retry attempts for failed LLM calls
  - Default: `3`

- **`retry_min_wait`** (int): Minimum wait time (seconds) between retries
  - Default: `2`

- **`retry_max_wait`** (int): Maximum wait time (seconds) between retries
  - Default: `30`

- **`retry_multiplier`** (int): Multiplier for exponential backoff
  - Default: `1`

### Logging Settings

- **`log_level`** (str): Logging level
  - Default: `"INFO"`
  - Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

### API Keys (Optional)

While provider-specific API keys (e.g., `OPENAI_API_KEY`) are handled by the underlying `any-llm-sdk`, you can also set them via GlueLLM settings:

- **`openai_api_key`** (str, optional): OpenAI API key
- **`anthropic_api_key`** (str, optional): Anthropic API key
- **`xai_api_key`** (str, optional): xAI API key

## Usage in Code

### Using Default Settings

When you don't specify parameters, GlueLLM uses the configured defaults:

```python
from gluellm import GlueLLM, complete

# Uses settings.default_model and settings.default_system_prompt
client = GlueLLM()
result = client.complete("What is 2+2?")

# Same for convenience functions
result = complete("What is 2+2?")
```

### Overriding Settings

You can override settings per request:

```python
from gluellm import GlueLLM

# Override the default model
client = GlueLLM(
    model="anthropic:claude-3-5-sonnet-20241022",
    system_prompt="You are a specialized math tutor.",
    max_tool_iterations=20,
)
```

### Reloading Settings

If you modify environment variables or the `.env` file while your application is running:

```python
from gluellm import reload_settings

# Reload settings from environment
new_settings = reload_settings()
```

## Best Practices

1. **Use `.env` for local development**: Keep sensitive API keys and local preferences in a `.env` file (don't commit it!)

2. **Use environment variables in production**: Set configuration via environment variables in your deployment environment

3. **Override per request when needed**: Use default settings for most cases, but override when you need different behavior:

```python
# Most requests use default settings
client = GlueLLM()

# Special case: need more iterations
result = client.complete(
    "Complex task requiring many tool calls",
)

# For this specific call, use a different model
from gluellm import complete
result = complete(
    "Quick question",
    model="openai:gpt-4o-mini",  # Faster, cheaper model
)
```

4. **Configure retry settings based on your needs**: If you're dealing with rate limits, adjust the retry settings:

```bash
GLUELLM_RETRY_MAX_ATTEMPTS=5
GLUELLM_RETRY_MIN_WAIT=5
GLUELLM_RETRY_MAX_WAIT=60
```

## Example: Multi-Environment Setup

### Development (.env)
```bash
GLUELLM_DEFAULT_MODEL=openai:gpt-4o-mini
GLUELLM_LOG_LEVEL=DEBUG
OPENAI_API_KEY=your-dev-key
```

### Production (environment variables)
```bash
export GLUELLM_DEFAULT_MODEL=openai:gpt-4o
export GLUELLM_LOG_LEVEL=WARNING
export GLUELLM_RETRY_MAX_ATTEMPTS=5
export OPENAI_API_KEY=your-production-key
```

### Staging (custom settings)
```python
from gluellm import GlueLLMSettings, get_settings

# Load base settings
base_settings = get_settings()

# Override for staging
staging_settings = GlueLLMSettings(
    default_model="openai:gpt-4o-mini",
    log_level="INFO",
    retry_max_attempts=3,
)
```
