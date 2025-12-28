"""Configuration Examples

This script demonstrates different ways to configure GlueLLM.
"""

import os

from source.api import GlueLLM
from source.config import GlueLLMSettings, settings

print("=" * 60)
print("GlueLLM Configuration Examples")
print("=" * 60)

# ============================================================================
# Example 1: Using Global Settings
# ============================================================================
print("\n1. Using Global Settings")
print("-" * 60)

print(f"Default Model: {settings.default_model}")
print(f"Default System Prompt: {settings.default_system_prompt}")
print(f"Max Tool Iterations: {settings.max_tool_iterations}")
print(f"Retry Max Attempts: {settings.retry_max_attempts}")

# ============================================================================
# Example 2: Using Defaults (from settings)
# ============================================================================
print("\n2. Using Defaults from Settings")
print("-" * 60)

# When no parameters are provided, uses settings.default_model
# and settings.default_system_prompt
client = GlueLLM()
print(f"Client Model: {client.model}")
print(f"Client System Prompt: {client.system_prompt}")
print(f"Client Max Tool Iterations: {client.max_tool_iterations}")

# ============================================================================
# Example 3: Overriding Settings Per Request
# ============================================================================
print("\n3. Overriding Settings Per Request")
print("-" * 60)

custom_client = GlueLLM(
    model="anthropic:claude-3-5-sonnet-20241022",
    system_prompt="You are a coding assistant.",
    max_tool_iterations=15,
)
print(f"Custom Client Model: {custom_client.model}")
print(f"Custom Client System Prompt: {custom_client.system_prompt}")
print(f"Custom Client Max Tool Iterations: {custom_client.max_tool_iterations}")

# ============================================================================
# Example 4: Modifying Settings at Runtime
# ============================================================================
print("\n4. Modifying Settings at Runtime")
print("-" * 60)

original_model = settings.default_model
print(f"Original Default Model: {original_model}")

# Modify settings
settings.default_model = "openai:gpt-4o"
print(f"Modified Default Model: {settings.default_model}")

# New clients will use the modified default
new_client = GlueLLM()
print(f"New Client Model: {new_client.model}")

# Restore original
settings.default_model = original_model
print(f"Restored Default Model: {settings.default_model}")

# ============================================================================
# Example 5: Creating Custom Settings Instance
# ============================================================================
print("\n5. Creating Custom Settings Instance")
print("-" * 60)

custom_settings = GlueLLMSettings(
    default_model="openai:gpt-4o",
    default_system_prompt="You are a specialized assistant.",
    max_tool_iterations=20,
    retry_max_attempts=5,
)

print(f"Custom Settings Model: {custom_settings.default_model}")
print(f"Custom Settings System Prompt: {custom_settings.default_system_prompt}")
print(f"Custom Settings Max Iterations: {custom_settings.max_tool_iterations}")
print(f"Custom Settings Retry Attempts: {custom_settings.retry_max_attempts}")

# ============================================================================
# Example 6: Environment Variables
# ============================================================================
print("\n6. Checking Environment Variables")
print("-" * 60)

# Check if environment variables are set
env_vars = [
    "GLUELLM_DEFAULT_MODEL",
    "GLUELLM_DEFAULT_SYSTEM_PROMPT",
    "GLUELLM_MAX_TOOL_ITERATIONS",
    "GLUELLM_RETRY_MAX_ATTEMPTS",
]

print("Current environment variables:")
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"  {var}: {value}")
    else:
        print(f"  {var}: (not set)")

# ============================================================================
# Example 7: Configuration Best Practices
# ============================================================================
print("\n7. Configuration Best Practices")
print("-" * 60)

print("""
Best Practices:
1. Use .env file for local development
2. Use environment variables in production
3. Override per request when you need different behavior
4. Keep sensitive API keys in environment, not in code

Example .env file:
```
GLUELLM_DEFAULT_MODEL=openai:gpt-4o-mini
GLUELLM_MAX_TOOL_ITERATIONS=10
OPENAI_API_KEY=your-api-key-here
```

Example usage:
```python
# Uses default settings from .env
client = GlueLLM()

# Override for specific use case
special_client = GlueLLM(
    model="openai:gpt-4o",
    max_tool_iterations=20,
)
```
""")

print("\n" + "=" * 60)
print("Configuration examples completed!")
print("=" * 60)
