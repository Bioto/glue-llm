# GlueLLM Agents

Agents encapsulate LLM configuration (model, system prompt, tools, limits) for use in workflows and executors.

## Agent Base Class

```python
from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt

agent = Agent(
    name="Research Assistant",
    description="Helps with research tasks",
    system_prompt=SystemPrompt(content="You are a research assistant."),
    tools=[search_web, read_file],
    max_tool_iterations=5,
    model="openai:gpt-4o-mini",
    max_tokens=4096,
)
```

### Agent Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent identifier |
| `description` | `str` | Purpose and capabilities |
| `system_prompt` | `SystemPrompt` | Behavior definition |
| `tools` | `list[Callable]` | Available tools |
| `model` | `str` | LLM model (defaults to `settings.default_model`) |
| `max_tool_iterations` | `int` | Max tool rounds (default 10) |
| `max_tokens` | `int \| None` | Max completion tokens |

## GenericAgent

Pre-configured generic agent for quick setup or subclassing.

```python
from gluellm.agents.generic import GenericAgent

agent = GenericAgent()
# Default: "Generic Agent", pirate-themed system prompt, no tools
```

GenericAgent provides:
- Name: "Generic Agent"
- Description: "A generic agent that can use any tool"
- System prompt: "You are a generic agent that can use any tool. You are a pirate"
- Empty tools list
- 10 max tool iterations

## Using Agents with Executors

Agents are used via `AgentExecutor` in workflows:

```python
from gluellm.agents.generic import GenericAgent
from gluellm.executors import AgentExecutor
from gluellm.workflows.pipeline import PipelineWorkflow

agent = GenericAgent()
executor = AgentExecutor(agent)

workflow = PipelineWorkflow(stages=[("agent", executor)])
result = await workflow.execute("Hello")
```

## Creating Custom Agents

Subclass `Agent` or `GenericAgent`:

```python
from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt

class ResearchAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Research Agent",
            description="Searches and synthesizes information",
            system_prompt=SystemPrompt(
                content="You are a careful researcher. Cite sources and verify claims."
            ),
            tools=[search_web, fetch_url],
            max_tool_iterations=10,
            model="anthropic:claude-3-5-sonnet-20241022",
        )
```

## See Also

- [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md) - Workflow usage
- [MODELS.md](MODELS.md) - Agent model reference
- [TOOL_EXECUTION.md](TOOL_EXECUTION.md) - Tool definition
