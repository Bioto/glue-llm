# Extending GlueLLM

How to extend GlueLLM with custom workflows, hooks, guardrails, and providers.

## Custom Workflows

### 1. Extend Workflow Base Class

```python
from gluellm.workflows._base import Workflow, WorkflowResult
from gluellm.executors._base import Executor

class CustomWorkflow(Workflow):
    def __init__(self, agents: list[Executor], **kwargs):
        super().__init__(**kwargs)
        self.agents = agents

    async def _execute_internal(
        self,
        initial_input: str,
        context: dict | None = None,
    ) -> WorkflowResult:
        current = initial_input
        interactions = []
        for i, agent in enumerate(self.agents):
            output = await agent.execute(current)
            interactions.append({"agent": i, "input": current, "output": output})
            current = output
        return WorkflowResult(
            final_output=current,
            iterations=len(self.agents),
            agent_interactions=interactions,
        )

    def validate_config(self) -> bool:
        return len(self.agents) > 0
```

### 2. Create Config Model (Optional)

```python
from gluellm.models.workflow import BaseModel, Field

class CustomConfig(BaseModel):
    max_steps: int = Field(default=10, gt=0)
```

## Custom Hooks

### Add a Hook

```python
from gluellm.models.hook import HookConfig, HookContext, HookStage, HookErrorStrategy

def my_hook(context: HookContext) -> HookContext:
    context.content = context.content.upper()
    return context

config = HookConfig(
    handler=my_hook,
    name="uppercase",
    error_strategy=HookErrorStrategy.SKIP,
)
registry.add_hook(HookStage.PRE_EXECUTOR, config)
```

## Custom Guardrails

### Custom Input/Output Callables

```python
from gluellm import GuardrailsConfig

def redact_secrets(content: str) -> str:
    return content.replace("SECRET", "[REDACTED]")

guardrails = GuardrailsConfig(
    enabled=True,
    custom_input=[redact_secrets],
    custom_output=[redact_secrets],
)
```

### Prompt-Guided Guardrail

```python
from gluellm.guardrails.config import GuardrailsConfig, PromptGuidedConfig

def evaluator(content: str, prompt: str) -> tuple[bool, str | None]:
    # Call LLM or rule-based check
    passed = check_policy(content, prompt)
    return (passed, None if passed else "Violates policy")

guardrails = GuardrailsConfig(
    prompt_guided=PromptGuidedConfig(
        prompt="Content must be professional and on-topic.",
        evaluator=evaluator,
    ),
)
```

## Custom Executors

```python
from gluellm.executors._base import Executor

class MyExecutor(Executor):
    async def execute(self, prompt: str, context: dict | None = None) -> str:
        # Your logic here
        return "response"

    async def execute_with_context(self, prompt: str, context: dict) -> str:
        return await self.execute(prompt, context)
```

## Custom Evaluation Store

```python
from gluellm.eval.store import EvalStore
from gluellm.models.eval import EvalRecord

class MyEvalStore:
    async def record(self, record: EvalRecord) -> None:
        await self._persist(record)

    async def close(self) -> None:
        pass
```

## Custom Event Sinks

```python
from gluellm.events import Sink, ProcessEvent

class MySink(Sink):
    def emit(self, event: ProcessEvent) -> None:
        # Send to your system
        self._send_to_analytics(event)
```

## Adding Providers

GlueLLM uses [any-llm](https://github.com/BerkYeni/any_llm_client) for provider abstraction. To add a provider:

1. Ensure any-llm supports it
2. Add API key to settings (e.g., `gluellm_settings.your_provider_api_key`)
3. Use model format: `provider:model_name`

## See Also

- [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md) - Creating custom workflows
- [HOOKS.md](HOOKS.md) - Hook system
- [GUARDRAILS.md](GUARDRAILS.md) - Guardrails config
