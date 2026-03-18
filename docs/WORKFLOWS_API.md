# GlueLLM Workflows API Reference

Detailed API reference for workflow implementations. For pattern selection and usage guidance, see [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md).

## Base Interface

### Workflow (Abstract)

All workflows extend `Workflow` and implement:

```python
async def _execute_internal(self, initial_input: str, context: dict | None = None) -> WorkflowResult
def validate_config(self) -> bool
```

### WorkflowResult

| Field | Type |
|-------|------|
| `final_output` | `str` |
| `iterations` | `int` |
| `agent_interactions` | `list[dict]` |
| `metadata` | `dict` |
| `hooks_executed` | `int` |
| `hook_errors` | `list[dict]` |

## Executors

Workflows use executors (agents) for LLM calls:

- `SimpleExecutor` - Direct LLM execution
- `AgentExecutor` - Agent with tools
- Custom executor implementing `Executor.execute(prompt) -> str`

## Workflow Implementations

### PipelineWorkflow

Sequential stages; each output feeds the next.

```python
from gluellm.workflows.pipeline import PipelineWorkflow
from gluellm.executors import SimpleExecutor

workflow = PipelineWorkflow(
    stages=[
        ("research", researcher_executor),
        ("write", writer_executor),
        ("edit", editor_executor),
    ]
)
result = await workflow.execute("Write an article about X")
```

### IterativeWorkflow

Refinement with critics.

```python
from gluellm.workflows.iterative import IterativeWorkflow
from gluellm.models.workflow import IterativeConfig

workflow = IterativeWorkflow(
    generator=generator_executor,
    critics=[critic_executor],
    config=IterativeConfig(max_iterations=5, min_quality_score=0.8),
)
```

**IterativeConfig:** `max_iterations`, `min_quality_score`, `convergence_threshold`, `quality_evaluator`

### ReflectionWorkflow

Agent reflects on its own output.

```python
from gluellm.workflows.reflection import ReflectionWorkflow
from gluellm.models.workflow import ReflectionConfig

workflow = ReflectionWorkflow(
    generator=generator,
    reflector=reflector,  # Can be same as generator
    config=ReflectionConfig(max_reflections=3),
)
```

**ReflectionConfig:** `max_reflections`, `min_improvement_threshold`, `reflection_prompt_template`

### DebateWorkflow

Multi-agent adversarial discussion.

```python
from gluellm.workflows.debate import DebateWorkflow
from gluellm.models.workflow import CriticConfig

workflow = DebateWorkflow(
    proposer=pro_executor,
    critics=[("opponent", con_executor, CriticConfig())],
    moderator=moderator_executor,
)
```

**CriticConfig:** `executor`, `specialty`, `goal`, `weight`

### ConsensusWorkflow

Proposers vote until agreement.

```python
from gluellm.workflows.consensus import ConsensusWorkflow
from gluellm.models.workflow import ConsensusConfig

workflow = ConsensusWorkflow(
    proposers=[("Engineer", e), ("Designer", d), ("PM", p)],
    config=ConsensusConfig(min_agreement_ratio=0.7, max_rounds=5, allow_abstention=True),
)
```

### RoundRobinWorkflow

Agents contribute in turns.

```python
from gluellm.workflows.round_robin import RoundRobinWorkflow
from gluellm.models.workflow import RoundRobinConfig

workflow = RoundRobinWorkflow(
    participants=[("Creative", c), ("Technical", t), ("Business", b)],
    config=RoundRobinConfig(rounds=3, build_on_previous=True),
)
```

### ChatRoomWorkflow

Natural discussion with moderator and synthesis.

```python
from gluellm.workflows.chat_room import ChatRoomWorkflow, ChatRoomConfig

workflow = ChatRoomWorkflow(
    participants=[("Alice", a), ("Bob", b), ("Charlie", c)],
    moderator=moderator,
    config=ChatRoomConfig(max_rounds=10, synthesis_rounds=2, allow_moderator_interjection=True),
)
```

### HierarchicalWorkflow

Manager delegates to specialist workers.

```python
from gluellm.workflows.hierarchical import HierarchicalWorkflow
from gluellm.models.workflow import HierarchicalConfig

workflow = HierarchicalWorkflow(
    manager=manager_executor,
    workers=[("frontend", f), ("backend", b), ("devops", d)],
    config=HierarchicalConfig(max_delegations=3, parallel_workers=True, synthesis_strategy="summarize"),
)
```

**HierarchicalConfig:** `max_subtasks`, `parallel_workers`, `synthesis_strategy` ("concatenate"|"summarize"|"merge")

### MapReduceWorkflow

Parallel map, then reduce.

```python
from gluellm.workflows.map_reduce import MapReduceWorkflow
from gluellm.models.workflow import MapReduceConfig

workflow = MapReduceWorkflow(
    mapper=mapper_executor,
    reducer=reducer_executor,
    config=MapReduceConfig(chunk_size=1000, chunk_overlap=100, reduce_strategy="summarize"),
)
```

**MapReduceConfig:** `chunk_size`, `chunk_overlap`, `max_parallel_chunks`, `reduce_strategy`

### ChainOfDensityWorkflow

Progressive summarization.

```python
from gluellm.workflows.chain_of_density import ChainOfDensityWorkflow
from gluellm.models.workflow import ChainOfDensityConfig

workflow = ChainOfDensityWorkflow(
    generator=summarizer_executor,
    config=ChainOfDensityConfig(num_iterations=5, target_length=100),
)
```

### SocraticWorkflow

Guided questioning.

```python
from gluellm.workflows.socratic import SocraticWorkflow
from gluellm.models.workflow import SocraticConfig

workflow = SocraticWorkflow(
    questioner=questioner_executor,
    responder=responder_executor,
    config=SocraticConfig(max_exchanges=5),
)
```

### RAGWorkflow

Retrieval-augmented generation.

```python
from gluellm.workflows.rag import RAGWorkflow
from gluellm.models.workflow import RAGConfig

async def retriever(query: str) -> list[str]:
    return await my_vector_store.search(query, top_k=5)

workflow = RAGWorkflow(
    retriever=retriever,
    generator=generator_executor,
    config=RAGConfig(top_k=5, include_sources=True),
)
```

### ReActWorkflow

Reasoning and acting with tools.

```python
from gluellm.workflows.react import ReActWorkflow
from gluellm.models.workflow import ReActConfig

workflow = ReActWorkflow(
    agent=reasoning_executor,
    tools=[search_web, calculator, db_query],
    config=ReActConfig(max_steps=10, stop_on_final_answer=True),
)
```

**ReActConfig:** `max_steps`, `thought_prefix`, `action_prefix`, `observation_prefix`, `stop_on_final_answer`

### MixtureOfExpertsWorkflow

Router selects experts.

```python
from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow
from gluellm.models.workflow import MoEConfig, ExpertConfig

workflow = MixtureOfExpertsWorkflow(
    router=router_executor,
    experts=[
        ExpertConfig(name="legal", executor=legal_exec, keywords=["law", "contract"]),
        ExpertConfig(name="technical", executor=tech_exec, keywords=["code", "api"]),
        ExpertConfig(name="general", executor=general_exec, keywords=[]),
    ],
    config=MoEConfig(use_all_experts=False),
)
```

### ConstitutionalWorkflow

Principles-based refinement.

```python
from gluellm.workflows.constitutional import ConstitutionalWorkflow
from gluellm.models.workflow import ConstitutionalConfig, Principle

principles = [
    Principle(name="helpfulness", description="Responses should be helpful"),
    Principle(name="safety", description="Responses should not encourage harm"),
]
workflow = ConstitutionalWorkflow(
    generator=generator_executor,
    critic=critic_executor,
    config=ConstitutionalConfig(principles=principles, max_revisions=3),
)
```

### TreeOfThoughtsWorkflow

Branching reasoning with evaluation.

```python
from gluellm.workflows.tree_of_thoughts import TreeOfThoughtsWorkflow
from gluellm.models.workflow import TreeOfThoughtsConfig

workflow = TreeOfThoughtsWorkflow(
    generator=thought_generator_executor,
    evaluator=evaluator_executor,
    config=TreeOfThoughtsConfig(branching_factor=3, max_depth=4, beam_width=2),
)
```

## Hooks

Workflows support pre/post hooks via `HookRegistry`:

- `PRE_WORKFLOW` - Before workflow starts
- `POST_WORKFLOW` - After workflow completes

See [HOOKS.md](HOOKS.md) for details.

## See Also

- [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md) - Pattern guide and examples
- [EXTENDING.md](EXTENDING.md) - Custom workflows
