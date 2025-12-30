# GlueLLM Workflow Patterns Guide

GlueLLM provides 15 pre-built multi-agent workflow patterns for orchestrating complex LLM interactions. This guide explains when and how to use each pattern.

## Table of Contents

- [Overview](#overview)
- [Choosing the Right Workflow](#choosing-the-right-workflow)
- [Workflow Patterns](#workflow-patterns)
  - [Pipeline](#pipeline-workflow)
  - [Iterative Refinement](#iterative-refinement-workflow)
  - [Reflection](#reflection-workflow)
  - [Debate](#debate-workflow)
  - [Consensus](#consensus-workflow)
  - [Round Robin](#round-robin-workflow)
  - [Hierarchical](#hierarchical-workflow)
  - [MapReduce](#mapreduce-workflow)
  - [Chain of Density](#chain-of-density-workflow)
  - [Socratic](#socratic-workflow)
  - [RAG (Retrieval-Augmented Generation)](#rag-workflow)
  - [ReAct](#react-workflow)
  - [Mixture of Experts](#mixture-of-experts-workflow)
  - [Constitutional AI](#constitutional-ai-workflow)
  - [Tree of Thoughts](#tree-of-thoughts-workflow)
- [Creating Custom Workflows](#creating-custom-workflows)

## Overview

All workflows in GlueLLM follow a common pattern:

```python
from gluellm.workflows import SomeWorkflow
from gluellm.executors import SimpleExecutor

# Create executors (agents)
agent = SimpleExecutor(system_prompt="You are a helpful assistant")

# Create workflow
workflow = SomeWorkflow(agents=[agent], config=SomeConfig())

# Execute
result = await workflow.execute("Your task here")

# Access results
print(result.final_output)
print(f"Iterations: {result.iterations}")
print(f"Interactions: {result.agent_interactions}")
```

## Choosing the Right Workflow

Use this decision tree to select the appropriate workflow:

```
Is your task...
│
├─ Sequential processing? ──────────────────────► Pipeline
│
├─ Needs quality improvement? ──────────────────► Iterative / Reflection
│
├─ Requires diverse perspectives?
│   ├─ Adversarial debate? ─────────────────────► Debate
│   ├─ Agreement needed? ───────────────────────► Consensus
│   └─ Equal participation? ────────────────────► Round Robin
│
├─ Complex/large input? ────────────────────────► MapReduce / Hierarchical
│
├─ Summarization? ──────────────────────────────► Chain of Density
│
├─ Teaching/exploration? ───────────────────────► Socratic
│
├─ External knowledge needed? ──────────────────► RAG
│
├─ Dynamic tool use? ───────────────────────────► ReAct
│
├─ Specialized expertise? ──────────────────────► Mixture of Experts
│
├─ Safety/ethics critical? ─────────────────────► Constitutional AI
│
└─ Complex reasoning? ──────────────────────────► Tree of Thoughts
```

## Workflow Patterns

### Pipeline Workflow

**Purpose:** Sequential processing where each agent's output feeds into the next.

**Use When:**
- Tasks have clear sequential stages
- Each stage transforms or enriches the input
- Order of operations matters

**Example Use Cases:**
- Content creation: Research → Draft → Edit → Format
- Data processing: Extract → Transform → Validate → Load
- Translation: Translate → Localize → Review

```python
from gluellm.workflows.pipeline import PipelineWorkflow
from gluellm.executors import SimpleExecutor

# Create specialized agents
researcher = SimpleExecutor(system_prompt="Research the given topic thoroughly")
writer = SimpleExecutor(system_prompt="Write engaging content based on research")
editor = SimpleExecutor(system_prompt="Edit for clarity and grammar")

# Create pipeline
pipeline = PipelineWorkflow(
    stages=[
        ("research", researcher),
        ("write", writer),
        ("edit", editor),
    ]
)

result = await pipeline.execute("Write an article about quantum computing")
```

### Iterative Refinement Workflow

**Purpose:** Improve output quality through multiple refinement cycles.

**Use When:**
- Initial output needs polish
- Quality improves with iteration
- You have clear improvement criteria

**Example Use Cases:**
- Essay writing with revisions
- Code optimization
- Design iteration

```python
from gluellm.workflows.iterative import IterativeWorkflow
from gluellm.models.workflow import IterativeConfig

workflow = IterativeWorkflow(
    generator=generator_agent,
    critics=[critic_agent],
    config=IterativeConfig(
        max_iterations=5,
        quality_threshold=0.8,
    )
)

result = await workflow.execute("Write a compelling product description")
```

### Reflection Workflow

**Purpose:** Agent reflects on its own output to improve it.

**Use When:**
- Self-improvement is sufficient
- You want meta-cognitive behavior
- Single-agent refinement is appropriate

**Example Use Cases:**
- Code review and self-correction
- Answer verification
- Writing improvement

```python
from gluellm.workflows.reflection import ReflectionWorkflow
from gluellm.models.workflow import ReflectionConfig

workflow = ReflectionWorkflow(
    generator=generator,
    reflector=reflector,  # Can be same as generator
    config=ReflectionConfig(
        max_reflections=3,
    )
)

result = await workflow.execute("Solve this math problem: ...")
```

### Debate Workflow

**Purpose:** Multiple agents debate to explore different perspectives.

**Use When:**
- Topic has multiple valid viewpoints
- You want comprehensive analysis
- Adversarial testing improves output

**Example Use Cases:**
- Policy analysis
- Risk assessment
- Decision evaluation

```python
from gluellm.workflows.debate import DebateWorkflow
from gluellm.models.workflow import CriticConfig

pro_agent = SimpleExecutor(system_prompt="Argue in favor of the position")
con_agent = SimpleExecutor(system_prompt="Argue against the position")
moderator = SimpleExecutor(system_prompt="Summarize the key points from both sides")

workflow = DebateWorkflow(
    proposer=pro_agent,
    critics=[("opponent", con_agent, CriticConfig())],
    moderator=moderator,
)

result = await workflow.execute("Should companies mandate return-to-office?")
```

### Consensus Workflow

**Purpose:** Multiple agents propose solutions and vote until agreement.

**Use When:**
- Group agreement is valuable
- Multiple valid approaches exist
- Collective wisdom improves decisions

**Example Use Cases:**
- Team decision making
- Solution selection
- Priority ranking

```python
from gluellm.workflows.consensus import ConsensusWorkflow
from gluellm.models.workflow import ConsensusConfig

workflow = ConsensusWorkflow(
    proposers=[
        ("Engineer", engineer_agent),
        ("Designer", designer_agent),
        ("PM", pm_agent),
    ],
    config=ConsensusConfig(
        min_agreement_ratio=0.7,
        max_rounds=5,
        allow_abstention=True,
    )
)

result = await workflow.execute("How should we prioritize these features?")
```

### Round Robin Workflow

**Purpose:** Each agent contributes equally in turns.

**Use When:**
- All perspectives should be heard
- Order of contribution matters
- Building on previous responses

**Example Use Cases:**
- Brainstorming sessions
- Story building
- Collaborative writing

```python
from gluellm.workflows.round_robin import RoundRobinWorkflow
from gluellm.models.workflow import RoundRobinConfig

workflow = RoundRobinWorkflow(
    participants=[
        ("Creative", creative_agent),
        ("Technical", technical_agent),
        ("Business", business_agent),
    ],
    config=RoundRobinConfig(
        rounds=3,
        build_on_previous=True,
    )
)

result = await workflow.execute("Generate innovative product ideas")
```

### Hierarchical Workflow

**Purpose:** Manager agent coordinates specialist workers.

**Use When:**
- Task needs decomposition
- Specialists should focus on subtasks
- Coordination is required

**Example Use Cases:**
- Complex project planning
- Multi-domain analysis
- Research coordination

```python
from gluellm.workflows.hierarchical import HierarchicalWorkflow
from gluellm.models.workflow import HierarchicalConfig

workflow = HierarchicalWorkflow(
    manager=manager_agent,
    workers=[
        ("frontend", frontend_specialist),
        ("backend", backend_specialist),
        ("devops", devops_specialist),
    ],
    config=HierarchicalConfig(
        max_delegations=3,
    )
)

result = await workflow.execute("Design a scalable web application architecture")
```

### MapReduce Workflow

**Purpose:** Process large inputs in parallel chunks, then combine.

**Use When:**
- Input is too large for single call
- Task is parallelizable
- Results need aggregation

**Example Use Cases:**
- Document summarization
- Large dataset analysis
- Multi-file code review

```python
from gluellm.workflows.map_reduce import MapReduceWorkflow
from gluellm.models.workflow import MapReduceConfig

workflow = MapReduceWorkflow(
    mapper=chunk_processor,
    reducer=aggregator,
    config=MapReduceConfig(
        chunk_size=1000,  # characters per chunk
        overlap=100,      # overlap between chunks
    )
)

result = await workflow.execute(very_long_document)
```

### Chain of Density Workflow

**Purpose:** Progressive summarization that increases density.

**Use When:**
- Creating concise summaries
- Information density matters
- Iterative condensation needed

**Example Use Cases:**
- Article summarization
- Abstract generation
- Key points extraction

```python
from gluellm.workflows.chain_of_density import ChainOfDensityWorkflow
from gluellm.models.workflow import ChainOfDensityConfig

workflow = ChainOfDensityWorkflow(
    generator=summarizer,
    config=ChainOfDensityConfig(
        num_iterations=5,
        target_length=100,  # words
    )
)

result = await workflow.execute(long_article)
```

### Socratic Workflow

**Purpose:** Explore topics through guided questioning.

**Use When:**
- Teaching or learning scenarios
- Deep exploration needed
- Understanding should emerge

**Example Use Cases:**
- Educational content
- Philosophy exploration
- Problem decomposition

```python
from gluellm.workflows.socratic import SocraticWorkflow
from gluellm.models.workflow import SocraticConfig

workflow = SocraticWorkflow(
    questioner=socratic_questioner,
    responder=student_agent,
    config=SocraticConfig(
        max_exchanges=5,
    )
)

result = await workflow.execute("What is consciousness?")
```

### RAG Workflow

**Purpose:** Augment generation with retrieved context.

**Use When:**
- External knowledge is needed
- Accuracy is critical
- Current/specific information required

**Example Use Cases:**
- Q&A with documents
- Knowledge base queries
- Fact-checked generation

```python
from gluellm.workflows.rag import RAGWorkflow
from gluellm.models.workflow import RAGConfig

async def retriever(query: str) -> list[str]:
    # Your retrieval logic here
    return relevant_documents

workflow = RAGWorkflow(
    retriever=retriever,
    generator=generator,
    config=RAGConfig(
        top_k=5,
        include_sources=True,
    )
)

result = await workflow.execute("What are our company's vacation policies?")
```

### ReAct Workflow

**Purpose:** Reason and Act - dynamic tool use with reasoning traces.

**Use When:**
- Complex multi-step tasks
- Dynamic tool selection needed
- Reasoning transparency important

**Example Use Cases:**
- Research tasks
- Data gathering
- Problem solving with tools

```python
from gluellm.workflows.react import ReActWorkflow
from gluellm.models.workflow import ReActConfig

tools = [search_web, calculator, database_query]

workflow = ReActWorkflow(
    agent=reasoning_agent,
    tools=tools,
    config=ReActConfig(
        max_steps=10,
    )
)

result = await workflow.execute("What is the current stock price of AAPL times the number of employees?")
```

### Mixture of Experts Workflow

**Purpose:** Route to specialized experts based on query type.

**Use When:**
- Different query types need different expertise
- Specialization improves quality
- Dynamic routing beneficial

**Example Use Cases:**
- Customer support routing
- Multi-domain Q&A
- Specialized content generation

```python
from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow
from gluellm.models.workflow import MoEConfig, ExpertConfig

workflow = MixtureOfExpertsWorkflow(
    router=router_agent,
    experts=[
        ExpertConfig(name="legal", executor=legal_expert, keywords=["law", "contract"]),
        ExpertConfig(name="technical", executor=tech_expert, keywords=["code", "api"]),
        ExpertConfig(name="general", executor=general_agent, keywords=[]),
    ],
    config=MoEConfig(
        use_all_experts=False,  # Route to single expert
    )
)

result = await workflow.execute("How do I implement OAuth 2.0?")
```

### Constitutional AI Workflow

**Purpose:** Ensure outputs align with defined principles.

**Use When:**
- Safety and ethics are critical
- Content policies must be enforced
- Outputs need verification

**Example Use Cases:**
- Content moderation
- Safe AI assistants
- Policy-compliant generation

```python
from gluellm.workflows.constitutional import ConstitutionalWorkflow
from gluellm.models.workflow import ConstitutionalConfig, Principle

principles = [
    Principle(
        name="helpfulness",
        description="Responses should be helpful and accurate",
    ),
    Principle(
        name="safety",
        description="Responses should not encourage harmful behavior",
    ),
    Principle(
        name="honesty",
        description="Responses should be truthful and not misleading",
    ),
]

workflow = ConstitutionalWorkflow(
    generator=generator,
    critic=constitutional_critic,
    config=ConstitutionalConfig(
        principles=principles,
        max_revisions=3,
    )
)

result = await workflow.execute("Write advice about health")
```

### Tree of Thoughts Workflow

**Purpose:** Explore multiple reasoning paths and select the best.

**Use When:**
- Complex reasoning required
- Multiple valid approaches exist
- Exploration improves solutions

**Example Use Cases:**
- Mathematical proofs
- Strategic planning
- Complex problem solving

```python
from gluellm.workflows.tree_of_thoughts import TreeOfThoughtsWorkflow
from gluellm.models.workflow import TreeOfThoughtsConfig

workflow = TreeOfThoughtsWorkflow(
    generator=thought_generator,
    evaluator=thought_evaluator,
    config=TreeOfThoughtsConfig(
        branching_factor=3,  # Generate 3 thoughts per step
        max_depth=4,         # Explore 4 levels deep
        beam_width=2,        # Keep top 2 paths
    )
)

result = await workflow.execute("Find the optimal solution to this puzzle: ...")
```

## Creating Custom Workflows

You can create custom workflows by extending the base `Workflow` class:

```python
from gluellm.workflows._base import Workflow, WorkflowResult
from gluellm.executors._base import Executor

class CustomWorkflow(Workflow):
    def __init__(
        self,
        agents: list[Executor],
        hook_registry=None,
    ):
        super().__init__(hook_registry=hook_registry)
        self.agents = agents

    async def _execute_internal(
        self,
        initial_input: str,
        context: dict | None = None
    ) -> WorkflowResult:
        interactions = []
        current_output = initial_input

        # Your custom logic here
        for i, agent in enumerate(self.agents):
            output = await agent.execute(current_output)
            interactions.append({
                "agent_index": i,
                "input": current_output,
                "output": output,
            })
            current_output = output

        return WorkflowResult(
            final_output=current_output,
            iterations=len(self.agents),
            agent_interactions=interactions,
            metadata={"custom_field": "value"},
        )

    def validate_config(self) -> bool:
        return len(self.agents) > 0
```

## Best Practices

1. **Start Simple:** Begin with Pipeline or Iterative before using complex workflows.

2. **Monitor Costs:** Multi-agent workflows multiply API calls. Use token tracking.

3. **Set Limits:** Always configure `max_iterations`, `max_rounds`, etc. to prevent runaway loops.

4. **Use Hooks:** Add pre/post hooks for logging, validation, and safety.

5. **Test Incrementally:** Test each agent independently before combining into workflows.

6. **Handle Errors:** Workflows may fail at any stage. Handle partial results gracefully.

## Further Reading

- [Examples Directory](../examples/) - Working code examples
- [API Reference](./API.md) - Detailed API documentation
- [Hooks Guide](../examples/hooks_example.py) - Hook system usage
