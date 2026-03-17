"""Examples of workflow patterns in GlueLLM.

Demonstrates ReflectionWorkflow, DebateWorkflow, ConsensusWorkflow,
MapReduceWorkflow, ReActWorkflow, and ChainOfDensityWorkflow.
"""

import asyncio

from gluellm.executors import SimpleExecutor
from gluellm.models.workflow import (
    ChainOfDensityConfig,
    ConsensusConfig,
    MapReduceConfig,
    ReActConfig,
    ReflectionConfig,
)
from gluellm.workflows import (
    ChainOfDensityWorkflow,
    ConsensusWorkflow,
    DebateConfig,
    DebateWorkflow,
    MapReduceWorkflow,
    ReActWorkflow,
    ReflectionWorkflow,
)


def _simple_executor(system_prompt: str, tools: list | None = None):
    return SimpleExecutor(system_prompt=system_prompt, tools=tools or [])


async def example_reflection():
    """ReflectionWorkflow: self-critique and improvement."""
    print("=" * 60)
    print("Example 1: ReflectionWorkflow")
    print("=" * 60)

    workflow = ReflectionWorkflow(
        generator=_simple_executor("You are a concise writer. Write briefly."),
        config=ReflectionConfig(max_reflections=2),
    )
    result = await workflow.execute("Write one sentence about Python.")
    print(f"Output: {result.final_output[:200]}...")
    print(f"Iterations: {result.iterations}")
    print()


async def example_debate():
    """DebateWorkflow: multi-agent debate with judge."""
    print("=" * 60)
    print("Example 2: DebateWorkflow")
    print("=" * 60)

    pro = _simple_executor("You argue FOR the topic. Give one short argument.")
    con = _simple_executor("You argue AGAINST the topic. Give one short argument.")
    judge = _simple_executor("You are the judge. Summarize both sides in one sentence.")

    workflow = DebateWorkflow(
        participants=[("Pro", pro), ("Con", con)],
        judge=judge,
        config=DebateConfig(max_rounds=2),
    )
    result = await workflow.execute("Is remote work better than office work?")
    print(f"Verdict: {result.final_output[:200]}...")
    print(f"Rounds: {result.iterations}")
    print()


async def example_consensus():
    """ConsensusWorkflow: multiple agents propose and agree."""
    print("=" * 60)
    print("Example 3: ConsensusWorkflow")
    print("=" * 60)

    a1 = _simple_executor("You propose solutions. Be brief.")
    a2 = _simple_executor("You propose solutions. Be brief.")

    workflow = ConsensusWorkflow(
        proposers=[("Agent1", a1), ("Agent2", a2)],
        config=ConsensusConfig(max_rounds=2, min_agreement_ratio=0.5),
    )
    result = await workflow.execute("What is one benefit of open source software?")
    print(f"Consensus: {result.final_output[:200]}...")
    print(f"Rounds: {result.iterations}")
    print()


async def example_map_reduce():
    """MapReduceWorkflow: parallel processing and aggregation."""
    print("=" * 60)
    print("Example 4: MapReduceWorkflow")
    print("=" * 60)

    mapper = _simple_executor("Summarize the given text in one sentence.")
    reducer = _simple_executor("Combine the summaries into one coherent summary.")

    workflow = MapReduceWorkflow(
        mapper=mapper,
        reducer=reducer,
        config=MapReduceConfig(chunk_size=50, max_parallel_chunks=2),
    )
    text = "Python is great. JavaScript is popular. Go is fast. Rust is safe. "
    result = await workflow.execute(text)
    print(f"Reduced: {result.final_output[:200]}...")
    print(f"Interactions: {len(result.agent_interactions)}")
    print()


async def example_react():
    """ReActWorkflow: reasoning and acting with tools."""
    print("=" * 60)
    print("Example 5: ReActWorkflow")
    print("=" * 60)

    def get_answer(question: str) -> str:
        """Get an answer (simulated)."""
        return "42" if "meaning" in question.lower() or "life" in question.lower() else "unknown"

    reasoner = SimpleExecutor(
        system_prompt="Think step by step. Use tools when needed. End with Final Answer:",
        tools=[get_answer],
    )
    workflow = ReActWorkflow(
        reasoner=reasoner,
        config=ReActConfig(max_steps=3, stop_on_final_answer=True),
    )
    result = await workflow.execute("What is the answer to the ultimate question?")
    print(f"Answer: {result.final_output[:200]}...")
    print(f"Steps: {result.iterations}")
    print()


async def example_chain_of_density():
    """ChainOfDensityWorkflow: iteratively increase content density."""
    print("=" * 60)
    print("Example 6: ChainOfDensityWorkflow")
    print("=" * 60)

    generator = _simple_executor("You write concise summaries. Add density with each iteration.")

    workflow = ChainOfDensityWorkflow(
        generator=generator,
        config=ChainOfDensityConfig(num_iterations=2, density_increment="entities"),
    )
    result = await workflow.execute("Summarize: Machine learning uses data to make predictions.")
    print(f"Dense output: {result.final_output[:200]}...")
    print(f"Iterations: {result.iterations}")
    print()


async def main():
    await example_reflection()
    await example_debate()
    await example_consensus()
    await example_map_reduce()
    await example_react()
    await example_chain_of_density()


if __name__ == "__main__":
    asyncio.run(main())
