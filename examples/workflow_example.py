"""Example demonstrating multi-agent workflows in GlueLLM.

This example shows how to use IterativeRefinementWorkflow with multiple
specialized critics to iteratively improve content.
"""

import asyncio

from source.executors import AgentExecutor
from source.models.agent import Agent
from source.models.prompt import SystemPrompt
from source.models.workflow import CriticConfig, IterativeConfig
from source.workflows.iterative import IterativeRefinementWorkflow


async def main():
    """Demonstrate iterative refinement workflow with multiple critics."""
    # Define the producer agent (writer)
    writer = Agent(
        name="Writer",
        description="Technical writer who creates content",
        system_prompt=SystemPrompt(
            content="""You are a technical writer. You write clear, engaging, and well-structured content.
            When given feedback, you carefully consider it and revise your work to address the concerns raised."""
        ),
        tools=[],
        max_tool_iterations=5,
    )

    # Define specialized critic agents
    grammar_critic = Agent(
        name="Grammar Critic",
        description="Reviews grammar, spelling, and clarity",
        system_prompt=SystemPrompt(
            content="""You are a grammar and clarity critic. You focus on:
            - Grammar and spelling errors
            - Sentence structure and clarity
            - Readability and flow
            Provide specific, actionable feedback."""
        ),
        tools=[],
        max_tool_iterations=5,
    )

    style_critic = Agent(
        name="Style Critic",
        description="Reviews writing style and engagement",
        system_prompt=SystemPrompt(
            content="""You are a writing style critic. You focus on:
            - Writing style and tone
            - Engagement and narrative flow
            - Structure and organization
            Provide specific, actionable feedback."""
        ),
        tools=[],
        max_tool_iterations=5,
    )

    accuracy_critic = Agent(
        name="Accuracy Critic",
        description="Verifies technical accuracy",
        system_prompt=SystemPrompt(
            content="""You are a technical accuracy critic. You focus on:
            - Technical correctness of claims
            - Factual accuracy
            - Logical consistency
            Provide specific, actionable feedback."""
        ),
        tools=[],
        max_tool_iterations=5,
    )

    # Create workflow with multiple specialized critics
    workflow = IterativeRefinementWorkflow(
        producer=AgentExecutor(writer),
        critics=[
            CriticConfig(
                executor=AgentExecutor(grammar_critic),
                specialty="grammar and clarity",
                goal="Optimize for readability and eliminate errors",
            ),
            CriticConfig(
                executor=AgentExecutor(style_critic),
                specialty="writing style and flow",
                goal="Ensure engaging, well-structured narrative",
            ),
            CriticConfig(
                executor=AgentExecutor(accuracy_critic),
                specialty="technical accuracy",
                goal="Verify all technical claims are correct and logical",
            ),
        ],
        config=IterativeConfig(max_iterations=3),
    )

    # Execute workflow
    print("Starting iterative refinement workflow...")
    print("=" * 60)

    result = await workflow.execute("Write a 300-word article about the benefits of async programming in Python")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(result.final_output)
    print("\n" + "=" * 60)
    print(f"Iterations completed: {result.iterations}")
    print(f"Converged: {result.metadata.get('converged', False)}")
    print(f"Number of critics: {result.metadata.get('num_critics', 0)}")
    print("\n" + "=" * 60)
    print("INTERACTION HISTORY:")
    print("=" * 60)

    for i, interaction in enumerate(result.agent_interactions, 1):
        agent_name = interaction.get("agent", "unknown")
        iteration = interaction.get("iteration", "?")
        print(f"\n[{i}] Iteration {iteration} - {agent_name}")
        if "specialty" in interaction:
            print(f"    Specialty: {interaction['specialty']}")
            print(f"    Goal: {interaction['goal']}")
        print(f"    Output length: {len(interaction.get('output', ''))} chars")


if __name__ == "__main__":
    asyncio.run(main())
