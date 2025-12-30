"""Workflow-related CLI commands.

Commands for testing various multi-agent workflow patterns.
"""

import click

from gluellm.cli.utils import (
    console,
    print_error,
    print_header,
    print_result,
    print_success,
    run_async,
)


def create_simple_executor(system_prompt: str = "You are a helpful assistant"):
    """Create a SimpleExecutor for workflow testing."""
    from gluellm.executors import SimpleExecutor

    return SimpleExecutor(system_prompt=system_prompt)


@click.command("test-iterative-workflow")
@click.option("--input", "-i", default="Topic: The benefits of async programming", help="Input prompt")
@click.option("--iterations", "-n", default=2, type=int, help="Max iterations")
def test_iterative_workflow(input: str, iterations: int) -> None:
    """Test iterative refinement workflow."""
    from gluellm.models.workflow import IterativeConfig
    from gluellm.workflows.iterative import IterativeWorkflow

    print_header("Test Iterative Workflow", f"Max iterations: {iterations}")

    async def run_test():
        generator = create_simple_executor("You are a content writer.")
        critic = create_simple_executor("You are a constructive critic.")

        workflow = IterativeWorkflow(
            generator=generator,
            critics=[critic],
            config=IterativeConfig(max_iterations=iterations),
        )
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        console.print(f"  Iterations: {result.iterations}")
        print_success("Iterative workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-pipeline-workflow")
@click.option("--input", "-i", default="Write an article about Python", help="Input prompt")
def test_pipeline_workflow(input: str) -> None:
    """Test pipeline workflow."""
    from gluellm.workflows.pipeline import PipelineWorkflow

    print_header("Test Pipeline Workflow")

    async def run_test():
        stages = [
            ("research", create_simple_executor("You are a researcher.")),
            ("write", create_simple_executor("You are a writer.")),
            ("edit", create_simple_executor("You are an editor.")),
        ]
        workflow = PipelineWorkflow(stages=stages)
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Pipeline workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-debate-workflow")
@click.option("--topic", "-t", default="Should AI be regulated?", help="Debate topic")
def test_debate_workflow(topic: str) -> None:
    """Test debate workflow."""
    from gluellm.models.workflow import CriticConfig
    from gluellm.workflows.debate import DebateWorkflow

    print_header("Test Debate Workflow", f"Topic: {topic[:50]}")

    async def run_test():
        proposer = create_simple_executor("You argue in favor.")
        critic = create_simple_executor("You argue against.")
        moderator = create_simple_executor("You are a neutral moderator.")

        workflow = DebateWorkflow(
            proposer=proposer,
            critics=[("opponent", critic, CriticConfig())],
            moderator=moderator,
        )
        return await workflow.execute(topic)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Debate workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-reflection-workflow")
@click.option("--input", "-i", default="Write an article about Python", help="Input prompt")
def test_reflection_workflow(input: str) -> None:
    """Test reflection workflow."""
    from gluellm.models.workflow import ReflectionConfig
    from gluellm.workflows.reflection import ReflectionWorkflow

    print_header("Test Reflection Workflow")

    async def run_test():
        generator = create_simple_executor("You are a content writer.")
        reflector = create_simple_executor("You are a thoughtful reflector.")

        workflow = ReflectionWorkflow(
            generator=generator,
            reflector=reflector,
            config=ReflectionConfig(max_reflections=2),
        )
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Reflection workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-chain-of-density-workflow")
@click.option("--input", "-i", default="Summarize this article", help="Input prompt")
def test_chain_of_density_workflow(input: str) -> None:
    """Test Chain of Density summarization workflow."""
    from gluellm.models.workflow import ChainOfDensityConfig
    from gluellm.workflows.chain_of_density import ChainOfDensityWorkflow

    print_header("Test Chain of Density Workflow")

    async def run_test():
        generator = create_simple_executor("You are a summarization expert.")
        workflow = ChainOfDensityWorkflow(
            generator=generator,
            config=ChainOfDensityConfig(num_iterations=3),
        )
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Summary", result.final_output[:500])
        print_success("Chain of Density workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-socratic-workflow")
@click.option("--topic", "-t", default="What is artificial intelligence?", help="Topic to explore")
def test_socratic_workflow(topic: str) -> None:
    """Test Socratic questioning workflow."""
    from gluellm.models.workflow import SocraticConfig
    from gluellm.workflows.socratic import SocraticWorkflow

    print_header("Test Socratic Workflow", f"Topic: {topic[:50]}")

    async def run_test():
        questioner = create_simple_executor("You ask probing questions.")
        responder = create_simple_executor("You respond thoughtfully.")

        workflow = SocraticWorkflow(
            questioner=questioner,
            responder=responder,
            config=SocraticConfig(max_exchanges=3),
        )
        return await workflow.execute(topic)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Socratic workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-consensus-workflow")
@click.option("--problem", "-p", default="Design a solution", help="Problem to solve")
def test_consensus_workflow(problem: str) -> None:
    """Test consensus building workflow."""
    from gluellm.models.workflow import ConsensusConfig
    from gluellm.workflows.consensus import ConsensusWorkflow

    print_header("Test Consensus Workflow")

    async def run_test():
        proposers = [
            ("Engineer", create_simple_executor("You are an engineer.")),
            ("Designer", create_simple_executor("You are a designer.")),
            ("PM", create_simple_executor("You are a product manager.")),
        ]

        workflow = ConsensusWorkflow(
            proposers=proposers,
            config=ConsensusConfig(max_rounds=2, min_agreement_ratio=0.5),
        )
        return await workflow.execute(problem)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Consensus workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-round-robin-workflow")
@click.option("--input", "-i", default="Write an article", help="Input task")
def test_round_robin_workflow(input: str) -> None:
    """Test round robin workflow."""
    from gluellm.models.workflow import RoundRobinConfig
    from gluellm.workflows.round_robin import RoundRobinWorkflow

    print_header("Test Round Robin Workflow")

    async def run_test():
        participants = [
            ("Creative", create_simple_executor("You are creative.")),
            ("Technical", create_simple_executor("You are technical.")),
            ("Business", create_simple_executor("You focus on business.")),
        ]

        workflow = RoundRobinWorkflow(
            participants=participants,
            config=RoundRobinConfig(rounds=2),
        )
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Round robin workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-hierarchical-workflow")
@click.option("--task", "-t", default="Research and write a report", help="Task to decompose")
def test_hierarchical_workflow(task: str) -> None:
    """Test hierarchical delegation workflow."""
    from gluellm.models.workflow import HierarchicalConfig
    from gluellm.workflows.hierarchical import HierarchicalWorkflow

    print_header("Test Hierarchical Workflow")

    async def run_test():
        manager = create_simple_executor("You are a project manager.")
        workers = [
            ("researcher", create_simple_executor("You are a researcher.")),
            ("writer", create_simple_executor("You are a writer.")),
        ]

        workflow = HierarchicalWorkflow(
            manager=manager,
            workers=workers,
            config=HierarchicalConfig(max_delegations=2),
        )
        return await workflow.execute(task)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Hierarchical workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-map-reduce-workflow")
@click.option("--input", "-i", default="Process this long document", help="Input to process")
def test_map_reduce_workflow(input: str) -> None:
    """Test MapReduce workflow."""
    from gluellm.models.workflow import MapReduceConfig
    from gluellm.workflows.map_reduce import MapReduceWorkflow

    print_header("Test MapReduce Workflow")

    async def run_test():
        mapper = create_simple_executor("You process chunks of text.")
        reducer = create_simple_executor("You combine and summarize.")

        workflow = MapReduceWorkflow(
            mapper=mapper,
            reducer=reducer,
            config=MapReduceConfig(chunk_size=500),
        )
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("MapReduce workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-react-workflow")
@click.option("--question", "-q", default="What is the weather in Paris?", help="Question to solve")
def test_react_workflow(question: str) -> None:
    """Test ReAct (Reason + Act) workflow."""
    from gluellm.cli.utils import get_weather
    from gluellm.models.workflow import ReActConfig
    from gluellm.workflows.react import ReActWorkflow

    print_header("Test ReAct Workflow")

    async def run_test():
        agent = create_simple_executor("You reason step by step.")

        workflow = ReActWorkflow(
            agent=agent,
            tools=[get_weather],
            config=ReActConfig(max_steps=3),
        )
        return await workflow.execute(question)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("ReAct workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-mixture-of-experts-workflow")
@click.option("--query", "-q", default="Calculate something", help="Query to route")
def test_mixture_of_experts_workflow(query: str) -> None:
    """Test Mixture of Experts workflow."""
    from gluellm.models.workflow import ExpertConfig, MoEConfig
    from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow

    print_header("Test Mixture of Experts Workflow")

    async def run_test():
        router = create_simple_executor("You route queries to experts.")
        experts = [
            ExpertConfig(
                name="math", executor=create_simple_executor("You are a math expert."), keywords=["calculate", "math"]
            ),
            ExpertConfig(name="general", executor=create_simple_executor("You are a general assistant."), keywords=[]),
        ]

        workflow = MixtureOfExpertsWorkflow(
            router=router,
            experts=experts,
            config=MoEConfig(),
        )
        return await workflow.execute(query)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Mixture of Experts workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-constitutional-workflow")
@click.option("--input", "-i", default="Write about AI safety", help="Input to generate")
def test_constitutional_workflow(input: str) -> None:
    """Test Constitutional AI workflow."""
    from gluellm.models.workflow import ConstitutionalConfig, Principle
    from gluellm.workflows.constitutional import ConstitutionalWorkflow

    print_header("Test Constitutional AI Workflow")

    async def run_test():
        generator = create_simple_executor("You are a content writer.")
        critic = create_simple_executor("You evaluate content for safety.")

        principles = [
            Principle(name="helpful", description="Content should be helpful"),
            Principle(name="safe", description="Content should be safe"),
        ]

        workflow = ConstitutionalWorkflow(
            generator=generator,
            critic=critic,
            config=ConstitutionalConfig(principles=principles, max_revisions=2),
        )
        return await workflow.execute(input)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Constitutional AI workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-tree-of-thoughts-workflow")
@click.option("--problem", "-p", default="Solve this problem", help="Problem to solve")
def test_tree_of_thoughts_workflow(problem: str) -> None:
    """Test Tree of Thoughts workflow."""
    from gluellm.models.workflow import TreeOfThoughtsConfig
    from gluellm.workflows.tree_of_thoughts import TreeOfThoughtsWorkflow

    print_header("Test Tree of Thoughts Workflow")

    async def run_test():
        generator = create_simple_executor("You generate thoughts.")
        evaluator = create_simple_executor("You evaluate thoughts.")

        workflow = TreeOfThoughtsWorkflow(
            generator=generator,
            evaluator=evaluator,
            config=TreeOfThoughtsConfig(branching_factor=2, max_depth=2),
        )
        return await workflow.execute(problem)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("Tree of Thoughts workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-rag-workflow")
@click.option("--query", "-q", default="What is Python?", help="Query to answer")
def test_rag_workflow(query: str) -> None:
    """Test RAG (Retrieval-Augmented Generation) workflow."""
    from gluellm.models.workflow import RAGConfig
    from gluellm.workflows.rag import RAGWorkflow

    print_header("Test RAG Workflow")

    async def mock_retriever(q: str) -> list[str]:
        return [
            "Python is a high-level programming language.",
            "Python was created by Guido van Rossum in 1991.",
        ]

    async def run_test():
        generator = create_simple_executor("You answer questions using context.")

        workflow = RAGWorkflow(
            retriever=mock_retriever,
            generator=generator,
            config=RAGConfig(top_k=2),
        )
        return await workflow.execute(query)

    try:
        result = run_async(run_test())
        print_result("Final Output", result.final_output[:500])
        print_success("RAG workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


@click.command("test-chat-room-workflow")
@click.option("--query", "-q", default="How should we design a new API?", help="Query to discuss")
@click.option("--rounds", "-r", default=2, type=int, help="Max discussion rounds")
@click.option("--synthesis-rounds", "-s", default=1, type=int, help="Synthesis refinement rounds")
def test_chat_room_workflow(query: str, rounds: int, synthesis_rounds: int) -> None:
    """Test chat room discussion workflow with moderator."""
    from gluellm.models.workflow import ChatRoomConfig
    from gluellm.workflows.chat_room import ChatRoomWorkflow

    print_header("Test Chat Room Workflow", f"Max rounds: {rounds}")

    async def run_test():
        participants = [
            ("Alice", create_simple_executor("You are Alice, a technical architect focused on scalability.")),
            ("Bob", create_simple_executor("You are Bob, a UX designer focused on user experience.")),
            ("Charlie", create_simple_executor("You are Charlie, a security expert focused on safety.")),
        ]
        moderator = create_simple_executor(
            "You are a moderator. Decide if discussion is complete by responding with CONTINUE or CONCLUDE."
        )

        workflow = ChatRoomWorkflow(
            participants=participants,
            moderator=moderator,
            config=ChatRoomConfig(max_rounds=rounds, synthesis_rounds=synthesis_rounds),
        )
        return await workflow.execute(query)

    try:
        result = run_async(run_test())
        print_result("Final Answer", result.final_output[:500])
        console.print(f"  Discussion rounds: {result.metadata.get('discussion_rounds', 0)}")
        console.print(f"  Moderator concluded: {result.metadata.get('moderator_concluded', False)}")
        print_success("Chat room workflow test passed")
    except Exception as e:
        print_error(f"Test failed: {e}")


# Export all commands
workflows_commands = [
    test_iterative_workflow,
    test_pipeline_workflow,
    test_debate_workflow,
    test_reflection_workflow,
    test_chain_of_density_workflow,
    test_socratic_workflow,
    test_consensus_workflow,
    test_round_robin_workflow,
    test_hierarchical_workflow,
    test_map_reduce_workflow,
    test_react_workflow,
    test_mixture_of_experts_workflow,
    test_constitutional_workflow,
    test_tree_of_thoughts_workflow,
    test_rag_workflow,
    test_chat_room_workflow,
]
