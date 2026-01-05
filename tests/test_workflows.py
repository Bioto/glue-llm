"""Tests for workflow implementations."""

import pytest
from pydantic import ValidationError

from gluellm.executors._base import Executor
from gluellm.models.workflow import (
    ChainOfDensityConfig,
    ChatRoomConfig,
    ConsensusConfig,
    ConstitutionalConfig,
    CriticConfig,
    ExpertConfig,
    HierarchicalConfig,
    IterativeConfig,
    MapReduceConfig,
    MoEConfig,
    Principle,
    RAGConfig,
    ReActConfig,
    ReflectionConfig,
    RoundRobinConfig,
    SocraticConfig,
    TreeOfThoughtsConfig,
)
from gluellm.workflows._base import WorkflowResult
from gluellm.workflows.chain_of_density import ChainOfDensityWorkflow
from gluellm.workflows.chat_room import ChatRoomWorkflow
from gluellm.workflows.consensus import ConsensusWorkflow
from gluellm.workflows.constitutional import ConstitutionalWorkflow
from gluellm.workflows.debate import DebateConfig, DebateWorkflow
from gluellm.workflows.hierarchical import HierarchicalWorkflow
from gluellm.workflows.iterative import IterativeRefinementWorkflow
from gluellm.workflows.map_reduce import MapReduceWorkflow
from gluellm.workflows.mixture_of_experts import MixtureOfExpertsWorkflow
from gluellm.workflows.pipeline import PipelineWorkflow
from gluellm.workflows.rag import RAGWorkflow
from gluellm.workflows.react import ReActWorkflow
from gluellm.workflows.reflection import ReflectionWorkflow
from gluellm.workflows.round_robin import RoundRobinWorkflow
from gluellm.workflows.socratic import SocraticWorkflow
from gluellm.workflows.tree_of_thoughts import TreeOfThoughtsWorkflow


class MockExecutor(Executor):
    """Mock executor for testing."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize mock executor with optional response sequence.

        Args:
            responses: Optional list of responses to return in sequence
        """
        super().__init__()
        self.responses = responses or []
        self.call_count = 0

    async def _execute_internal(self, query: str) -> str:
        """Execute query and return mock response.

        Args:
            query: The query string

        Returns:
            Mock response string
        """
        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
        else:
            response = f"Mock response to: {query[:50]}"
        self.call_count += 1
        return response


@pytest.mark.asyncio
async def test_iterative_workflow_single_critic():
    """Test iterative workflow with single critic."""
    producer = MockExecutor(["Draft 1", "Draft 2", "Final Draft"])
    critic = MockExecutor(["Fix grammar", "Fix style", "Looks good"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(
            executor=critic,
            specialty="grammar",
            goal="Fix errors",
        ),
        config=IterativeConfig(max_iterations=2),
    )

    result = await workflow.execute("Write an article")

    assert result.iterations == 2
    assert result.final_output == "Draft 2"
    assert len(result.agent_interactions) == 4  # 2 producer + 2 critic
    assert producer.call_count == 2
    assert critic.call_count == 2


@pytest.mark.asyncio
async def test_iterative_workflow_multiple_critics():
    """Test iterative workflow with multiple critics executing in parallel."""
    producer = MockExecutor(["Draft 1", "Draft 2"])
    critic1 = MockExecutor(["Grammar feedback"])
    critic2 = MockExecutor(["Style feedback"])
    critic3 = MockExecutor(["Accuracy feedback"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=[
            CriticConfig(executor=critic1, specialty="grammar", goal="Fix grammar"),
            CriticConfig(executor=critic2, specialty="style", goal="Improve style"),
            CriticConfig(executor=critic3, specialty="accuracy", goal="Verify accuracy"),
        ],
        config=IterativeConfig(max_iterations=2),
    )

    result = await workflow.execute("Write an article")

    assert result.iterations == 2
    assert result.final_output == "Draft 2"
    assert result.metadata["num_critics"] == 3
    # Should have 2 producer calls + 2 rounds * 3 critics = 8 total interactions
    assert len(result.agent_interactions) == 8
    assert producer.call_count == 2
    assert critic1.call_count == 2
    assert critic2.call_count == 2
    assert critic3.call_count == 2


@pytest.mark.asyncio
async def test_iterative_workflow_feedback_formatting():
    """Test that feedback from multiple critics is properly formatted."""
    producer = MockExecutor(["Draft 1", "Draft 2"])
    critic1 = MockExecutor(["Grammar is good"])
    critic2 = MockExecutor(["Style needs work"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=[
            CriticConfig(executor=critic1, specialty="grammar", goal="Check grammar"),
            CriticConfig(executor=critic2, specialty="style", goal="Check style"),
        ],
        config=IterativeConfig(max_iterations=2),
    )

    result = await workflow.execute("Write something")

    # Check that producer received formatted feedback in second iteration
    producer_calls = [call for call in result.agent_interactions if call["agent"] == "producer"]
    assert len(producer_calls) == 2
    # Second producer call should have feedback
    producer_input = producer_calls[1]["input"]
    assert "Feedback from critics" in producer_input
    assert "Grammar Critic" in producer_input or "grammar" in producer_input.lower()
    assert "Style Critic" in producer_input or "style" in producer_input.lower()


@pytest.mark.asyncio
async def test_iterative_workflow_convergence():
    """Test workflow convergence with quality evaluator."""
    producer = MockExecutor(["Draft 1", "Draft 2"])
    critic = MockExecutor(["Feedback"])

    def quality_evaluator(content: str, feedback: dict) -> float:
        """Mock quality evaluator."""
        return 0.9  # High quality, should converge

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Improve"),
        config=IterativeConfig(
            max_iterations=5,
            min_quality_score=0.8,
            quality_evaluator=quality_evaluator,
        ),
    )

    result = await workflow.execute("Write something")

    # Should converge early due to high quality score
    assert result.iterations <= 5
    assert result.metadata["converged"] is True


@pytest.mark.asyncio
async def test_pipeline_workflow():
    """Test pipeline workflow with sequential stages."""
    stage1 = MockExecutor(["Research output"])
    stage2 = MockExecutor(["Written content"])
    stage3 = MockExecutor(["Edited content"])

    workflow = PipelineWorkflow(
        stages=[
            ("research", stage1),
            ("write", stage2),
            ("edit", stage3),
        ]
    )

    result = await workflow.execute("Topic: AI")

    assert result.iterations == 3
    assert result.final_output == "Edited content"
    assert len(result.agent_interactions) == 3
    assert stage1.call_count == 1
    assert stage2.call_count == 1
    assert stage3.call_count == 1
    assert result.metadata["stages"] == ["research", "write", "edit"]


@pytest.mark.asyncio
async def test_debate_workflow():
    """Test debate workflow with participants and judge."""
    pro = MockExecutor(["Pro argument 1", "Pro argument 2"])
    con = MockExecutor(["Con argument 1", "Con argument 2"])
    judge = MockExecutor(["Final judgment"])

    workflow = DebateWorkflow(
        participants=[
            ("Pro", pro),
            ("Con", con),
        ],
        judge=judge,
        config=DebateConfig(max_rounds=2),
    )

    result = await workflow.execute("Should AI be regulated?")

    assert result.iterations == 2
    assert "judgment" in result.final_output.lower() or "judgment" in str(result.agent_interactions[-1])
    assert len(result.agent_interactions) == 5  # 2 rounds * 2 participants + 1 judge
    assert pro.call_count == 2
    assert con.call_count == 2
    assert judge.call_count == 1
    assert result.metadata["judge_used"] is True


@pytest.mark.asyncio
async def test_debate_workflow_no_judge():
    """Test debate workflow without judge."""
    pro = MockExecutor(["Pro argument"])
    con = MockExecutor(["Con argument"])

    workflow = DebateWorkflow(
        participants=[
            ("Pro", pro),
            ("Con", con),
        ],
        judge=None,
        config=DebateConfig(max_rounds=1, judge_decides=False),
    )

    result = await workflow.execute("Topic")

    assert result.iterations == 1
    assert len(result.agent_interactions) == 2  # 1 round * 2 participants
    assert result.metadata["judge_used"] is False


def test_workflow_validation():
    """Test workflow configuration validation."""
    producer = MockExecutor()
    critic = MockExecutor()

    # Valid workflow
    workflow1 = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="test", goal="test"),
        config=IterativeConfig(max_iterations=3),
    )
    assert workflow1.validate_config() is True

    # Invalid workflow (no critics)
    workflow2 = IterativeRefinementWorkflow(
        producer=producer,
        critics=[],
        config=IterativeConfig(max_iterations=3),
    )
    assert workflow2.validate_config() is False

    # Valid pipeline
    pipeline = PipelineWorkflow(stages=[("stage1", producer)])
    assert pipeline.validate_config() is True

    # Invalid pipeline
    pipeline_invalid = PipelineWorkflow(stages=[])
    assert pipeline_invalid.validate_config() is False

    # Valid debate
    debate = DebateWorkflow(
        participants=[("A", producer), ("B", critic)],
        config=DebateConfig(max_rounds=2),
    )
    assert debate.validate_config() is True

    # Invalid debate (not enough participants)
    debate_invalid = DebateWorkflow(
        participants=[("A", producer)],
        config=DebateConfig(max_rounds=2),
    )
    assert debate_invalid.validate_config() is False


@pytest.mark.asyncio
async def test_iterative_workflow_critic_error_handling():
    """Test that workflow handles critic errors gracefully."""
    producer = MockExecutor(["Draft"])
    critic_good = MockExecutor(["Good feedback"])

    # Create a critic that raises an exception
    class FailingExecutor(Executor):
        async def _execute_internal(self, query: str) -> str:
            raise Exception("Critic failed")

    critic_bad = FailingExecutor()

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=[
            CriticConfig(executor=critic_good, specialty="good", goal="Provide feedback"),
            CriticConfig(executor=critic_bad, specialty="bad", goal="This will fail"),
        ],
        config=IterativeConfig(max_iterations=1),
    )

    result = await workflow.execute("Write something")

    # Should still complete despite one critic failing
    assert result.iterations == 1
    # Should have producer + 2 critics (one with error)
    assert len(result.agent_interactions) == 3
    # Check that error is recorded
    error_interactions = [i for i in result.agent_interactions if "Error" in str(i.get("output", ""))]
    assert len(error_interactions) > 0


@pytest.mark.asyncio
async def test_iterative_workflow_max_iterations():
    """Test that workflow stops at max iterations."""
    producer = MockExecutor(["Draft 1", "Draft 2", "Draft 3", "Draft 4"])
    critic = MockExecutor(["Keep improving"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Improve"),
        config=IterativeConfig(max_iterations=3),
    )

    result = await workflow.execute("Write something")

    assert result.iterations == 3
    assert result.final_output == "Draft 3"
    assert producer.call_count == 3
    assert critic.call_count == 3
    assert result.metadata["converged"] is False  # Hit max iterations


@pytest.mark.asyncio
async def test_iterative_workflow_single_iteration():
    """Test workflow with single iteration."""
    producer = MockExecutor(["Draft"])
    critic = MockExecutor(["Feedback"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Review"),
        config=IterativeConfig(max_iterations=1),
    )

    result = await workflow.execute("Write something")

    assert result.iterations == 1
    assert result.final_output == "Draft"
    assert producer.call_count == 1
    assert critic.call_count == 1


@pytest.mark.asyncio
async def test_iterative_workflow_empty_input():
    """Test workflow with empty input."""
    producer = MockExecutor(["Generated content"])
    critic = MockExecutor(["Feedback"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Review"),
        config=IterativeConfig(max_iterations=1),
    )

    result = await workflow.execute("")

    assert result.iterations == 1
    assert result.final_output == "Generated content"


@pytest.mark.asyncio
async def test_iterative_workflow_quality_evaluator_low_score():
    """Test workflow with quality evaluator that returns low scores."""
    producer = MockExecutor(["Draft 1", "Draft 2", "Draft 3"])
    critic = MockExecutor(["Needs work"])

    call_count = {"count": 0}

    def quality_evaluator(content: str, feedback: dict) -> float:
        """Mock quality evaluator that always returns low scores."""
        call_count["count"] += 1
        return 0.3  # Low quality, should not converge

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Review"),
        config=IterativeConfig(
            max_iterations=3,
            min_quality_score=0.8,
            quality_evaluator=quality_evaluator,
        ),
    )

    result = await workflow.execute("Write something")

    # Should run all iterations since quality never meets threshold
    assert result.iterations == 3
    assert call_count["count"] == 3
    assert result.metadata["converged"] is False


@pytest.mark.asyncio
async def test_iterative_workflow_quality_evaluator_exception():
    """Test workflow handles quality evaluator exceptions gracefully."""
    producer = MockExecutor(["Draft 1", "Draft 2"])
    critic = MockExecutor(["Feedback"])

    def failing_evaluator(content: str, feedback: dict) -> float:
        """Mock quality evaluator that raises exception."""
        raise ValueError("Evaluator failed")

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Review"),
        config=IterativeConfig(
            max_iterations=2,
            min_quality_score=0.8,
            quality_evaluator=failing_evaluator,
        ),
    )

    result = await workflow.execute("Write something")

    # Should complete all iterations despite evaluator failure
    assert result.iterations == 2
    assert result.final_output == "Draft 2"


@pytest.mark.asyncio
async def test_iterative_workflow_context_parameter():
    """Test workflow with context parameter (should be accepted but unused)."""
    producer = MockExecutor(["Draft"])
    critic = MockExecutor(["Feedback"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Review"),
        config=IterativeConfig(max_iterations=1),
    )

    context = {"extra": "data"}
    result = await workflow.execute("Write something", context=context)

    assert result.iterations == 1
    # Context is accepted but not used in current implementation
    assert result.final_output == "Draft"


@pytest.mark.asyncio
async def test_iterative_workflow_critic_weight():
    """Test that critic weight is stored in config (even if not used yet)."""
    producer = MockExecutor(["Draft"])
    critic1 = MockExecutor(["Feedback 1"])
    critic2 = MockExecutor(["Feedback 2"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=[
            CriticConfig(executor=critic1, specialty="grammar", goal="Check", weight=1.0),
            CriticConfig(executor=critic2, specialty="style", goal="Check", weight=2.0),
        ],
        config=IterativeConfig(max_iterations=1),
    )

    result = await workflow.execute("Write something")

    assert result.iterations == 1
    # Verify critics were called
    assert critic1.call_count == 1
    assert critic2.call_count == 1


@pytest.mark.asyncio
async def test_pipeline_workflow_single_stage():
    """Test pipeline workflow with single stage."""
    stage = MockExecutor(["Output"])

    workflow = PipelineWorkflow(stages=[("single", stage)])

    result = await workflow.execute("Input")

    assert result.iterations == 1
    assert result.final_output == "Output"
    assert stage.call_count == 1


@pytest.mark.asyncio
async def test_pipeline_workflow_empty_input():
    """Test pipeline workflow with empty input."""
    stage1 = MockExecutor(["Output 1"])
    stage2 = MockExecutor(["Output 2"])

    workflow = PipelineWorkflow(stages=[("stage1", stage1), ("stage2", stage2)])

    result = await workflow.execute("")

    assert result.iterations == 2
    assert result.final_output == "Output 2"


@pytest.mark.asyncio
async def test_pipeline_workflow_context_parameter():
    """Test pipeline workflow with context parameter."""
    stage = MockExecutor(["Output"])

    workflow = PipelineWorkflow(stages=[("stage", stage)])

    context = {"extra": "data"}
    result = await workflow.execute("Input", context=context)

    assert result.iterations == 1
    assert result.final_output == "Output"


@pytest.mark.asyncio
async def test_pipeline_workflow_interaction_history():
    """Test pipeline workflow interaction history structure."""
    stage1 = MockExecutor(["Output 1"])
    stage2 = MockExecutor(["Output 2"])

    workflow = PipelineWorkflow(stages=[("stage1", stage1), ("stage2", stage2)])

    result = await workflow.execute("Input")

    assert len(result.agent_interactions) == 2
    assert result.agent_interactions[0]["stage"] == "stage1"
    assert result.agent_interactions[0]["input"] == "Input"
    assert result.agent_interactions[0]["output"] == "Output 1"
    assert result.agent_interactions[1]["stage"] == "stage2"
    assert result.agent_interactions[1]["input"] == "Output 1"
    assert result.agent_interactions[1]["output"] == "Output 2"


@pytest.mark.asyncio
async def test_debate_workflow_single_round():
    """Test debate workflow with single round."""
    pro = MockExecutor(["Pro argument"])
    con = MockExecutor(["Con argument"])

    workflow = DebateWorkflow(
        participants=[("Pro", pro), ("Con", con)],
        config=DebateConfig(max_rounds=1),
    )

    result = await workflow.execute("Topic")

    assert result.iterations == 1
    assert len(result.agent_interactions) == 2
    assert pro.call_count == 1
    assert con.call_count == 1


@pytest.mark.asyncio
async def test_debate_workflow_many_participants():
    """Test debate workflow with many participants."""
    participants = [(f"Participant{i}", MockExecutor([f"Argument {i}"])) for i in range(5)]

    workflow = DebateWorkflow(
        participants=participants,
        config=DebateConfig(max_rounds=2),
    )

    result = await workflow.execute("Topic")

    assert result.iterations == 2
    # 2 rounds * 5 participants = 10 interactions
    assert len(result.agent_interactions) == 10
    assert result.metadata["participants"] == [f"Participant{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_debate_workflow_judge_without_decides():
    """Test debate workflow with judge but judge_decides=False."""
    pro = MockExecutor(["Pro argument"])
    con = MockExecutor(["Con argument"])
    judge = MockExecutor(["Judge comment"])

    workflow = DebateWorkflow(
        participants=[("Pro", pro), ("Con", con)],
        judge=judge,
        config=DebateConfig(max_rounds=1, judge_decides=False),
    )

    result = await workflow.execute("Topic")

    assert result.iterations == 1
    assert len(result.agent_interactions) == 2  # Only participants, no judge
    assert judge.call_count == 0
    assert result.metadata["judge_used"] is False


@pytest.mark.asyncio
async def test_debate_workflow_context_parameter():
    """Test debate workflow with context parameter."""
    pro = MockExecutor(["Pro argument"])
    con = MockExecutor(["Con argument"])

    workflow = DebateWorkflow(
        participants=[("Pro", pro), ("Con", con)],
        config=DebateConfig(max_rounds=1),
    )

    context = {"extra": "data"}
    result = await workflow.execute("Topic", context=context)

    assert result.iterations == 1


@pytest.mark.asyncio
async def test_debate_workflow_argument_history():
    """Test that debate workflow builds argument history correctly."""
    pro = MockExecutor(["Pro round 1", "Pro round 2"])
    con = MockExecutor(["Con round 1", "Con round 2"])

    workflow = DebateWorkflow(
        participants=[("Pro", pro), ("Con", con)],
        config=DebateConfig(max_rounds=2),
    )

    result = await workflow.execute("Topic")

    # Check that second round participants see first round arguments
    # The final output should contain all arguments
    assert "Pro round 1" in result.final_output or "Pro round 2" in result.final_output
    assert "Con round 1" in result.final_output or "Con round 2" in result.final_output


def test_workflow_result_empty_interactions():
    """Test WorkflowResult with empty interactions."""
    result = WorkflowResult(
        final_output="Output",
        iterations=0,
        agent_interactions=[],
        metadata={},
    )

    assert result.final_output == "Output"
    assert result.iterations == 0
    assert len(result.agent_interactions) == 0
    assert result.metadata == {}


def test_workflow_result_metadata():
    """Test WorkflowResult metadata handling."""
    metadata = {"key1": "value1", "key2": 42, "nested": {"inner": "value"}}

    result = WorkflowResult(
        final_output="Output",
        iterations=1,
        agent_interactions=[],
        metadata=metadata,
    )

    assert result.metadata == metadata
    assert result.metadata["key1"] == "value1"
    assert result.metadata["key2"] == 42
    assert result.metadata["nested"]["inner"] == "value"


def test_iterative_config_validation():
    """Test IterativeConfig validation."""
    # Valid config
    config1 = IterativeConfig(max_iterations=5)
    assert config1.max_iterations == 5

    # Invalid: max_iterations must be > 0
    with pytest.raises(ValidationError):
        IterativeConfig(max_iterations=0)

    # Invalid: min_quality_score out of range
    with pytest.raises(ValidationError):
        IterativeConfig(max_iterations=3, min_quality_score=1.5)

    # Valid: min_quality_score in range
    config2 = IterativeConfig(max_iterations=3, min_quality_score=0.5)
    assert config2.min_quality_score == 0.5


def test_critic_config_defaults():
    """Test CriticConfig default values."""
    executor = MockExecutor()

    config = CriticConfig(
        executor=executor,
        specialty="test",
        goal="test goal",
    )

    assert config.specialty == "test"
    assert config.goal == "test goal"
    assert config.weight == 1.0  # Default weight


def test_critic_config_custom_weight():
    """Test CriticConfig with custom weight."""
    executor = MockExecutor()

    config = CriticConfig(
        executor=executor,
        specialty="test",
        goal="test goal",
        weight=2.5,
    )

    assert config.weight == 2.5


@pytest.mark.asyncio
async def test_iterative_workflow_critic_prompt_formatting():
    """Test that critic prompts are properly formatted with specialty and goal."""
    producer = MockExecutor(["Draft"])
    captured_prompts = []

    class CapturingExecutor(Executor):
        async def _execute_internal(self, query: str) -> str:
            captured_prompts.append(query)
            return "Feedback"

    critic = CapturingExecutor()

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(
            executor=critic,
            specialty="technical accuracy",
            goal="Verify all claims are correct",
        ),
        config=IterativeConfig(max_iterations=1),
    )

    await workflow.execute("Write about Python")

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "technical accuracy" in prompt.lower()
    assert "Verify all claims are correct" in prompt
    assert "Draft" in prompt


@pytest.mark.asyncio
async def test_iterative_workflow_feedback_formatting_multiple():
    """Test feedback formatting with multiple critics."""
    producer = MockExecutor(["Draft 1", "Draft 2"])
    critic1 = MockExecutor(["Grammar: fix comma"])
    critic2 = MockExecutor(["Style: improve flow"])
    critic3 = MockExecutor(["Accuracy: verify facts"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=[
            CriticConfig(executor=critic1, specialty="grammar", goal="Fix grammar"),
            CriticConfig(executor=critic2, specialty="style", goal="Improve style"),
            CriticConfig(executor=critic3, specialty="accuracy", goal="Verify accuracy"),
        ],
        config=IterativeConfig(max_iterations=2),
    )

    result = await workflow.execute("Write something")

    # Get second producer call which should have formatted feedback
    producer_calls = [i for i in result.agent_interactions if i["agent"] == "producer"]
    second_producer_input = producer_calls[1]["input"]

    # Check all three critics' feedback is included
    assert "Grammar Critic" in second_producer_input or "grammar" in second_producer_input.lower()
    assert "Style Critic" in second_producer_input or "style" in second_producer_input.lower()
    assert "Accuracy Critic" in second_producer_input or "accuracy" in second_producer_input.lower()


@pytest.mark.asyncio
async def test_iterative_workflow_no_feedback_first_iteration():
    """Test that first iteration doesn't include feedback."""
    producer = MockExecutor(["Draft 1", "Draft 2"])
    critic = MockExecutor(["Feedback"])

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=CriticConfig(executor=critic, specialty="general", goal="Review"),
        config=IterativeConfig(max_iterations=2),
    )

    result = await workflow.execute("Write something")

    producer_calls = [i for i in result.agent_interactions if i["agent"] == "producer"]
    first_producer_input = producer_calls[0]["input"]
    second_producer_input = producer_calls[1]["input"]

    # First iteration should not have feedback
    assert "Feedback from critics" not in first_producer_input
    # Second iteration should have feedback
    assert "Feedback from critics" in second_producer_input


@pytest.mark.asyncio
async def test_pipeline_workflow_data_flow():
    """Test that data flows correctly through pipeline stages."""

    # Each stage appends its name to the input
    class AppendExecutor(Executor):
        def __init__(self, name: str):
            super().__init__()
            self.name = name

        async def _execute_internal(self, query: str) -> str:
            return f"{query} -> {self.name}"

    stage1 = AppendExecutor("stage1")
    stage2 = AppendExecutor("stage2")
    stage3 = AppendExecutor("stage3")

    workflow = PipelineWorkflow(
        stages=[
            ("stage1", stage1),
            ("stage2", stage2),
            ("stage3", stage3),
        ]
    )

    result = await workflow.execute("start")

    assert result.final_output == "start -> stage1 -> stage2 -> stage3"
    assert result.agent_interactions[0]["output"] == "start -> stage1"
    assert result.agent_interactions[1]["input"] == "start -> stage1"
    assert result.agent_interactions[1]["output"] == "start -> stage1 -> stage2"


# ============================================================================
# Tests for new workflows
# ============================================================================


@pytest.mark.asyncio
async def test_reflection_workflow_basic():
    """Test reflection workflow with basic execution."""
    generator = MockExecutor(["Initial output", "Improved output", "Final output"])
    reflector = MockExecutor(["Add more detail", "Looks good"])

    workflow = ReflectionWorkflow(
        generator=generator,
        reflector=reflector,
        config=ReflectionConfig(max_reflections=2),
    )

    result = await workflow.execute("Write an article")

    assert result.iterations == 2
    assert len(result.agent_interactions) >= 2
    assert generator.call_count >= 2


@pytest.mark.asyncio
async def test_reflection_workflow_same_agent():
    """Test reflection workflow with same agent for generator and reflector."""
    agent = MockExecutor(["Output 1", "Reflection 1", "Output 2"])

    workflow = ReflectionWorkflow(
        generator=agent,
        reflector=None,  # Should default to generator
        config=ReflectionConfig(max_reflections=1),
    )

    result = await workflow.execute("Write something")

    assert result.iterations == 1
    assert agent.call_count >= 1


@pytest.mark.asyncio
async def test_reflection_workflow_validate_config():
    """Test reflection workflow configuration validation."""
    generator = MockExecutor()
    workflow = ReflectionWorkflow(generator=generator, config=ReflectionConfig(max_reflections=3))
    assert workflow.validate_config() is True

    workflow.config.max_reflections = 0
    assert workflow.validate_config() is False


@pytest.mark.asyncio
async def test_chain_of_density_workflow_basic():
    """Test chain of density workflow."""
    generator = MockExecutor(["Sparse summary", "More detailed", "Very detailed", "Extremely detailed"])

    workflow = ChainOfDensityWorkflow(
        generator=generator,
        config=ChainOfDensityConfig(num_iterations=3),
    )

    result = await workflow.execute("Summarize this")

    assert result.iterations == 3
    assert len(result.agent_interactions) == 3
    assert generator.call_count == 3


@pytest.mark.asyncio
async def test_chain_of_density_workflow_density_types():
    """Test chain of density with different increment types."""
    generator = MockExecutor(["Output"] * 5)

    for increment_type in ["entities", "details", "examples"]:
        workflow = ChainOfDensityWorkflow(
            generator=generator,
            config=ChainOfDensityConfig(num_iterations=2, density_increment=increment_type),
        )
        result = await workflow.execute("Test")
        assert result.iterations == 2
        assert result.metadata["density_increment"] == increment_type


@pytest.mark.asyncio
async def test_socratic_workflow_basic():
    """Test Socratic workflow with basic execution."""
    questioner = MockExecutor(["What is AI?", "How does it work?", "What are applications?"])
    responder = MockExecutor(["AI is...", "It works by...", "Applications include..."])

    workflow = SocraticWorkflow(
        questioner=questioner,
        responder=responder,
        config=SocraticConfig(max_exchanges=2),
    )

    result = await workflow.execute("Tell me about AI")

    assert result.iterations == 2
    assert questioner.call_count == 2
    # Responder is called 2 times during exchanges + 1 time for synthesis (default)
    assert responder.call_count == 3


@pytest.mark.asyncio
async def test_socratic_workflow_modes():
    """Test Socratic workflow with different modes."""
    questioner = MockExecutor(["Question"])
    responder = MockExecutor(["Answer"])

    for mode in ["teacher_student", "peer"]:
        workflow = SocraticWorkflow(
            questioner=questioner,
            responder=responder,
            config=SocraticConfig(max_exchanges=1, mode=mode),
        )
        result = await workflow.execute("Topic")
        assert result.metadata["mode"] == mode


@pytest.mark.asyncio
async def test_socratic_workflow_synthesis():
    """Test Socratic workflow with synthesis."""
    questioner = MockExecutor(["Q1"])
    responder = MockExecutor(["A1", "Synthesis"])

    workflow = SocraticWorkflow(
        questioner=questioner,
        responder=responder,
        config=SocraticConfig(max_exchanges=1, synthesis_at_end=True),
    )

    result = await workflow.execute("Topic")
    assert result.metadata["synthesized"] is True


@pytest.mark.asyncio
async def test_rag_workflow_basic():
    """Test RAG workflow with basic retrieval."""

    def mock_retriever(query: str) -> list[dict]:
        return [
            {"content": "Context about Python", "source": "doc1"},
            {"content": "More Python info", "source": "doc2"},
        ]

    generator = MockExecutor(["Answer using context"])

    workflow = RAGWorkflow(
        retriever=mock_retriever,
        generator=generator,
        config=RAGConfig(max_retrieved_chunks=2),
    )

    result = await workflow.execute("What is Python?")

    assert result.iterations == 1
    assert generator.call_count == 1
    assert result.metadata["retrieved_chunks"] == 2


@pytest.mark.asyncio
async def test_rag_workflow_no_context_fallback():
    """Test RAG workflow fallback when no context retrieved."""

    def empty_retriever(query: str) -> list[dict]:
        return []

    generator = MockExecutor(["Fallback answer"])

    workflow = RAGWorkflow(
        retriever=empty_retriever,
        generator=generator,
        config=RAGConfig(fallback_on_no_context=True),
    )

    result = await workflow.execute("Query")
    assert len(result.final_output) > 0


@pytest.mark.asyncio
async def test_rag_workflow_verification():
    """Test RAG workflow with fact verification."""

    def mock_retriever(query: str) -> list[dict]:
        return [{"content": "Context", "source": "doc1"}]

    generator = MockExecutor(["Generated answer"])
    verifier = MockExecutor(["Verification: looks good"])

    workflow = RAGWorkflow(
        retriever=mock_retriever,
        generator=generator,
        verifier=verifier,
        config=RAGConfig(verify_facts=True),
    )

    result = await workflow.execute("Query")
    assert result.metadata["verified"] is True
    assert verifier.call_count == 1


@pytest.mark.asyncio
async def test_round_robin_workflow_basic():
    """Test round-robin workflow."""
    agent1 = MockExecutor(["Contribution 1", "Contribution 4"])
    agent2 = MockExecutor(["Contribution 2", "Contribution 5"])
    agent3 = MockExecutor(["Contribution 3", "Contribution 6"])

    workflow = RoundRobinWorkflow(
        agents=[("Agent1", agent1), ("Agent2", agent2), ("Agent3", agent3)],
        config=RoundRobinConfig(max_rounds=2),
    )

    result = await workflow.execute("Task")

    assert result.iterations == 2
    assert len(result.agent_interactions) >= 6  # 2 rounds * 3 agents


@pytest.mark.asyncio
async def test_round_robin_workflow_contribution_styles():
    """Test round-robin with different contribution styles."""
    agent = MockExecutor(["Contribution"])

    for style in ["extend", "refine", "challenge"]:
        workflow = RoundRobinWorkflow(
            agents=[("Agent", agent)],
            config=RoundRobinConfig(max_rounds=1, contribution_style=style),
        )
        result = await workflow.execute("Task")
        assert result.metadata["contribution_style"] == style


@pytest.mark.asyncio
async def test_consensus_workflow_basic():
    """Test consensus workflow."""
    agent1 = MockExecutor(["Proposal 1", "Vote: 0"])
    agent2 = MockExecutor(["Proposal 2", "Vote: 1"])
    agent3 = MockExecutor(["Proposal 3", "Vote: 0"])

    workflow = ConsensusWorkflow(
        proposers=[("Agent1", agent1), ("Agent2", agent2), ("Agent3", agent3)],
        config=ConsensusConfig(max_rounds=1, min_agreement_ratio=0.5),
    )

    result = await workflow.execute("Problem")

    assert result.iterations == 1
    assert len(result.agent_interactions) > 0


@pytest.mark.asyncio
async def test_consensus_workflow_validate_config():
    """Test consensus workflow configuration validation."""
    agent1 = MockExecutor()
    agent2 = MockExecutor()

    workflow = ConsensusWorkflow(
        proposers=[("Agent1", agent1), ("Agent2", agent2)],
        config=ConsensusConfig(max_rounds=3),
    )
    assert workflow.validate_config() is True

    workflow.proposers = [("Agent1", agent1)]
    assert workflow.validate_config() is False


@pytest.mark.asyncio
async def test_hierarchical_workflow_basic():
    """Test hierarchical workflow."""
    manager = MockExecutor(["1. Task A\n2. Task B\n3. Task C", "Synthesized result"])
    worker1 = MockExecutor(["Result A"])
    worker2 = MockExecutor(["Result B"])

    workflow = HierarchicalWorkflow(
        manager=manager,
        workers=[("Worker1", worker1), ("Worker2", worker2)],
        config=HierarchicalConfig(max_subtasks=3, parallel_workers=True),
    )

    result = await workflow.execute("Complex task")

    assert manager.call_count >= 2  # Decomposition + synthesis
    assert result.metadata["subtasks_created"] > 0


@pytest.mark.asyncio
async def test_hierarchical_workflow_sequential():
    """Test hierarchical workflow with sequential execution."""
    manager = MockExecutor(["1. Task A\n2. Task B", "Synthesis"])
    worker = MockExecutor(["Result"])

    workflow = HierarchicalWorkflow(
        manager=manager,
        workers=[("Worker", worker)],
        config=HierarchicalConfig(parallel_workers=False),
    )

    result = await workflow.execute("Task")
    assert result.metadata["parallel_execution"] is False


@pytest.mark.asyncio
async def test_map_reduce_workflow_basic():
    """Test MapReduce workflow."""
    mapper = MockExecutor(["Mapped chunk 1", "Mapped chunk 2"])
    reducer = MockExecutor(["Reduced result"])

    workflow = MapReduceWorkflow(
        mapper=mapper,
        reducer=reducer,
        config=MapReduceConfig(chunk_size=100, reduce_strategy="summarize"),
    )

    long_input = "x" * 250  # Will be split into chunks
    result = await workflow.execute(long_input)

    assert mapper.call_count > 0
    assert reducer.call_count == 1
    assert result.metadata["chunks_processed"] > 0


@pytest.mark.asyncio
async def test_map_reduce_workflow_no_chunking():
    """Test MapReduce workflow without chunking."""
    mapper = MockExecutor(["Mapped"])
    reducer = MockExecutor(["Reduced"])

    workflow = MapReduceWorkflow(
        mapper=mapper,
        reducer=reducer,
        config=MapReduceConfig(chunk_size=None),  # No chunking
    )

    result = await workflow.execute("Short input")
    assert result.metadata["chunks_processed"] == 1


@pytest.mark.asyncio
async def test_react_workflow_basic():
    """Test ReAct workflow."""
    reasoner = MockExecutor(
        ["Thought: I need to find information\nAction: search\nObservation: Found info\nFinal Answer: The answer is X"]
    )

    workflow = ReActWorkflow(
        reasoner=reasoner,
        config=ReActConfig(max_steps=3, stop_on_final_answer=True),
    )

    result = await workflow.execute("Question")

    assert result.iterations <= 3
    assert len(result.agent_interactions) > 0


@pytest.mark.asyncio
async def test_react_workflow_no_final_answer():
    """Test ReAct workflow without final answer detection."""
    reasoner = MockExecutor(["Thought: thinking\nAction: act"])

    workflow = ReActWorkflow(
        reasoner=reasoner,
        config=ReActConfig(max_steps=2, stop_on_final_answer=False),
    )

    result = await workflow.execute("Question")
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_mixture_of_experts_workflow_basic():
    """Test Mixture of Experts workflow."""
    expert1 = MockExecutor(["Expert 1 response"])
    expert2 = MockExecutor(["Expert 2 response"])
    combiner = MockExecutor(["Combined response"])

    workflow = MixtureOfExpertsWorkflow(
        experts=[
            ExpertConfig(
                executor=expert1,
                specialty="math",
                description="Math expert",
                activation_keywords=["calculate", "math"],
            ),
            ExpertConfig(
                executor=expert2,
                specialty="code",
                description="Code expert",
                activation_keywords=["code", "program"],
            ),
        ],
        combiner=combiner,
        config=MoEConfig(routing_strategy="all", combine_strategy="synthesize"),
    )

    result = await workflow.execute("Calculate something")

    assert result.iterations >= 1
    assert result.metadata["experts_used"] > 0


@pytest.mark.asyncio
async def test_mixture_of_experts_workflow_keyword_routing():
    """Test MoE workflow with keyword routing."""
    expert1 = MockExecutor(["Math response"])
    expert2 = MockExecutor(["Code response"])

    workflow = MixtureOfExpertsWorkflow(
        experts=[
            ExpertConfig(
                executor=expert1,
                specialty="math",
                description="Math",
                activation_keywords=["calculate"],
            ),
            ExpertConfig(
                executor=expert2,
                specialty="code",
                description="Code",
                activation_keywords=["code"],
            ),
        ],
        config=MoEConfig(routing_strategy="keyword"),
    )

    result = await workflow.execute("Calculate the sum")
    assert expert1.call_count >= 1


@pytest.mark.asyncio
async def test_constitutional_workflow_basic():
    """Test Constitutional AI workflow."""
    generator = MockExecutor(["Generated content", "Revised content"])
    critic = MockExecutor(["PASS: harmless\nPASS: helpful"])

    workflow = ConstitutionalWorkflow(
        generator=generator,
        critic=critic,
        config=ConstitutionalConfig(
            principles=[
                Principle(name="harmless", description="Should not harm", severity="critical"),
                Principle(name="helpful", description="Should be helpful", severity="error"),
            ],
            max_revisions=2,
        ),
    )

    result = await workflow.execute("Generate content")

    assert result.iterations >= 1
    assert len(result.agent_interactions) > 0


@pytest.mark.asyncio
async def test_constitutional_workflow_requires_config():
    """Test that Constitutional workflow requires config."""
    generator = MockExecutor()

    with pytest.raises(ValueError):
        ConstitutionalWorkflow(generator=generator, config=None)


@pytest.mark.asyncio
async def test_tree_of_thoughts_workflow_basic():
    """Test Tree of Thoughts workflow."""
    thinker = MockExecutor(["Thought 1\nThought 2\nThought 3"])
    evaluator = MockExecutor(["0.8", "0.6", "0.9"])

    workflow = TreeOfThoughtsWorkflow(
        thinker=thinker,
        evaluator=evaluator,
        config=TreeOfThoughtsConfig(branching_factor=3, max_depth=2, evaluation_strategy="score"),
    )

    result = await workflow.execute("Problem")

    assert result.iterations == 2
    assert result.metadata["max_depth"] == 2
    assert result.metadata["branching_factor"] == 3


@pytest.mark.asyncio
async def test_tree_of_thoughts_workflow_evaluation_strategies():
    """Test ToT workflow with different evaluation strategies."""
    thinker = MockExecutor(["Thought"])
    evaluator = MockExecutor(["0.7"])

    for strategy in ["vote", "score", "best_first"]:
        workflow = TreeOfThoughtsWorkflow(
            thinker=thinker,
            evaluator=evaluator,
            config=TreeOfThoughtsConfig(evaluation_strategy=strategy),
        )
        result = await workflow.execute("Problem")
        assert result.metadata["evaluation_strategy"] == strategy


@pytest.mark.asyncio
async def test_tree_of_thoughts_workflow_validate_config():
    """Test ToT workflow configuration validation."""
    thinker = MockExecutor()

    workflow = TreeOfThoughtsWorkflow(
        thinker=thinker,
        config=TreeOfThoughtsConfig(branching_factor=2, max_depth=2),
    )
    assert workflow.validate_config() is True

    workflow.config.branching_factor = 0
    assert workflow.validate_config() is False


# Chat Room Workflow Tests


@pytest.mark.asyncio
async def test_chat_room_workflow_basic():
    """Test basic chat room workflow with discussion and synthesis."""
    alice = MockExecutor(["Alice's first comment", "Alice's second comment", "Alice drafts answer"])
    bob = MockExecutor(["Bob's first comment", "Bob's second comment", "Bob refines answer"])
    charlie = MockExecutor(["Charlie's first comment", "Charlie's second comment", "Charlie refines final answer"])
    moderator = MockExecutor(["CONTINUE", "CONCLUDE"])

    workflow = ChatRoomWorkflow(
        participants=[
            ("Alice", alice),
            ("Bob", bob),
            ("Charlie", charlie),
        ],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=3, synthesis_rounds=1),
    )

    result = await workflow.execute("How should we design a new API?")

    # Should have 2 rounds (moderator said CONTINUE then CONCLUDE)
    assert result.iterations >= 2
    # Final output should be the refined answer
    assert "answer" in result.final_output.lower()
    # Should have discussion interactions + moderator evaluations + synthesis
    assert len(result.agent_interactions) > 6
    # Verify metadata
    assert result.metadata["participants"] == ["Alice", "Bob", "Charlie"]
    assert result.metadata["discussion_rounds"] >= 2


@pytest.mark.asyncio
async def test_chat_room_workflow_moderator_concludes_immediately():
    """Test chat room workflow when moderator concludes after first round."""
    alice = MockExecutor(["Alice's comment", "Alice drafts answer"])
    bob = MockExecutor(["Bob's comment"])
    moderator = MockExecutor(["CONCLUDE"])

    workflow = ChatRoomWorkflow(
        participants=[
            ("Alice", alice),
            ("Bob", bob),
        ],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=5, synthesis_rounds=1),
    )

    result = await workflow.execute("Simple question?")

    # Should stop after 1 round
    assert result.iterations == 1
    # Should have 2 discussion + 1 moderator + 2 synthesis (draft + refine)
    assert len(result.agent_interactions) >= 4
    assert result.metadata["moderator_concluded"] is True


@pytest.mark.asyncio
async def test_chat_room_workflow_max_rounds_reached():
    """Test chat room workflow when max rounds is reached."""
    alice = MockExecutor(["Comment"] * 10)
    bob = MockExecutor(["Comment"] * 10)
    moderator = MockExecutor(["CONTINUE"] * 10)

    workflow = ChatRoomWorkflow(
        participants=[
            ("Alice", alice),
            ("Bob", bob),
        ],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=3, synthesis_rounds=1),
    )

    result = await workflow.execute("Question?")

    # Should stop at max_rounds
    assert result.iterations == 3
    # Moderator kept saying continue but we hit the limit
    assert result.metadata["moderator_concluded"] is False


@pytest.mark.asyncio
async def test_chat_room_workflow_multiple_synthesis_rounds():
    """Test chat room workflow with multiple synthesis refinement rounds."""
    alice = MockExecutor(["Discussion", "Draft v1", "Refine v2"])
    bob = MockExecutor(["Discussion", "Refine v1", "Refine v3"])
    charlie = MockExecutor(["Discussion", "Refine v2", "Refine v4"])
    moderator = MockExecutor(["CONCLUDE"])

    workflow = ChatRoomWorkflow(
        participants=[
            ("Alice", alice),
            ("Bob", bob),
            ("Charlie", charlie),
        ],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=2, synthesis_rounds=2),
    )

    result = await workflow.execute("Question?")

    # Should have synthesis interactions
    synthesis_interactions = [i for i in result.agent_interactions if i.get("stage") == "synthesis"]
    # 1 draft + (2 participants * 2 rounds) = 5 synthesis interactions
    assert len(synthesis_interactions) == 5
    # Final output should be the last refinement
    assert "Refine" in result.final_output


@pytest.mark.asyncio
async def test_chat_room_workflow_moderator_decision_parsing():
    """Test moderator decision parsing with various responses."""
    alice = MockExecutor(["Comment"])
    bob = MockExecutor(["Comment"])

    # Test various moderator responses
    test_cases = [
        ("CONTINUE", True),
        ("continue", True),
        ("Let's CONTINUE the discussion", True),
        ("CONCLUDE", False),
        ("conclude", False),
        ("The discussion is DONE", False),
        ("This is COMPLETE", False),
        ("We are READY to synthesize", False),
        ("SUFFICIENT information gathered", False),
    ]

    for moderator_response, should_continue in test_cases:
        moderator = MockExecutor([moderator_response])
        workflow = ChatRoomWorkflow(
            participants=[("Alice", alice), ("Bob", bob)],
            moderator=moderator,
            config=ChatRoomConfig(max_rounds=5, synthesis_rounds=1),
        )

        result = await workflow.execute("Question?")

        # Check if moderator decision was parsed correctly
        moderator_interactions = [i for i in result.agent_interactions if i.get("stage") == "moderator_evaluation"]
        if moderator_interactions:
            assert moderator_interactions[0]["should_continue"] == should_continue


@pytest.mark.asyncio
async def test_chat_room_workflow_discussion_history_context():
    """Test that participants receive full discussion history."""
    # Track what prompts participants receive
    received_prompts = []

    class TrackingExecutor(Executor):
        def __init__(self, response: str):
            super().__init__()
            self.response = response

        async def _execute_internal(self, query: str) -> str:
            received_prompts.append(query)
            return self.response

    alice = TrackingExecutor("Alice's comment")
    bob = TrackingExecutor("Bob's comment")
    moderator = MockExecutor(["CONCLUDE"])

    workflow = ChatRoomWorkflow(
        participants=[
            ("Alice", alice),
            ("Bob", bob),
        ],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=2, synthesis_rounds=1),
    )

    await workflow.execute("Test question?")

    # Alice's second turn should include Bob's first comment in the prompt
    # Find Alice's second prompt (should be after Bob's first)
    alice_prompts = [p for p in received_prompts if "Alice" in p]
    if len(alice_prompts) > 1:
        second_prompt = alice_prompts[1]
        assert "Bob" in second_prompt
        assert "Discussion so far" in second_prompt or "Bob's comment" in second_prompt


@pytest.mark.asyncio
async def test_chat_room_workflow_validate_config():
    """Test chat room workflow configuration validation."""
    alice = MockExecutor()
    bob = MockExecutor()
    moderator = MockExecutor()

    # Valid configuration
    workflow = ChatRoomWorkflow(
        participants=[("Alice", alice), ("Bob", bob)],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=5, synthesis_rounds=1),
    )
    assert workflow.validate_config() is True

    # Invalid: only one participant
    workflow_invalid1 = ChatRoomWorkflow(
        participants=[("Alice", alice)],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=5, synthesis_rounds=1),
    )
    assert workflow_invalid1.validate_config() is False

    # Invalid: no moderator
    workflow_invalid2 = ChatRoomWorkflow(
        participants=[("Alice", alice), ("Bob", bob)],
        moderator=None,
        config=ChatRoomConfig(max_rounds=5, synthesis_rounds=1),
    )
    assert workflow_invalid2.validate_config() is False

    # Invalid: max_rounds = 0
    workflow_invalid3 = ChatRoomWorkflow(
        participants=[("Alice", alice), ("Bob", bob)],
        moderator=moderator,
    )
    workflow_invalid3.config.max_rounds = 0
    assert workflow_invalid3.validate_config() is False


@pytest.mark.asyncio
async def test_chat_room_workflow_config_defaults():
    """Test chat room workflow configuration defaults."""
    config = ChatRoomConfig()

    assert config.max_rounds == 10
    assert config.allow_moderator_interjection is True
    assert config.synthesis_rounds == 1


@pytest.mark.asyncio
async def test_chat_room_workflow_empty_participants():
    """Test chat room workflow with empty participants list."""
    moderator = MockExecutor()

    workflow = ChatRoomWorkflow(
        participants=[],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=2, synthesis_rounds=1),
    )

    assert workflow.validate_config() is False


@pytest.mark.asyncio
async def test_chat_room_workflow_synthesis_phase():
    """Test that synthesis phase properly drafts and refines."""
    alice = MockExecutor(["Discussion 1", "Initial draft answer"])
    bob = MockExecutor(["Discussion 1", "Refined answer v1"])
    charlie = MockExecutor(["Discussion 1", "Refined answer v2"])
    moderator = MockExecutor(["CONCLUDE"])

    workflow = ChatRoomWorkflow(
        participants=[
            ("Alice", alice),
            ("Bob", bob),
            ("Charlie", charlie),
        ],
        moderator=moderator,
        config=ChatRoomConfig(max_rounds=2, synthesis_rounds=1),
    )

    result = await workflow.execute("Question?")

    # Check synthesis interactions
    synthesis_interactions = [i for i in result.agent_interactions if i.get("stage") == "synthesis"]

    # Should have: 1 draft (Alice) + 2 refines (Bob, Charlie)
    assert len(synthesis_interactions) == 3

    # First should be draft by Alice
    assert synthesis_interactions[0]["action"] == "draft"
    assert synthesis_interactions[0]["participant"] == "Alice"

    # Others should be refine
    assert synthesis_interactions[1]["action"] == "refine"
    assert synthesis_interactions[2]["action"] == "refine"

    # Final output should be the last refinement
    assert result.final_output == "Refined answer v2"
