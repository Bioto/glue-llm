"""Tests for workflow implementations."""

import pytest
from pydantic import ValidationError

from source.executors._base import Executor
from source.models.workflow import CriticConfig, IterativeConfig
from source.workflows._base import WorkflowResult
from source.workflows.debate import DebateConfig, DebateWorkflow
from source.workflows.iterative import IterativeRefinementWorkflow
from source.workflows.pipeline import PipelineWorkflow


class MockExecutor(Executor):
    """Mock executor for testing."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize mock executor with optional response sequence.

        Args:
            responses: Optional list of responses to return in sequence
        """
        self.responses = responses or []
        self.call_count = 0

    async def execute(self, query: str) -> str:
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
        async def execute(self, query: str) -> str:
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
        async def execute(self, query: str) -> str:
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
            self.name = name

        async def execute(self, query: str) -> str:
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
