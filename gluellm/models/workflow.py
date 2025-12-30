"""Workflow configuration models for multi-agent workflows.

This module provides configuration models for defining workflows,
including critic configurations and iterative refinement settings.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gluellm.executors._base import Executor


class CriticConfig(BaseModel):
    """Configuration for a specialized critic in a workflow.

    Attributes:
        executor: The executor to use for this critic
        specialty: The specialty/focus area of this critic (e.g., "grammar", "technical accuracy", "tone")
        goal: The specific goal this critic should optimize for
        weight: Optional weight for this critic's feedback (defaults to 1.0)

    Example:
        >>> from gluellm.executors import AgentExecutor
        >>> from gluellm.models.workflow import CriticConfig
        >>>
        >>> critic_config = CriticConfig(
        ...     executor=AgentExecutor(my_agent),
        ...     specialty="grammar and clarity",
        ...     goal="Optimize for readability and eliminate errors"
        ... )
    """

    model_config = {"arbitrary_types_allowed": True}

    executor: "Executor" = Field(description="The executor to use for this critic")
    specialty: str = Field(description="The specialty/focus area of this critic")
    goal: str = Field(description="The specific goal this critic should optimize for")
    weight: float = Field(default=1.0, description="Optional weight for this critic's feedback")


class IterativeConfig(BaseModel):
    """Configuration for iterative refinement workflows.

    Attributes:
        max_iterations: Maximum number of refinement iterations
        min_quality_score: Optional minimum quality score threshold for early stopping
        convergence_threshold: Optional convergence threshold for stopping early
        quality_evaluator: Optional callable to evaluate quality (content, feedback) -> float

    Example:
        >>> from gluellm.models.workflow import IterativeConfig
        >>>
        >>> config = IterativeConfig(
        ...     max_iterations=5,
        ...     min_quality_score=0.8
        ... )
    """

    max_iterations: int = Field(default=3, description="Maximum number of refinement iterations", gt=0)
    min_quality_score: float | None = Field(
        default=None, description="Optional minimum quality score threshold for early stopping", ge=0.0, le=1.0
    )
    convergence_threshold: float | None = Field(
        default=None, description="Optional convergence threshold for stopping early", ge=0.0, le=1.0
    )
    quality_evaluator: Any | None = Field(
        default=None, description="Optional callable to evaluate quality (content, feedback) -> float"
    )


class ReActConfig(BaseModel):
    """Configuration for ReAct (Reasoning + Acting) workflow.

    Attributes:
        max_steps: Maximum number of reasoning/action steps
        thought_prefix: Prefix for thought steps
        action_prefix: Prefix for action steps
        observation_prefix: Prefix for observation steps
        stop_on_final_answer: Whether to stop when final answer is detected
    """

    max_steps: int = Field(default=10, description="Maximum number of reasoning/action steps", gt=0)
    thought_prefix: str = Field(default="Thought:", description="Prefix for thought steps")
    action_prefix: str = Field(default="Action:", description="Prefix for action steps")
    observation_prefix: str = Field(default="Observation:", description="Prefix for observation steps")
    stop_on_final_answer: bool = Field(default=True, description="Whether to stop when final answer is detected")


class ReflectionConfig(BaseModel):
    """Configuration for reflection workflow.

    Attributes:
        max_reflections: Maximum number of reflection iterations
        min_improvement_threshold: Optional minimum improvement threshold for early stopping
        reflection_prompt_template: Optional custom prompt template for reflection
    """

    max_reflections: int = Field(default=3, description="Maximum number of reflection iterations", gt=0)
    min_improvement_threshold: float | None = Field(
        default=None, description="Optional minimum improvement threshold for early stopping", ge=0.0, le=1.0
    )
    reflection_prompt_template: str | None = Field(
        default=None, description="Optional custom prompt template for reflection"
    )


class HierarchicalConfig(BaseModel):
    """Configuration for hierarchical task workflow.

    Attributes:
        max_subtasks: Maximum number of subtasks to create
        parallel_workers: Whether to execute worker tasks in parallel
        synthesis_strategy: Strategy for synthesizing worker outputs
    """

    max_subtasks: int = Field(default=5, description="Maximum number of subtasks to create", gt=0)
    parallel_workers: bool = Field(default=True, description="Whether to execute worker tasks in parallel")
    synthesis_strategy: Literal["concatenate", "summarize", "merge"] = Field(
        default="summarize", description="Strategy for synthesizing worker outputs"
    )


class MapReduceConfig(BaseModel):
    """Configuration for MapReduce workflow.

    Attributes:
        chunk_size: Optional chunk size for splitting input
        chunk_overlap: Overlap between chunks
        max_parallel_chunks: Maximum number of chunks to process in parallel
        reduce_strategy: Strategy for reducing mapped results
    """

    chunk_size: int | None = Field(default=None, description="Optional chunk size for splitting input", gt=0)
    chunk_overlap: int = Field(default=0, description="Overlap between chunks", ge=0)
    max_parallel_chunks: int | None = Field(
        default=None, description="Maximum number of chunks to process in parallel", gt=0
    )
    reduce_strategy: Literal["concatenate", "summarize", "hierarchical"] = Field(
        default="summarize", description="Strategy for reducing mapped results"
    )


class TreeOfThoughtsConfig(BaseModel):
    """Configuration for Tree of Thoughts workflow.

    Attributes:
        branching_factor: Number of branches to explore at each level
        max_depth: Maximum depth of the tree
        evaluation_strategy: Strategy for evaluating and selecting paths
        prune_threshold: Threshold for pruning low-scoring paths
    """

    branching_factor: int = Field(default=3, description="Number of branches to explore at each level", gt=0)
    max_depth: int = Field(default=3, description="Maximum depth of the tree", gt=0)
    evaluation_strategy: Literal["vote", "score", "best_first"] = Field(
        default="score", description="Strategy for evaluating and selecting paths"
    )
    prune_threshold: float = Field(default=0.3, description="Threshold for pruning low-scoring paths", ge=0.0, le=1.0)


class ExpertConfig(BaseModel):
    """Configuration for an expert in Mixture of Experts workflow.

    Attributes:
        executor: The executor for this expert
        specialty: The specialty/domain of this expert
        description: Description of what this expert does
        activation_keywords: Keywords that activate this expert
    """

    model_config = {"arbitrary_types_allowed": True}

    executor: "Executor" = Field(description="The executor for this expert")
    specialty: str = Field(description="The specialty/domain of this expert")
    description: str = Field(description="Description of what this expert does")
    activation_keywords: list[str] = Field(default_factory=list, description="Keywords that activate this expert")


class MoEConfig(BaseModel):
    """Configuration for Mixture of Experts workflow.

    Attributes:
        routing_strategy: Strategy for routing queries to experts
        top_k: Number of top experts to use when routing_strategy is 'top_k'
        combine_strategy: Strategy for combining expert outputs
    """

    routing_strategy: Literal["keyword", "semantic", "all", "top_k"] = Field(
        default="semantic", description="Strategy for routing queries to experts"
    )
    top_k: int = Field(default=2, description="Number of top experts to use when routing_strategy is 'top_k'", gt=0)
    combine_strategy: Literal["concatenate", "synthesize", "vote"] = Field(
        default="synthesize", description="Strategy for combining expert outputs"
    )


class Principle(BaseModel):
    """A principle for Constitutional AI workflow.

    Attributes:
        name: Name of the principle
        description: Description of what the principle checks
        severity: Severity level of violations
    """

    name: str = Field(description="Name of the principle")
    description: str = Field(description="Description of what the principle checks")
    severity: Literal["warning", "error", "critical"] = Field(
        default="error", description="Severity level of violations"
    )


class ConstitutionalConfig(BaseModel):
    """Configuration for Constitutional AI workflow.

    Attributes:
        principles: List of principles to check against
        max_revisions: Maximum number of revision iterations
        require_all_pass: Whether all principles must pass
    """

    principles: list[Principle] = Field(description="List of principles to check against")
    max_revisions: int = Field(default=3, description="Maximum number of revision iterations", gt=0)
    require_all_pass: bool = Field(default=True, description="Whether all principles must pass before completion")


class SocraticConfig(BaseModel):
    """Configuration for Socratic dialogue workflow.

    Attributes:
        max_exchanges: Maximum number of question-answer exchanges
        mode: Mode of dialogue (teacher_student or peer)
        synthesis_at_end: Whether to synthesize the dialogue at the end
    """

    max_exchanges: int = Field(default=5, description="Maximum number of question-answer exchanges", gt=0)
    mode: Literal["teacher_student", "peer"] = Field(
        default="peer", description="Mode of dialogue (teacher_student or peer)"
    )
    synthesis_at_end: bool = Field(default=True, description="Whether to synthesize the dialogue at the end")


class ConsensusConfig(BaseModel):
    """Configuration for consensus building workflow.

    Attributes:
        min_agreement_ratio: Minimum ratio of agents that must agree
        max_rounds: Maximum number of consensus rounds
        voting_strategy: Strategy for voting/agreement
        allow_abstention: Whether agents can abstain from voting
    """

    min_agreement_ratio: float = Field(
        default=0.7, description="Minimum ratio of agents that must agree", ge=0.0, le=1.0
    )
    max_rounds: int = Field(default=5, description="Maximum number of consensus rounds", gt=0)
    voting_strategy: Literal["majority", "unanimous", "weighted"] = Field(
        default="majority", description="Strategy for voting/agreement"
    )
    allow_abstention: bool = Field(default=False, description="Whether agents can abstain from voting")


class RoundRobinConfig(BaseModel):
    """Configuration for round-robin workflow.

    Attributes:
        max_rounds: Maximum number of rounds
        contribution_style: Style of contribution (extend, refine, challenge)
        final_synthesis: Whether to synthesize contributions at the end
    """

    max_rounds: int = Field(default=3, description="Maximum number of rounds", gt=0)
    contribution_style: Literal["extend", "refine", "challenge"] = Field(
        default="extend", description="Style of contribution (extend, refine, challenge)"
    )
    final_synthesis: bool = Field(default=True, description="Whether to synthesize contributions at the end")


class ChainOfDensityConfig(BaseModel):
    """Configuration for Chain of Density workflow.

    Attributes:
        num_iterations: Number of density-increasing iterations
        density_increment: Type of density increment (entities, details, examples)
        preserve_length: Whether to preserve approximate length
    """

    num_iterations: int = Field(default=5, description="Number of density-increasing iterations", gt=0)
    density_increment: Literal["entities", "details", "examples"] = Field(
        default="entities", description="Type of density increment (entities, details, examples)"
    )
    preserve_length: bool = Field(default=True, description="Whether to preserve approximate length")


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) workflow.

    Attributes:
        max_retrieved_chunks: Maximum number of chunks to retrieve
        include_sources: Whether to include source information
        verify_facts: Whether to verify facts in the generated response
        fallback_on_no_context: Whether to fallback if no context is retrieved
    """

    max_retrieved_chunks: int = Field(default=5, description="Maximum number of chunks to retrieve", gt=0)
    include_sources: bool = Field(default=True, description="Whether to include source information")
    verify_facts: bool = Field(default=False, description="Whether to verify facts in the generated response")
    fallback_on_no_context: bool = Field(default=True, description="Whether to fallback if no context is retrieved")


class ChatRoomConfig(BaseModel):
    """Configuration for chat room workflow.

    Attributes:
        max_rounds: Maximum number of discussion rounds (safety limit)
        allow_moderator_interjection: Whether moderator can guide discussion with comments
        synthesis_rounds: Number of refinement passes for collaborative final answer
    """

    max_rounds: int = Field(default=10, description="Maximum number of discussion rounds (safety limit)", gt=0)
    allow_moderator_interjection: bool = Field(
        default=True, description="Whether moderator can guide discussion with comments"
    )
    synthesis_rounds: int = Field(
        default=1, description="Number of refinement passes for collaborative final answer", gt=0
    )


# Rebuild models after Executor is available to resolve forward references
def _rebuild_models():
    """Rebuild Pydantic models to resolve forward references."""
    try:
        from gluellm.executors._base import Executor

        # Rebuild models with Executor in the namespace
        CriticConfig.model_rebuild(_types_namespace={"Executor": Executor})
        ExpertConfig.model_rebuild(_types_namespace={"Executor": Executor})
    except (ImportError, Exception):
        # Executor not available yet or circular import issue
        import sys

        if "pytest" in sys.modules or "unittest" in sys.modules:
            # In tests, try again later
            pass
        else:
            # During normal import, this is expected
            pass


# Try to rebuild immediately if Executor is already available
_rebuild_models()
