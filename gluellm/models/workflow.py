"""Workflow configuration models for multi-agent workflows.

This module provides configuration models for defining workflows,
including critic configurations and iterative refinement settings.
"""

from typing import TYPE_CHECKING, Any

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


# Rebuild models after Executor is available to resolve forward references
def _rebuild_models():
    """Rebuild Pydantic models to resolve forward references."""
    try:
        from gluellm.executors._base import Executor  # noqa: F401

        CriticConfig.model_rebuild()
    except ImportError:
        # Executor not available yet, will be rebuilt when imported
        pass


# Try to rebuild immediately if Executor is already available
_rebuild_models()
