"""Workflow implementations for multi-agent orchestration.

This module provides workflow implementations for orchestrating multiple
agents in various patterns like iterative refinement, pipelines, and debates.
"""

from gluellm.workflows._base import Workflow, WorkflowResult
from gluellm.workflows.debate import DebateConfig, DebateWorkflow
from gluellm.workflows.iterative import IterativeRefinementWorkflow
from gluellm.workflows.pipeline import PipelineWorkflow

__all__ = [
    "Workflow",
    "WorkflowResult",
    "IterativeRefinementWorkflow",
    "PipelineWorkflow",
    "DebateWorkflow",
    "DebateConfig",
]
