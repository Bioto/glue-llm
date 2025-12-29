"""Workflow implementations for multi-agent orchestration.

This module provides workflow implementations for orchestrating multiple
agents in various patterns like iterative refinement, pipelines, and debates.
"""

from source.workflows._base import Workflow, WorkflowResult
from source.workflows.debate import DebateConfig, DebateWorkflow
from source.workflows.iterative import IterativeRefinementWorkflow
from source.workflows.pipeline import PipelineWorkflow

__all__ = [
    "Workflow",
    "WorkflowResult",
    "IterativeRefinementWorkflow",
    "PipelineWorkflow",
    "DebateWorkflow",
    "DebateConfig",
]
