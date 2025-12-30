"""Workflow implementations for multi-agent orchestration.

This module provides workflow implementations for orchestrating multiple
agents in various patterns like iterative refinement, pipelines, debates,
and many other advanced patterns.
"""

from gluellm.models.workflow import ChatRoomConfig
from gluellm.workflows._base import Workflow, WorkflowResult
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

__all__ = [
    "Workflow",
    "WorkflowResult",
    # Original workflows
    "IterativeRefinementWorkflow",
    "PipelineWorkflow",
    "DebateWorkflow",
    "DebateConfig",
    # New workflows
    "ReflectionWorkflow",
    "ChainOfDensityWorkflow",
    "SocraticWorkflow",
    "RAGWorkflow",
    "RoundRobinWorkflow",
    "ConsensusWorkflow",
    "HierarchicalWorkflow",
    "MapReduceWorkflow",
    "ReActWorkflow",
    "MixtureOfExpertsWorkflow",
    "ConstitutionalWorkflow",
    "TreeOfThoughtsWorkflow",
    "ChatRoomWorkflow",
    "ChatRoomConfig",
]
