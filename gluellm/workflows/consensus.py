"""Consensus building workflow for multi-agent agreement.

This module provides the ConsensusWorkflow, which enables multiple agents
to propose solutions and iterate until consensus is reached.
"""

import asyncio
import re
from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ConsensusConfig
from gluellm.observability.logging_config import get_logger
from gluellm.workflows._base import Workflow, WorkflowResult

logger = get_logger(__name__)


class ConsensusWorkflow(Workflow):
    """Workflow for building consensus among multiple agents.

    This workflow orchestrates multiple agents proposing solutions, voting
    on proposals, and iterating until consensus is reached.

    Attributes:
        proposers: List of (agent_name, executor) tuples for proposing solutions
        config: Configuration for the consensus process

    Example:
        >>> from gluellm.workflows.consensus import ConsensusWorkflow, ConsensusConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = ConsensusWorkflow(
        ...     proposers=[
        ...         ("Agent1", AgentExecutor(agent1)),
        ...         ("Agent2", AgentExecutor(agent2)),
        ...         ("Agent3", AgentExecutor(agent3)),
        ...     ],
        ...     config=ConsensusConfig(min_agreement_ratio=0.7, max_rounds=5)
        ... )
        >>>
        >>> result = await workflow.execute("Design a solution for X")
    """

    def __init__(
        self,
        proposers: list[tuple[str, Executor]],
        config: ConsensusConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a ConsensusWorkflow.

        Args:
            proposers: List of (agent_name, executor) tuples
            config: Optional configuration for consensus process
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.proposers = proposers
        self.config = config or ConsensusConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute consensus workflow.

        Args:
            initial_input: The problem/question to solve
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        proposals_history = []
        current_proposals = []

        for round_num in range(self.config.max_rounds):
            # Each agent proposes a solution
            round_proposals = []
            proposal_tasks = []

            for agent_name, executor in self.proposers:
                prompt = self._build_proposal_prompt(initial_input, proposals_history, round_num + 1, agent_name)
                task = self._execute_proposal(executor, prompt, agent_name, round_num + 1)
                proposal_tasks.append((agent_name, task))

            # Execute proposals in parallel
            results = await asyncio.gather(*[task for _, task in proposal_tasks], return_exceptions=True)

            for (agent_name, _), result in zip(proposal_tasks, results, strict=False):
                proposal = f"Error: {type(result).__name__}: {str(result)}" if isinstance(result, Exception) else result
                round_proposals.append((agent_name, proposal))
                current_proposals.append(proposal)

                interactions.append(
                    {
                        "round": round_num + 1,
                        "agent": agent_name,
                        "stage": "proposal",
                        "input": self._build_proposal_prompt(
                            initial_input, proposals_history, round_num + 1, agent_name
                        ),
                        "output": proposal,
                    }
                )

            proposals_history.append((round_num + 1, round_proposals))

            # Vote on proposals
            votes = await self._collect_votes(round_proposals, initial_input, interactions, round_num + 1)

            # Check consensus
            agreement_ratio = self._calculate_agreement(votes, round_proposals)
            interactions.append(
                {
                    "round": round_num + 1,
                    "stage": "voting",
                    "votes": votes,
                    "agreement_ratio": agreement_ratio,
                }
            )

            if agreement_ratio >= self.config.min_agreement_ratio:
                # Consensus reached
                winning_proposal = self._select_winning_proposal(votes, round_proposals)
                return WorkflowResult(
                    final_output=winning_proposal,
                    iterations=round_num + 1,
                    agent_interactions=interactions,
                    metadata={
                        "consensus_reached": True,
                        "agreement_ratio": agreement_ratio,
                        "winning_proposal_agent": self._get_winning_agent(votes, round_proposals),
                    },
                )

        # Max rounds reached, return best proposal
        final_votes = await self._collect_votes(
            proposals_history[-1][1] if proposals_history else [], initial_input, interactions, round_num + 1
        )
        winning_proposal = self._select_winning_proposal(
            final_votes, proposals_history[-1][1] if proposals_history else []
        )

        return WorkflowResult(
            final_output=winning_proposal or "No consensus reached.",
            iterations=self.config.max_rounds,
            agent_interactions=interactions,
            metadata={
                "consensus_reached": False,
                "final_agreement_ratio": self._calculate_agreement(
                    final_votes, proposals_history[-1][1] if proposals_history else []
                ),
            },
        )

    async def _execute_proposal(self, executor: Executor, prompt: str, agent_name: str, round_num: int) -> str:
        """Execute a proposal from an agent.

        Args:
            executor: The executor for the agent
            prompt: The proposal prompt
            agent_name: Name of the agent
            round_num: Round number

        Returns:
            The proposal
        """
        return await executor.execute(prompt)

    def _build_proposal_prompt(
        self, initial_input: str, proposals_history: list, round_num: int, agent_name: str
    ) -> str:
        """Build a proposal prompt.

        Args:
            initial_input: Original problem/question
            proposals_history: History of previous rounds' proposals
            round_num: Current round number
            agent_name: Name of the proposing agent

        Returns:
            Formatted proposal prompt
        """
        if round_num == 1:
            return f"""Problem: {initial_input}

As {agent_name}, propose a solution to this problem. Be specific and actionable."""

        # Include previous proposals
        history_text = ""
        for prev_round, prev_proposals in proposals_history:
            history_text += f"\n\nRound {prev_round} proposals:\n"
            for prop_agent, prop_text in prev_proposals:
                history_text += f"- {prop_agent}: {prop_text[:200]}...\n"

        return f"""Problem: {initial_input}

Previous rounds:
{history_text}

As {agent_name}, propose a refined solution that addresses feedback and builds on previous proposals.
Consider what worked and what didn't in earlier rounds."""

    async def _collect_votes(
        self,
        proposals: list[tuple[str, str]],
        initial_input: str,
        interactions: list,
        round_num: int,
    ) -> dict[str, list[str]]:
        """Collect votes from ALL agents on proposals.

        Each agent evaluates all proposals and votes for the best one.
        Agents can abstain if configured to allow abstention.

        Args:
            proposals: List of (agent_name, proposal) tuples
            initial_input: Original problem
            interactions: Interactions list to append to
            round_num: Round number

        Returns:
            Dictionary mapping proposal index to list of voting agent names
        """
        if not proposals:
            return {}

        votes: dict[str, list[str]] = {str(i): [] for i in range(len(proposals))}
        abstentions: list[str] = []

        # Build voting prompt once (same for all agents)
        voting_prompt = self._build_voting_prompt(initial_input, proposals)

        # Collect votes from all agents in parallel
        async def _get_agent_vote(agent_name: str, executor: Executor) -> tuple[str, str]:
            """Execute voting for a single agent."""
            try:
                vote_response = await executor.execute(voting_prompt)
                return (agent_name, vote_response)
            except Exception as e:
                logger.warning(f"Agent {agent_name} failed to vote: {e}")
                return (agent_name, f"Error: {e}")

        # Execute voting in parallel
        vote_tasks = [_get_agent_vote(name, executor) for name, executor in self.proposers]
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        # Process each vote
        for result in vote_results:
            if isinstance(result, Exception):
                logger.error(f"Voting task failed: {result}")
                continue

            agent_name, vote_response = result

            # Parse vote - check for abstention first
            vote_response_lower = vote_response.lower()
            if "abstain" in vote_response_lower:
                if self.config.allow_abstention:
                    abstentions.append(agent_name)
                    logger.debug(f"Agent {agent_name} abstained from voting")
                    interactions.append(
                        {
                            "round": round_num,
                            "agent": agent_name,
                            "stage": "vote",
                            "vote": "abstain",
                            "vote_response": vote_response,
                        }
                    )
                    continue
                # If abstention not allowed, try to find a number anyway
                logger.debug(f"Agent {agent_name} tried to abstain but abstention is not allowed")

            # Extract proposal number from response
            vote_numbers = re.findall(r"\d+", vote_response)
            if vote_numbers:
                voted_index = int(vote_numbers[0])
                # Validate the vote is within range
                if 0 <= voted_index < len(proposals):
                    votes[str(voted_index)].append(agent_name)
                    logger.debug(f"Agent {agent_name} voted for proposal {voted_index}")
                    interactions.append(
                        {
                            "round": round_num,
                            "agent": agent_name,
                            "stage": "vote",
                            "vote": voted_index,
                            "vote_response": vote_response,
                        }
                    )
                else:
                    # Invalid vote index, log warning
                    logger.warning(
                        f"Agent {agent_name} voted for invalid proposal {voted_index} "
                        f"(valid range: 0-{len(proposals) - 1})"
                    )
                    interactions.append(
                        {
                            "round": round_num,
                            "agent": agent_name,
                            "stage": "vote",
                            "vote": "invalid",
                            "vote_response": vote_response,
                            "error": f"Invalid proposal index: {voted_index}",
                        }
                    )
            else:
                # Could not parse vote
                logger.warning(f"Could not parse vote from agent {agent_name}: {vote_response[:100]}")
                interactions.append(
                    {
                        "round": round_num,
                        "agent": agent_name,
                        "stage": "vote",
                        "vote": "unparseable",
                        "vote_response": vote_response,
                    }
                )

        # Log voting summary
        total_votes = sum(len(voters) for voters in votes.values())
        logger.info(f"Round {round_num} voting complete: {total_votes} votes cast, {len(abstentions)} abstentions")

        return votes

    def _build_voting_prompt(self, initial_input: str, proposals: list[tuple[str, str]]) -> str:
        """Build a voting prompt.

        Args:
            initial_input: Original problem
            proposals: List of (agent_name, proposal) tuples

        Returns:
            Formatted voting prompt
        """
        proposals_text = ""
        for i, (agent_name, proposal) in enumerate(proposals):
            proposals_text += f"\nProposal {i} (by {agent_name}):\n{proposal}\n"

        return f"""Problem: {initial_input}

Proposals to vote on:
{proposals_text}

Review all proposals and vote for the best one. Respond with the proposal number (0-{len(proposals) - 1})
or "abstain" if you cannot decide."""

    def _calculate_agreement(self, votes: dict[str, list[str]], proposals: list[tuple[str, str]]) -> float:
        """Calculate agreement ratio from votes.

        Args:
            votes: Dictionary mapping proposal index to list of voters
            proposals: List of proposals

        Returns:
            Agreement ratio (0.0 to 1.0)
        """
        if not votes or not proposals:
            return 0.0

        # Find proposal with most votes
        max_votes = max(len(voters) for voters in votes.values()) if votes else 0
        total_voters = sum(len(voters) for voters in votes.values())

        if total_voters == 0:
            return 0.0

        return max_votes / total_voters

    def _select_winning_proposal(self, votes: dict[str, list[str]], proposals: list[tuple[str, str]]) -> str:
        """Select the winning proposal based on votes.

        Args:
            votes: Dictionary mapping proposal index to list of voters
            proposals: List of (agent_name, proposal) tuples

        Returns:
            The winning proposal text
        """
        if not votes or not proposals:
            return proposals[0][1] if proposals else ""

        # Find proposal with most votes
        winning_index = max(votes.keys(), key=lambda k: len(votes[k]))
        return proposals[int(winning_index)][1]

    def _get_winning_agent(self, votes: dict[str, list[str]], proposals: list[tuple[str, str]]) -> str:
        """Get the name of the agent with the winning proposal.

        Args:
            votes: Dictionary mapping proposal index to list of voters
            proposals: List of (agent_name, proposal) tuples

        Returns:
            Name of winning agent
        """
        if not votes or not proposals:
            return proposals[0][0] if proposals else ""

        winning_index = max(votes.keys(), key=lambda k: len(votes[k]))
        return proposals[int(winning_index)][0]

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.proposers) >= 2 and self.config.max_rounds > 0
