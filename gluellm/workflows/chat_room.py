"""Chat room workflow for multi-agent discussions.

This module provides the ChatRoomWorkflow, which orchestrates multiple agents
discussing a topic in a chat room style, with a moderator controlling when
the discussion concludes and agents collaboratively producing a final answer.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import ChatRoomConfig
from gluellm.observability.logging_config import get_logger
from gluellm.workflows._base import Workflow, WorkflowResult

logger = get_logger(__name__)


class ChatRoomWorkflow(Workflow):
    """Workflow for multi-agent chat room discussions with moderator.

    This workflow orchestrates multiple agents discussing a user's query in a
    natural chat room style. Agents take turns speaking, with a moderator
    deciding when the discussion is complete. The agents then collaboratively
    draft a final answer.

    Attributes:
        participants: List of (participant_name, executor) tuples
        moderator: Moderator executor that decides when discussion is complete
        config: Configuration for the chat room

    Example:
        >>> from gluellm.workflows.chat_room import ChatRoomWorkflow, ChatRoomConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = ChatRoomWorkflow(
        ...     participants=[
        ...         ("Alice", AgentExecutor(alice_agent)),
        ...         ("Bob", AgentExecutor(bob_agent)),
        ...         ("Charlie", AgentExecutor(charlie_agent)),
        ...     ],
        ...     moderator=AgentExecutor(moderator_agent),
        ...     config=ChatRoomConfig(max_rounds=10, synthesis_rounds=2)
        ... )
        >>>
        >>> result = await workflow.execute("How should we design a new API?")
    """

    def __init__(
        self,
        participants: list[tuple[str, Executor]],
        moderator: Executor,
        config: ChatRoomConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a ChatRoomWorkflow.

        Args:
            participants: List of (participant_name, executor) tuples
            moderator: Moderator executor that decides when discussion is complete
            config: Optional configuration for the chat room
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.participants = participants
        self.moderator = moderator
        self.config = config or ChatRoomConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute chat room workflow.

        Args:
            initial_input: The user's query/question to discuss
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        discussion_history = []

        # Discussion phase: agents take turns, moderator evaluates after each round
        for round_num in range(self.config.max_rounds):
            round_interactions = []

            # Each participant speaks in fixed order
            for participant_name, executor in self.participants:
                prompt = self._build_discussion_prompt(initial_input, discussion_history, participant_name)
                response = await executor.execute(prompt)
                discussion_history.append((participant_name, response))
                round_interactions.append(
                    {
                        "round": round_num + 1,
                        "participant": participant_name,
                        "message": response,
                    }
                )

            interactions.extend(round_interactions)

            # Moderator evaluates whether to continue
            moderator_response = await self._check_moderator(initial_input, discussion_history, round_num + 1)
            should_continue = self._parse_moderator_decision(moderator_response)

            interactions.append(
                {
                    "round": round_num + 1,
                    "stage": "moderator_evaluation",
                    "moderator_response": moderator_response,
                    "should_continue": should_continue,
                }
            )

            if not should_continue:
                logger.info(f"Moderator concluded discussion after round {round_num + 1}")
                break

        # Track discussion rounds before synthesis
        discussion_rounds = len(
            {i.get("round", 0) for i in interactions if i.get("round") and i.get("stage") != "synthesis"}
        )

        # Collaborative synthesis phase: agents work together to create final answer
        final_answer = await self._collaborative_synthesis(initial_input, discussion_history, interactions)

        return WorkflowResult(
            final_output=final_answer,
            iterations=discussion_rounds,
            agent_interactions=interactions,
            metadata={
                "participants": [name for name, _ in self.participants],
                "discussion_rounds": discussion_rounds,
                "moderator_concluded": not should_continue if "should_continue" in locals() else False,
            },
        )

    def _build_discussion_prompt(
        self, initial_input: str, discussion_history: list[tuple[str, str]], participant_name: str
    ) -> str:
        """Build a prompt for a participant's turn in the discussion.

        Args:
            initial_input: The original user query
            discussion_history: List of (participant_name, message) tuples
            participant_name: Name of the current participant

        Returns:
            Formatted discussion prompt
        """
        if not discussion_history:
            return f"""User Query: {initial_input}

You are {participant_name} in a collaborative discussion. Share your thoughts, ideas, or questions about this query.
Be conversational and natural - you can reference what others have said, ask clarifying questions, or build on ideas.
Your response:"""

        # Format discussion history
        history_text = "\n\n".join([f"{name}: {message}" for name, message in discussion_history])

        return f"""User Query: {initial_input}

Discussion so far:
{history_text}

You are {participant_name}. Continue the discussion naturally. You can:
- Respond to points made by others
- Ask clarifying questions
- Build on ideas presented
- Share new perspectives
- Express agreement or disagreement

Your response:"""

    async def _check_moderator(
        self, initial_input: str, discussion_history: list[tuple[str, str]], round_num: int
    ) -> str:
        """Check with moderator whether discussion should continue.

        Args:
            initial_input: The original user query
            discussion_history: List of (participant_name, message) tuples
            round_num: Current round number

        Returns:
            Moderator's response
        """
        history_text = "\n\n".join([f"{name}: {message}" for name, message in discussion_history])

        if self.config.allow_moderator_interjection:
            prompt = f"""User Query: {initial_input}

Discussion (Round {round_num}):
{history_text}

As the moderator, evaluate whether the discussion has reached a sufficient depth and quality to answer the user's query.

Respond with:
- "CONTINUE" if the discussion needs more depth or exploration
- "CONCLUDE" if the discussion is ready to synthesize into a final answer

You may optionally provide brief guidance or comments before your decision.

Your evaluation:"""
        else:
            prompt = f"""User Query: {initial_input}

Discussion (Round {round_num}):
{history_text}

As the moderator, evaluate whether the discussion has reached a sufficient depth and quality to answer the user's query.

Respond with:
- "CONTINUE" if the discussion needs more depth or exploration
- "CONCLUDE" if the discussion is ready to synthesize into a final answer

Your evaluation:"""

        return await self.moderator.execute(prompt)

    def _parse_moderator_decision(self, moderator_response: str) -> bool:
        """Parse moderator response to determine if discussion should continue.

        Args:
            moderator_response: The moderator's response text

        Returns:
            True if discussion should continue, False if it should conclude
        """
        response_upper = moderator_response.upper().strip()

        # Check for explicit CONCLUDE signal
        if "CONCLUDE" in response_upper:
            return False

        # Check for explicit CONTINUE signal
        if "CONTINUE" in response_upper:
            return True

        # Default: if response is very short and contains "yes" or "continue", continue
        # Otherwise, if it contains "done", "complete", "ready", "sufficient", conclude
        if any(word in response_upper for word in ["DONE", "COMPLETE", "READY", "SUFFICIENT", "FINISHED"]):
            return False

        # Default to continue if unclear
        logger.warning(
            f"Could not clearly parse moderator decision, defaulting to continue. Response: {moderator_response[:100]}"
        )
        return True

    async def _collaborative_synthesis(
        self, initial_input: str, discussion_history: list[tuple[str, str]], interactions: list[dict[str, Any]]
    ) -> str:
        """Collaboratively synthesize final answer from discussion.

        Args:
            initial_input: The original user query
            discussion_history: List of (participant_name, message) tuples
            interactions: All interactions for metadata

        Returns:
            Final synthesized answer
        """
        history_text = "\n\n".join([f"{name}: {message}" for name, message in discussion_history])

        # First participant drafts the initial answer
        draft_prompt = f"""User Query: {initial_input}

Full Discussion:
{history_text}

As {self.participants[0][0]}, synthesize the discussion into a clear, comprehensive answer to the user's query.
Incorporate the key points, insights, and conclusions from the discussion.
Your draft answer:"""

        current_answer = await self.participants[0][1].execute(draft_prompt)
        interactions.append(
            {
                "stage": "synthesis",
                "participant": self.participants[0][0],
                "action": "draft",
                "output": current_answer,
            }
        )

        # Subsequent participants refine the answer
        for _synthesis_round in range(self.config.synthesis_rounds):
            for _idx, (participant_name, executor) in enumerate(self.participants[1:], start=1):
                refine_prompt = f"""User Query: {initial_input}

Original Discussion:
{history_text}

Current Draft Answer:
{current_answer}

As {participant_name}, refine and improve this answer. You can:
- Add missing important points from the discussion
- Clarify or improve wording
- Ensure completeness and accuracy
- Polish the presentation

Your refined answer:"""

                refined_answer = await executor.execute(refine_prompt)
                interactions.append(
                    {
                        "stage": "synthesis",
                        "participant": participant_name,
                        "action": "refine",
                        "input": current_answer,
                        "output": refined_answer,
                    }
                )
                current_answer = refined_answer

        return current_answer

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return (
            len(self.participants) >= 2
            and self.moderator is not None
            and self.config.max_rounds > 0
            and self.config.synthesis_rounds > 0
        )
