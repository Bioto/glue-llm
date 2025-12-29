"""Socratic dialogue workflow for question-answer exploration.

This module provides the SocraticWorkflow, which enables two agents to
engage in a Socratic dialogue through iterative questioning and answering.
"""

from typing import Any

from gluellm.executors._base import Executor
from gluellm.models.hook import HookRegistry
from gluellm.models.workflow import SocraticConfig
from gluellm.workflows._base import Workflow, WorkflowResult


class SocraticWorkflow(Workflow):
    """Workflow for Socratic dialogue between two agents.

    This workflow orchestrates a question-answer dialogue between two agents,
    exploring a topic through iterative questioning. Supports teacher-student
    and peer-to-peer modes.

    Attributes:
        questioner: The executor for asking questions
        responder: The executor for answering questions
        config: Configuration for the Socratic dialogue

    Example:
        >>> from gluellm.workflows.socratic import SocraticWorkflow, SocraticConfig
        >>> from gluellm.executors import AgentExecutor
        >>>
        >>> workflow = SocraticWorkflow(
        ...     questioner=AgentExecutor(questioner_agent),
        ...     responder=AgentExecutor(responder_agent),
        ...     config=SocraticConfig(max_exchanges=5, mode="peer")
        ... )
        >>>
        >>> result = await workflow.execute("What is artificial intelligence?")
    """

    def __init__(
        self,
        questioner: Executor,
        responder: Executor,
        config: SocraticConfig | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize a SocraticWorkflow.

        Args:
            questioner: The executor for asking questions
            responder: The executor for answering questions
            config: Optional configuration for Socratic dialogue
            hook_registry: Optional webhook registry for this workflow
        """
        super().__init__(hook_registry=hook_registry)
        self.questioner = questioner
        self.responder = responder
        self.config = config or SocraticConfig()

    async def _execute_internal(self, initial_input: str, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute Socratic dialogue workflow.

        Args:
            initial_input: The initial topic/question to explore
            context: Optional context dictionary (currently unused)

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        interactions = []
        dialogue_history = []

        for exchange_num in range(self.config.max_exchanges):
            # Build context for questioner
            if exchange_num == 0:
                questioner_prompt = self._build_initial_question_prompt(initial_input)
            else:
                questioner_prompt = self._build_followup_question_prompt(initial_input, dialogue_history)

            # Questioner asks a question
            question = await self.questioner.execute(questioner_prompt)
            interactions.append(
                {
                    "exchange": exchange_num + 1,
                    "agent": "questioner",
                    "input": questioner_prompt,
                    "output": question,
                }
            )

            # Build context for responder
            responder_prompt = self._build_answer_prompt(initial_input, dialogue_history, question)

            # Responder answers
            answer = await self.responder.execute(responder_prompt)
            interactions.append(
                {
                    "exchange": exchange_num + 1,
                    "agent": "responder",
                    "input": responder_prompt,
                    "output": answer,
                }
            )

            # Record exchange
            dialogue_history.append((question, answer))

        # Build final output
        if self.config.synthesis_at_end:
            synthesis_prompt = self._build_synthesis_prompt(initial_input, dialogue_history)
            synthesizer = self.responder  # Use responder for synthesis
            final_output = await synthesizer.execute(synthesis_prompt)
            interactions.append(
                {
                    "stage": "synthesis",
                    "agent": "synthesizer",
                    "input": synthesis_prompt,
                    "output": final_output,
                }
            )
        else:
            # Concatenate dialogue
            dialogue_text = []
            for i, (q, a) in enumerate(dialogue_history, 1):
                dialogue_text.append(f"Q{i}: {q}\nA{i}: {a}")
            final_output = "\n\n".join(dialogue_text)

        return WorkflowResult(
            final_output=final_output,
            iterations=self.config.max_exchanges,
            agent_interactions=interactions,
            metadata={
                "mode": self.config.mode,
                "exchanges": len(dialogue_history),
                "synthesized": self.config.synthesis_at_end,
            },
        )

    def _build_initial_question_prompt(self, topic: str) -> str:
        """Build the initial question prompt.

        Args:
            topic: The initial topic

        Returns:
            Formatted prompt for initial question
        """
        if self.config.mode == "teacher_student":
            return f"""You are a Socratic teacher exploring a topic with a student.

Topic: {topic}

Ask an initial question that will help explore this topic deeply. Your question should
guide the student to think critically and discover insights themselves."""
        # peer
        return f"""You are engaging in a Socratic dialogue with a peer to explore a topic.

Topic: {topic}

Ask an initial question that will help both of you explore this topic deeply through
dialogue. Your question should encourage thoughtful discussion."""

    def _build_followup_question_prompt(self, topic: str, dialogue_history: list[tuple[str, str]]) -> str:
        """Build a follow-up question prompt.

        Args:
            topic: The original topic
            dialogue_history: List of (question, answer) tuples

        Returns:
            Formatted prompt for follow-up question
        """
        history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in dialogue_history])

        if self.config.mode == "teacher_student":
            return f"""You are a Socratic teacher exploring a topic with a student.

Original topic: {topic}

Dialogue so far:
{history_text}

Ask a follow-up question that builds on the previous answers and deepens the exploration.
Guide the student to discover more insights."""
        # peer
        return f"""You are engaging in a Socratic dialogue with a peer.

Original topic: {topic}

Dialogue so far:
{history_text}

Ask a follow-up question that builds on the previous answers and continues the exploration.
Encourage deeper thinking and discussion."""

    def _build_answer_prompt(self, topic: str, dialogue_history: list[tuple[str, str]], question: str) -> str:
        """Build an answer prompt.

        Args:
            topic: The original topic
            dialogue_history: Previous (question, answer) tuples
            question: The current question to answer

        Returns:
            Formatted prompt for answering
        """
        history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in dialogue_history])

        if self.config.mode == "teacher_student":
            return f"""You are a student engaged in a Socratic dialogue with a teacher.

Original topic: {topic}

Previous dialogue:
{history_text}

Current question:
{question}

Provide a thoughtful answer that demonstrates your understanding and thinking process."""
        # peer
        return f"""You are engaging in a Socratic dialogue with a peer.

Original topic: {topic}

Previous dialogue:
{history_text}

Current question:
{question}

Provide a thoughtful answer that contributes to the exploration of the topic."""

    def _build_synthesis_prompt(self, topic: str, dialogue_history: list[tuple[str, str]]) -> str:
        """Build a synthesis prompt.

        Args:
            topic: The original topic
            dialogue_history: All (question, answer) tuples

        Returns:
            Formatted prompt for synthesis
        """
        dialogue_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in dialogue_history])

        return f"""Synthesize the following Socratic dialogue into a comprehensive exploration
of the topic.

Topic: {topic}

Dialogue:
{dialogue_text}

Provide a synthesis that captures the key insights, questions, and conclusions from
this dialogue."""

    def validate_config(self) -> bool:
        """Validate workflow configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return self.config.max_exchanges > 0
