"""Exception classes for guardrails system."""


class GuardrailBlockedError(Exception):
    """Raised when a guardrail blocks content and no retry is possible.

    This exception is raised for:
    - Input guardrail failures (user input is blocked)
    - Output guardrail failures after max retries are exhausted

    Attributes:
        reason: Human-readable reason why the content was blocked
        guardrail_name: Optional name/type of the guardrail that failed
    """

    def __init__(self, reason: str, guardrail_name: str | None = None):
        """Initialize GuardrailBlockedError.

        Args:
            reason: Human-readable reason why content was blocked
            guardrail_name: Optional name/type of the guardrail that failed
        """
        self.reason = reason
        self.guardrail_name = guardrail_name
        message = f"Content blocked by guardrail: {reason}"
        if guardrail_name:
            message += f" (guardrail: {guardrail_name})"
        super().__init__(message)


class GuardrailRejectedError(Exception):
    """Raised when an output guardrail rejects content, triggering a retry.

    This exception is caught by the completion path to request a new
    response from the LLM with feedback about why the previous response
    was rejected.

    Attributes:
        reason: Model-friendly reason why the response was rejected
        guardrail_name: Optional name/type of the guardrail that failed
    """

    def __init__(self, reason: str, guardrail_name: str | None = None):
        """Initialize GuardrailRejectedError.

        Args:
            reason: Model-friendly reason why response was rejected
            guardrail_name: Optional name/type of the guardrail that failed
        """
        self.reason = reason
        self.guardrail_name = guardrail_name
        message = f"Response rejected by guardrail: {reason}"
        if guardrail_name:
            message += f" (guardrail: {guardrail_name})"
        super().__init__(message)
