"""Configuration model for guardrails."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field


class BlocklistConfig(BaseModel):
    """Configuration for blocklist guardrail.

    Attributes:
        patterns: List of patterns to block (substrings or regex patterns)
        case_sensitive: Whether pattern matching is case-sensitive
        on_input: Action to take on input matches ("block" raises, "redact" replaces)
        on_output: Action to take on output matches ("block" raises, "redact" replaces)
    """

    patterns: list[str] = Field(description="List of patterns to block")
    case_sensitive: bool = Field(default=False, description="Whether matching is case-sensitive")
    on_input: Literal["block", "redact"] = Field(
        default="block", description="Action for input matches: block (raise) or redact (replace)"
    )
    on_output: Literal["block", "redact"] = Field(
        default="redact", description="Action for output matches: block (raise) or redact (replace)"
    )


class PIIConfig(BaseModel):
    """Configuration for PII redaction guardrail.

    Attributes:
        redact_emails: Whether to redact email addresses
        redact_phones: Whether to redact phone numbers
        redact_ssn: Whether to redact SSN-like patterns
        redact_credit_cards: Whether to redact credit card numbers
        custom_patterns: Optional list of custom regex patterns to redact
        replacement: String to use as replacement for redacted content
    """

    redact_emails: bool = Field(default=True, description="Redact email addresses")
    redact_phones: bool = Field(default=True, description="Redact phone numbers")
    redact_ssn: bool = Field(default=True, description="Redact SSN-like patterns")
    redact_credit_cards: bool = Field(default=True, description="Redact credit card numbers")
    custom_patterns: list[str] | None = Field(default=None, description="Custom regex patterns to redact")
    replacement: str = Field(default="[REDACTED]", description="Replacement string for redacted content")


class MaxLengthConfig(BaseModel):
    """Configuration for max length guardrail.

    Attributes:
        max_input_length: Maximum input length in characters (None = no limit)
        max_output_length: Maximum output length in characters (None = no limit)
        strategy: What to do when limit exceeded: "truncate" or "block"
    """

    max_input_length: int | None = Field(default=None, description="Max input length in chars (None = no limit)")
    max_output_length: int | None = Field(default=None, description="Max output length in chars (None = no limit)")
    strategy: Literal["truncate", "block"] = Field(
        default="block", description="Action when limit exceeded: truncate or block"
    )


class PromptGuidedConfig(BaseModel):
    """Configuration for prompt-guided guardrail.

    Uses an evaluator callable (e.g. backed by an LLM) to check content against
    a natural-language prompt/criteria. The evaluator receives (content, prompt)
    and returns (passed, failure_reason). Useful for policy, tone, or safety checks.

    Attributes:
        prompt: Instruction or criteria the content must satisfy (e.g. "Responses must be professional and on-topic.")
        evaluator: Callable (content: str, prompt: str) -> (passed: bool, reason: str | None).
            If passed is False, reason is used as the rejection/block message.
    """

    prompt: str = Field(description="Instruction or criteria the content must satisfy")
    evaluator: Callable[[str, str], tuple[bool, str | None]] = Field(
        description="(content, prompt) -> (passed, failure_reason_if_not_passed)"
    )

    model_config = {"arbitrary_types_allowed": True}


class GuardrailsConfig(BaseModel):
    """Configuration for guardrails system.

    Attributes:
        enabled: Whether guardrails are enabled
        blocklist: Optional blocklist configuration
        pii: Optional PII redaction configuration
        max_length: Optional max length configuration
        prompt_guided: Optional prompt-guided guardrail (evaluator + prompt)
        custom_input: Optional list of custom input guardrail callables
        custom_output: Optional list of custom output guardrail callables
        max_output_guardrail_retries: Maximum retries for output guardrail failures (default 3)
    """

    enabled: bool = Field(default=True, description="Whether guardrails are enabled")
    blocklist: BlocklistConfig | None = Field(default=None, description="Blocklist configuration")
    pii: PIIConfig | None = Field(default=None, description="PII redaction configuration")
    max_length: MaxLengthConfig | None = Field(default=None, description="Max length configuration")
    prompt_guided: PromptGuidedConfig | None = Field(
        default=None, description="Prompt-guided guardrail (evaluator checks content against prompt)"
    )
    custom_input: list[Callable[[str], str]] | None = Field(
        default=None, description="Custom input guardrail callables (content: str) -> str"
    )
    custom_output: list[Callable[[str], str]] | None = Field(
        default=None, description="Custom output guardrail callables (content: str) -> str"
    )
    max_output_guardrail_retries: int = Field(
        default=3, ge=0, description="Maximum retries for output guardrail failures"
    )

    model_config = {"arbitrary_types_allowed": True}
