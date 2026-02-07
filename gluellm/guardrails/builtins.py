"""Built-in guardrail implementations."""

import re

from gluellm.guardrails.config import BlocklistConfig, MaxLengthConfig, PIIConfig
from gluellm.guardrails.exceptions import GuardrailBlockedError, GuardrailRejectedError

# PII regex patterns
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_PATTERN = r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
CREDIT_CARD_PATTERN = r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"


def _check_blocklist(content: str, config: BlocklistConfig, is_input: bool) -> tuple[str, str | None]:
    """Check content against blocklist patterns.

    Args:
        content: Content to check
        config: Blocklist configuration
        is_input: Whether this is input (True) or output (False)

    Returns:
        Tuple of (processed_content, rejection_reason)
        If content passes, returns (content, None)
        If content fails and action is "block", raises GuardrailBlockedError or GuardrailRejectedError
        If content fails and action is "redact", returns (redacted_content, None)
    """
    action = config.on_input if is_input else config.on_output
    flags = 0 if config.case_sensitive else re.IGNORECASE

    for pattern in config.patterns:
        if re.search(pattern, content, flags):
            if action == "block":
                reason = f"Content contains blocklisted pattern: {pattern}"
                if is_input:
                    raise GuardrailBlockedError(reason, guardrail_name="blocklist")
                raise GuardrailRejectedError(reason, guardrail_name="blocklist")
            # redact
            # Replace matches with [REDACTED]
            content = re.sub(pattern, "[REDACTED]", content, flags=flags)

    return content, None


def _redact_pii(content: str, config: PIIConfig) -> tuple[str, str | None]:
    """Redact PII from content.

    Args:
        content: Content to process
        config: PII configuration

    Returns:
        Tuple of (processed_content, None) - PII redaction never rejects, only transforms
    """
    patterns_to_check = []

    if config.redact_emails:
        patterns_to_check.append(("email", EMAIL_PATTERN))
    if config.redact_phones:
        patterns_to_check.append(("phone", PHONE_PATTERN))
    if config.redact_ssn:
        patterns_to_check.append(("SSN", SSN_PATTERN))
    if config.redact_credit_cards:
        patterns_to_check.append(("credit card", CREDIT_CARD_PATTERN))

    if config.custom_patterns:
        for pattern in config.custom_patterns:
            patterns_to_check.append(("custom", pattern))

    for _pii_type, pattern in patterns_to_check:
        content = re.sub(pattern, config.replacement, content, flags=re.IGNORECASE)

    return content, None


def _check_max_length(content: str, config: MaxLengthConfig, is_input: bool) -> tuple[str, str | None]:
    """Check content length against limits.

    Args:
        content: Content to check
        config: Max length configuration
        is_input: Whether this is input (True) or output (False)

    Returns:
        Tuple of (processed_content, rejection_reason)
        If content passes, returns (content, None)
        If content fails and strategy is "block", raises GuardrailBlockedError or GuardrailRejectedError
        If content fails and strategy is "truncate", returns (truncated_content, None)
    """
    max_length = config.max_input_length if is_input else config.max_output_length

    if max_length is None:
        return content, None

    if len(content) <= max_length:
        return content, None

    if config.strategy == "truncate":
        truncated = content[:max_length]
        return truncated, None
    # block
    reason = f"Content length {len(content)} exceeds maximum {max_length} characters"
    if is_input:
        raise GuardrailBlockedError(reason, guardrail_name="max_length")
    raise GuardrailRejectedError(reason, guardrail_name="max_length")


def apply_builtin_guardrails(
    content: str, config: BlocklistConfig | PIIConfig | MaxLengthConfig, is_input: bool
) -> tuple[str, str | None]:
    """Apply a built-in guardrail to content.

    Args:
        content: Content to process
        config: Guardrail configuration (BlocklistConfig, PIIConfig, or MaxLengthConfig)
        is_input: Whether this is input (True) or output (False)

    Returns:
        Tuple of (processed_content, rejection_reason)
        If content passes, returns (content, None)
        If guardrail rejects, raises GuardrailBlockedError (input) or GuardrailRejectedError (output)
    """
    if isinstance(config, BlocklistConfig):
        return _check_blocklist(content, config, is_input)
    if isinstance(config, PIIConfig):
        return _redact_pii(content, config)
    if isinstance(config, MaxLengthConfig):
        return _check_max_length(content, config, is_input)
    # Should not happen, but return as-is
    return content, None
