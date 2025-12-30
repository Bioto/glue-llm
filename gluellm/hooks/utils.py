"""Utility hooks for common use cases.

This module provides pre-built hook functions for common tasks like
PII removal, content validation, safety checks, and formatting.
"""

import re

from gluellm.models.hook import HookContext

# PII Removal Hooks


def remove_emails(context: HookContext) -> HookContext:
    """Remove email addresses from content.

    Args:
        context: The hook context

    Returns:
        HookContext with emails removed
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    context.content = re.sub(email_pattern, "[EMAIL_REDACTED]", context.content)
    return context


def remove_phone_numbers(context: HookContext) -> HookContext:
    """Remove phone numbers from content.

    Args:
        context: The hook context

    Returns:
        HookContext with phone numbers removed
    """
    # Match various phone number formats
    phone_patterns = [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US format: 123-456-7890
        r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",  # International
    ]
    for pattern in phone_patterns:
        context.content = re.sub(pattern, "[PHONE_REDACTED]", context.content)
    return context


def remove_ssn(context: HookContext) -> HookContext:
    """Remove Social Security Numbers from content.

    Args:
        context: The hook context

    Returns:
        HookContext with SSNs removed
    """
    # Match SSN format: XXX-XX-XXXX
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    context.content = re.sub(ssn_pattern, "[SSN_REDACTED]", context.content)
    return context


def remove_credit_cards(context: HookContext) -> HookContext:
    """Remove credit card numbers from content.

    Args:
        context: The hook context

    Returns:
        HookContext with credit card numbers removed
    """
    # Match credit card patterns (13-19 digits, possibly with dashes/spaces)
    cc_pattern = r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"
    context.content = re.sub(cc_pattern, "[CARD_REDACTED]", context.content)
    return context


def remove_pii(context: HookContext) -> HookContext:
    """Remove all common PII types from content.

    Args:
        context: The hook context

    Returns:
        HookContext with all PII removed
    """
    context = remove_emails(context)
    context = remove_phone_numbers(context)
    context = remove_ssn(context)
    return remove_credit_cards(context)


# Content Validation Hooks


def validate_length_factory(min_len: int | None = None, max_len: int | None = None):
    """Create a hook that validates content length.

    Args:
        min_len: Minimum length in characters (None for no minimum)
        max_len: Maximum length in characters (None for no maximum)

    Returns:
        A hook function that validates length
    """

    def validate_length(context: HookContext) -> HookContext:
        """Validate content length with clear error messages."""
        length = len(context.content)

        if min_len is not None and length < min_len:
            raise ValueError(f"content too short: {length} < {min_len}")

        if max_len is not None and length > max_len:
            raise ValueError(f"content too long: {length} > {max_len}")

        return context

    return validate_length


def validate_no_profanity(context: HookContext) -> HookContext:
    """Validate that content contains no profanity.

    This is a basic keyword-based check. For production use, consider
    using a more sophisticated profanity detection library.

    Args:
        context: The hook context

    Returns:
        HookContext (raises ValueError if profanity detected)
    """
    # Basic profanity list - expand as needed
    profanity_words = ["damn", "hell"]  # Add more as needed
    content_lower = context.content.lower()
    for word in profanity_words:
        if word in content_lower:
            raise ValueError(f"Profanity detected: {word}")
    return context


def validate_language_factory(allowed_languages: list[str] | None = None):
    """Create a hook that validates content language.

    Note: This is a placeholder. For production, use a language detection library.

    Args:
        allowed_languages: List of allowed language codes (e.g., ['en', 'es'])

    Returns:
        A hook function that validates language
    """

    def validate_language(context: HookContext) -> HookContext:
        """Validate content language.

        Args:
            context: The hook context

        Returns:
            HookContext (raises ValueError if language not allowed)
        """
        # Placeholder - in production, use langdetect or similar
        if allowed_languages:
            # For now, just pass through
            # In production: detect language and check against allowed_languages
            pass
        return context

    return validate_language


# Content Safety Hooks


def check_toxicity_factory(threshold: float = 0.5):
    """Create a hook that checks content toxicity.

    Note: This is a basic keyword-based check. For production, use a
    proper toxicity detection API or model.

    Args:
        threshold: Toxicity threshold (0.0 to 1.0)

    Returns:
        A hook function that checks toxicity
    """
    # Basic toxic keywords
    toxic_keywords = ["hate", "violence"]  # Add more as needed

    def check_toxicity(context: HookContext) -> HookContext:
        """Check content for toxicity.

        Args:
            context: The hook context

        Returns:
            HookContext (raises ValueError if toxicity exceeds threshold)
        """
        content_lower = context.content.lower()
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in content_lower)
        toxicity_score = toxic_count / max(len(context.content.split()), 1)
        if toxicity_score > threshold:
            raise ValueError(f"Toxicity score {toxicity_score:.2f} exceeds threshold {threshold}")
        return context

    return check_toxicity


def require_citations(context: HookContext) -> HookContext:
    """Require that content includes citations.

    Args:
        context: The hook context

    Returns:
        HookContext (raises ValueError if no citations found)
    """
    # Check for common citation patterns
    citation_patterns = [
        r"\[.*?\]",  # [1], [source], etc.
        r"\(.*?\)",  # (Author, 2023), etc.
        r"http[s]?://",  # URLs
    ]
    has_citations = any(re.search(pattern, context.content) for pattern in citation_patterns)
    if not has_citations:
        raise ValueError("Content must include citations")
    return context


# Formatting Hooks


def normalize_whitespace(context: HookContext) -> HookContext:
    """Normalize whitespace in content.

    Args:
        context: The hook context

    Returns:
        HookContext with normalized whitespace
    """
    # Replace multiple spaces with single space
    context.content = re.sub(r" +", " ", context.content)
    # Replace multiple newlines with double newline
    context.content = re.sub(r"\n{3,}", "\n\n", context.content)
    # Strip leading/trailing whitespace
    context.content = context.content.strip()
    return context


def escape_html(context: HookContext) -> HookContext:
    """Escape HTML characters in content.

    Args:
        context: The hook context

    Returns:
        HookContext with HTML escaped
    """
    html_escape_map = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
    }
    for char, escaped in html_escape_map.items():
        context.content = context.content.replace(char, escaped)
    return context


def truncate_output_factory(max_chars: int | None = None, max_tokens: int | None = None):
    """Create a hook that truncates output.

    Args:
        max_chars: Maximum characters (None for no limit)
        max_tokens: Maximum tokens (approximate, None for no limit)

    Returns:
        A hook function that truncates content
    """

    def truncate_output(context: HookContext) -> HookContext:
        """Truncate content if it exceeds limits.

        Args:
            context: The hook context

        Returns:
            HookContext with truncated content
        """
        if max_chars and len(context.content) > max_chars:
            context.content = context.content[:max_chars] + "..."
        elif max_tokens:
            # Approximate token count (rough estimate: 1 token ≈ 4 characters)
            approx_tokens = len(context.content) // 4
            if approx_tokens > max_tokens:
                max_chars_approx = max_tokens * 4
                context.content = context.content[:max_chars_approx] + "..."
        return context

    return truncate_output
