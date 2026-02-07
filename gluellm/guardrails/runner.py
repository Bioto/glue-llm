"""Guardrails runner that applies guardrails to content."""

from gluellm.guardrails.config import GuardrailsConfig
from gluellm.guardrails.exceptions import GuardrailBlockedError, GuardrailRejectedError
from gluellm.observability.logging_config import get_logger

from .builtins import apply_builtin_guardrails

logger = get_logger(__name__)


def run_input_guardrails(content: str, config: GuardrailsConfig) -> str:
    """Run input guardrails on content.

    Input guardrails run before sending content to the LLM.
    If a guardrail fails, raises GuardrailBlockedError (no retry).

    Args:
        content: User input content to validate
        config: Guardrails configuration

    Returns:
        Processed content (may be transformed by guardrails)

    Raises:
        GuardrailBlockedError: If any input guardrail blocks the content
    """
    if not config.enabled:
        return content

    logger.debug(f"Running input guardrails on content (length={len(content)})")

    current_content = content

    # Run built-in guardrails in order: blocklist → PII → max_length
    if config.blocklist:
        try:
            current_content, _ = apply_builtin_guardrails(current_content, config.blocklist, is_input=True)
        except GuardrailBlockedError:
            raise  # Re-raise as-is
        except GuardrailRejectedError as e:
            # Should not happen for input, but convert to GuardrailBlockedError
            raise GuardrailBlockedError(e.reason, guardrail_name=e.guardrail_name) from e

    if config.pii:
        current_content, _ = apply_builtin_guardrails(current_content, config.pii, is_input=True)

    if config.max_length:
        try:
            current_content, _ = apply_builtin_guardrails(current_content, config.max_length, is_input=True)
        except GuardrailBlockedError:
            raise  # Re-raise as-is
        except GuardrailRejectedError as e:
            # Should not happen for input, but convert to GuardrailBlockedError
            raise GuardrailBlockedError(e.reason, guardrail_name=e.guardrail_name) from e

    # Run prompt-guided guardrail (evaluator checks content against prompt)
    if config.prompt_guided:
        pg = config.prompt_guided
        try:
            passed, reason = pg.evaluator(current_content, pg.prompt)
            if not passed and reason:
                raise GuardrailBlockedError(reason, guardrail_name="prompt_guided")
            if not passed:
                raise GuardrailBlockedError(
                    "Content did not satisfy prompt-guided criteria",
                    guardrail_name="prompt_guided",
                )
        except (GuardrailBlockedError, GuardrailRejectedError):
            raise
        except Exception as e:
            raise GuardrailBlockedError(f"Prompt-guided guardrail failed: {e}", guardrail_name="prompt_guided") from e

    # Run custom input guardrails
    if config.custom_input:
        for i, guardrail_func in enumerate(config.custom_input):
            try:
                result = guardrail_func(current_content)
                if not isinstance(result, str):
                    logger.warning(
                        f"Custom input guardrail {i} returned non-string type {type(result)}, using original content"
                    )
                    continue
                current_content = result
            except Exception as e:
                # Custom guardrails can raise to block, or return transformed content
                # If they raise, wrap in GuardrailBlockedError
                if isinstance(e, (GuardrailBlockedError, GuardrailRejectedError)):
                    if isinstance(e, GuardrailRejectedError):
                        raise GuardrailBlockedError(e.reason, guardrail_name=e.guardrail_name) from e
                    raise
                # Other exceptions are treated as blocking
                raise GuardrailBlockedError(
                    f"Custom input guardrail {i} failed: {e}", guardrail_name=f"custom_input_{i}"
                ) from e

    logger.debug(f"Input guardrails completed: final_length={len(current_content)}")
    return current_content


def run_output_guardrails(content: str, config: GuardrailsConfig) -> str:
    """Run output guardrails on content.

    Output guardrails run after receiving content from the LLM.
    If a guardrail fails, raises GuardrailRejectedError to trigger retry with feedback.

    Args:
        content: LLM output content to validate
        config: Guardrails configuration

    Returns:
        Processed content (may be transformed by guardrails)

    Raises:
        GuardrailRejectedError: If any output guardrail rejects the content (triggers retry)
    """
    if not config.enabled:
        return content

    logger.debug(f"Running output guardrails on content (length={len(content)})")

    current_content = content

    # Run built-in guardrails in order: blocklist → PII → max_length
    if config.blocklist:
        current_content, _ = apply_builtin_guardrails(current_content, config.blocklist, is_input=False)

    if config.pii:
        current_content, _ = apply_builtin_guardrails(current_content, config.pii, is_input=False)

    if config.max_length:
        current_content, _ = apply_builtin_guardrails(current_content, config.max_length, is_input=False)

    # Run prompt-guided guardrail (evaluator checks content against prompt)
    if config.prompt_guided:
        pg = config.prompt_guided
        try:
            passed, reason = pg.evaluator(current_content, pg.prompt)
            if not passed and reason:
                raise GuardrailRejectedError(reason, guardrail_name="prompt_guided")
            if not passed:
                raise GuardrailRejectedError(
                    "Response did not satisfy prompt-guided criteria",
                    guardrail_name="prompt_guided",
                )
        except (GuardrailBlockedError, GuardrailRejectedError):
            raise
        except Exception as e:
            raise GuardrailRejectedError(f"Prompt-guided guardrail failed: {e}", guardrail_name="prompt_guided") from e

    # Run custom output guardrails
    if config.custom_output:
        for i, guardrail_func in enumerate(config.custom_output):
            try:
                result = guardrail_func(current_content)
                if not isinstance(result, str):
                    logger.warning(
                        f"Custom output guardrail {i} returned non-string type {type(result)}, using original content"
                    )
                    continue
                current_content = result
            except Exception as e:
                # Custom guardrails can raise GuardrailRejectedError to trigger retry,
                # or GuardrailBlockedError to block immediately
                if isinstance(e, (GuardrailBlockedError, GuardrailRejectedError)):
                    raise
                # Other exceptions are treated as rejection (will trigger retry)
                raise GuardrailRejectedError(
                    f"Custom output guardrail {i} failed: {e}", guardrail_name=f"custom_output_{i}"
                ) from e

    logger.debug(f"Output guardrails completed: final_length={len(current_content)}")
    return current_content
