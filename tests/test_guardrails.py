"""Tests for guardrails system."""

import pytest

from gluellm import GuardrailBlockedError, GuardrailRejectedError, GuardrailsConfig
from gluellm.guardrails.config import BlocklistConfig, MaxLengthConfig, PIIConfig, PromptGuidedConfig
from gluellm.guardrails.runner import run_input_guardrails, run_output_guardrails


class TestGuardrailsConfig:
    """Test GuardrailsConfig model."""

    def test_default_config(self):
        """Test default guardrails config."""
        config = GuardrailsConfig()
        assert config.enabled is True
        assert config.max_output_guardrail_retries == 3
        assert config.blocklist is None
        assert config.pii is None
        assert config.max_length is None

    def test_config_with_builtins(self):
        """Test config with built-in guardrails."""
        config = GuardrailsConfig(
            blocklist=BlocklistConfig(patterns=["badword"]),
            pii=PIIConfig(redact_emails=True),
            max_length=MaxLengthConfig(max_input_length=100),
        )
        assert config.blocklist is not None
        assert config.pii is not None
        assert config.max_length is not None

    def test_config_with_prompt_guided(self):
        """Test config with prompt-guided guardrail."""

        def evaluator(content: str, prompt: str) -> tuple[bool, str | None]:
            return ("professional" in content.lower(), None)

        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Responses must be professional.",
                evaluator=evaluator,
            )
        )
        assert config.prompt_guided is not None
        assert config.prompt_guided.prompt == "Responses must be professional."
        ok, _ = config.prompt_guided.evaluator("This is professional.", config.prompt_guided.prompt)
        assert ok is True
        ok2, _ = config.prompt_guided.evaluator("casual stuff", config.prompt_guided.prompt)
        assert ok2 is False

    def test_custom_guardrails(self):
        """Test config with custom guardrails."""

        def custom_input(content: str) -> str:
            return content.upper()

        def custom_output(content: str) -> str:
            return content.lower()

        config = GuardrailsConfig(
            custom_input=[custom_input],
            custom_output=[custom_output],
        )
        assert len(config.custom_input) == 1
        assert len(config.custom_output) == 1


class TestInputGuardrails:
    """Test input guardrails."""

    def test_blocklist_block(self):
        """Test blocklist blocks input."""
        config = GuardrailsConfig(blocklist=BlocklistConfig(patterns=["badword"], on_input="block"))
        with pytest.raises(GuardrailBlockedError) as exc_info:
            run_input_guardrails("This contains badword", config)
        assert "blocklisted" in str(exc_info.value).lower()

    def test_blocklist_redact(self):
        """Test blocklist redacts input."""
        config = GuardrailsConfig(blocklist=BlocklistConfig(patterns=["badword"], on_input="redact"))
        result = run_input_guardrails("This contains badword", config)
        assert "[REDACTED]" in result
        assert "badword" not in result

    def test_pii_redaction(self):
        """Test PII redaction on input."""
        config = GuardrailsConfig(pii=PIIConfig(redact_emails=True))
        result = run_input_guardrails("Contact me at test@example.com", config)
        assert "[REDACTED]" in result
        assert "test@example.com" not in result

    def test_max_length_block(self):
        """Test max length blocks input."""
        config = GuardrailsConfig(max_length=MaxLengthConfig(max_input_length=10, strategy="block"))
        with pytest.raises(GuardrailBlockedError) as exc_info:
            run_input_guardrails("This is too long", config)
        assert "exceeds maximum" in str(exc_info.value).lower()

    def test_max_length_truncate(self):
        """Test max length truncates input."""
        config = GuardrailsConfig(max_length=MaxLengthConfig(max_input_length=10, strategy="truncate"))
        result = run_input_guardrails("This is too long", config)
        assert len(result) == 10

    def test_custom_input_guardrail(self):
        """Test custom input guardrail."""

        def custom_guardrail(content: str) -> str:
            return content.replace("foo", "bar")

        config = GuardrailsConfig(custom_input=[custom_guardrail])
        result = run_input_guardrails("foo bar", config)
        assert result == "bar bar"

    def test_custom_input_guardrail_raises(self):
        """Test custom input guardrail that raises."""

        def custom_guardrail(content: str) -> str:
            if "blocked" in content:
                raise ValueError("Content is blocked")
            return content

        config = GuardrailsConfig(custom_input=[custom_guardrail])
        with pytest.raises(GuardrailBlockedError):
            run_input_guardrails("This is blocked", config)

    def test_disabled_guardrails(self):
        """Test disabled guardrails pass through."""
        config = GuardrailsConfig(enabled=False, blocklist=BlocklistConfig(patterns=["badword"]))
        result = run_input_guardrails("badword", config)
        assert result == "badword"

    def test_prompt_guided_input_pass(self):
        """Test prompt-guided guardrail passes when evaluator returns True."""
        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Must be professional.",
                evaluator=lambda content, prompt: (True, None),
            )
        )
        result = run_input_guardrails("Professional response here.", config)
        assert result == "Professional response here."

    def test_prompt_guided_input_block(self):
        """Test prompt-guided guardrail blocks when evaluator returns False."""
        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Must be professional.",
                evaluator=lambda content, prompt: (False, "Content is not professional enough."),
            )
        )
        with pytest.raises(GuardrailBlockedError) as exc_info:
            run_input_guardrails("yo whats up", config)
        assert "not professional" in str(exc_info.value).lower()
        assert exc_info.value.guardrail_name == "prompt_guided"

    def test_prompt_guided_input_block_no_reason(self):
        """Test prompt-guided guardrail blocks with default reason when reason is None."""
        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Must be on-topic.",
                evaluator=lambda content, prompt: (False, None),
            )
        )
        with pytest.raises(GuardrailBlockedError) as exc_info:
            run_input_guardrails("off topic", config)
        assert "prompt-guided criteria" in str(exc_info.value).lower()


class TestOutputGuardrails:
    """Test output guardrails."""

    def test_blocklist_reject(self):
        """Test blocklist rejects output."""
        config = GuardrailsConfig(blocklist=BlocklistConfig(patterns=["badword"], on_output="block"))
        with pytest.raises(GuardrailRejectedError) as exc_info:
            run_output_guardrails("This contains badword", config)
        assert "blocklisted" in str(exc_info.value).lower()

    def test_blocklist_redact(self):
        """Test blocklist redacts output."""
        config = GuardrailsConfig(blocklist=BlocklistConfig(patterns=["badword"], on_output="redact"))
        result = run_output_guardrails("This contains badword", config)
        assert "[REDACTED]" in result
        assert "badword" not in result

    def test_pii_redaction_output(self):
        """Test PII redaction on output."""
        config = GuardrailsConfig(pii=PIIConfig(redact_phones=True))
        result = run_output_guardrails("Call me at 555-123-4567", config)
        assert "[REDACTED]" in result
        assert "555-123-4567" not in result

    def test_max_length_reject(self):
        """Test max length rejects output."""
        config = GuardrailsConfig(max_length=MaxLengthConfig(max_output_length=10, strategy="block"))
        with pytest.raises(GuardrailRejectedError) as exc_info:
            run_output_guardrails("This is too long", config)
        assert "exceeds maximum" in str(exc_info.value).lower()

    def test_custom_output_guardrail(self):
        """Test custom output guardrail."""

        def custom_guardrail(content: str) -> str:
            return content.replace("foo", "bar")

        config = GuardrailsConfig(custom_output=[custom_guardrail])
        result = run_output_guardrails("foo bar", config)
        assert result == "bar bar"

    def test_custom_output_guardrail_raises_rejected(self):
        """Test custom output guardrail that raises GuardrailRejectedError."""

        def custom_guardrail(content: str) -> str:
            if "reject" in content:
                raise GuardrailRejectedError("Content rejected", guardrail_name="custom")
            return content

        config = GuardrailsConfig(custom_output=[custom_guardrail])
        with pytest.raises(GuardrailRejectedError) as exc_info:
            run_output_guardrails("This should reject", config)
        assert exc_info.value.reason == "Content rejected"

    def test_disabled_guardrails_output(self):
        """Test disabled guardrails pass through."""
        config = GuardrailsConfig(enabled=False, blocklist=BlocklistConfig(patterns=["badword"]))
        result = run_output_guardrails("badword", config)
        assert result == "badword"

    def test_prompt_guided_output_pass(self):
        """Test prompt-guided guardrail passes output when evaluator returns True."""
        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Response must be helpful.",
                evaluator=lambda content, prompt: (True, None),
            )
        )
        result = run_output_guardrails("Here is a helpful answer.", config)
        assert result == "Here is a helpful answer."

    def test_prompt_guided_output_reject(self):
        """Test prompt-guided guardrail rejects output when evaluator returns False."""
        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Response must be helpful.",
                evaluator=lambda content, prompt: (False, "Response was not helpful."),
            )
        )
        with pytest.raises(GuardrailRejectedError) as exc_info:
            run_output_guardrails("I refuse to help.", config)
        assert exc_info.value.reason == "Response was not helpful."
        assert exc_info.value.guardrail_name == "prompt_guided"

    def test_prompt_guided_output_reject_no_reason(self):
        """Test prompt-guided guardrail rejects with default reason when reason is None."""
        config = GuardrailsConfig(
            prompt_guided=PromptGuidedConfig(
                prompt="Stay on topic.",
                evaluator=lambda content, prompt: (False, None),
            )
        )
        with pytest.raises(GuardrailRejectedError) as exc_info:
            run_output_guardrails("Random tangent.", config)
        assert "prompt-guided criteria" in str(exc_info.value).lower()


class TestPromptGuidedOrder:
    """Test prompt-guided runs with other guardrails."""

    def test_prompt_guided_after_blocklist_input(self):
        """Prompt-guided runs after blocklist on input."""
        # Blocklist blocks first, so prompt_guided never runs
        config = GuardrailsConfig(
            blocklist=BlocklistConfig(patterns=["badword"], on_input="block"),
            prompt_guided=PromptGuidedConfig(
                prompt="Be nice.",
                evaluator=lambda c, p: (False, "not nice"),
            ),
        )
        with pytest.raises(GuardrailBlockedError) as exc_info:
            run_input_guardrails("badword", config)
        assert "blocklisted" in str(exc_info.value).lower()

    def test_prompt_guided_after_blocklist_output(self):
        """Prompt-guided runs after blocklist on output."""
        config = GuardrailsConfig(
            blocklist=BlocklistConfig(patterns=["badword"], on_output="redact"),
            prompt_guided=PromptGuidedConfig(
                prompt="Be professional.",
                evaluator=lambda c, p: ("professional" in c.lower(), None),
            ),
        )
        # Content gets redacted then prompt_guided sees "[REDACTED] hello professional"
        result = run_output_guardrails("badword hello professional", config)
        assert "[REDACTED]" in result
        assert "professional" in result


class TestGuardrailOrder:
    """Test guardrail execution order."""

    def test_input_order(self):
        """Test input guardrails run in correct order."""
        # Blocklist should run first, so if it blocks, PII never runs
        config = GuardrailsConfig(
            blocklist=BlocklistConfig(patterns=["badword"], on_input="block"),
            pii=PIIConfig(redact_emails=True),
        )
        with pytest.raises(GuardrailBlockedError):
            run_input_guardrails("badword test@example.com", config)

    def test_output_order(self):
        """Test output guardrails run in correct order."""
        # Blocklist should run first, then PII redaction
        config = GuardrailsConfig(
            blocklist=BlocklistConfig(patterns=["badword"], on_output="redact"),
            pii=PIIConfig(redact_emails=True),
        )
        result = run_output_guardrails("badword test@example.com", config)
        assert "[REDACTED]" in result
        assert "badword" not in result
        assert "test@example.com" not in result
