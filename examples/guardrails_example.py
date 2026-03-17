"""Examples of guardrails with GlueLLM.

Demonstrates GuardrailsConfig, BlocklistConfig, PIIConfig, PromptGuidedConfig,
and error handling for GuardrailBlockedError and GuardrailRejectedError.
"""

import asyncio
import os

from gluellm import (
    GuardrailBlockedError,
    GuardrailRejectedError,
    GuardrailsConfig,
    complete,
)
from gluellm.guardrails.config import BlocklistConfig, PIIConfig, PromptGuidedConfig
from gluellm.guardrails.runner import run_input_guardrails, run_output_guardrails


async def example_blocklist_block_input():
    """Blocklist with on_input='block' raises GuardrailBlockedError."""
    print("=" * 60)
    print("Example 1: Blocklist Blocks Input")
    print("=" * 60)

    config = GuardrailsConfig(
        blocklist=BlocklistConfig(patterns=[r"secret", r"confidential"], on_input="block"),
    )

    try:
        run_input_guardrails("What is the capital of France?", config)
        print("Clean input passed.")
    except GuardrailBlockedError as e:
        print(f"Would block: {e}")

    try:
        run_input_guardrails("Tell me the secret password", config)
    except GuardrailBlockedError as e:
        print(f"Blocked: {e.reason}")
    print()


async def example_blocklist_redact_input():
    """Blocklist with on_input='redact' replaces matches."""
    print("=" * 60)
    print("Example 2: Blocklist Redacts Input")
    print("=" * 60)

    config = GuardrailsConfig(
        blocklist=BlocklistConfig(patterns=[r"foo"], on_input="redact"),
    )
    result = run_input_guardrails("The foo bar and foo baz", config)
    print("Original: 'The foo bar and foo baz'")
    print(f"Redacted: '{result}'")
    print()


async def example_pii_redaction():
    """PII config redacts emails and phone numbers."""
    print("=" * 60)
    print("Example 3: PII Redaction")
    print("=" * 60)

    config = GuardrailsConfig(pii=PIIConfig(redact_emails=True, redact_phones=True))
    text = "Contact John at john@example.com or call 555-123-4567."
    result = run_input_guardrails(text, config)
    print(f"Original: {text}")
    print(f"Redacted: {result}")
    print()


async def example_prompt_guided():
    """PromptGuidedConfig uses an evaluator to check content against criteria."""
    print("=" * 60)
    print("Example 4: Prompt-Guided Guardrail")
    print("=" * 60)

    def evaluator(content: str, prompt: str) -> tuple[bool, str | None]:
        # Simple rule: response must be under 80 chars and non-empty
        if not content.strip():
            return False, "Response is empty"
        if len(content) > 80:
            return False, "Response exceeds 80 characters"
        return True, None

    config = GuardrailsConfig(
        prompt_guided=PromptGuidedConfig(
            prompt="Responses must be concise (under 80 chars).",
            evaluator=evaluator,
        )
    )

    short = "Yes, that works."
    long_text = "This is a very long response that exceeds the maximum allowed length and should be rejected by the prompt-guided guardrail."
    print("Checking short response:", run_output_guardrails(short, config) == short)

    try:
        run_output_guardrails(long_text, config)
    except GuardrailRejectedError as e:
        print(f"Rejected long response: {e.reason}")
    print()


async def example_complete_with_guardrails():
    """Use guardrails with complete() - input redaction, output redaction."""
    print("=" * 60)
    print("Example 5: complete() with Guardrails")
    print("=" * 60)

    config = GuardrailsConfig(
        blocklist=BlocklistConfig(
            patterns=[r"internal"],
            on_input="redact",
            on_output="redact",
        ),
        pii=PIIConfig(redact_emails=True),
    )

    result = await complete(
        "What is 2+2? Reply in one word.",
        system_prompt="Be brief.",
        guardrails=config,
    )
    print(f"Response: {result.final_response}")
    print()


async def main():
    await example_blocklist_block_input()
    await example_blocklist_redact_input()
    await example_pii_redaction()
    await example_prompt_guided()
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY") != "sk-test":
        await example_complete_with_guardrails()
    else:
        print("(Skipping example 5: OPENAI_API_KEY not set)")


if __name__ == "__main__":
    asyncio.run(main())
