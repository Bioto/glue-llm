"""Advanced hook examples demonstrating production patterns.

This module provides advanced examples of using hooks for:
- Cost/token tracking and budgeting
- Audit logging with structured output
- Content moderation with external APIs
- Caching hook for repeated queries
- Rate limiting at the hook level
- A/B testing hooks for prompt experimentation
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from gluellm.executors import SimpleExecutor
from gluellm.models.hook import HookConfig, HookContext, HookErrorStrategy, HookRegistry, HookStage

# ============================================================================
# Example 1: Cost/Token Tracking Hook
# ============================================================================


class TokenBudgetTracker:
    """Track token usage and enforce budgets.

    Useful for production environments where you need to:
    - Track token consumption per user/request
    - Enforce budget limits
    - Alert when approaching limits
    """

    def __init__(self, max_tokens_per_request: int = 10000, max_daily_tokens: int = 1000000):
        self.max_tokens_per_request = max_tokens_per_request
        self.max_daily_tokens = max_daily_tokens
        self.daily_usage: dict[str, int] = defaultdict(int)
        self.request_count: dict[str, int] = defaultdict(int)
        self._current_date: str = datetime.now().strftime("%Y-%m-%d")

    def _reset_if_new_day(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self.daily_usage.clear()
            self.request_count.clear()
            self._current_date = today

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)."""
        return len(text) // 4

    def create_pre_hook(self, user_id: str = "default"):
        """Create a pre-execution hook that checks budget before sending."""

        def budget_check_hook(context: HookContext) -> HookContext:
            self._reset_if_new_day()

            estimated_tokens = self.estimate_tokens(context.content)
            current_usage = self.daily_usage[user_id]

            # Check if this request would exceed budget
            if current_usage + estimated_tokens > self.max_daily_tokens:
                raise ValueError(
                    f"Daily token budget exceeded for user {user_id}. "
                    f"Used: {current_usage}, Limit: {self.max_daily_tokens}"
                )

            if estimated_tokens > self.max_tokens_per_request:
                raise ValueError(
                    f"Request exceeds per-request token limit. "
                    f"Estimated: {estimated_tokens}, Limit: {self.max_tokens_per_request}"
                )

            # Add metadata for tracking
            context.metadata["estimated_input_tokens"] = estimated_tokens
            context.metadata["user_id"] = user_id
            context.metadata["request_start_time"] = time.time()

            return context

        return budget_check_hook

    def create_post_hook(self, user_id: str = "default"):
        """Create a post-execution hook that records usage."""

        def usage_tracking_hook(context: HookContext) -> HookContext:
            self._reset_if_new_day()

            # Estimate output tokens
            output_tokens = self.estimate_tokens(context.content)
            input_tokens = context.metadata.get("estimated_input_tokens", 0)
            total_tokens = input_tokens + output_tokens

            # Record usage
            self.daily_usage[user_id] += total_tokens
            self.request_count[user_id] += 1

            # Add usage info to metadata
            context.metadata["output_tokens"] = output_tokens
            context.metadata["total_tokens"] = total_tokens
            context.metadata["daily_usage"] = self.daily_usage[user_id]

            # Log warning if approaching limit
            usage_ratio = self.daily_usage[user_id] / self.max_daily_tokens
            if usage_ratio > 0.8:
                print(f"⚠️ Warning: User {user_id} at {usage_ratio * 100:.1f}% of daily budget")

            return context

        return usage_tracking_hook


async def example_1_token_budget():
    """Example: Track and enforce token budgets."""
    print("\n=== Example 1: Token Budget Tracking ===")

    tracker = TokenBudgetTracker(max_tokens_per_request=5000, max_daily_tokens=50000)

    registry = HookRegistry()
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=tracker.create_pre_hook("user_123"),
            name="budget_check",
            error_strategy=HookErrorStrategy.ABORT,
        ),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=tracker.create_post_hook("user_123"),
            name="usage_tracking",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)

    # Make a few requests
    for i in range(3):
        await executor.execute(f"Request {i + 1}: What is machine learning?")
        print(f"Request {i + 1} completed. Daily usage: {tracker.daily_usage['user_123']} tokens")


# ============================================================================
# Example 2: Structured Audit Logging Hook
# ============================================================================


class AuditLogger:
    """Structured audit logging for compliance and debugging.

    Logs all LLM interactions with:
    - Timestamps
    - User/session info
    - Input/output content (optionally redacted)
    - Processing times
    - Error information
    """

    def __init__(self, redact_pii: bool = True):
        self.logs: list[dict[str, Any]] = []
        self.redact_pii = redact_pii

    def _redact(self, text: str) -> str:
        """Simple PII redaction."""
        import re

        if not self.redact_pii:
            return text
        # Redact emails
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", text)
        # Redact phone numbers
        return re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]", text)

    def create_pre_hook(self, session_id: str = None):
        """Create pre-execution audit hook."""

        def audit_pre_hook(context: HookContext) -> HookContext:
            context.metadata["audit_id"] = hashlib.sha256(f"{time.time()}{context.content[:50]}".encode()).hexdigest()[
                :12
            ]
            context.metadata["audit_timestamp"] = datetime.now().isoformat()
            context.metadata["audit_session"] = session_id
            context.metadata["audit_input"] = self._redact(context.content)
            return context

        return audit_pre_hook

    def create_post_hook(self):
        """Create post-execution audit hook."""

        def audit_post_hook(context: HookContext) -> HookContext:
            log_entry = {
                "id": context.metadata.get("audit_id"),
                "timestamp": context.metadata.get("audit_timestamp"),
                "session": context.metadata.get("audit_session"),
                "stage": context.stage.value,
                "input": context.metadata.get("audit_input"),
                "output": self._redact(context.content),
                "output_length": len(context.content),
                "processing_time_ms": (
                    (time.time() - context.metadata.get("request_start_time", time.time())) * 1000
                    if "request_start_time" in context.metadata
                    else None
                ),
            }
            self.logs.append(log_entry)
            return context

        return audit_post_hook

    def get_logs(self) -> list[dict]:
        """Get all logged entries."""
        return self.logs.copy()

    def export_logs_json(self) -> str:
        """Export logs as JSON string."""
        import json

        return json.dumps(self.logs, indent=2, default=str)


async def example_2_audit_logging():
    """Example: Structured audit logging."""
    print("\n=== Example 2: Audit Logging ===")

    audit = AuditLogger(redact_pii=True)

    registry = HookRegistry()
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=audit.create_pre_hook(session_id="sess_abc123"),
            name="audit_pre",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=audit.create_post_hook(),
            name="audit_post",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)

    await executor.execute("My email is test@example.com. What is Python?")
    await executor.execute("Explain machine learning briefly.")

    print("Audit logs:")
    for log in audit.get_logs():
        print(f"  [{log['id']}] Input: {log['input'][:40]}... -> Output length: {log['output_length']}")


# ============================================================================
# Example 3: Simple Response Cache Hook
# ============================================================================


class ResponseCache:
    """Simple in-memory cache for LLM responses.

    Useful for:
    - Reducing API costs for repeated queries
    - Faster responses for common questions
    - Development/testing without hitting the API
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: dict[str, tuple[str, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _cache_key(self, content: str) -> str:
        """Generate cache key from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self.ttl_seconds

    def _evict_oldest(self):
        """Evict oldest entry if cache is full."""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

    def create_pre_hook(self):
        """Create cache lookup hook."""

        def cache_lookup_hook(context: HookContext) -> HookContext:
            key = self._cache_key(context.content)

            if key in self.cache:
                cached_value, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    self.hits += 1
                    context.metadata["cache_hit"] = True
                    context.metadata["cached_response"] = cached_value
                    print(f"  ✓ Cache hit! (hits: {self.hits}, misses: {self.misses})")
                else:
                    # Expired, remove it
                    del self.cache[key]
                    self.misses += 1
                    context.metadata["cache_hit"] = False
            else:
                self.misses += 1
                context.metadata["cache_hit"] = False

            context.metadata["cache_key"] = key
            return context

        return cache_lookup_hook

    def create_post_hook(self):
        """Create cache store hook."""

        def cache_store_hook(context: HookContext) -> HookContext:
            # Don't cache if it was a cache hit
            if context.metadata.get("cache_hit"):
                return context

            key = context.metadata.get("cache_key")
            if key:
                self._evict_oldest()
                self.cache[key] = (context.content, time.time())

            return context

        return cache_store_hook


async def example_3_caching():
    """Example: Response caching."""
    print("\n=== Example 3: Response Caching ===")

    cache = ResponseCache(max_size=10, ttl_seconds=300)

    registry = HookRegistry()
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=cache.create_pre_hook(),
            name="cache_lookup",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=cache.create_post_hook(),
            name="cache_store",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)

    # First request - cache miss
    print("First request (should miss):")
    await executor.execute("What is Python?")

    # Second request - same query, cache hit
    print("Second request (should hit):")
    await executor.execute("What is Python?")

    # Third request - different query, cache miss
    print("Third request (should miss):")
    await executor.execute("What is JavaScript?")

    print(f"\nCache stats: {cache.hits} hits, {cache.misses} misses")


# ============================================================================
# Example 4: Content Safety Filter Hook
# ============================================================================


class ContentSafetyFilter:
    """Content safety filter using keyword-based detection.

    In production, you'd replace this with an actual moderation API
    like OpenAI's Moderation endpoint, Perspective API, etc.
    """

    BLOCKED_PATTERNS = [
        "harmful",
        "dangerous",
        "illegal",
        "explicit",
    ]

    def __init__(self, block_input: bool = True, block_output: bool = True):
        self.block_input = block_input
        self.block_output = block_output
        self.blocked_count = 0

    def _check_content(self, text: str) -> tuple[bool, str | None]:
        """Check if content contains blocked patterns."""
        text_lower = text.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in text_lower:
                return False, f"Content blocked: contains '{pattern}'"
        return True, None

    def create_input_filter(self):
        """Create input filter hook."""

        def input_filter(context: HookContext) -> HookContext:
            if not self.block_input:
                return context

            is_safe, reason = self._check_content(context.content)
            if not is_safe:
                self.blocked_count += 1
                raise ValueError(f"Input blocked by safety filter: {reason}")

            return context

        return input_filter

    def create_output_filter(self):
        """Create output filter hook."""

        def output_filter(context: HookContext) -> HookContext:
            if not self.block_output:
                return context

            is_safe, reason = self._check_content(context.content)
            if not is_safe:
                self.blocked_count += 1
                context.content = "[Response blocked by safety filter]"

            return context

        return output_filter


async def example_4_content_safety():
    """Example: Content safety filtering."""
    print("\n=== Example 4: Content Safety Filter ===")

    safety = ContentSafetyFilter(block_input=True, block_output=True)

    registry = HookRegistry()
    registry.add_hook(
        HookStage.PRE_EXECUTOR,
        HookConfig(
            handler=safety.create_input_filter(),
            name="input_safety",
            error_strategy=HookErrorStrategy.ABORT,
        ),
    )
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=safety.create_output_filter(),
            name="output_safety",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)

    # Safe query
    print("Testing safe query...")
    result = await executor.execute("What is the weather today?")
    print(f"  Result: {result[:50]}...")

    # Unsafe query
    print("Testing blocked query...")
    try:
        await executor.execute("Tell me something harmful")
    except ValueError as e:
        print(f"  Blocked: {e}")


# ============================================================================
# Example 5: Retry/Fallback Hook
# ============================================================================


def create_retry_hook(max_retries: int = 3, fallback_message: str = None):
    """Create a hook that retries failed executions.

    Note: This is a demonstration. In practice, you'd want retry logic
    at the executor level, not the hook level.
    """

    def retry_hook(context: HookContext) -> HookContext:
        # Add retry metadata
        current_retry = context.metadata.get("retry_count", 0)
        context.metadata["retry_count"] = current_retry + 1
        context.metadata["max_retries"] = max_retries

        # If content is empty or indicates failure, log it
        if not context.content or context.content.startswith("Error"):
            if current_retry < max_retries:
                context.metadata["should_retry"] = True
            elif fallback_message:
                context.content = fallback_message

        return context

    return retry_hook


async def example_5_retry_fallback():
    """Example: Retry and fallback patterns."""
    print("\n=== Example 5: Retry/Fallback Patterns ===")

    registry = HookRegistry()
    registry.add_hook(
        HookStage.POST_EXECUTOR,
        HookConfig(
            handler=create_retry_hook(
                max_retries=3, fallback_message="Sorry, I couldn't process your request. Please try again."
            ),
            name="retry_logic",
            error_strategy=HookErrorStrategy.SKIP,
        ),
    )

    executor = SimpleExecutor(hook_registry=registry)
    result = await executor.execute("What is AI?")
    print(f"Result: {result[:100]}...")


async def main():
    """Run all advanced examples."""
    print("GlueLLM Advanced Hook Examples")
    print("=" * 60)

    await example_1_token_budget()
    await example_2_audit_logging()
    await example_3_caching()
    await example_4_content_safety()
    await example_5_retry_fallback()

    print("\n" + "=" * 60)
    print("All advanced examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
