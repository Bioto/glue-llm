"""Cost tracking for LLM API calls.

This module provides cost tracking and budgeting functionality for LLM usage.
It integrates with the pricing data module to calculate real-time costs.

Features:
    - Track costs per request, session, or globally
    - Set cost limits and alerts
    - Get cost breakdowns by model/provider
    - Export cost reports

Example:
    >>> from gluellm.costing import CostTracker, get_global_tracker
    >>>
    >>> # Use global tracker
    >>> tracker = get_global_tracker()
    >>> tracker.set_budget(daily_limit=10.00)  # $10/day
    >>>
    >>> # Track a call
    >>> tracker.record_usage(
    ...     model="openai:gpt-4o-mini",
    ...     input_tokens=1000,
    ...     output_tokens=500,
    ... )
    >>>
    >>> # Check costs
    >>> print(f"Today's cost: ${tracker.get_daily_cost():.4f}")
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gluellm.costing.pricing_data import calculate_cost
from gluellm.observability.logging_config import get_logger
from gluellm.rate_limiting.api_key_pool import extract_provider_from_model

logger = get_logger(__name__)


@dataclass
class UsageRecord:
    """Record of a single LLM API call."""

    timestamp: datetime
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float | None
    request_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostSummary:
    """Summary of costs over a period."""

    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    request_count: int
    cost_by_model: dict[str, float]
    cost_by_provider: dict[str, float]
    start_time: datetime | None
    end_time: datetime | None


class CostTracker:
    """Track and manage LLM API costs.

    Thread-safe cost tracking with budget limits and alerts.

    Attributes:
        daily_limit: Maximum daily spend in USD (None = no limit)
        session_limit: Maximum session spend in USD (None = no limit)
        alert_threshold: Percentage of budget to trigger warnings (0.0-1.0)
    """

    def __init__(
        self,
        daily_limit: float | None = None,
        session_limit: float | None = None,
        alert_threshold: float = 0.8,
    ):
        """Initialize the cost tracker.

        Args:
            daily_limit: Maximum daily spend in USD
            session_limit: Maximum session spend in USD
            alert_threshold: Percentage of budget to trigger warnings
        """
        self.daily_limit = daily_limit
        self.session_limit = session_limit
        self.alert_threshold = alert_threshold

        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._session_start = datetime.now()
        self._current_date = datetime.now().date()

        # Aggregated stats for quick lookups
        self._daily_cost: float = 0.0
        self._session_cost: float = 0.0
        self._daily_tokens: dict[str, int] = defaultdict(int)

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if a new day has started."""
        today = datetime.now().date()
        if today != self._current_date:
            logger.info(f"New day detected, resetting daily cost counter (previous: ${self._daily_cost:.4f})")
            self._daily_cost = 0.0
            self._daily_tokens.clear()
            self._current_date = today

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        request_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record a usage event.

        Args:
            model: Model identifier (e.g., "openai:gpt-4o-mini")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_input_tokens: Number of cached input tokens
            request_id: Optional request/correlation ID
            user_id: Optional user identifier
            metadata: Optional additional metadata

        Returns:
            The created UsageRecord

        Raises:
            ValueError: If budget limit is exceeded (when limits are set)
        """
        with self._lock:
            self._reset_daily_if_needed()

            # Extract provider and model name
            provider = extract_provider_from_model(model)
            model_name = model.split(":", 1)[1] if ":" in model else model

            # Calculate cost
            cost = calculate_cost(
                provider=provider,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )

            # Create record
            record = UsageRecord(
                timestamp=datetime.now(),
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cost_usd=cost,
                request_id=request_id,
                user_id=user_id,
                metadata=metadata or {},
            )

            self._records.append(record)

            # Update aggregates
            if cost is not None:
                self._daily_cost += cost
                self._session_cost += cost

                # Check limits
                self._check_limits(cost)

            self._daily_tokens["input"] += input_tokens
            self._daily_tokens["output"] += output_tokens
            self._daily_tokens["cached"] += cached_input_tokens

            logger.debug(
                f"Recorded usage: model={model}, tokens={input_tokens}+{output_tokens}, "
                f"cost=${cost:.6f if cost else 'N/A'}, daily_total=${self._daily_cost:.4f}"
            )

            return record

    def _check_limits(self, new_cost: float) -> None:
        """Check if limits are approached or exceeded."""
        # Check daily limit
        if self.daily_limit is not None:
            ratio = self._daily_cost / self.daily_limit
            if ratio >= 1.0:
                logger.warning(f"Daily budget EXCEEDED: ${self._daily_cost:.4f} / ${self.daily_limit:.2f}")
            elif ratio >= self.alert_threshold:
                logger.warning(
                    f"Daily budget alert: ${self._daily_cost:.4f} / ${self.daily_limit:.2f} ({ratio * 100:.1f}%)"
                )

        # Check session limit
        if self.session_limit is not None:
            ratio = self._session_cost / self.session_limit
            if ratio >= 1.0:
                logger.warning(f"Session budget EXCEEDED: ${self._session_cost:.4f} / ${self.session_limit:.2f}")
            elif ratio >= self.alert_threshold:
                logger.warning(
                    f"Session budget alert: ${self._session_cost:.4f} / ${self.session_limit:.2f} ({ratio * 100:.1f}%)"
                )

    def set_budget(
        self,
        daily_limit: float | None = None,
        session_limit: float | None = None,
    ) -> None:
        """Set budget limits.

        Args:
            daily_limit: Maximum daily spend in USD
            session_limit: Maximum session spend in USD
        """
        with self._lock:
            if daily_limit is not None:
                self.daily_limit = daily_limit
                logger.info(f"Daily budget set to ${daily_limit:.2f}")
            if session_limit is not None:
                self.session_limit = session_limit
                logger.info(f"Session budget set to ${session_limit:.2f}")

    def get_daily_cost(self) -> float:
        """Get total cost for today.

        Returns:
            Total cost in USD for the current day
        """
        with self._lock:
            self._reset_daily_if_needed()
            return self._daily_cost

    def get_session_cost(self) -> float:
        """Get total cost for the current session.

        Returns:
            Total cost in USD since tracker initialization
        """
        with self._lock:
            return self._session_cost

    def get_remaining_daily_budget(self) -> float | None:
        """Get remaining daily budget.

        Returns:
            Remaining budget in USD, or None if no limit set
        """
        if self.daily_limit is None:
            return None
        with self._lock:
            self._reset_daily_if_needed()
            return max(0.0, self.daily_limit - self._daily_cost)

    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CostSummary:
        """Get a summary of costs over a time period.

        Args:
            start_time: Start of period (None = session start)
            end_time: End of period (None = now)

        Returns:
            CostSummary with aggregated data
        """
        with self._lock:
            start = start_time or self._session_start
            end = end_time or datetime.now()

            # Filter records
            filtered = [r for r in self._records if start <= r.timestamp <= end]

            # Aggregate
            total_cost = 0.0
            total_input = 0
            total_output = 0
            total_cached = 0
            cost_by_model: dict[str, float] = defaultdict(float)
            cost_by_provider: dict[str, float] = defaultdict(float)

            for record in filtered:
                if record.cost_usd:
                    total_cost += record.cost_usd
                    cost_by_model[record.model] += record.cost_usd
                    cost_by_provider[record.provider] += record.cost_usd
                total_input += record.input_tokens
                total_output += record.output_tokens
                total_cached += record.cached_input_tokens

            return CostSummary(
                total_cost=total_cost,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_cached_tokens=total_cached,
                request_count=len(filtered),
                cost_by_model=dict(cost_by_model),
                cost_by_provider=dict(cost_by_provider),
                start_time=start,
                end_time=end,
            )

    def get_records(
        self,
        limit: int | None = None,
        model: str | None = None,
        user_id: str | None = None,
    ) -> list[UsageRecord]:
        """Get usage records with optional filtering.

        Args:
            limit: Maximum number of records to return
            model: Filter by model
            user_id: Filter by user ID

        Returns:
            List of UsageRecord objects
        """
        with self._lock:
            records = self._records.copy()

        # Apply filters
        if model:
            records = [r for r in records if r.model == model]
        if user_id:
            records = [r for r in records if r.user_id == user_id]

        # Sort by timestamp (most recent first)
        records.sort(key=lambda r: r.timestamp, reverse=True)

        if limit:
            records = records[:limit]

        return records

    def reset_session(self) -> CostSummary:
        """Reset the session and return final summary.

        Returns:
            Summary of the session being reset
        """
        with self._lock:
            summary = self.get_summary()
            self._records.clear()
            self._session_cost = 0.0
            self._session_start = datetime.now()
            logger.info(f"Session reset. Previous session cost: ${summary.total_cost:.4f}")
            return summary

    def export_records(self, format: str = "dict") -> list[dict] | str:
        """Export records in various formats.

        Args:
            format: Export format ("dict", "csv", "json")

        Returns:
            Records in the requested format
        """
        with self._lock:
            records = self._records.copy()

        if format == "dict":
            return [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "model": r.model,
                    "provider": r.provider,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_input_tokens": r.cached_input_tokens,
                    "cost_usd": r.cost_usd,
                    "request_id": r.request_id,
                    "user_id": r.user_id,
                }
                for r in records
            ]
        if format == "json":
            import json

            return json.dumps(self.export_records("dict"), indent=2)
        if format == "csv":
            lines = ["timestamp,model,provider,input_tokens,output_tokens,cached_tokens,cost_usd,request_id,user_id"]
            for r in records:
                lines.append(
                    f"{r.timestamp.isoformat()},{r.model},{r.provider},"
                    f"{r.input_tokens},{r.output_tokens},{r.cached_input_tokens},"
                    f"{r.cost_usd or ''},{r.request_id or ''},{r.user_id or ''}"
                )
            return "\n".join(lines)
        raise ValueError(f"Unknown format: {format}")


# Global tracker instance
_global_tracker: CostTracker | None = None
_global_tracker_lock = threading.Lock()


def get_global_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Creates a new instance if one doesn't exist.

    Returns:
        The global CostTracker instance
    """
    global _global_tracker
    with _global_tracker_lock:
        if _global_tracker is None:
            _global_tracker = CostTracker()
            logger.debug("Created global cost tracker")
        return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global cost tracker instance."""
    global _global_tracker
    with _global_tracker_lock:
        _global_tracker = None


def configure_global_tracker(
    daily_limit: float | None = None,
    session_limit: float | None = None,
    alert_threshold: float = 0.8,
) -> CostTracker:
    """Configure the global cost tracker.

    Args:
        daily_limit: Maximum daily spend in USD
        session_limit: Maximum session spend in USD
        alert_threshold: Percentage of budget to trigger warnings

    Returns:
        The configured global CostTracker instance
    """
    global _global_tracker
    with _global_tracker_lock:
        _global_tracker = CostTracker(
            daily_limit=daily_limit,
            session_limit=session_limit,
            alert_threshold=alert_threshold,
        )
        logger.info(
            f"Global cost tracker configured: daily_limit=${daily_limit or 'None'}, "
            f"session_limit=${session_limit or 'None'}, alert_threshold={alert_threshold}"
        )
        return _global_tracker


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Estimate the cost of an LLM call without recording it.

    Args:
        model: Model identifier (e.g., "openai:gpt-4o-mini")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens

    Returns:
        Estimated cost in USD, or None if pricing not available
    """
    provider = extract_provider_from_model(model)
    model_name = model.split(":", 1)[1] if ":" in model else model

    return calculate_cost(
        provider=provider,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )
