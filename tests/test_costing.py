"""Tests for the costing module — pricing data, cost calculation, and cost tracking."""

import json
from datetime import datetime

import pytest

from gluellm.costing.cost_tracker import (
    CostSummary,
    CostTracker,
    UsageRecord,
    configure_global_tracker,
    estimate_cost,
    get_global_tracker,
    reset_global_tracker,
)
from gluellm.costing.pricing_data import (
    ANTHROPIC_PRICING,
    OPENAI_PRICING,
    PRICING_BY_PROVIDER,
    XAI_PRICING,
    calculate_cost,
    calculate_embedding_cost,
    get_embedding_pricing,
    get_model_pricing,
    list_available_models,
)


class TestModelPricing:
    """Tests for pricing data lookups."""

    def test_openai_model_exact_match(self):
        pricing = get_model_pricing("openai", "gpt-4o-mini")
        assert pricing is not None
        assert pricing.input_price_per_million == 0.15
        assert pricing.output_price_per_million == 0.60

    def test_anthropic_model_exact_match(self):
        pricing = get_model_pricing("anthropic", "claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing.input_price_per_million == 3.00

    def test_xai_model_exact_match(self):
        pricing = get_model_pricing("xai", "grok-2-1212")
        assert pricing is not None
        assert pricing.input_price_per_million == 2.00

    def test_unknown_provider_returns_none(self):
        assert get_model_pricing("unknown_provider", "some-model") is None

    def test_unknown_model_returns_none(self):
        assert get_model_pricing("openai", "nonexistent-model-xyz") is None

    def test_partial_match_versioned_model(self):
        pricing = get_model_pricing("openai", "gpt-4o-2024-11-20")
        assert pricing is not None

    def test_case_insensitive_provider(self):
        pricing = get_model_pricing("OpenAI", "gpt-4o-mini")
        assert pricing is not None

    def test_cached_input_pricing(self):
        pricing = get_model_pricing("openai", "gpt-4o")
        assert pricing is not None
        assert pricing.cached_input_price_per_million == 1.25

    def test_model_without_cached_pricing(self):
        pricing = get_model_pricing("openai", "gpt-4")
        assert pricing is not None
        assert pricing.cached_input_price_per_million is None

    def test_pricing_dicts_not_empty(self):
        assert len(OPENAI_PRICING) > 0
        assert len(ANTHROPIC_PRICING) > 0
        assert len(XAI_PRICING) > 0

    def test_all_providers_in_combined_dict(self):
        assert "openai" in PRICING_BY_PROVIDER
        assert "anthropic" in PRICING_BY_PROVIDER
        assert "xai" in PRICING_BY_PROVIDER


class TestCalculateCost:
    """Tests for cost calculation."""

    def test_basic_cost_calculation(self):
        cost = calculate_cost("openai", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost is not None
        assert cost == pytest.approx(0.15 + 0.60)

    def test_zero_tokens_returns_zero(self):
        cost = calculate_cost("openai", "gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_unknown_model_returns_none(self):
        cost = calculate_cost("openai", "nonexistent-model", input_tokens=100, output_tokens=50)
        assert cost is None

    def test_cached_tokens_included(self):
        cost_without_cache = calculate_cost("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
        cost_with_cache = calculate_cost(
            "openai", "gpt-4o", input_tokens=1000, output_tokens=500, cached_input_tokens=500
        )
        assert cost_with_cache > cost_without_cache

    def test_cached_tokens_ignored_when_no_pricing(self):
        cost = calculate_cost("openai", "gpt-4", input_tokens=1000, output_tokens=500, cached_input_tokens=500)
        cost_no_cache = calculate_cost("openai", "gpt-4", input_tokens=1000, output_tokens=500)
        assert cost == cost_no_cache


class TestEmbeddingPricing:
    """Tests for embedding pricing."""

    def test_openai_embedding_pricing(self):
        price = get_embedding_pricing("openai", "text-embedding-3-small")
        assert price is not None
        assert price == 0.02

    def test_unknown_embedding_model(self):
        assert get_embedding_pricing("openai", "nonexistent-embedding") is None

    def test_unknown_provider_embedding(self):
        assert get_embedding_pricing("unknown", "text-embedding-3-small") is None

    def test_calculate_embedding_cost(self):
        cost = calculate_embedding_cost("openai", "text-embedding-3-small", input_tokens=1_000_000)
        assert cost == pytest.approx(0.02)

    def test_calculate_embedding_cost_unknown(self):
        assert calculate_embedding_cost("openai", "nonexistent", input_tokens=1000) is None


class TestListAvailableModels:
    """Tests for listing models."""

    def test_list_all_models(self):
        models = list_available_models()
        assert len(models) > 0
        assert any(m.startswith("openai:") for m in models)
        assert any(m.startswith("anthropic:") for m in models)
        assert any(m.startswith("xai:") for m in models)

    def test_list_models_by_provider(self):
        models = list_available_models("openai")
        assert len(models) > 0
        assert all(m.startswith("openai:") for m in models)

    def test_list_models_unknown_provider(self):
        models = list_available_models("nonexistent")
        assert models == []


class TestCostTracker:
    """Tests for the CostTracker class."""

    def test_record_usage(self):
        tracker = CostTracker()
        record = tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        assert isinstance(record, UsageRecord)
        assert record.model == "openai:gpt-4o-mini"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost_usd is not None

    def test_daily_cost_accumulates(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert tracker.get_daily_cost() > 0

    def test_session_cost_accumulates(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=1000, output_tokens=500)
        cost1 = tracker.get_session_cost()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=1000, output_tokens=500)
        cost2 = tracker.get_session_cost()
        assert cost2 == pytest.approx(cost1 * 2)

    def test_set_budget(self):
        tracker = CostTracker()
        tracker.set_budget(daily_limit=10.0, session_limit=50.0)
        assert tracker.daily_limit == 10.0
        assert tracker.session_limit == 50.0

    def test_remaining_daily_budget(self):
        tracker = CostTracker(daily_limit=10.0)
        remaining = tracker.get_remaining_daily_budget()
        assert remaining == 10.0

    def test_remaining_daily_budget_no_limit(self):
        tracker = CostTracker()
        assert tracker.get_remaining_daily_budget() is None

    def test_get_summary(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.record_usage("anthropic:claude-3-5-sonnet-20241022", input_tokens=200, output_tokens=100)
        summary = tracker.get_summary()
        assert isinstance(summary, CostSummary)
        assert summary.request_count == 2
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 150
        assert "openai:gpt-4o-mini" in summary.cost_by_model
        assert "openai" in summary.cost_by_provider

    def test_get_records_filter_by_model(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.record_usage("openai:gpt-4o", input_tokens=100, output_tokens=50)
        records = tracker.get_records(model="openai:gpt-4o-mini")
        assert len(records) == 1
        assert records[0].model == "openai:gpt-4o-mini"

    def test_get_records_filter_by_user_id(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50, user_id="user-1")
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50, user_id="user-2")
        records = tracker.get_records(user_id="user-1")
        assert len(records) == 1

    def test_get_records_with_limit(self):
        tracker = CostTracker()
        for _ in range(5):
            tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        records = tracker.get_records(limit=3)
        assert len(records) == 3

    def test_reset_session(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        summary = tracker.reset_session()
        assert isinstance(summary, CostSummary)
        assert summary.request_count == 1
        assert tracker.get_session_cost() == 0.0
        assert tracker.get_records() == []

    def test_export_records_dict(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        exported = tracker.export_records("dict")
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert "model" in exported[0]
        assert "cost_usd" in exported[0]

    def test_export_records_json(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        exported = tracker.export_records("json")
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert len(parsed) == 1

    def test_export_records_csv(self):
        tracker = CostTracker()
        tracker.record_usage("openai:gpt-4o-mini", input_tokens=100, output_tokens=50)
        exported = tracker.export_records("csv")
        assert isinstance(exported, str)
        lines = exported.strip().split("\n")
        assert len(lines) == 2  # header + 1 record

    def test_export_records_invalid_format(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="Unknown format"):
            tracker.export_records("xml")

    def test_record_with_metadata(self):
        tracker = CostTracker()
        record = tracker.record_usage(
            "openai:gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            request_id="req-123",
            user_id="user-1",
            metadata={"purpose": "testing"},
        )
        assert record.request_id == "req-123"
        assert record.user_id == "user-1"
        assert record.metadata == {"purpose": "testing"}

    def test_unknown_model_cost_is_none(self):
        tracker = CostTracker()
        record = tracker.record_usage("unknown:fake-model", input_tokens=100, output_tokens=50)
        assert record.cost_usd is None


class TestGlobalTracker:
    """Tests for global tracker management."""

    def setup_method(self):
        reset_global_tracker()

    def teardown_method(self):
        reset_global_tracker()

    def test_get_global_tracker_creates_instance(self):
        tracker = get_global_tracker()
        assert isinstance(tracker, CostTracker)

    def test_get_global_tracker_returns_same_instance(self):
        t1 = get_global_tracker()
        t2 = get_global_tracker()
        assert t1 is t2

    def test_reset_global_tracker(self):
        t1 = get_global_tracker()
        reset_global_tracker()
        t2 = get_global_tracker()
        assert t1 is not t2

    def test_configure_global_tracker(self):
        tracker = configure_global_tracker(daily_limit=5.0, session_limit=20.0)
        assert tracker.daily_limit == 5.0
        assert tracker.session_limit == 20.0
        assert get_global_tracker() is tracker


class TestEstimateCost:
    """Tests for the estimate_cost helper."""

    def test_estimate_known_model(self):
        cost = estimate_cost("openai:gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert cost > 0

    def test_estimate_unknown_model(self):
        cost = estimate_cost("unknown:fake", input_tokens=1000, output_tokens=500)
        assert cost is None
