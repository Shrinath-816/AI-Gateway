"""
test_usage_tracker.py
======================

Unit tests for UsageTracker and RequestTimer.
"""

import time

import pytest

from app.usage_tracker import RequestTimer, UsageTracker


def test_record_stores_and_computes_total_tokens() -> None:
    """total_tokens on a stored log should equal prompt + completion tokens."""
    tracker = UsageTracker()
    log = tracker.record(
        model="test-model", prompt_tokens=100, completion_tokens=50, cost_usd=0.01, latency_ms=250.0
    )
    assert log.total_tokens == 150


def test_aggregate_totals_across_multiple_requests() -> None:
    """total_cost_usd and total_tokens should sum across all recorded requests."""
    tracker = UsageTracker()
    tracker.record(model="m", prompt_tokens=100, completion_tokens=50, cost_usd=0.01, latency_ms=100.0)
    tracker.record(model="m", prompt_tokens=200, completion_tokens=100, cost_usd=0.02, latency_ms=100.0)

    assert tracker.total_tokens() == 450
    assert tracker.total_cost_usd() == pytest.approx(0.03)
    assert len(tracker.all_logs()) == 2


def test_request_timer_measures_elapsed_time() -> None:
    """RequestTimer.stop() should report a positive elapsed duration."""
    timer = RequestTimer()
    timer.start()
    time.sleep(0.01)
    elapsed_ms = timer.stop()
    assert elapsed_ms > 0


def test_request_timer_raises_if_stopped_before_started() -> None:
    """Calling stop() before start() is a programming error and should raise."""
    timer = RequestTimer()
    with pytest.raises(RuntimeError):
        timer.stop()
