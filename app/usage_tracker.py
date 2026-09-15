"""
usage_tracker.py
=================

Captures per-request telemetry: which model served it, how many tokens
were used, what it cost, and how long it took.

Why a dedicated component instead of print statements in the route
handler:
This is the seed of what becomes proper AI observability in Week 5
(tracing, dashboards, regression detection). Starting with a clean,
structured RequestLog now means swapping the in-memory store for a
real database or a tracing backend (e.g. Langfuse) later is a
one-file change, not a rewrite.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RequestLog:
    """
    A single structured record of one completed LLM request.

    Attributes:
        model: Model slug that served the request.
        prompt_tokens: Input tokens consumed.
        completion_tokens: Output tokens generated.
        total_tokens: prompt_tokens + completion_tokens.
        cost_usd: Computed cost of the request.
        latency_ms: Total wall-clock duration of the request, in milliseconds.
        timestamp: UTC time the request completed.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RequestTimer:
    """
    Small context-manager-style helper for measuring wall-clock latency
    around an LLM call.

    Usage:
        timer = RequestTimer()
        timer.start()
        ... do the LLM call ...
        elapsed_ms = timer.stop()
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        """Record the start time using a monotonic clock (immune to
        system clock adjustments, which wall-clock time is not)."""
        self._start = time.monotonic()

    def stop(self) -> float:
        """
        Returns:
            Elapsed time in milliseconds since start() was called.

        Raises:
            RuntimeError: If stop() is called before start().
        """
        if self._start is None:
            raise RuntimeError("RequestTimer.stop() called before start()")
        return (time.monotonic() - self._start) * 1000


class UsageTracker:
    """
    In-memory store of RequestLog entries.

    This is intentionally the simplest possible implementation — a
    list guarded by nothing fancy — because the point of Week 1 is
    establishing *where* usage tracking lives in the architecture, not
    building a production telemetry pipeline. Swap `_logs` for a
    database write or a call to an observability platform once that's
    covered later in the program.
    """

    def __init__(self) -> None:
        self._logs: list[RequestLog] = []

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> RequestLog:
        """
        Create and store a RequestLog entry.

        Args:
            model: Model slug that served the request.
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens generated.
            cost_usd: Computed cost of the request.
            latency_ms: Total request latency in milliseconds.

        Returns:
            The RequestLog entry that was stored.
        """
        log_entry = RequestLog(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self._logs.append(log_entry)
        return log_entry

    def all_logs(self) -> list[RequestLog]:
        """Return all recorded request logs, oldest first."""
        return list(self._logs)

    def total_cost_usd(self) -> float:
        """Return the sum of cost_usd across every recorded request."""
        return round(sum(log.cost_usd for log in self._logs), 6)

    def total_tokens(self) -> int:
        """Return the sum of total_tokens across every recorded request."""
        return sum(log.total_tokens for log in self._logs)
