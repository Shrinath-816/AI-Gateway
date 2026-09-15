"""
model_router.py
================

Model tiering: routing a request to a specific model based on how
demanding the task is, instead of hardcoding one model for every
request.

Why this matters (backend framing, not ML framing):
This is the same category of decision as picking a database read
replica size, or choosing which cache tier serves a request — you are
trading cost and latency against capability, per request, based on
workload characteristics.

IMPORTANT: The model slugs below are examples. OpenRouter's catalog of
available models and their exact slugs changes over time. Before
running this project, check https://openrouter.ai/models and update
DEFAULT_TIER_MAP with model slugs you actually have access to and want
to use.
"""

from app.schemas import TaskComplexity

# Default mapping of task complexity -> OpenRouter model slug.
#
# This is intentionally a plain dict (not hardcoded inline in the
# router class) so it can be swapped out entirely — e.g. loaded from a
# config file or a database — without touching the routing logic
# itself.
DEFAULT_TIER_MAP: dict[TaskComplexity, str] = {
    # Cheap, fast model — good for classification, extraction, short Q&A.
    TaskComplexity.SIMPLE: "openai/gpt-4o-mini",
    # Balanced default for everyday chat and general reasoning.
    TaskComplexity.STANDARD: "anthropic/claude-3.5-sonnet",
    # Strongest/most expensive tier — multi-step reasoning, coding, long analysis.
    TaskComplexity.COMPLEX: "anthropic/claude-3-opus",
}


class ModelTierRouter:
    """
    Selects a model slug for a request based on a TaskComplexity hint.

    This class is deliberately tiny. The point of Week 1 is not a
    sophisticated routing algorithm (that comes later, e.g. Week 8's
    confidence-based escalation) — it's establishing the *pattern*:
    routing decisions belong in one dedicated, swappable component,
    not scattered through request handlers.
    """

    def __init__(self, tier_map: dict[TaskComplexity, str] | None = None) -> None:
        """
        Args:
            tier_map: Optional override of the complexity->model mapping.
                Defaults to DEFAULT_TIER_MAP if not provided. Passing a
                custom map is mainly useful for tests.
        """
        self._tier_map = tier_map or DEFAULT_TIER_MAP

    def select_model(self, complexity: TaskComplexity) -> str:
        """
        Return the model slug that should handle a request of the given
        complexity.

        Args:
            complexity: The task complexity hint from the incoming request.

        Returns:
            The OpenRouter model slug to call.

        Raises:
            KeyError: If the complexity value has no entry in the tier
                map. This should not happen in practice, since
                TaskComplexity is a closed enum and DEFAULT_TIER_MAP
                covers every member — but we don't silently fall back,
                because silently routing to the wrong model tier is a
                cost/quality bug that's hard to notice in production.
        """
        return self._tier_map[complexity]
