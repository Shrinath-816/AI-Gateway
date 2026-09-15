"""
test_model_router.py
=====================

Unit tests for ModelTierRouter. These don't touch the network — the
router is pure logic (a dict lookup), so it should be tested as such.
"""

import pytest

from app.model_router import ModelTierRouter
from app.schemas import TaskComplexity


def test_select_model_returns_expected_model_for_each_tier() -> None:
    """Each complexity tier should route to a distinct, expected model."""
    custom_map = {
        TaskComplexity.SIMPLE: "cheap-model",
        TaskComplexity.STANDARD: "balanced-model",
        TaskComplexity.COMPLEX: "strong-model",
    }
    router = ModelTierRouter(tier_map=custom_map)

    assert router.select_model(TaskComplexity.SIMPLE) == "cheap-model"
    assert router.select_model(TaskComplexity.STANDARD) == "balanced-model"
    assert router.select_model(TaskComplexity.COMPLEX) == "strong-model"


def test_default_tier_map_covers_every_complexity_value() -> None:
    """
    The default router must have an entry for every TaskComplexity
    member — a missing entry would raise KeyError in production the
    first time that tier was requested.
    """
    router = ModelTierRouter()
    for complexity in TaskComplexity:
        # Should not raise.
        model = router.select_model(complexity)
        assert isinstance(model, str)
        assert model  # non-empty


def test_missing_tier_raises_key_error() -> None:
    """
    Routing a complexity value with no entry in the tier map should
    raise loudly rather than silently defaulting to some model.
    """
    router = ModelTierRouter(tier_map={TaskComplexity.SIMPLE: "only-simple"})

    with pytest.raises(KeyError):
        router.select_model(TaskComplexity.COMPLEX)
