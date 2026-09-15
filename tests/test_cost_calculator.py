"""
test_cost_calculator.py
========================

Unit tests for CostCalculator. Pure arithmetic — no network calls.
"""

import pytest

from app.pricing import CostCalculator, ModelPrice


@pytest.fixture
def calculator() -> CostCalculator:
    """A CostCalculator with a small, known pricing table for predictable assertions."""
    pricing_table = {
        "test-model": ModelPrice(input_per_million=1.0, output_per_million=2.0),
    }
    return CostCalculator(pricing_table=pricing_table)


def test_calculate_basic_cost(calculator: CostCalculator) -> None:
    """1,000,000 prompt tokens + 1,000,000 completion tokens should cost input+output price exactly."""
    cost = calculator.calculate(model="test-model", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(3.0)  # $1.00 input + $2.00 output


def test_calculate_small_token_counts_are_not_rounded_to_zero(calculator: CostCalculator) -> None:
    """
    A tiny request (e.g. 100 tokens) still produces a nonzero cost given
    the rounding precision used — this guards against the common bug of
    rounding to 2 decimal places and making small requests look free.
    """
    cost = calculator.calculate(model="test-model", prompt_tokens=100, completion_tokens=100)
    assert cost > 0


def test_calculate_unknown_model_raises_key_error(calculator: CostCalculator) -> None:
    """An unpriced model should raise rather than silently returning $0."""
    with pytest.raises(KeyError):
        calculator.calculate(model="unknown-model", prompt_tokens=100, completion_tokens=100)
