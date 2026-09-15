"""
pricing.py
==========

Per-model pricing and cost calculation.

IMPORTANT — pricing is fast-changing data, not a durable fact:
The numbers in PRICING_TABLE below are placeholders illustrating the
*shape* of the data (dollars per million tokens, input vs. output
priced separately). Before relying on this for real cost tracking,
replace them with current figures from https://openrouter.ai/models
(each model's page lists prompt/completion price per token). Treat
this table the same way you'd treat a cloud provider's price list —
something you refresh periodically, not something you hardcode once
and forget.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """
    Price for one model, in USD per 1 million tokens.

    Providers (and OpenRouter, which aggregates them) typically price
    input (prompt) and output (completion) tokens differently —
    output tokens are usually more expensive since they involve actual
    generation compute, not just a forward pass over the prompt.

    Attributes:
        input_per_million: USD cost per 1,000,000 prompt tokens.
        output_per_million: USD cost per 1,000,000 completion tokens.
    """

    input_per_million: float
    output_per_million: float


# Placeholder pricing table — VERIFY AND UPDATE against
# https://openrouter.ai/models before using this for real cost tracking.
PRICING_TABLE: dict[str, ModelPrice] = {
    "openai/gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
    "anthropic/claude-3.5-sonnet": ModelPrice(input_per_million=3.00, output_per_million=15.00),
    "anthropic/claude-3-opus": ModelPrice(input_per_million=15.00, output_per_million=75.00),
}


class CostCalculator:
    """
    Computes the USD cost of a single request from its token usage and
    the model that served it.
    """

    def __init__(self, pricing_table: dict[str, ModelPrice] | None = None) -> None:
        """
        Args:
            pricing_table: Optional override of the model->price mapping.
                Defaults to PRICING_TABLE. Injectable for testing.
        """
        self._pricing_table = pricing_table or PRICING_TABLE

    def calculate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost, in USD, of a single request.

        Args:
            model: The model slug that served the request.
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.

        Returns:
            Cost in USD, rounded to 6 decimal places (token-level costs
            are fractions of a cent — rounding to 2 decimals would
            silently zero out most individual requests).

        Raises:
            KeyError: If the model has no entry in the pricing table.
                We raise rather than default to $0 or a guessed price,
                because a silent $0 cost is a worse failure mode than a
                loud error — it would make a cost dashboard quietly wrong.
        """
        price = self._pricing_table[model]
        input_cost = (prompt_tokens / 1_000_000) * price.input_per_million
        output_cost = (completion_tokens / 1_000_000) * price.output_per_million
        return round(input_cost + output_cost, 6)
