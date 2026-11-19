"""Token accounting and cost estimates.

Prices are USD per million tokens, Anthropic first-party API, as of PRICES_AS_OF.
They are estimates for logging and the README, not billing; check the pricing page
before budgeting. Haiku 5.5 rates are for prompts up to 100K tokens, which all of
our requests are.
"""

from dataclasses import dataclass

from anthropic.types import Usage

PRICES_AS_OF = "2026-11"
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_write: float  # 5-minute cache writes: 1.25x input
    cache_read: float  # 0.1x input


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
}


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, usage: Usage) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_tokens += usage.cache_read_input_tokens or 0

    def cost_usd(self, model: str, *, batch: bool = False) -> float | None:
        """Estimated cost, or None for a model missing from the price table."""
        price = PRICES.get(model)
        if price is None:
            return None
        total = (
            self.input_tokens * price.input
            + self.output_tokens * price.output
            + self.cache_write_tokens * price.cache_write
            + self.cache_read_tokens * price.cache_read
        ) / 1_000_000
        return total * BATCH_DISCOUNT if batch else total

    def describe(self, model: str, *, batch: bool = False) -> str:
        cost = self.cost_usd(model, batch=batch)
        cost_text = f"${cost:.4f}" if cost is not None else "unknown (model not in price table)"
        return (
            f"{self.requests} requests | input {self.input_tokens:,} | "
            f"cache write {self.cache_write_tokens:,} | cache read {self.cache_read_tokens:,} | "
            f"output {self.output_tokens:,} | est. cost {cost_text}"
            + (" (batch -50%)" if batch else "")
        )
