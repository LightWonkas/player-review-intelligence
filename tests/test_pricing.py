import pytest
from anthropic.types import Usage

from review_intel.llm.pricing import UsageTotals


def _usage(input_tokens: int, output_tokens: int, cache_write: int, cache_read: int) -> Usage:
    return Usage.model_validate(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
        }
    )


def test_totals_add_up_across_requests() -> None:
    totals = UsageTotals()
    totals.add(_usage(100, 50, 2000, 0))
    totals.add(_usage(100, 50, 0, 2000))

    assert totals.requests == 2
    assert (totals.input_tokens, totals.output_tokens) == (200, 100)
    assert (totals.cache_write_tokens, totals.cache_read_tokens) == (2000, 2000)


def test_haiku_cost_uses_each_token_type_rate() -> None:
    totals = UsageTotals()
    totals.add(_usage(1_000_000, 1_000_000, 1_000_000, 1_000_000))

    # 0.10 input + 0.50 output + 0.125 cache write + 0.01 cache read
    assert totals.cost_usd("claude-haiku-5-5") == pytest.approx(0.735)
    assert totals.cost_usd("claude-haiku-5-5", batch=True) == pytest.approx(0.3675)


def test_unknown_model_has_no_price() -> None:
    assert UsageTotals().cost_usd("some-future-model") is None
