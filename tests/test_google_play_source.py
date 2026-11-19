from datetime import UTC, datetime
from email.message import Message
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from review_intel.apps_config import StoreListing
from review_intel.collectors.google_play import (
    GooglePlaySource,
    is_retryable_play_error,
    local_naive_to_utc,
)
from review_intel.domain import Store
from review_intel.retry import Pacer, RetryPolicy

from .fakes import load_play_items

LISTING = StoreListing(store=Store.GOOGLE_PLAY, store_app_id="com.critical.strike2")
FAST_RETRY = RetryPolicy(attempts=3, base_delay=0.0, jitter=0.0)


class FakeReviews:
    """Stands in for google_play_scraper.reviews: serves pages, then an empty page."""

    def __init__(self, pages: list[list[dict[str, Any]]], errors: list[Exception] | None = None):
        self.pages = pages
        self.errors = errors or []
        self.calls: list[dict[str, Any]] = []
        self.served = 0

    def __call__(self, app_id: str, **kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
        self.calls.append({"app_id": app_id, **kwargs})
        if self.errors:
            raise self.errors.pop(0)
        if self.served < len(self.pages):
            self.served += 1
            return self.pages[self.served - 1], f"token-{self.served}"
        return [], None


def _source(fake: FakeReviews) -> GooglePlaySource:
    return GooglePlaySource(
        reviews_fn=fake, pacer=Pacer(0.0), retry_policy=FAST_RETRY, sleep=lambda _: None
    )


def test_maps_recorded_reviews() -> None:
    items = load_play_items()
    fake = FakeReviews([items])

    reviews = list(_source(fake).fetch(LISTING, limit=10))

    assert [r.store_review_id for r in reviews] == [item["reviewId"] for item in items]
    first = reviews[1]
    assert first.rating == items[1]["score"]
    assert first.app_version == items[1]["reviewCreatedVersion"]
    assert first.title is None
    assert first.language == "en"


def test_converts_scraper_local_time_to_true_utc() -> None:
    # The scraper calls datetime.fromtimestamp(ts): naive local time on whatever machine runs it.
    ts = 1_795_000_000
    naive_local = datetime.fromtimestamp(ts)

    assert local_naive_to_utc(naive_local) == datetime.fromtimestamp(ts, UTC)


def test_pages_with_continuation_token_and_respects_limit() -> None:
    items = load_play_items()
    page = [dict(items[0], reviewId=f"r{i}") for i in range(2)]
    page2 = [dict(items[0], reviewId=f"r{i}") for i in range(2, 4)]
    page3 = [dict(items[0], reviewId=f"r{i}") for i in range(4, 6)]
    fake = FakeReviews([page, page2, page3])

    reviews = list(_source(fake).fetch(LISTING, limit=5))

    assert [r.store_review_id for r in reviews] == ["r0", "r1", "r2", "r3", "r4"]
    assert [call["continuation_token"] for call in fake.calls] == [None, "token-1", "token-2"]
    assert [call["count"] for call in fake.calls] == [5, 3, 1]


def test_stops_when_store_returns_empty_page() -> None:
    fake = FakeReviews([load_play_items()])

    reviews = list(_source(fake).fetch(LISTING, limit=500))

    assert len(reviews) == 3
    assert len(fake.calls) == 2


def test_retries_network_errors() -> None:
    fake = FakeReviews([load_play_items()], errors=[URLError("dns hiccup")])

    reviews = list(_source(fake).fetch(LISTING, limit=3))

    assert len(reviews) == 3


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (HTTPError("u", 429, "slow down", Message(), None), True),
        (HTTPError("u", 503, "unavailable", Message(), None), True),
        (HTTPError("u", 404, "not found", Message(), None), False),
        (URLError("dns"), True),
        (TimeoutError(), True),
        (ValueError("parse"), False),
    ],
)
def test_retryable_play_errors(exc: Exception, expected: bool) -> None:
    assert is_retryable_play_error(exc) is expected
