"""Shared test doubles and fixture loaders. Nothing here touches the network."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_intel.apps_config import StoreListing
from review_intel.collectors.base import RawReview
from review_intel.domain import Store

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_play_items() -> list[dict[str, Any]]:
    """Recorded scraper output, with dates rebuilt exactly as the scraper builds them."""
    items: list[dict[str, Any]] = load_json("google_play_reviews.json")
    for item in items:
        item["at"] = datetime.fromtimestamp(item["at"])
    return items


def make_raw(review_id: str, rating: int = 3, body: str = "fine") -> RawReview:
    return RawReview(
        store_review_id=review_id,
        rating=rating,
        title=None,
        body=body,
        app_version="1.0",
        review_date=datetime(2026, 11, 1, tzinfo=UTC),
        language="en",
    )


class FakeSource:
    """A ReviewSource that serves canned reviews and can fail after N of them."""

    def __init__(
        self, store: Store, reviews: list[RawReview], fail_after: int | None = None
    ) -> None:
        self.store = store
        self._reviews = reviews
        self._fail_after = fail_after
        self.calls = 0

    def fetch(self, listing: StoreListing, limit: int) -> Iterator[RawReview]:
        self.calls += 1
        for index, review in enumerate(self._reviews[:limit]):
            if self._fail_after is not None and index == self._fail_after:
                raise ConnectionError("store went away")
            yield review
