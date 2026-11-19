"""The common shape every store collector produces, and the interface it implements."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from review_intel.apps_config import StoreListing
from review_intel.domain import Store


@dataclass(frozen=True, slots=True)
class RawReview:
    """A store review normalised to our schema.

    There are deliberately no author, username or avatar fields: personal data is
    dropped at the collector boundary, so it can never reach the database or a prompt.
    """

    store_review_id: str
    rating: int
    title: str | None
    body: str
    app_version: str | None
    review_date: datetime
    language: str | None

    def __post_init__(self) -> None:
        if not 1 <= self.rating <= 5:
            raise ValueError(f"rating must be 1-5, got {self.rating}")
        if self.review_date.tzinfo is None:
            raise ValueError("review_date must be timezone-aware")


class ReviewSource(Protocol):
    """Anything that can list a store listing's newest reviews.

    Implementations handle paging, rate limiting and retries; callers only see
    a stream of `RawReview`s, newest first, at most `limit` of them.
    """

    store: Store

    def fetch(self, listing: StoreListing, limit: int) -> Iterator[RawReview]: ...
