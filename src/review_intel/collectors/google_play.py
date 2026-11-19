"""Google Play reviews via the `google-play-scraper` package.

The package is untyped and does no retrying, so this module is the boundary that
adds types, paging, pacing and retries around it.
"""

import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import partial
from typing import Any
from urllib.error import HTTPError, URLError

from google_play_scraper import Sort
from google_play_scraper import reviews as scraper_reviews

from review_intel.apps_config import StoreListing
from review_intel.collectors.base import RawReview
from review_intel.domain import Store
from review_intel.retry import DEFAULT_RETRY, Pacer, RetryPolicy, Sleep, with_retries

PAGE_SIZE = 100

# Signature of google_play_scraper.reviews: returns (reviews, continuation_token).
ReviewsFn = Callable[..., tuple[list[dict[str, Any]], Any]]


def is_retryable_play_error(exc: Exception) -> bool:
    """The scraper uses urllib: retry throttling, server errors and network failures."""
    if isinstance(exc, HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, URLError | TimeoutError | ConnectionError)


def local_naive_to_utc(value: datetime) -> datetime:
    """The scraper builds dates with `datetime.fromtimestamp()`: naive *local* time.

    `astimezone()` on a naive datetime assumes local time, which reverses exactly
    that, so the result is the true moment in UTC whatever machine runs this.
    """
    return value.astimezone(UTC)


def to_raw_review(item: dict[str, Any], language: str) -> RawReview:
    return RawReview(
        store_review_id=str(item["reviewId"]),
        rating=int(item["score"]),
        title=None,  # Play reviews have no title.
        body=(item.get("content") or "").strip(),
        # reviewCreatedVersion is the version the review was written on; often missing.
        app_version=item.get("reviewCreatedVersion") or item.get("appVersion"),
        review_date=local_naive_to_utc(item["at"]),
        language=language,
    )


class GooglePlaySource:
    store = Store.GOOGLE_PLAY

    def __init__(
        self,
        *,
        reviews_fn: ReviewsFn = scraper_reviews,
        pacer: Pacer | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._reviews_fn = reviews_fn
        self._pacer = pacer or Pacer(1.0, sleep=sleep)
        self._retry_policy = retry_policy
        self._sleep = sleep

    def fetch(self, listing: StoreListing, limit: int) -> Iterator[RawReview]:
        token: Any = None
        fetched = 0
        while fetched < limit:
            self._pacer.wait()
            call = partial(
                self._reviews_fn,
                listing.store_app_id,
                lang=listing.language,
                country=listing.country,
                sort=Sort.NEWEST,
                count=min(PAGE_SIZE, limit - fetched),
                continuation_token=token,
            )
            batch, token = with_retries(
                call,
                is_retryable=is_retryable_play_error,
                policy=self._retry_policy,
                sleep=self._sleep,
                description=f"google_play page for {listing.store_app_id}",
            )
            if not batch:
                return
            for item in batch:
                yield to_raw_review(item, listing.language)
                fetched += 1
                if fetched >= limit:
                    return
