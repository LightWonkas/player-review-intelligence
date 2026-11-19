"""App Store reviews via Apple's public customer-reviews JSON feed.

The feed serves at most 10 pages of 50 reviews (500 per storefront), newest first.
Quirks handled here (all seen live):
- a page with one review returns `entry` as an object instead of a list;
- an empty page has no `entry` key at all;
- a page in the *middle* can be empty while later pages still have reviews, so an
  empty page is skipped, not treated as the end. The real end comes from the
  feed's own `rel="last"` link on page 1.
"""

import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from typing import Any

import httpx

from review_intel.apps_config import StoreListing
from review_intel.collectors.base import RawReview
from review_intel.domain import Store
from review_intel.retry import DEFAULT_RETRY, Pacer, RetryPolicy, Sleep, with_retries

FEED_URL = (
    "https://itunes.apple.com/{country}/rss/customerreviews"
    "/page={page}/id={app_id}/sortby=mostrecent/json"
)
MAX_PAGES = 10
USER_AGENT = "review-intel/0.1"


def is_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    # Timeouts, connection resets, DNS hiccups.
    return isinstance(exc, httpx.TransportError)


def _label(entry: dict[str, Any], key: str) -> str | None:
    value = entry.get(key)
    if isinstance(value, dict):
        label = value.get("label")
        return str(label) if label is not None else None
    return None


def feed_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    entry = data.get("feed", {}).get("entry")
    if entry is None:
        return []
    if isinstance(entry, dict):
        return [entry]
    return list(entry)


_PAGE_IN_URL = re.compile(r"/page=(\d+)/")


def feed_last_page(data: dict[str, Any]) -> int | None:
    """Read the page count from the feed's `rel="last"` link, if present."""
    links = data.get("feed", {}).get("link", [])
    if isinstance(links, dict):
        links = [links]
    for link in links:
        attributes = link.get("attributes", {}) if isinstance(link, dict) else {}
        if attributes.get("rel") == "last":
            match = _PAGE_IN_URL.search(str(attributes.get("href", "")))
            return int(match.group(1)) if match else None
    return None


def to_raw_review(entry: dict[str, Any]) -> RawReview | None:
    """Map one feed entry; returns None for entries that are not reviews."""
    review_id = _label(entry, "id")
    rating = _label(entry, "im:rating")
    updated = _label(entry, "updated")
    # Older feed versions put the app itself first, without a rating.
    if review_id is None or rating is None or updated is None:
        return None
    return RawReview(
        store_review_id=review_id,
        rating=int(rating),
        title=_label(entry, "title"),
        body=(_label(entry, "content") or "").strip(),
        app_version=_label(entry, "im:version"),
        review_date=datetime.fromisoformat(updated).astimezone(UTC),
        language=None,  # The feed reports the storefront, not the review language.
    )


class AppStoreSource:
    store = Store.APP_STORE

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        pacer: Pacer | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._client = client
        self._pacer = pacer or Pacer(1.0, sleep=sleep)
        self._retry_policy = retry_policy
        self._sleep = sleep

    def fetch(self, listing: StoreListing, limit: int) -> Iterator[RawReview]:
        if self._client is not None:
            yield from self._fetch_pages(self._client, listing, limit)
            return
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=15.0, headers=headers) as client:
            yield from self._fetch_pages(client, listing, limit)

    def _fetch_pages(
        self, client: httpx.Client, listing: StoreListing, limit: int
    ) -> Iterator[RawReview]:
        fetched = 0
        page, last_page = 1, MAX_PAGES
        while page <= last_page:
            self._pacer.wait()
            url = FEED_URL.format(country=listing.country, page=page, app_id=listing.store_app_id)
            data = with_retries(
                partial(_get_json, client, url),
                is_retryable=is_retryable_http_error,
                policy=self._retry_policy,
                sleep=self._sleep,
                description=f"app_store page {page} for {listing.store_app_id}",
            )
            entries = feed_entries(data)
            if page == 1:
                if not entries:
                    return  # no reviews at all for this storefront
                last_page = min(MAX_PAGES, feed_last_page(data) or MAX_PAGES)
            for entry in entries:
                review = to_raw_review(entry)
                if review is None:
                    continue
                yield review
                fetched += 1
                if fetched >= limit:
                    return
            page += 1


def _get_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data
