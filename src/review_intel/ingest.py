"""Turn collector output into database rows, idempotently.

Running a fetch twice must never create duplicates. We look up which review ids
already exist in one query, insert the new ones and update only those whose
content changed (players can edit a review). The DB unique constraint on
(app_id, store_review_id) is a second safety net behind this logic.
"""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.apps_config import AppsConfig
from review_intel.collectors.base import RawReview, ReviewSource
from review_intel.db.models import App, Review, utcnow
from review_intel.domain import Store

logger = logging.getLogger(__name__)

# Fields copied from a RawReview; a change in any of them means the player edited the review.
TRACKED_FIELDS = ("rating", "title", "body", "app_version", "review_date", "language")


@dataclass(frozen=True)
class UpsertResult:
    inserted: int
    updated: int
    unchanged: int


@dataclass(frozen=True)
class StoreFetchResult:
    store: Store
    fetched: int
    upsert: UpsertResult
    error: str | None = None


def upsert_reviews(session: Session, app: App, reviews: Iterable[RawReview]) -> UpsertResult:
    # Keyed by id: the same review can appear twice if the feed shifts between pages.
    incoming = {review.store_review_id: review for review in reviews}
    if not incoming:
        return UpsertResult(inserted=0, updated=0, unchanged=0)

    existing = {
        row.store_review_id: row
        for row in session.scalars(
            select(Review).where(
                Review.app_id == app.id, Review.store_review_id.in_(list(incoming))
            )
        )
    }

    now = utcnow()
    inserted = updated = unchanged = 0
    for review_id, raw in incoming.items():
        values: dict[str, Any] = {field: getattr(raw, field) for field in TRACKED_FIELDS}
        row = existing.get(review_id)
        if row is None:
            session.add(Review(app_id=app.id, store_review_id=review_id, fetched_at=now, **values))
            inserted += 1
        elif any(getattr(row, field) != value for field, value in values.items()):
            for field, value in values.items():
                setattr(row, field, value)
            row.fetched_at = now
            updated += 1
        else:
            unchanged += 1
    session.flush()
    return UpsertResult(inserted=inserted, updated=updated, unchanged=unchanged)


def fetch_and_store(
    session_factory: Callable[[], Session],
    config: AppsConfig,
    slug: str,
    stores: Sequence[Store],
    limit: int,
    sources: Mapping[Store, ReviewSource],
) -> list[StoreFetchResult]:
    """Fetch each requested store and save what arrived.

    A failure in one store does not stop the others, and reviews fetched before a
    failure are still saved, so a flaky page 7 doesn't throw away pages 1-6.
    """
    game = config.get_game(slug)
    results: list[StoreFetchResult] = []
    for listing in game.listings:
        if listing.store not in stores:
            continue

        fetched: list[RawReview] = []
        error: str | None = None
        try:
            for raw in sources[listing.store].fetch(listing, limit):
                fetched.append(raw)
        except Exception as exc:
            logger.exception("fetching %s from %s failed", slug, listing.store)
            error = f"{type(exc).__name__}: {exc}"

        with session_factory() as session, session.begin():
            app = session.scalars(
                select(App).where(App.slug == slug, App.store == listing.store)
            ).one()
            upsert = upsert_reviews(session, app, fetched)
        results.append(StoreFetchResult(listing.store, len(fetched), upsert, error))
    return results
