"""Per-version counts from SQL. Every number `ask` and `spikes` show comes from here,
never from the model."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from review_intel.db.models import App, Classification, Review
from review_intel.domain import Sentiment, Store
from review_intel.patch_notes import version_key


@dataclass(frozen=True)
class ClassifiedReview:
    review_id: int
    store: Store
    version: str | None
    review_date: datetime
    rating: int
    title: str | None
    body: str
    categories: list[str]
    primary_category: str
    severity: int
    sentiment: Sentiment
    summary: str
    is_bug_report: bool


@dataclass
class VersionCounts:
    version: str
    n: int = 0
    categories: Counter[str] = field(default_factory=Counter)  # reviews mentioning each
    first_seen: date | None = None


@dataclass(frozen=True)
class Coverage:
    store: Store
    reviews: int
    classified: int
    first: date
    last: date


def classified_reviews(
    session: Session,
    slug: str,
    *,
    model: str,
    prompt_version: str,
    store: Store | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    versions: Sequence[str] | None = None,
) -> list[ClassifiedReview]:
    """Classified reviews of one game, filtered in SQL: [start, end) and/or versions."""
    query = (
        select(Review, App.store, Classification)
        .join(App, Review.app_id == App.id)
        .join(Classification, Classification.review_id == Review.id)
        .where(
            App.slug == slug,
            Classification.model == model,
            Classification.prompt_version == prompt_version,
        )
        .order_by(Review.review_date)
    )
    if store is not None:
        query = query.where(App.store == store)
    if start is not None:
        query = query.where(Review.review_date >= start)
    if end is not None:
        query = query.where(Review.review_date < end)
    if versions is not None:
        query = query.where(Review.app_version.in_(list(versions)))
    return [
        ClassifiedReview(
            review_id=review.id,
            store=app_store,
            version=review.app_version,
            review_date=review.review_date,
            rating=review.rating,
            title=review.title,
            body=review.body,
            categories=list(result.categories),
            primary_category=result.primary_category,
            severity=result.severity,
            sentiment=result.sentiment,
            summary=result.short_summary,
            is_bug_report=result.is_bug_report,
        )
        for review, app_store, result in session.execute(query)
    ]


def counts_by_version(reviews: list[ClassifiedReview]) -> list[VersionCounts]:
    """Oldest version first. Reviews without a version are left out."""
    rows: dict[str, VersionCounts] = {}
    for review in reviews:
        if not review.version:
            continue
        row = rows.setdefault(review.version, VersionCounts(review.version))
        row.n += 1
        row.categories.update(set(review.categories))
        day = review.review_date.date()
        if row.first_seen is None or day < row.first_seen:
            row.first_seen = day
    return sorted(rows.values(), key=lambda row: version_key(row.version))


def coverage(session: Session, slug: str, *, model: str, prompt_version: str) -> list[Coverage]:
    """What the data covers per store, so answers can say what they can't see."""
    totals = session.execute(
        select(
            App.store,
            func.count(Review.id),
            func.min(Review.review_date),
            func.max(Review.review_date),
        )
        .join(App, Review.app_id == App.id)
        .where(App.slug == slug)
        .group_by(App.store)
    ).all()
    classified = {
        store: count
        for store, count in session.execute(
            select(App.store, func.count(Classification.id))
            .join(Review, Classification.review_id == Review.id)
            .join(App, Review.app_id == App.id)
            .where(
                App.slug == slug,
                Classification.model == model,
                Classification.prompt_version == prompt_version,
            )
            .group_by(App.store)
        )
    }
    found = {store: (n, first, last) for store, n, first, last in totals}
    return [
        Coverage(store, n, classified.get(store, 0), first.date(), last.date())
        for store in Store
        if store in found
        for n, first, last in [found[store]]
    ]
