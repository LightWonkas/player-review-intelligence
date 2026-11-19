"""Per-version counts from SQL. Every number `ask` and `spikes` show comes from here,
never from the model."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import select
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
) -> list[ClassifiedReview]:
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
    result = []
    for store in Store:
        dates = list(
            session.scalars(
                select(Review.review_date)
                .join(App, Review.app_id == App.id)
                .where(App.slug == slug, App.store == store)
            )
        )
        if not dates:
            continue
        classified = len(
            classified_reviews(
                session, slug, model=model, prompt_version=prompt_version, store=store
            )
        )
        result.append(Coverage(store, len(dates), classified, min(dates).date(), max(dates).date()))
    return result
