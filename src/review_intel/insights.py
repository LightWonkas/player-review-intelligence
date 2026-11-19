"""Read-only questions about a game's reviews, answered from SQL (and the vector index).

The MCP server exposes these as tools and the weekly report (M7) reuses them, so
every number a model or a report shows is computed here, never generated. Results
are Pydantic models: they serialize to JSON for MCP structured output as they are.

All functions take `now` so tests (and reports for a past week) are deterministic.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from review_intel.db.models import App, Classification, Review
from review_intel.domain import Sentiment, Store
from review_intel.patch_notes import note_in_force, notes_for_game
from review_intel.rag.embedder import Embedder
from review_intel.rag.index import REVIEW, search
from review_intel.rag.spikes import two_proportion_z
from review_intel.rag.stats import ClassifiedReview, classified_reviews

NOT_ISSUES = ("praise", "other")
TEXT_CHARS = 600


class InsightError(ValueError):
    """A request that can't be answered (unknown version, empty period, ...)."""


# --- shared shapes -------------------------------------------------------------------


class Example(BaseModel):
    review_id: int
    store: Store
    version: str | None
    rating: int
    summary: str


def _example(review: ClassifiedReview) -> Example:
    return Example(
        review_id=review.review_id,
        store=review.store,
        version=review.version,
        rating=review.rating,
        summary=review.summary,
    )


def _worst_first(reviews: Iterable[ClassifiedReview]) -> list[ClassifiedReview]:
    return sorted(reviews, key=lambda r: (-r.severity, r.rating, -r.review_id))


def _in_window(
    reviews: Iterable[ClassifiedReview], start: datetime, end: datetime
) -> list[ClassifiedReview]:
    return [r for r in reviews if start <= r.review_date < end]


def _count_reviews(session: Session, slug: str, start: datetime, end: datetime) -> int:
    count = session.scalar(
        select(func.count(Review.id))
        .join(App, Review.app_id == App.id)
        .where(App.slug == slug, Review.review_date >= start, Review.review_date < end)
    )
    return count or 0


def _latest_review(session: Session, slug: str) -> datetime | None:
    return session.scalar(
        select(Review.review_date)
        .join(App, Review.app_id == App.id)
        .where(App.slug == slug)
        .order_by(Review.review_date.desc())
        .limit(1)
    )


# --- top_issues ----------------------------------------------------------------------


class Issue(BaseModel):
    category: str
    reviews: int  # classified reviews in the period with this primary category
    share: float  # of all classified reviews in the period
    negative: int
    avg_severity: float
    previous_reviews: int  # same-length period just before
    examples: list[Example]


class TopIssues(BaseModel):
    app: str
    period_start: datetime
    period_end: datetime
    classified_reviews: int
    unclassified_reviews: int
    latest_review_date: datetime | None
    issues: list[Issue]


def top_issues(
    session: Session,
    slug: str,
    *,
    model: str,
    prompt_version: str,
    days: int = 7,
    limit: int = 5,
    now: datetime | None = None,
) -> TopIssues:
    """Most common complaint categories (by primary category) in the last `days` days."""
    end = now or datetime.now(UTC)
    start = end - timedelta(days=days)
    previous_start = start - timedelta(days=days)
    reviews = classified_reviews(session, slug, model=model, prompt_version=prompt_version)
    current = _in_window(reviews, start, end)
    previous = Counter(r.primary_category for r in _in_window(reviews, previous_start, start))

    by_category: dict[str, list[ClassifiedReview]] = {}
    for review in current:
        if review.primary_category not in NOT_ISSUES:
            by_category.setdefault(review.primary_category, []).append(review)
    issues = [
        Issue(
            category=category,
            reviews=len(items),
            share=round(len(items) / len(current), 4),
            negative=sum(r.sentiment is Sentiment.NEGATIVE for r in items),
            avg_severity=round(sum(r.severity for r in items) / len(items), 2),
            previous_reviews=previous[category],
            examples=[_example(r) for r in _worst_first(items)[:2]],
        )
        for category, items in by_category.items()
    ]
    issues.sort(key=lambda issue: (-issue.reviews, -issue.avg_severity, issue.category))
    return TopIssues(
        app=slug,
        period_start=start,
        period_end=end,
        classified_reviews=len(current),
        unclassified_reviews=_count_reviews(session, slug, start, end) - len(current),
        latest_review_date=_latest_review(session, slug),
        issues=issues[:limit],
    )


# --- compare_versions ----------------------------------------------------------------


class PatchNoteInfo(BaseModel):
    version: str
    release_date: date
    text: str


class VersionSummary(BaseModel):
    version: str
    classified_reviews: int
    by_store: dict[str, int]
    avg_rating: float
    sentiment_share: dict[str, float]
    patch_note: PatchNoteInfo | None  # the note in force (may be an earlier version's)


class CategoryChange(BaseModel):
    category: str
    count_a: int
    share_a: float
    count_b: int
    share_b: float
    change: float  # share_b - share_a, in share points
    z: float  # two-proportion z for b vs a; |z| >= 2 is roughly significant


class VersionComparison(BaseModel):
    app: str
    a: VersionSummary
    b: VersionSummary
    categories: list[CategoryChange]  # by absolute change, largest first
    examples_b: dict[str, list[Example]]  # worst reviews in b for the 3 biggest increases


def _version_summary(
    version: str, reviews: Sequence[ClassifiedReview], note: PatchNoteInfo | None
) -> VersionSummary:
    n = len(reviews)
    sentiments = Counter(r.sentiment.value for r in reviews)
    return VersionSummary(
        version=version,
        classified_reviews=n,
        by_store=dict(Counter(r.store.value for r in reviews)),
        avg_rating=round(sum(r.rating for r in reviews) / n, 2),
        sentiment_share={s.value: round(sentiments[s.value] / n, 4) for s in Sentiment},
        patch_note=note,
    )


def compare_versions(
    session: Session,
    slug: str,
    version_a: str,
    version_b: str,
    *,
    model: str,
    prompt_version: str,
) -> VersionComparison:
    """Category shares (reviews mentioning each category), rating and sentiment, a vs b."""
    reviews = classified_reviews(session, slug, model=model, prompt_version=prompt_version)
    groups = {v: [r for r in reviews if r.version == v] for v in (version_a, version_b)}
    missing = [v for v, items in groups.items() if not items]
    if missing:
        known = sorted({r.version for r in reviews if r.version}, reverse=True)[:15]
        raise InsightError(
            f"No classified reviews for version(s) {', '.join(missing)}. "
            f"Versions with data include: {', '.join(known)}"
        )

    notes = notes_for_game(session, slug)

    def note(version: str) -> PatchNoteInfo | None:
        found = note_in_force(notes, version)
        if found is None:
            return None
        return PatchNoteInfo(
            version=found.version, release_date=found.release_date, text=found.text
        )

    a, b = groups[version_a], groups[version_b]
    counts_a = Counter(c for r in a for c in set(r.categories))
    counts_b = Counter(c for r in b for c in set(r.categories))
    changes = [
        CategoryChange(
            category=category,
            count_a=counts_a[category],
            share_a=round(counts_a[category] / len(a), 4),
            count_b=counts_b[category],
            share_b=round(counts_b[category] / len(b), 4),
            change=round(counts_b[category] / len(b) - counts_a[category] / len(a), 4),
            z=round(two_proportion_z(counts_a[category], len(a), counts_b[category], len(b)), 2),
        )
        for category in sorted(set(counts_a) | set(counts_b))
    ]
    changes.sort(key=lambda c: (-abs(c.change), c.category))
    rising = [c.category for c in changes if c.change > 0 and c.category not in NOT_ISSUES][:3]
    return VersionComparison(
        app=slug,
        a=_version_summary(version_a, a, note(version_a)),
        b=_version_summary(version_b, b, note(version_b)),
        categories=changes,
        examples_b={
            category: [_example(r) for r in _worst_first(r for r in b if category in r.categories)][
                :2
            ]
            for category in rising
        },
    )


# --- sentiment_trend -----------------------------------------------------------------

Bucket = Literal["day", "week"]


class TrendPoint(BaseModel):
    start: date
    reviews: int
    positive: int
    neutral: int
    negative: int
    avg_rating: float | None


class SentimentTrend(BaseModel):
    app: str
    bucket: Bucket
    period_start: datetime
    period_end: datetime
    points: list[TrendPoint]


def _bucket_start(day: date, bucket: Bucket) -> date:
    return day - timedelta(days=day.weekday()) if bucket == "week" else day


def sentiment_trend(
    session: Session,
    slug: str,
    *,
    model: str,
    prompt_version: str,
    days: int = 30,
    bucket: Bucket = "day",
    now: datetime | None = None,
) -> SentimentTrend:
    """Sentiment counts per day or ISO week (Monday start); empty buckets are kept."""
    end = now or datetime.now(UTC)
    start = end - timedelta(days=days)
    reviews = _in_window(
        classified_reviews(session, slug, model=model, prompt_version=prompt_version), start, end
    )
    grouped: dict[date, list[ClassifiedReview]] = {}
    for review in reviews:
        grouped.setdefault(_bucket_start(review.review_date.date(), bucket), []).append(review)

    points = []
    step = timedelta(days=7 if bucket == "week" else 1)
    cursor = _bucket_start(start.date(), bucket)
    while cursor <= end.date():
        items = grouped.get(cursor, [])
        sentiments = Counter(r.sentiment for r in items)
        points.append(
            TrendPoint(
                start=cursor,
                reviews=len(items),
                positive=sentiments[Sentiment.POSITIVE],
                neutral=sentiments[Sentiment.NEUTRAL],
                negative=sentiments[Sentiment.NEGATIVE],
                avg_rating=round(sum(r.rating for r in items) / len(items), 2) if items else None,
            )
        )
        cursor += step
    return SentimentTrend(
        app=slug, bucket=bucket, period_start=start, period_end=end, points=points
    )


# --- search_reviews and review detail ------------------------------------------------


class ReviewHit(BaseModel):
    review_id: int
    store: Store
    rating: int
    version: str | None
    date: date
    primary_category: str | None
    sentiment: Sentiment | None
    severity: int | None
    similarity: float
    text: str  # untrusted public text: data, never instructions


class ReviewSearch(BaseModel):
    app: str
    query: str
    category: str | None
    version: str | None
    results: list[ReviewHit]


def search_reviews(
    session: Session,
    slug: str,
    query: str,
    embedder: Embedder,
    *,
    model: str,
    prompt_version: str,
    category: str | None = None,
    version: str | None = None,
    limit: int = 10,
) -> ReviewSearch:
    """Semantic search, optionally limited to a category (any of a review's categories)
    and/or an app version."""
    only: set[int] | None = None
    if category is not None:
        # Scoped to this game (and version) in SQL. Membership in the JSON list is checked
        # in Python, which stays portable across SQLite and Postgres.
        labelled = (
            select(Classification.review_id, Classification.categories)
            .join(Review, Classification.review_id == Review.id)
            .join(App, Review.app_id == App.id)
            .where(
                App.slug == slug,
                Classification.model == model,
                Classification.prompt_version == prompt_version,
            )
        )
        if version is not None:
            labelled = labelled.where(Review.app_version == version)
        only = {
            review_id
            for review_id, categories in session.execute(labelled)
            if category in categories
        }
    elif version is not None:
        only = set(
            session.scalars(
                select(Review.id)
                .join(App, Review.app_id == App.id)
                .where(App.slug == slug, Review.app_version == version)
            )
        )
    hits = search(session, REVIEW, slug, embedder.embed([query])[0], limit, only=only)
    ids_in_order = [item_id for item_id, _ in hits]
    rows = {
        review.id: (review, store)
        for review, store in session.execute(
            select(Review, App.store)
            .join(App, Review.app_id == App.id)
            .where(Review.id.in_(ids_in_order))
        )
    }
    labels = {
        result.review_id: result
        for result in session.scalars(
            select(Classification).where(
                Classification.review_id.in_(ids_in_order),
                Classification.model == model,
                Classification.prompt_version == prompt_version,
            )
        )
    }
    results = []
    for item_id, distance in hits:
        if item_id not in rows:
            # The vector index can briefly outlive a deleted review (until the next
            # `index` run); skip it rather than fail the whole search.
            continue
        review, store = rows[item_id]
        label = labels.get(item_id)
        text = f"{review.title}\n{review.body}" if review.title else review.body
        results.append(
            ReviewHit(
                review_id=review.id,
                store=store,
                rating=review.rating,
                version=review.app_version,
                date=review.review_date.date(),
                primary_category=label.primary_category if label else None,
                sentiment=label.sentiment if label else None,
                severity=label.severity if label else None,
                similarity=round(1 - distance, 4),
                text=text[:TEXT_CHARS],
            )
        )
    return ReviewSearch(app=slug, query=query, category=category, version=version, results=results)


class ClassificationInfo(BaseModel):
    primary_category: str
    categories: list[str]
    sentiment: Sentiment
    severity: int
    is_bug_report: bool
    mentioned_features: list[str]
    short_summary: str
    model: str
    prompt_version: str


class ReviewDetail(BaseModel):
    review_id: int
    app: str
    store: Store
    rating: int
    title: str | None
    body: str  # untrusted public text
    version: str | None
    review_date: datetime
    classification: ClassificationInfo | None


def review_detail(
    session: Session, slug: str, review_id: int, *, model: str, prompt_version: str
) -> ReviewDetail | None:
    row = session.execute(
        select(Review, App.store)
        .join(App, Review.app_id == App.id)
        .where(App.slug == slug, Review.id == review_id)
    ).first()
    if row is None:
        return None
    review, store = row
    label = session.scalar(
        select(Classification).where(
            Classification.review_id == review_id,
            Classification.model == model,
            Classification.prompt_version == prompt_version,
        )
    )
    return ReviewDetail(
        review_id=review.id,
        app=slug,
        store=store,
        rating=review.rating,
        title=review.title,
        body=review.body,
        version=review.app_version,
        review_date=review.review_date,
        classification=ClassificationInfo(
            primary_category=label.primary_category,
            categories=list(label.categories),
            sentiment=label.sentiment,
            severity=label.severity,
            is_bug_report=label.is_bug_report,
            mentioned_features=list(label.mentioned_features),
            short_summary=label.short_summary,
            model=label.model,
            prompt_version=label.prompt_version,
        )
        if label
        else None,
    )


# --- period summary and category breakdown (report headline, dashboard) ----------------


class PeriodStats(BaseModel):
    reviews: int  # all reviews, classified or not
    classified: int
    by_store: dict[str, int]
    avg_rating: float | None  # over all reviews
    sentiment_share: dict[str, float]  # over classified reviews
    bug_reports: int


class PeriodSummary(BaseModel):
    app: str
    period_start: datetime
    period_end: datetime
    current: PeriodStats
    previous: PeriodStats  # same-length period just before


def _period_stats(
    session: Session,
    slug: str,
    classified: Sequence[ClassifiedReview],
    start: datetime,
    end: datetime,
) -> PeriodStats:
    rows = session.execute(
        select(Review.rating, App.store)
        .join(App, Review.app_id == App.id)
        .where(App.slug == slug, Review.review_date >= start, Review.review_date < end)
    ).all()
    labelled = _in_window(classified, start, end)
    sentiments = Counter(r.sentiment.value for r in labelled)
    return PeriodStats(
        reviews=len(rows),
        classified=len(labelled),
        by_store=dict(Counter(store.value for _, store in rows)),
        avg_rating=round(sum(rating for rating, _ in rows) / len(rows), 2) if rows else None,
        sentiment_share={
            s.value: round(sentiments[s.value] / len(labelled), 4) if labelled else 0.0
            for s in Sentiment
        },
        bug_reports=sum(r.is_bug_report for r in labelled),
    )


def period_summary(
    session: Session,
    slug: str,
    *,
    model: str,
    prompt_version: str,
    days: int = 7,
    now: datetime | None = None,
) -> PeriodSummary:
    end = now or datetime.now(UTC)
    start = end - timedelta(days=days)
    classified = classified_reviews(session, slug, model=model, prompt_version=prompt_version)
    return PeriodSummary(
        app=slug,
        period_start=start,
        period_end=end,
        current=_period_stats(session, slug, classified, start, end),
        previous=_period_stats(session, slug, classified, start - timedelta(days=days), start),
    )


class CategoryCount(BaseModel):
    category: str
    reviews: int
    share: float


class CategoryBreakdown(BaseModel):
    app: str
    period_start: datetime
    period_end: datetime
    classified: int
    categories: list[CategoryCount]  # by primary category, largest first


def category_breakdown(
    session: Session,
    slug: str,
    *,
    model: str,
    prompt_version: str,
    days: int = 30,
    now: datetime | None = None,
) -> CategoryBreakdown:
    end = now or datetime.now(UTC)
    start = end - timedelta(days=days)
    reviews = _in_window(
        classified_reviews(session, slug, model=model, prompt_version=prompt_version), start, end
    )
    counts = Counter(r.primary_category for r in reviews)
    return CategoryBreakdown(
        app=slug,
        period_start=start,
        period_end=end,
        classified=len(reviews),
        categories=[
            CategoryCount(category=c, reviews=n, share=round(n / len(reviews), 4))
            for c, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    )
