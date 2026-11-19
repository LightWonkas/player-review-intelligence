from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict

import pytest
from sqlalchemy.orm import Session

from review_intel.db.models import Review
from review_intel.domain import Store
from review_intel.insights import (
    InsightError,
    compare_versions,
    review_detail,
    search_reviews,
    sentiment_trend,
    top_issues,
)
from review_intel.rag.index import index_game

from .fake_embedder import FakeEmbedder
from .rag_data import MODEL, PROMPT, SLUG, Seeder


class Scope(TypedDict):
    model: str
    prompt_version: str


SCOPE: Scope = {"model": MODEL, "prompt_version": PROMPT}
# Seeder dates are 2026-10-01 + day; "now" is the end of day 14.
NOW = datetime(2026, 10, 15, tzinfo=UTC)


@pytest.fixture
def seed(session: Session, apps_config_path: Path) -> Seeder:
    seed = Seeder(session, apps_config_path)
    # previous week (days 1-7)
    seed.review("ads ads ads", categories=["ads"], day=2, version="1.0")
    seed.review("crash on start", categories=["bug_crash"], day=3, version="1.0", severity=5)
    seed.review("great", categories=["praise"], rating=5, day=4, version="1.0")
    # this week (days 8-14)
    for day in (8, 9, 10):
        seed.review("an ad after every match", categories=["ads"], day=day, version="1.1")
    seed.review(
        "lost my purchase after the crash",
        categories=["bug_crash", "monetization_iap"],
        severity=5,
        rating=1,
        day=11,
        version="1.1",
    )
    seed.review("love it", categories=["praise"], rating=5, day=12, version="1.1")
    seed.review("not classified yet", day=13, version="1.1")
    seed.note("1.1", "Lower ad frequency\nShop fixes", day=7)
    return seed


def test_top_issues_counts_this_week_against_last_week(seed: Seeder) -> None:
    result = top_issues(seed.session, SLUG, days=7, now=NOW, **SCOPE)

    assert (result.classified_reviews, result.unclassified_reviews) == (5, 1)
    assert [(i.category, i.reviews, i.previous_reviews) for i in result.issues] == [
        ("ads", 3, 1),
        ("bug_crash", 1, 1),
    ]  # praise is not an issue
    ads = result.issues[0]
    assert ads.share == pytest.approx(0.6)
    assert ads.negative == 3
    assert len(ads.examples) == 2
    assert result.latest_review_date is not None


def test_top_issues_limit(seed: Seeder) -> None:
    result = top_issues(seed.session, SLUG, days=7, limit=1, now=NOW, **SCOPE)

    assert [i.category for i in result.issues] == ["ads"]


def test_compare_versions(seed: Seeder) -> None:
    result = compare_versions(seed.session, SLUG, "1.0", "1.1", **SCOPE)

    assert (result.a.classified_reviews, result.b.classified_reviews) == (3, 5)
    assert result.b.patch_note is not None and result.b.patch_note.version == "1.1"
    assert result.a.patch_note is None
    ads = next(c for c in result.categories if c.category == "ads")
    assert (ads.count_a, ads.count_b) == (1, 3)
    assert ads.change == pytest.approx(3 / 5 - 1 / 3, abs=1e-4)
    assert "monetization_iap" in result.examples_b  # mentioned as a secondary category
    assert result.b.by_store == {"google_play": 5}


def test_compare_unknown_version_lists_known_ones(seed: Seeder) -> None:
    with pytest.raises(InsightError, match=r"9\.9.*1\.1"):
        compare_versions(seed.session, SLUG, "1.0", "9.9", **SCOPE)


def test_weekly_sentiment_trend_keeps_empty_buckets(seed: Seeder) -> None:
    result = sentiment_trend(seed.session, SLUG, days=21, bucket="week", now=NOW, **SCOPE)

    starts = [p.start for p in result.points]
    assert starts == [date(2026, 9, 21), date(2026, 9, 28), date(2026, 10, 5), date(2026, 10, 12)]
    # Oct 3-4 fall in the week of Sep 28; Oct 5 is a Monday and opens the next week.
    assert [p.reviews for p in result.points] == [0, 2, 4, 2]
    assert result.points[0].avg_rating is None
    assert result.points[-1].negative == 2  # the seeder labels every review negative


def test_search_reviews_with_filters(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder())

    plain = search_reviews(seed.session, SLUG, "crash", FakeEmbedder(), limit=2, **SCOPE)
    money = search_reviews(
        seed.session, SLUG, "crash", FakeEmbedder(), category="monetization_iap", **SCOPE
    )
    old = search_reviews(seed.session, SLUG, "ads", FakeEmbedder(), version="1.0", **SCOPE)

    assert len(plain.results) == 2
    assert "crash" in plain.results[0].text
    assert [r.text for r in money.results] == ["lost my purchase after the crash"]
    assert {r.version for r in old.results} == {"1.0"}
    assert money.results[0].primary_category == "bug_crash"


def test_review_detail(seed: Seeder) -> None:
    review = seed.review("needs detail", categories=["other"], store=Store.APP_STORE)

    detail = review_detail(seed.session, SLUG, review.id, **SCOPE)
    missing = review_detail(seed.session, SLUG, 99999, **SCOPE)

    assert detail is not None and detail.store is Store.APP_STORE
    assert detail.classification is not None and detail.classification.primary_category == "other"
    assert missing is None


def test_search_skips_reviews_deleted_since_indexing(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder())
    gone = search_reviews(seed.session, SLUG, "crash", FakeEmbedder(), limit=1, **SCOPE).results[0]
    review = seed.session.get(Review, gone.review_id)
    assert review is not None
    seed.session.delete(review)
    seed.session.flush()  # deleted, but `index` hasn't run again yet

    result = search_reviews(seed.session, SLUG, "crash", FakeEmbedder(), limit=10, **SCOPE)

    assert gone.review_id not in [hit.review_id for hit in result.results]
    assert result.results  # the other reviews still come back
