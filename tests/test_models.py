from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from review_intel.db.models import App, Classification, Review
from review_intel.domain import Sentiment, Store

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _app(session: Session, store: Store = Store.GOOGLE_PLAY) -> App:
    app = App(
        slug="game", name="Game", store=store, store_app_id="com.x", country="us", language="en"
    )
    session.add(app)
    session.flush()
    return app


def _review(app: App, store_review_id: str = "r1", rating: int = 3) -> Review:
    return Review(
        app=app, store_review_id=store_review_id, rating=rating, body="ok", review_date=NOW
    )


def _classification(review: Review, prompt_version: str = "v1") -> Classification:
    return Classification(
        review=review,
        categories=["bug_crash"],
        primary_category="bug_crash",
        sentiment=Sentiment.NEGATIVE,
        severity=4,
        is_bug_report=True,
        short_summary="Crashes on launch",
        model="claude-haiku-5-5",
        prompt_version=prompt_version,
    )


def test_duplicate_store_review_id_for_same_app_is_rejected(session: Session) -> None:
    app = _app(session)
    session.add_all([_review(app, "r1"), _review(app, "r1")])

    with pytest.raises(IntegrityError):
        session.flush()


def test_same_store_review_id_on_different_apps_is_allowed(session: Session) -> None:
    android, ios = _app(session, Store.GOOGLE_PLAY), _app(session, Store.APP_STORE)
    session.add_all([_review(android, "r1"), _review(ios, "r1")])
    session.flush()

    assert session.scalar(select(func.count()).select_from(Review)) == 2


@pytest.mark.parametrize("rating", [0, 6])
def test_rating_outside_1_to_5_is_rejected(session: Session, rating: int) -> None:
    session.add(_review(_app(session), rating=rating))

    with pytest.raises(IntegrityError):
        session.flush()


def test_datetimes_round_trip_as_aware_utc(session: Session) -> None:
    istanbul = timezone(timedelta(hours=3))
    review = _review(_app(session))
    review.review_date = datetime(2026, 11, 1, 15, 0, tzinfo=istanbul)
    session.add(review)
    session.commit()
    session.expire_all()

    loaded = session.scalars(select(Review)).one()

    assert loaded.review_date == NOW
    assert loaded.review_date.tzinfo is UTC
    assert loaded.fetched_at.tzinfo is UTC


def test_naive_datetime_is_rejected(session: Session) -> None:
    review = _review(_app(session))
    review.review_date = datetime(2026, 11, 1, 12, 0)
    session.add(review)

    with pytest.raises(StatementError, match="naive datetime"):
        session.flush()


def test_one_classification_per_review_and_prompt_version(session: Session) -> None:
    review = _review(_app(session))
    session.add_all([_classification(review, "v1"), _classification(review, "v2")])
    session.flush()

    session.add(_classification(review, "v1"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_app_cascades_to_reviews_and_classifications(session: Session) -> None:
    app = _app(session)
    session.add(_classification(_review(app)))
    session.commit()

    session.delete(app)
    session.commit()

    assert session.scalar(select(func.count()).select_from(Review)) == 0
    assert session.scalar(select(func.count()).select_from(Classification)) == 0


def test_enums_are_stored_as_their_values(session: Session) -> None:
    _app(session, Store.APP_STORE)
    session.commit()

    raw = session.connection().exec_driver_sql("SELECT store FROM apps").scalar_one()

    assert raw == "app_store"
