from dataclasses import replace

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from review_intel.apps_config import AppsConfig
from review_intel.collectors.base import ReviewSource
from review_intel.db.apps import sync_apps
from review_intel.db.models import App, Review
from review_intel.db.session import make_session_factory
from review_intel.domain import Store
from review_intel.ingest import fetch_and_store, upsert_reviews

from .fakes import FakeSource, make_raw

CONFIG = AppsConfig.model_validate(
    {
        "games": [
            {
                "slug": "game",
                "name": "Game",
                "listings": [
                    {"store": "google_play", "store_app_id": "com.game"},
                    {"store": "app_store", "store_app_id": "123"},
                ],
            }
        ]
    }
)


def _registered_app(session: Session, store: Store = Store.GOOGLE_PLAY) -> App:
    sync_apps(session, CONFIG)
    return session.scalars(select(App).where(App.store == store)).one()


def _count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Review)) or 0


def test_first_run_inserts_second_run_adds_nothing(session: Session) -> None:
    app = _registered_app(session)
    batch = [make_raw("r1"), make_raw("r2"), make_raw("r3")]

    first = upsert_reviews(session, app, batch)
    second = upsert_reviews(session, app, batch)

    assert (first.inserted, first.updated, first.unchanged) == (3, 0, 0)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 3)
    assert _count(session) == 3


def test_edited_review_is_updated_in_place(session: Session) -> None:
    app = _registered_app(session)
    upsert_reviews(session, app, [make_raw("r1", rating=1, body="crashes")])

    result = upsert_reviews(session, app, [make_raw("r1", rating=4, body="fixed now")])

    assert (result.inserted, result.updated) == (0, 1)
    row = session.scalars(select(Review)).one()
    assert (row.rating, row.body) == (4, "fixed now")


def test_duplicates_within_one_batch_are_stored_once(session: Session) -> None:
    app = _registered_app(session)

    result = upsert_reviews(session, app, [make_raw("r1"), make_raw("r1")])

    assert result.inserted == 1
    assert _count(session) == 1


def test_same_review_id_on_other_store_is_a_different_review(session: Session) -> None:
    android = _registered_app(session, Store.GOOGLE_PLAY)
    ios = session.scalars(select(App).where(App.store == Store.APP_STORE)).one()

    upsert_reviews(session, android, [make_raw("r1")])
    result = upsert_reviews(session, ios, [make_raw("r1")])

    assert result.inserted == 1


def test_empty_batch_is_a_no_op(session: Session) -> None:
    result = upsert_reviews(session, _registered_app(session), [])

    assert (result.inserted, result.updated, result.unchanged) == (0, 0, 0)


def _prepare(engine: Engine) -> None:
    with make_session_factory(engine)() as session:
        sync_apps(session, CONFIG)
        session.commit()


def test_fetch_and_store_keeps_partial_results_when_a_store_fails(engine: Engine) -> None:
    _prepare(engine)
    sources: dict[Store, ReviewSource] = {
        Store.GOOGLE_PLAY: FakeSource(
            Store.GOOGLE_PLAY, [make_raw("a1"), make_raw("a2"), make_raw("a3")], fail_after=2
        ),
        Store.APP_STORE: FakeSource(Store.APP_STORE, [make_raw("i1")]),
    }

    results = fetch_and_store(
        make_session_factory(engine), CONFIG, "game", list(Store), 10, sources
    )

    by_store = {result.store: result for result in results}
    assert by_store[Store.GOOGLE_PLAY].upsert.inserted == 2
    error = by_store[Store.GOOGLE_PLAY].error
    assert error is not None
    assert "ConnectionError" in error
    assert by_store[Store.APP_STORE].upsert.inserted == 1
    assert by_store[Store.APP_STORE].error is None


def test_fetch_and_store_only_touches_requested_stores(engine: Engine) -> None:
    _prepare(engine)
    android = FakeSource(Store.GOOGLE_PLAY, [make_raw("a1")])
    ios = FakeSource(Store.APP_STORE, [replace(make_raw("i1"), title="t")])

    results = fetch_and_store(
        make_session_factory(engine),
        CONFIG,
        "game",
        [Store.APP_STORE],
        10,
        {Store.GOOGLE_PLAY: android, Store.APP_STORE: ios},
    )

    assert [result.store for result in results] == [Store.APP_STORE]
    assert android.calls == 0
