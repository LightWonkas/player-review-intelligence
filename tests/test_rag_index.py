from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.db.models import App, EmbeddedItem, Review
from review_intel.domain import Store
from review_intel.rag.index import (
    PATCH_CHUNK,
    REVIEW,
    IndexMissingError,
    index_game,
    search,
)

from .fake_embedder import FakeEmbedder
from .rag_data import SLUG, Seeder


@pytest.fixture
def seed(session: Session, apps_config_path: Path) -> Seeder:
    seed = Seeder(session, apps_config_path)
    seed.review("matchmaking puts me against level 200 players")
    seed.review("too many ads after every match")
    seed.review("the game crashes when I open the shop")
    seed.note("15.004", "New Weapon Blueprint: False Mercy\nLower ad frequency")
    return seed


def _query(session: Session, kind: str, text: str, k: int = 1, **kw: object) -> list[int]:
    vector = FakeEmbedder().embed([text])[0]
    return [item_id for item_id, _ in search(session, kind, SLUG, vector, k, **kw)]  # type: ignore[arg-type]


def test_index_then_search_finds_the_closest_review(seed: Seeder) -> None:
    result = index_game(seed.session, SLUG, FakeEmbedder())

    assert (result.reviews_embedded, result.chunks_embedded) == (3, 2)
    ads = seed.session.scalar(select(Review.id).where(Review.body.contains("ads")))
    assert _query(seed.session, REVIEW, "so many ads") == [ads]


def test_patch_lines_are_searchable(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder())

    (hit,) = _query(seed.session, PATCH_CHUNK, "the false mercy weapon")
    hits = search(seed.session, PATCH_CHUNK, SLUG, FakeEmbedder().embed(["mercy"])[0], 2)
    assert hit == hits[0][0]
    assert hits[0][1] < hits[1][1]  # cosine distance: closer first


def test_reindex_only_embeds_new_or_changed_text(seed: Seeder) -> None:
    embedder = FakeEmbedder()
    index_game(seed.session, SLUG, embedder)

    unchanged = index_game(seed.session, SLUG, embedder)
    review = seed.session.scalars(select(Review)).first()
    assert review is not None
    review.body = "edited review text"
    seed.review("a brand new review")
    changed = index_game(seed.session, SLUG, embedder)

    assert (unchanged.reviews_embedded, unchanged.unchanged) == (0, 5)
    assert changed.reviews_embedded == 2


def test_new_embedding_model_reembeds_everything(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder())

    result = index_game(seed.session, SLUG, FakeEmbedder(name="other-model"))

    assert (result.reviews_embedded, result.chunks_embedded) == (3, 2)


def test_new_dimension_rebuilds_the_table(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder(dim=64))

    index_game(seed.session, SLUG, FakeEmbedder(dim=32, name="small"))
    vector = FakeEmbedder(dim=32).embed(["ads"])[0]

    assert len(search(seed.session, REVIEW, SLUG, vector, 3)) == 3


def test_deleted_reviews_leave_the_index(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder())
    review = seed.session.scalars(select(Review)).first()
    assert review is not None
    seed.session.delete(review)
    seed.session.flush()

    result = index_game(seed.session, SLUG, FakeEmbedder())

    assert result.removed == 1
    assert (
        seed.session.scalar(
            select(EmbeddedItem).where(
                EmbeddedItem.kind == REVIEW, EmbeddedItem.item_id == review.id
            )
        )
        is None
    )
    assert review.id not in _query(seed.session, REVIEW, "anything", k=10)


def test_search_is_limited_to_one_game(seed: Seeder) -> None:
    other = App(
        slug="other-game",
        name="Other",
        store=Store.GOOGLE_PLAY,
        store_app_id="com.other.game",
        country="us",
        language="en",
    )
    seed.session.add(other)
    seed.session.flush()
    seed.session.add(
        Review(
            app_id=other.id,
            store_review_id="x",
            rating=1,
            title=None,
            body="too many ads after every match",
            app_version="1",
            review_date=seed.session.scalars(select(Review.review_date)).first(),
        )
    )
    seed.session.flush()
    index_game(seed.session, SLUG, FakeEmbedder())
    index_game(seed.session, "other-game", FakeEmbedder())

    ours = {r.id for r in seed.session.scalars(select(Review).where(Review.app_id != other.id))}
    assert set(_query(seed.session, REVIEW, "too many ads", k=10)) == ours


def test_only_filter_restricts_results(seed: Seeder) -> None:
    index_game(seed.session, SLUG, FakeEmbedder())
    crash = seed.session.scalar(select(Review.id).where(Review.body.contains("crashes")))
    assert crash is not None

    assert _query(seed.session, REVIEW, "too many ads", k=3, only={crash}) == [crash]


def test_search_before_index_explains_what_to_run(seed: Seeder) -> None:
    with pytest.raises(IndexMissingError, match="review-intel index"):
        _query(seed.session, REVIEW, "ads")
