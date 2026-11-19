"""The vector index: review and patch-note-line embeddings in sqlite-vec tables.

The vectors live in the same SQLite file as the reviews, so search results join
straight back to SQL rows (ratings, versions, classifications) without a second
store to keep in sync. `game` is a vec0 metadata column, so a search for one game
never ranks another game's reviews.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import sqlite_vec
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from review_intel.db.models import App, EmbeddedItem, PatchNote, PatchNoteChunk, Review
from review_intel.rag.embedder import Embedder

REVIEW = "review"
PATCH_CHUNK = "patch_chunk"
# kind -> (vec0 table, its id column)
TABLES = {REVIEW: ("vec_reviews", "review_id"), PATCH_CHUNK: ("vec_patch_chunks", "chunk_id")}
MAX_K = 4096  # sqlite-vec's limit for one KNN query
BATCH = 256


class IndexMissingError(Exception):
    """Search ran before `review-intel index` built the tables."""


@dataclass(frozen=True)
class IndexResult:
    reviews_embedded: int
    chunks_embedded: int
    unchanged: int
    removed: int


def review_text(title: str | None, body: str) -> str:
    return f"{title}\n{body}" if title else body


def _hash(model: str, content: str) -> str:
    return hashlib.sha256(f"{model}\x00{content}".encode()).hexdigest()


def _table_dimension(session: Session, table: str) -> int | None:
    sql = session.execute(
        text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :name"),
        {"name": table},
    ).scalar()
    if sql is None:
        return None
    match = re.search(r"float\[(\d+)\]", sql, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _ensure_table(session: Session, kind: str, dim: int) -> None:
    """Create the vec0 table, or rebuild it when the embedding size changed (new model)."""
    table, id_column = TABLES[kind]
    current = _table_dimension(session, table)
    if current == dim:
        return
    if current is not None:
        session.execute(text(f"DROP TABLE {table}"))
        session.execute(delete(EmbeddedItem).where(EmbeddedItem.kind == kind))
    session.execute(
        text(
            f"CREATE VIRTUAL TABLE {table} USING vec0("
            f"{id_column} INTEGER PRIMARY KEY, game TEXT, "
            f"embedding FLOAT[{dim}] distance_metric=cosine)"
        )
    )


def _write(
    session: Session,
    kind: str,
    game: str,
    items: Sequence[tuple[int, str]],
    embedder: Embedder,
) -> int:
    table, id_column = TABLES[kind]
    written = 0
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        vectors = embedder.embed([content for _, content in batch])
        _ensure_table(session, kind, len(vectors[0]))
        for (item_id, content), vector in zip(batch, vectors, strict=True):
            session.execute(text(f"DELETE FROM {table} WHERE {id_column} = :id"), {"id": item_id})
            session.execute(
                text(f"INSERT INTO {table}({id_column}, game, embedding) VALUES (:id, :game, :v)"),
                {"id": item_id, "game": game, "v": sqlite_vec.serialize_float32(vector)},
            )
            record = session.scalar(
                select(EmbeddedItem).where(
                    EmbeddedItem.kind == kind, EmbeddedItem.item_id == item_id
                )
            )
            digest = _hash(embedder.name, content)
            if record is None:
                session.add(EmbeddedItem(kind=kind, item_id=item_id, text_hash=digest))
            else:
                record.text_hash = digest
        written += len(batch)
    return written


def _remove_stale(session: Session) -> int:
    """Drop vectors whose source row is gone (e.g. a patch note was re-split into lines)."""
    removed = 0
    sources = {REVIEW: select(Review.id), PATCH_CHUNK: select(PatchNoteChunk.id)}
    for kind, live_ids in sources.items():
        stale = list(
            session.scalars(
                select(EmbeddedItem.item_id).where(
                    EmbeddedItem.kind == kind, EmbeddedItem.item_id.not_in(live_ids)
                )
            )
        )
        if not stale:
            continue
        table, id_column = TABLES[kind]
        if _table_dimension(session, table) is not None:
            for item_id in stale:
                session.execute(
                    text(f"DELETE FROM {table} WHERE {id_column} = :id"), {"id": item_id}
                )
        session.execute(
            delete(EmbeddedItem).where(EmbeddedItem.kind == kind, EmbeddedItem.item_id.in_(stale))
        )
        removed += len(stale)
    return removed


def index_game(session: Session, slug: str, embedder: Embedder) -> IndexResult:
    """Embed every review and patch-note line of a game that is new or changed."""
    removed = _remove_stale(session)
    reviews = [
        (review_id, review_text(title, body))
        for review_id, title, body in session.execute(
            select(Review.id, Review.title, Review.body)
            .join(App, Review.app_id == App.id)
            .where(App.slug == slug)
            .order_by(Review.id)
        )
    ]
    chunks = [
        (chunk_id, content)
        for chunk_id, content in session.execute(
            select(PatchNoteChunk.id, PatchNoteChunk.text)
            .join(PatchNote, PatchNoteChunk.patch_note_id == PatchNote.id)
            .join(App, PatchNote.app_id == App.id)
            .where(App.slug == slug)
            .order_by(PatchNoteChunk.id)
        )
    ]
    done = {
        (kind, item_id): digest
        for kind, item_id, digest in session.execute(
            select(EmbeddedItem.kind, EmbeddedItem.item_id, EmbeddedItem.text_hash)
        )
    }

    def todo(kind: str, items: list[tuple[int, str]]) -> list[tuple[int, str]]:
        return [
            (item_id, content)
            for item_id, content in items
            if done.get((kind, item_id)) != _hash(embedder.name, content)
        ]

    new_reviews, new_chunks = todo(REVIEW, reviews), todo(PATCH_CHUNK, chunks)
    unchanged = len(reviews) + len(chunks) - len(new_reviews) - len(new_chunks)
    embedded_reviews = _write(session, REVIEW, slug, new_reviews, embedder)
    embedded_chunks = _write(session, PATCH_CHUNK, slug, new_chunks, embedder)
    session.flush()
    return IndexResult(embedded_reviews, embedded_chunks, unchanged, removed)


def search(
    session: Session,
    kind: str,
    slug: str,
    vector: Sequence[float],
    k: int,
    *,
    only: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Nearest items as (id, cosine distance), closest first.

    `only` restricts results to given ids (e.g. reviews of one category). vec0 can't
    join, so we over-fetch and filter, which is fine at this data size.
    """
    table, id_column = TABLES[kind]
    dim = _table_dimension(session, table)
    if dim is None:
        raise IndexMissingError("No vector index yet. Run `review-intel index --app <slug>` first.")
    if len(vector) != dim:
        raise IndexMissingError(
            f"Index has {dim}-dim vectors but the query has {len(vector)}; re-run `index`."
        )
    fetch = k if only is None else min(MAX_K, max(k * 20, 200))
    rows = session.execute(
        text(
            f"SELECT {id_column}, distance FROM {table} "
            "WHERE embedding MATCH :v AND k = :k AND game = :game ORDER BY distance"
        ),
        {"v": sqlite_vec.serialize_float32(list(vector)), "k": min(fetch, MAX_K), "game": slug},
    ).all()
    hits = [(int(item_id), float(distance)) for item_id, distance in rows]
    if only is not None:
        hits = [hit for hit in hits if hit[0] in only]
    return hits[:k]
