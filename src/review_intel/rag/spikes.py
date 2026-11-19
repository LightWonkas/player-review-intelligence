"""Spike detector: categories whose share of reviews jumped in a new version.

Pure statistics, no LLM: for each store, a version's share of reviews mentioning a
category is compared with the pooled previous versions, using a two-proportion
z-test. Stores are kept apart because they differ in who writes reviews (Google
Play skews to short praise), so mixing them would create fake spikes whenever the
store mix shifts. A spike is linked to the patch note in force for that version,
and the note's lines are ranked by similarity to the complaints.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.db.models import PatchNoteChunk
from review_intel.domain import Store
from review_intel.patch_notes import note_in_force, notes_for_game
from review_intel.rag.embedder import Embedder
from review_intel.rag.index import review_text
from review_intel.rag.stats import (
    ClassifiedReview,
    VersionCounts,
    classified_reviews,
    counts_by_version,
)

DEFAULT_IGNORED = ("praise", "other")


@dataclass(frozen=True)
class SpikeSettings:
    min_reviews: int = 15  # per side (the version, and the pooled baseline)
    min_count: int = 5  # reviews in the category in the new version
    min_increase: float = 0.10  # share points, e.g. 12% -> 22%
    min_z: float = 2.0  # one-sided, roughly p < 0.025
    max_baseline_versions: int = 3
    ignored: tuple[str, ...] = DEFAULT_IGNORED


@dataclass(frozen=True)
class Spike:
    store: Store
    category: str
    version: str
    first_seen: date | None
    baseline_versions: list[str]
    before_count: int
    before_n: int
    after_count: int
    after_n: int
    z: float
    note_version: str | None = None
    note_date: date | None = None
    note_lines: list[tuple[str, float]] = field(default_factory=list)
    example_review_ids: list[int] = field(default_factory=list)

    @property
    def before_share(self) -> float:
        return self.before_count / self.before_n

    @property
    def after_share(self) -> float:
        return self.after_count / self.after_n


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """z for p2 > p1 with a pooled standard error. 0 when undefined."""
    if not n1 or not n2:
        return 0.0
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (x2 / n2 - x1 / n1) / se if se else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def _rank_note_lines(
    lines: list[str], complaints: list[ClassifiedReview], embedder: Embedder, top: int = 3
) -> list[tuple[str, float]]:
    """Which lines of the note look most like the complaints (cosine to their centroid)."""
    if not lines or not complaints:
        return []
    review_vectors = embedder.embed([review_text(r.title, r.body) for r in complaints[:30]])
    centroid = [sum(column) / len(review_vectors) for column in zip(*review_vectors, strict=True)]
    scored = [
        (line, _cosine(centroid, vector))
        for line, vector in zip(lines, embedder.embed(lines), strict=True)
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top]


def _store_spikes(
    reviews: list[ClassifiedReview], store: Store, settings: SpikeSettings
) -> list[Spike]:
    versions = counts_by_version(reviews)
    spikes = []
    for i, current in enumerate(versions):
        if current.n < settings.min_reviews:
            continue
        baseline: list[VersionCounts] = []
        before_n = 0
        for previous in reversed(versions[:i]):
            if len(baseline) == settings.max_baseline_versions or before_n >= settings.min_reviews:
                break
            baseline.append(previous)
            before_n += previous.n
        if before_n < settings.min_reviews:
            continue
        for category, after_count in current.categories.items():
            if category in settings.ignored or after_count < settings.min_count:
                continue
            before_count = sum(row.categories[category] for row in baseline)
            increase = after_count / current.n - before_count / before_n
            z = two_proportion_z(before_count, before_n, after_count, current.n)
            if increase >= settings.min_increase and z >= settings.min_z:
                spikes.append(
                    Spike(
                        store=store,
                        category=category,
                        version=current.version,
                        first_seen=current.first_seen,
                        baseline_versions=[row.version for row in reversed(baseline)],
                        before_count=before_count,
                        before_n=before_n,
                        after_count=after_count,
                        after_n=current.n,
                        z=z,
                    )
                )
    return spikes


def detect_spikes(
    session: Session,
    slug: str,
    *,
    model: str,
    prompt_version: str,
    settings: SpikeSettings | None = None,
    embedder: Embedder | None = None,
) -> list[Spike]:
    """Spikes for every store, strongest first, each linked to its patch note."""
    settings = settings or SpikeSettings()
    notes = notes_for_game(session, slug)
    found: list[Spike] = []
    for store in Store:
        reviews = classified_reviews(
            session, slug, model=model, prompt_version=prompt_version, store=store
        )
        for spike in _store_spikes(reviews, store, settings):
            complaints = sorted(
                (
                    r
                    for r in reviews
                    if r.version == spike.version and spike.category in r.categories
                ),
                key=lambda r: (-r.severity, r.rating, r.review_id),
            )
            note = note_in_force(notes, spike.version)
            lines: list[tuple[str, float]] = []
            if note is not None and embedder is not None:
                note_text = list(
                    session.scalars(
                        select(PatchNoteChunk.text)
                        .where(PatchNoteChunk.patch_note_id == note.id)
                        .order_by(PatchNoteChunk.position)
                    )
                )
                lines = _rank_note_lines(note_text, complaints, embedder)
            found.append(
                replace(
                    spike,
                    note_version=note.version if note else None,
                    note_date=note.release_date if note else None,
                    note_lines=lines,
                    example_review_ids=[r.review_id for r in complaints[:3]],
                )
            )
    return sorted(found, key=lambda spike: spike.z, reverse=True)
