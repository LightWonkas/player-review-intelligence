"""The labeled eval set: how it is sampled, split and stored.

Design notes:
- Each item carries the review text itself, so the eval runs from a fresh clone
  without the database, and results stay reproducible after reviews are re-fetched.
- Sampling never looks at model predictions; otherwise the eval would partly grade
  the model against its own choices.
- The dev/test split is a hash of the review key, so it never changes between runs.
  Tune prompts on `dev`; report `test`.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.db.models import App, Review
from review_intel.domain import Sentiment, Store
from review_intel.llm.prompts import ReviewInput

DEFAULT_LABELS_PATH = Path("evals/labeled_reviews.jsonl")
Split = Literal["dev", "test"]
TEST_SHARE_PERCENT = 30

# Sampling mix: some reviews at random (the real distribution, mostly praise) and
# more from low-rated reviews with real text, where the hard categories live.
RANDOM_SHARE = 0.4
COMPLAINT_MAX_RATING = 3
COMPLAINT_MIN_CHARS = 20
MIN_APP_STORE_SHARE = 0.25


def review_key(store: Store, store_review_id: str) -> str:
    """Stable across databases, unlike the autoincrement review id."""
    return f"{store.value}:{store_review_id}"


def split_for(key: str) -> Split:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < TEST_SHARE_PERCENT else "dev"


class LabeledReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    store: Store
    rating: int = Field(ge=1, le=5)
    title: str | None
    body: str
    app_version: str | None
    primary_category: str
    categories: list[str] = Field(min_length=1)
    sentiment: Sentiment
    split: Split
    labeled_at: datetime
    # Who set the label: "human", or a model id when a stronger model labeled blind.
    labeler: str = "human"

    def to_input(self, position: int) -> ReviewInput:
        return ReviewInput(
            review_id=position,
            store=self.store,
            rating=self.rating,
            title=self.title,
            body=self.body,
            app_version=self.app_version,
        )


@dataclass(frozen=True)
class Candidate:
    """A stored review offered to the labeler."""

    key: str
    store: Store
    rating: int
    title: str | None
    body: str
    app_version: str | None


def load_labels(path: Path = DEFAULT_LABELS_PATH) -> list[LabeledReview]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LabeledReview.model_validate_json(line) for line in lines if line.strip()]


def append_label(label: LabeledReview, path: Path = DEFAULT_LABELS_PATH) -> None:
    """Append one line and flush, so quitting mid-session loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.model_dump(mode="json"), ensure_ascii=False) + "\n")


def make_label(
    candidate: Candidate,
    primary: str,
    extra: list[str],
    sentiment: Sentiment,
    labeler: str = "human",
) -> LabeledReview:
    return LabeledReview(
        key=candidate.key,
        store=candidate.store,
        rating=candidate.rating,
        title=candidate.title,
        body=candidate.body,
        app_version=candidate.app_version,
        primary_category=primary,
        categories=list(dict.fromkeys([primary, *extra])),
        sentiment=sentiment,
        split=split_for(candidate.key),
        labeled_at=datetime.now(UTC),
        labeler=labeler,
    )


def _is_complaint(candidate: Candidate) -> bool:
    return (
        candidate.rating <= COMPLAINT_MAX_RATING
        and len(candidate.body.strip()) >= COMPLAINT_MIN_CHARS
    )


def sample_candidates(
    session: Session, slug: str, n: int, *, seed: int = 7, exclude: set[str] | None = None
) -> list[Candidate]:
    """Deterministic sample of `n` reviews: ~40% random, ~60% complaints, >=25% App Store."""
    rows = session.execute(
        select(Review, App).join(App, Review.app_id == App.id).where(App.slug == slug)
    ).all()
    everyone = sorted(
        (
            Candidate(
                key=review_key(app.store, review.store_review_id),
                store=app.store,
                rating=review.rating,
                title=review.title,
                body=review.body,
                app_version=review.app_version,
            )
            for review, app in rows
        ),
        key=lambda c: c.key,
    )
    rng = random.Random(seed)
    rng.shuffle(everyone)
    complaints = [c for c in everyone if _is_complaint(c)]

    picked: dict[str, Candidate] = {}
    skip = exclude or set()

    def take(pool: list[Candidate], count: int, store: Store | None = None) -> None:
        for candidate in pool:
            if count <= 0:
                return
            if candidate.key in picked or (store is not None and candidate.store is not store):
                continue
            picked[candidate.key] = candidate
            count -= 1

    n_complaints = n - round(n * RANDOM_SHARE)
    n_random = n - n_complaints
    take(complaints, round(n_complaints * MIN_APP_STORE_SHARE), Store.APP_STORE)
    take(complaints, n_complaints - len(picked))
    before_random = len(picked)
    take(everyone, round(n_random * MIN_APP_STORE_SHARE), Store.APP_STORE)
    take(everyone, n_random - (len(picked) - before_random))
    take(everyone, n - len(picked))  # top up if a pool ran short

    ordered = list(picked.values())
    rng.shuffle(ordered)  # mix complaints and random picks while labeling
    return [c for c in ordered if c.key not in skip]
