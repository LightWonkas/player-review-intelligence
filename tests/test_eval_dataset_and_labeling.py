from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.apps_config import load_apps_config
from review_intel.db.apps import sync_apps
from review_intel.db.models import App, Review
from review_intel.domain import Sentiment, Store
from review_intel.evals.dataset import (
    Candidate,
    load_labels,
    make_label,
    sample_candidates,
    split_for,
)
from review_intel.evals.labeling import run_labeling
from review_intel.llm.taxonomy import load_taxonomy

from .conftest import REPO_ROOT

TAXONOMY = load_taxonomy(REPO_ROOT / "config" / "taxonomy.yaml")


@pytest.fixture
def seeded(session: Session, apps_config_path: Path) -> Session:
    sync_apps(session, load_apps_config(apps_config_path))
    apps = {app.store: app for app in session.scalars(select(App))}
    base = datetime(2026, 11, 1, tzinfo=UTC)
    for i in range(120):
        store = Store.APP_STORE if i % 4 == 0 else Store.GOOGLE_PLAY
        complaint = i % 2 == 0
        session.add(
            Review(
                app_id=apps[store].id,
                store_review_id=f"r{i}",
                rating=1 if complaint else 5,
                title=None,
                body="the game crashes all the time after update" if complaint else "nice",
                app_version="15.0",
                review_date=base + timedelta(minutes=i),
            )
        )
    session.flush()
    return session


def test_sampling_is_deterministic(seeded: Session) -> None:
    first = sample_candidates(seeded, "critical-strike", 20)
    second = sample_candidates(seeded, "critical-strike", 20)

    assert [c.key for c in first] == [c.key for c in second]
    assert len({c.key for c in first}) == 20


def test_sample_mixes_complaints_and_app_store(seeded: Session) -> None:
    sample = sample_candidates(seeded, "critical-strike", 40)

    complaints = sum(c.rating <= 3 for c in sample)
    assert complaints >= 24  # 60% complaints, plus any the random part picks up
    assert Counter(c.store for c in sample)[Store.APP_STORE] >= 10


def test_sampling_excludes_already_labeled(seeded: Session) -> None:
    full = sample_candidates(seeded, "critical-strike", 20)
    done = {full[0].key, full[1].key}

    rest = sample_candidates(seeded, "critical-strike", 20, exclude=done)

    assert [c.key for c in rest] == [c.key for c in full[2:]]


def test_split_is_stable_and_roughly_70_30() -> None:
    keys = [f"google_play:{i}" for i in range(1000)]
    splits = Counter(split_for(key) for key in keys)

    assert split_for("google_play:42") == split_for("google_play:42")
    assert 250 <= splits["test"] <= 350


CANDIDATE = Candidate(
    key="app_store:1", store=Store.APP_STORE, rating=1, title="t", body="b", app_version="1"
)


def _scripted(answers: list[str]) -> Callable[[str], str]:
    remaining: Iterator[str] = iter(answers)
    return lambda _prompt: next(remaining)


def test_labeling_saves_answers_and_reasks_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    said: list[str] = []
    # "99" is not a category and "x" not a sentiment: both are asked again.
    ask = _scripted(["99", "1", "6, 12", "x", "n"])

    result = run_labeling(
        [CANDIDATE], TAXONOMY, path, already_labeled=0, target=5, ask=ask, say=said.append
    )

    assert result.labeled == 1
    (label,) = load_labels(path)
    assert label.primary_category == "bug_crash"
    assert label.categories == ["bug_crash", "ads", "other"]
    assert label.sentiment is Sentiment.NEGATIVE
    assert label.split == split_for(CANDIDATE.key)
    assert said.count("  Not understood, try again.") == 2


def test_labeling_skip_and_quit(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    second = Candidate("app_store:2", Store.APP_STORE, 5, None, "great", "1")
    third = Candidate("app_store:3", Store.APP_STORE, 5, None, "great", "1")

    result = run_labeling(
        [CANDIDATE, second, third],
        TAXONOMY,
        path,
        already_labeled=0,
        target=5,
        ask=_scripted(["s", "praise", "", "p", "q"]),
        say=lambda _: None,
    )

    assert (result.labeled, result.skipped, result.quit) == (1, 1, True)
    assert [label.key for label in load_labels(path)] == ["app_store:2"]


def test_labeling_stops_at_target(tmp_path: Path) -> None:
    result = run_labeling(
        [CANDIDATE],
        TAXONOMY,
        tmp_path / "labels.jsonl",
        already_labeled=5,
        target=5,
        ask=_scripted([]),
        say=lambda _: None,
    )

    assert result.labeled == 0


def test_make_label_puts_primary_first_without_duplicates() -> None:
    label = make_label(CANDIDATE, "ads", ["ads", "praise"], Sentiment.NEUTRAL)

    assert label.categories == ["ads", "praise"]
