from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from review_intel.domain import Store
from review_intel.rag.spikes import SpikeSettings, detect_spikes, two_proportion_z

from .fake_embedder import FakeEmbedder
from .rag_data import MODEL, PROMPT, SLUG, Seeder


def test_two_proportion_z() -> None:
    # 2/20 before, 12/20 after: pooled p = 0.35, se = 0.1508, diff = 0.5
    assert two_proportion_z(2, 20, 12, 20) == pytest.approx(3.3150, abs=1e-3)
    assert two_proportion_z(0, 0, 1, 1) == 0.0
    assert two_proportion_z(0, 10, 0, 10) == 0.0  # no variance


def _seed_versions(
    seed: Seeder,
    store: Store,
    ads_before: int,
    ads_after: int,
    after_categories: tuple[str, ...] = ("ads", "monetization_iap"),
) -> None:
    for i in range(20):
        ads = i < ads_before
        seed.review(
            "too many ads" if ads else "great game",
            store=store,
            version="1.0",
            categories=["ads"] if ads else ["praise"],
            day=i,
        )
    for i in range(20):
        ads = i < ads_after
        seed.review(
            "an ad after every single match" if ads else "love the new map",
            store=store,
            version="1.1",
            categories=list(after_categories) if ads else ["praise"],
            severity=5 if i == 3 else 3,
            day=30 + i,
        )


def test_flags_a_jump_and_links_the_patch_note(session: Session, apps_config_path: Path) -> None:
    seed = Seeder(session, apps_config_path)
    _seed_versions(seed, Store.GOOGLE_PLAY, ads_before=2, ads_after=12)
    _seed_versions(seed, Store.APP_STORE, 5, 5, after_categories=("ads",))  # flat: no spike
    seed.note("1.1", "New Map: Station\nMore ad rewards after every match\nBug fixes", day=29)

    spikes = detect_spikes(
        session, SLUG, model=MODEL, prompt_version=PROMPT, embedder=FakeEmbedder()
    )

    # Both categories of the new complaints jump on Google Play; App Store stays flat.
    assert {(s.store, s.category, s.version) for s in spikes} == {
        (Store.GOOGLE_PLAY, "ads", "1.1"),
        (Store.GOOGLE_PLAY, "monetization_iap", "1.1"),
    }
    spike = next(s for s in spikes if s.category == "ads")
    assert (spike.before_count, spike.before_n, spike.after_count, spike.after_n) == (2, 20, 12, 20)
    assert spike.before_share == pytest.approx(0.1) and spike.after_share == pytest.approx(0.6)
    assert spike.baseline_versions == ["1.0"]
    assert spike.note_version == "1.1"
    assert spike.note_lines[0][0] == "More ad rewards after every match"
    assert len(spike.example_review_ids) == 3


def test_praise_is_ignored_and_small_versions_are_skipped(
    session: Session, apps_config_path: Path
) -> None:
    seed = Seeder(session, apps_config_path)
    _seed_versions(seed, Store.GOOGLE_PLAY, ads_before=2, ads_after=12)

    strict = SpikeSettings(min_reviews=25)
    assert detect_spikes(session, SLUG, model=MODEL, prompt_version=PROMPT, settings=strict) == []

    spikes = detect_spikes(session, SLUG, model=MODEL, prompt_version=PROMPT)
    assert {s.category for s in spikes} == {"ads", "monetization_iap"}  # no praise


def test_without_embedder_the_note_is_linked_but_lines_are_not_ranked(
    session: Session, apps_config_path: Path
) -> None:
    seed = Seeder(session, apps_config_path)
    _seed_versions(seed, Store.GOOGLE_PLAY, ads_before=2, ads_after=12)
    seed.note("1.0", "Launch", day=0)

    spike = detect_spikes(session, SLUG, model=MODEL, prompt_version=PROMPT)[0]

    assert spike.note_version == "1.0"  # the note in force for 1.1
    assert spike.note_lines == []
