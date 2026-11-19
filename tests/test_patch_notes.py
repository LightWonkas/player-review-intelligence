from datetime import date
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.db.models import PatchNoteChunk
from review_intel.domain import Store
from review_intel.patch_notes import (
    EmptyPageError,
    ReleaseNote,
    chunk_lines,
    fetch_app_store_history,
    note_in_force,
    notes_for_game,
    parse_app_store_history,
    parse_markdown,
    store_notes,
    version_key,
)
from review_intel.retry import RetryPolicy

from .fakes import FIXTURES
from .rag_data import SLUG, Seeder

PAGE = (FIXTURES / "app_store_versions.html").read_text(encoding="utf-8")


def test_parses_recorded_version_history() -> None:
    notes = parse_app_store_history(PAGE)

    assert [(n.version, n.release_date) for n in notes] == [
        ("15.052", date(2026, 11, 2)),
        ("15.005", date(2026, 10, 6)),
        ("15.004", date(2026, 10, 5)),
    ]
    mercy = notes[2].text
    assert "New Weapon Blueprints: False Mercy, Redeemer & Divine Judgement" in mercy  # unescaped
    assert "http" not in mercy and "Discord" not in mercy  # social links dropped


def test_an_earlier_html_block_does_not_swallow_the_first_entry() -> None:
    # The real page has a long description block before the history; a greedy match
    # once glued 130 KB of it onto the newest version's notes.
    description = "<!-- HTML_TAG_START -->About this game...<!-- HTML_TAG_END --></span></p>"

    notes = parse_app_store_history(description + "<div>other</div>" + PAGE)

    assert notes[0].text.startswith("New Map: Manor")


def test_repeated_versions_are_kept_once() -> None:
    assert len(parse_app_store_history(PAGE + PAGE)) == 3


def test_chunk_lines_strips_bullets_and_duplicates() -> None:
    assert chunk_lines("- New map\n\n* New map\n• Bug fixes \n") == ["New map", "Bug fixes"]


def test_parse_markdown_sections() -> None:
    text = """# Notes kept by hand

## 15.008 (2026-10-16)
- Fixed a crash in the shop
- See https://example.com

## v15.010 - 2026-10-20
Lower ad frequency
"""
    notes = parse_markdown(text)

    assert notes == [
        ReleaseNote("15.008", date(2026, 10, 16), "- Fixed a crash in the shop"),
        ReleaseNote("15.010", date(2026, 10, 20), "Lower ad frequency"),
    ]


def test_fetch_retries_a_bare_shell_page_then_parses() -> None:
    pages = iter(["<html><body></body></html>", PAGE])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text=next(pages))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    notes = fetch_app_store_history("1467648713", "us", client=client, sleep=lambda _: None)

    assert len(notes) == 3
    assert seen == ["https://apps.apple.com/us/app/id1467648713"] * 2


def test_fetch_gives_up_when_the_page_never_has_history() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="")))

    with pytest.raises(EmptyPageError):
        fetch_app_store_history(
            "1", "us", client=client, retry_policy=RetryPolicy(attempts=2), sleep=lambda _: None
        )


def test_store_notes_upserts_and_keeps_chunks_in_step(
    session: Session, apps_config_path: Path
) -> None:
    app = Seeder(session, apps_config_path).apps[Store.APP_STORE]
    first = ReleaseNote("15.004", date(2026, 10, 5), "New map\nBug fixes")

    created = store_notes(session, app, [first])
    again = store_notes(session, app, [first])
    changed = store_notes(session, app, [ReleaseNote("15.004", date(2026, 10, 5), "New gun")])

    assert (created.created, again.unchanged, changed.updated) == (1, 1, 1)
    assert list(session.scalars(select(PatchNoteChunk.text))) == ["New gun"]


def test_version_order_and_note_in_force(session: Session, apps_config_path: Path) -> None:
    seed = Seeder(session, apps_config_path)
    seed.note("15.004", "Season start", day=4)
    seed.note("14.9501", "Old season", day=-20)
    seed.note("15.004", "Same version on Play", store=Store.GOOGLE_PLAY, day=6)

    notes = notes_for_game(session, SLUG)

    assert version_key("14.9501") < version_key("15.004") < version_key("15.052")
    assert [(n.version, n.text) for n in notes] == [
        ("14.9501", "Old season"),
        ("15.004", "Season start"),  # earliest release of a version wins
    ]
    hotfix = note_in_force(notes, "15.008")
    assert hotfix is not None and hotfix.version == "15.004"
    assert note_in_force(notes, "14.0") is None
