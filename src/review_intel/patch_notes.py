"""Patch notes: where they come from, how they are stored and split into lines.

Sources:
- The App Store page lists the last ~25 versions with release date and notes. It is
  the only public history with dates, so it is fetched (one request per run).
- A markdown file, for notes entered by hand (e.g. Google Play, which only shows
  the current version's "What's new").

Version numbers are shared across stores, so notes are matched to reviews from
either store by version.
"""

import html
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from review_intel.collectors.app_store import USER_AGENT, is_retryable_http_error
from review_intel.db.models import App, PatchNote, PatchNoteChunk
from review_intel.retry import DEFAULT_RETRY, RetryPolicy, with_retries

APP_PAGE_URL = "https://apps.apple.com/{country}/app/id{app_id}"

# One version-history entry: the notes, then the version and an ISO date. The notes
# group must not run past another entry's start, hence the tempered dot.
_ENTRY = re.compile(
    r"<!-- HTML_TAG_START -->((?:(?!HTML_TAG_START).)*?)<!-- HTML_TAG_END -->\s*</span></p>\s*"
    r'<div class="metadata[^"]*"><span[^>]*>([\w.\-]+)</span>\s*'
    r'<time datetime="(\d{4}-\d{2}-\d{2})"',
    re.DOTALL,
)
_MARKDOWN_HEADING = re.compile(
    r"^##\s+v?(?P<version>[\w.\-]+)\s*[(\-]\s*(?P<date>\d{4}-\d{2}-\d{2})\)?\s*$"
)
# Lines that carry no information about the release itself.
_NOISE = re.compile(r"https?://|^(facebook|discord|twitter|instagram|youtube)\b", re.IGNORECASE)


class EmptyPageError(Exception):
    """The store served a page without version history (it sometimes serves a bare shell)."""


@dataclass(frozen=True)
class ReleaseNote:
    version: str
    release_date: date
    text: str


def clean_text(text: str) -> str:
    lines = (line.strip() for line in html.unescape(text).splitlines())
    return "\n".join(line for line in lines if line and not _NOISE.search(line))


def chunk_lines(text: str) -> list[str]:
    """Split a note into lines, dropping bullets and duplicates."""
    seen: dict[str, None] = {}
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*• ").strip()
        if stripped:
            seen.setdefault(stripped, None)
    return list(seen)


def parse_app_store_history(page: str) -> list[ReleaseNote]:
    """Newest first, one entry per version (the page can repeat a section)."""
    notes: dict[str, ReleaseNote] = {}
    for raw, version, day in _ENTRY.findall(page):
        if version not in notes:
            notes[version] = ReleaseNote(version, date.fromisoformat(day), clean_text(raw))
    return list(notes.values())


def fetch_app_store_history(
    app_id: str,
    country: str,
    *,
    client: httpx.Client | None = None,
    retry_policy: RetryPolicy = DEFAULT_RETRY,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ReleaseNote]:
    owned = client is None
    http = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
    )

    def get() -> list[ReleaseNote]:
        response = http.get(APP_PAGE_URL.format(country=country, app_id=app_id))
        response.raise_for_status()
        notes = parse_app_store_history(response.text)
        if not notes:
            raise EmptyPageError("App Store page had no version history")
        return notes

    try:
        return with_retries(
            get,
            is_retryable=lambda exc: (
                isinstance(exc, EmptyPageError) or is_retryable_http_error(exc)
            ),
            policy=retry_policy,
            sleep=sleep,
            description="App Store version history",
        )
    finally:
        if owned:
            http.close()


def parse_markdown(text: str) -> list[ReleaseNote]:
    """Read notes written as `## 15.008 (2026-10-16)` headings followed by the text."""
    notes: list[ReleaseNote] = []
    current: tuple[str, date] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            notes.append(ReleaseNote(current[0], current[1], clean_text("\n".join(body))))

    for line in text.splitlines():
        match = _MARKDOWN_HEADING.match(line.strip())
        if match:
            flush()
            current = (match["version"], date.fromisoformat(match["date"]))
            body = []
        elif current is not None:
            body.append(line)
    flush()
    return notes


def load_markdown(path: Path) -> list[ReleaseNote]:
    return parse_markdown(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class NotesResult:
    created: int
    updated: int
    unchanged: int


def store_notes(session: Session, app: App, notes: Iterable[ReleaseNote]) -> NotesResult:
    """Upsert by (app, version) and keep the line chunks in step with the text."""
    existing = {
        note.version: note
        for note in session.scalars(select(PatchNote).where(PatchNote.app_id == app.id))
    }
    created = updated = unchanged = 0
    for note in notes:
        row = existing.get(note.version)
        if row is None:
            row = PatchNote(
                app_id=app.id, version=note.version, release_date=note.release_date, text=""
            )
            session.add(row)
            created += 1
        elif (row.release_date, row.text) == (note.release_date, note.text):
            unchanged += 1
            continue
        else:
            updated += 1
        row.release_date = note.release_date
        row.text = note.text
        session.flush()
        session.execute(delete(PatchNoteChunk).where(PatchNoteChunk.patch_note_id == row.id))
        session.add_all(
            PatchNoteChunk(patch_note_id=row.id, position=i, text=line)
            for i, line in enumerate(chunk_lines(note.text))
        )
        existing[note.version] = row
    session.flush()
    return NotesResult(created, updated, unchanged)


def version_key(version: str) -> tuple[int, ...]:
    """Order "14.9501" < "15.004" < "15.052". Non-numeric parts sort as 0."""
    return tuple(int(part) if part.isdigit() else 0 for part in re.split(r"[.\-]", version))


def notes_for_game(session: Session, slug: str) -> list[PatchNote]:
    """One note per version across the game's stores (the earliest release wins), oldest first."""
    rows = session.scalars(
        select(PatchNote)
        .join(App, PatchNote.app_id == App.id)
        .where(App.slug == slug)
        .order_by(PatchNote.release_date)
    )
    by_version: dict[str, PatchNote] = {}
    for note in rows:
        by_version.setdefault(note.version, note)
    return sorted(by_version.values(), key=lambda note: version_key(note.version))


def note_in_force(notes: list[PatchNote], version: str) -> PatchNote | None:
    """The note for `version`, or else the latest earlier one (a hotfix ships under the
    previous release's notes)."""
    key = version_key(version)
    candidates = [note for note in notes if version_key(note.version) <= key]
    return candidates[-1] if candidates else None
