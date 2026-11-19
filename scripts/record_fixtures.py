"""Record small, real API responses as test fixtures (run by hand, never in CI).

    uv run python scripts/record_fixtures.py

Keeps 3 reviews per store and replaces author names and avatars with "REDACTED",
because fixtures are committed to a public repo. Tests then run fully offline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from google_play_scraper import Sort, reviews

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
PLAY_APP_ID = "com.critical.strike2"
APP_STORE_URL = (
    "https://itunes.apple.com/us/rss/customerreviews/page=1/id=1467648713/sortby=mostrecent/json"
)
KEEP = 3
REDACTED = "REDACTED"


def record_google_play() -> None:
    items, _ = reviews(PLAY_APP_ID, lang="en", country="us", sort=Sort.NEWEST, count=KEEP)
    cleaned: list[dict[str, Any]] = []
    for item in items:
        item = dict(item)
        item["userName"] = REDACTED
        item["userImage"] = REDACTED
        # The scraper returns naive local datetimes; store the epoch seconds it was built from.
        for key in ("at", "repliedAt"):
            value = item.get(key)
            if isinstance(value, datetime):
                item[key] = int(value.timestamp())
        cleaned.append(item)
    _write("google_play_reviews.json", cleaned)


def record_app_store() -> None:
    response = httpx.get(APP_STORE_URL, timeout=15.0, headers={"User-Agent": "review-intel/0.1"})
    response.raise_for_status()
    data = response.json()
    entries = data["feed"]["entry"][:KEEP]
    for entry in entries:
        entry["author"] = {"name": {"label": REDACTED}, "uri": {"label": REDACTED}}
    data["feed"]["entry"] = entries
    data["feed"]["author"] = {"name": {"label": "iTunes Store"}}
    _write("app_store_page.json", data)


def _write(name: str, payload: Any) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(FIXTURES.parents[1])}")


if __name__ == "__main__":
    record_google_play()
    record_app_store()
