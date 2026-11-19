"""Seed helpers for RAG tests: reviews with classifications and patch notes."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.apps_config import load_apps_config
from review_intel.db.apps import sync_apps
from review_intel.db.models import App, Classification, Review
from review_intel.domain import Sentiment, Store
from review_intel.patch_notes import ReleaseNote, store_notes

SLUG = "critical-strike"
MODEL = "claude-haiku-5-5"
PROMPT = "v1-test"


class Seeder:
    def __init__(self, session: Session, apps_config_path: Path) -> None:
        sync_apps(session, load_apps_config(apps_config_path))
        session.flush()
        self.session = session
        self.apps = {app.store: app for app in session.scalars(select(App))}
        self._next = 0

    def review(
        self,
        body: str,
        *,
        store: Store = Store.GOOGLE_PLAY,
        version: str | None = "1.0",
        categories: list[str] | None = None,
        rating: int = 2,
        severity: int = 3,
        day: int = 0,
    ) -> Review:
        self._next += 1
        review = Review(
            app_id=self.apps[store].id,
            store_review_id=f"r{self._next}",
            rating=rating,
            title=None,
            body=body,
            app_version=version,
            review_date=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=day),
        )
        self.session.add(review)
        self.session.flush()
        if categories:
            self.session.add(
                Classification(
                    review_id=review.id,
                    categories=categories,
                    primary_category=categories[0],
                    sentiment=Sentiment.NEGATIVE,
                    severity=severity,
                    is_bug_report=False,
                    mentioned_features=[],
                    short_summary=body[:50],
                    model=MODEL,
                    prompt_version=PROMPT,
                )
            )
            self.session.flush()
        return review

    def note(
        self, version: str, text: str, *, store: Store = Store.APP_STORE, day: int = 0
    ) -> None:
        released = date(2026, 10, 1) + timedelta(days=day)
        store_notes(self.session, self.apps[store], [ReleaseNote(version, released, text)])
