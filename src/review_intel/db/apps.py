"""Keep the `apps` table in step with `config/apps.yaml`.

The YAML file is the source of truth for which games we track; the table gives
reviews a stable foreign key. Syncing is an idempotent upsert keyed on (slug, store).
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from review_intel.apps_config import AppsConfig
from review_intel.db.models import App


@dataclass(frozen=True)
class SyncResult:
    created: int
    updated: int
    unchanged: int


def sync_apps(session: Session, config: AppsConfig) -> SyncResult:
    """Insert new listings and update changed ones. Never deletes: old reviews keep their app."""
    created = updated = unchanged = 0
    for game in config.games:
        for listing in game.listings:
            row = session.scalar(
                select(App).where(App.slug == game.slug, App.store == listing.store)
            )
            values = {
                "name": game.name,
                "store_app_id": listing.store_app_id,
                "country": listing.country,
                "language": listing.language,
            }
            if row is None:
                session.add(App(slug=game.slug, store=listing.store, **values))
                created += 1
            elif any(getattr(row, field) != value for field, value in values.items()):
                for field, value in values.items():
                    setattr(row, field, value)
                updated += 1
            else:
                unchanged += 1
    session.flush()
    return SyncResult(created=created, updated=updated, unchanged=unchanged)
