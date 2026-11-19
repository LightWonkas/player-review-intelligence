"""Load and validate `config/apps.yaml`, the list of games we track.

Validation happens once, at load time, so a bad store id fails with a clear
message here instead of as a confusing 404 halfway through a fetch.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from review_intel.domain import Store

_SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
_ISO_CODE_PATTERN = r"^[a-z]{2}$"


class StoreListing(BaseModel):
    """One game on one store, e.g. Critical Strike on Google Play."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store: Store
    store_app_id: str = Field(min_length=1)
    country: str = Field(default="us", pattern=_ISO_CODE_PATTERN)
    language: str = Field(default="en", pattern=_ISO_CODE_PATTERN)

    @model_validator(mode="after")
    def _check_id_matches_store(self) -> Self:
        # Catches the most common copy-paste mistake: swapping the two stores' ids.
        if self.store is Store.APP_STORE and not self.store_app_id.isdigit():
            raise ValueError(
                f"app_store id must be the numeric iTunes id, got {self.store_app_id!r}"
            )
        if self.store is Store.GOOGLE_PLAY and "." not in self.store_app_id:
            raise ValueError(
                f"google_play id must be a package name like com.example.game, "
                f"got {self.store_app_id!r}"
            )
        return self


class GameConfig(BaseModel):
    """A game and all of its store listings, addressed on the CLI by `slug`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(pattern=_SLUG_PATTERN)
    name: str = Field(min_length=1)
    developer: str | None = None
    listings: list[StoreListing] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_listing_per_store(self) -> Self:
        stores = [listing.store for listing in self.listings]
        if len(stores) != len(set(stores)):
            raise ValueError(f"game {self.slug!r} lists the same store more than once")
        return self


class AppsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    games: list[GameConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_slugs(self) -> Self:
        slugs = [game.slug for game in self.games]
        if len(slugs) != len(set(slugs)):
            raise ValueError("game slugs in apps.yaml must be unique")
        return self

    def get_game(self, slug: str) -> GameConfig:
        for game in self.games:
            if game.slug == slug:
                return game
        known = ", ".join(game.slug for game in self.games) or "none"
        raise KeyError(f"unknown app {slug!r}; configured apps: {known}")


def load_apps_config(path: Path) -> AppsConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppsConfig.model_validate(raw)
