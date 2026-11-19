from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from review_intel.apps_config import AppsConfig, load_apps_config
from review_intel.domain import Store


def _config(**listing_overrides: Any) -> dict[str, Any]:
    listing = {"store": "app_store", "store_app_id": "123"} | listing_overrides
    return {"games": [{"slug": "my-game", "name": "My Game", "listings": [listing]}]}


def test_repo_config_has_critical_strike_on_both_stores(apps_config_path: Path) -> None:
    config = load_apps_config(apps_config_path)

    game = config.get_game("critical-strike")

    assert {listing.store for listing in game.listings} == {Store.GOOGLE_PLAY, Store.APP_STORE}


def test_app_store_id_must_be_numeric() -> None:
    with pytest.raises(ValidationError, match="numeric iTunes id"):
        AppsConfig.model_validate(_config(store_app_id="com.example.game"))


def test_google_play_id_must_be_package_name() -> None:
    with pytest.raises(ValidationError, match="package name"):
        AppsConfig.model_validate(_config(store="google_play", store_app_id="123"))


def test_duplicate_slugs_are_rejected() -> None:
    game = _config()["games"][0]
    with pytest.raises(ValidationError, match="unique"):
        AppsConfig.model_validate({"games": [game, game]})


def test_same_store_twice_in_one_game_is_rejected() -> None:
    listing = {"store": "app_store", "store_app_id": "123"}
    raw = {"games": [{"slug": "g", "name": "G", "listings": [listing, listing]}]}
    with pytest.raises(ValidationError, match="same store"):
        AppsConfig.model_validate(raw)


def test_unknown_field_is_rejected() -> None:
    # Typos like `store_id:` should fail loudly, not be silently ignored.
    with pytest.raises(ValidationError):
        AppsConfig.model_validate(_config(store_id="123"))


def test_get_game_unknown_slug_lists_known_apps() -> None:
    config = AppsConfig.model_validate(_config())
    with pytest.raises(KeyError, match="my-game"):
        config.get_game("nope")
