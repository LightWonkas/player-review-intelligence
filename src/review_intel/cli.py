"""`review-intel` command-line interface.

The CLI is a thin layer: it reads settings, calls library functions and prints
results. Logic lives in modules that tests (and later the MCP server) call directly.
"""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import Engine

from review_intel import __version__
from review_intel.apps_config import AppsConfig, load_apps_config
from review_intel.collectors import build_sources
from review_intel.db.apps import SyncResult, sync_apps
from review_intel.db.session import init_db, make_engine, make_session_factory
from review_intel.domain import Store
from review_intel.ingest import fetch_and_store
from review_intel.settings import get_settings

app = typer.Typer(
    help="Player Review Intelligence: collect, classify and report on app-store reviews.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")

DatabaseUrlOption = Annotated[
    str | None, typer.Option(help="Override DATABASE_URL from the environment.")
]
AppsConfigOption = Annotated[
    Path | None, typer.Option(help="Override APPS_CONFIG_PATH (default config/apps.yaml).")
]


class StoreChoice(StrEnum):
    ALL = "all"
    GOOGLE_PLAY = "google_play"
    APP_STORE = "app_store"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"review-intel {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Player Review Intelligence: collect, classify and report on app-store reviews."""
    # WARNING level: retries and failures are visible, routine progress is not.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _prepare_db(
    database_url: str | None, apps_config: Path | None
) -> tuple[Engine, AppsConfig, SyncResult]:
    """Create tables and register configured games. Every command starts here, so a
    fresh clone never needs a separate `db init` step."""
    settings = get_settings()
    config = load_apps_config(apps_config or settings.apps_config_path)
    engine = make_engine(database_url or settings.database_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        sync = sync_apps(session, config)
        session.commit()
    return engine, config, sync


@db_app.command("init")
def db_init(database_url: DatabaseUrlOption = None, apps_config: AppsConfigOption = None) -> None:
    """Create database tables and register the games from apps.yaml. Safe to re-run."""
    engine, _, sync = _prepare_db(database_url, apps_config)
    tables = init_db(engine)
    # Hide any password so the URL is safe to print in logs and screenshots.
    safe_url = engine.url.render_as_string(hide_password=True)
    engine.dispose()

    typer.echo(f"Database ready: {safe_url}")
    typer.echo(f"Tables: {', '.join(tables)}")
    typer.echo(f"Apps: {sync.created} added, {sync.updated} updated, {sync.unchanged} unchanged")


@app.command()
def fetch(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    store: Annotated[StoreChoice, typer.Option(help="Which store to fetch.")] = StoreChoice.ALL,
    limit: Annotated[int, typer.Option(min=1, help="Max reviews per store, newest first.")] = 100,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Fetch the newest public reviews and save them. Re-running adds no duplicates."""
    engine, config, _ = _prepare_db(database_url, apps_config)
    try:
        game = config.get_game(app_slug)
    except KeyError as exc:
        engine.dispose()
        raise typer.BadParameter(str(exc.args[0]), param_hint="--app") from exc

    stores = list(Store) if store is StoreChoice.ALL else [Store(store.value)]
    configured = {listing.store for listing in game.listings}
    if not configured.intersection(stores):
        engine.dispose()
        raise typer.BadParameter(f"{app_slug} has no {store.value} listing", param_hint="--store")

    results = fetch_and_store(
        make_session_factory(engine), config, app_slug, stores, limit, build_sources()
    )
    engine.dispose()

    for result in results:
        up = result.upsert
        line = (
            f"{result.store.value:<12} fetched {result.fetched:>4} -> "
            f"{up.inserted} new, {up.updated} updated, {up.unchanged} unchanged"
        )
        if result.error:
            line += f"  [ERROR: {result.error}]"
        typer.echo(line)

    if any(result.error for result in results):
        raise typer.Exit(code=1)
