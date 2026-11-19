"""`review-intel` command-line interface.

The CLI is a thin layer: it reads settings, calls library functions and prints
results. Logic lives in modules that tests (and later the MCP server) call directly.
"""

from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import make_url

from review_intel import __version__
from review_intel.apps_config import load_apps_config
from review_intel.db.apps import sync_apps
from review_intel.db.session import init_db, make_engine, make_session_factory
from review_intel.settings import get_settings

app = typer.Typer(
    help="Player Review Intelligence: collect, classify and report on app-store reviews.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")


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


@db_app.command("init")
def db_init(
    database_url: Annotated[
        str | None, typer.Option(help="Override DATABASE_URL from the environment.")
    ] = None,
    apps_config: Annotated[
        Path | None, typer.Option(help="Override APPS_CONFIG_PATH (default config/apps.yaml).")
    ] = None,
) -> None:
    """Create database tables and register the games from apps.yaml. Safe to re-run."""
    settings = get_settings()
    url = database_url or settings.database_url
    config = load_apps_config(apps_config or settings.apps_config_path)

    engine = make_engine(url)
    tables = init_db(engine)
    with make_session_factory(engine)() as session:
        result = sync_apps(session, config)
        session.commit()
    engine.dispose()

    # Hide any password so the URL is safe to print in logs and screenshots.
    safe_url = make_url(url).render_as_string(hide_password=True)
    typer.echo(f"Database ready: {safe_url}")
    typer.echo(f"Tables: {', '.join(tables)}")
    typer.echo(
        f"Apps: {result.created} added, {result.updated} updated, {result.unchanged} unchanged"
    )
