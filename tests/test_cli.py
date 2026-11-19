from pathlib import Path

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from review_intel.cli import app
from review_intel.db.models import App

runner = CliRunner()

EXPECTED_TABLES = {"apps", "reviews", "classifications", "patch_notes", "reports"}


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "db" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "review-intel" in result.output


def test_db_init_creates_tables_and_registers_apps(db_url: str, apps_config_path: Path) -> None:
    args = ["db", "init", "--database-url", db_url, "--apps-config", str(apps_config_path)]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "2 added" in result.output
    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES
    with Session(engine) as session:
        slugs = session.scalars(select(App.slug)).all()
    engine.dispose()
    assert slugs == ["critical-strike", "critical-strike"]


def test_db_init_is_idempotent(db_url: str, apps_config_path: Path) -> None:
    args = ["db", "init", "--database-url", db_url, "--apps-config", str(apps_config_path)]
    runner.invoke(app, args)

    second = runner.invoke(app, args)

    assert second.exit_code == 0, second.output
    assert "0 added, 0 updated, 2 unchanged" in second.output


def test_db_init_creates_missing_sqlite_folder(tmp_path: Path, apps_config_path: Path) -> None:
    db_file = tmp_path / "nested" / "dir" / "review_intel.db"
    url = f"sqlite:///{db_file.as_posix()}"

    result = runner.invoke(
        app, ["db", "init", "--database-url", url, "--apps-config", str(apps_config_path)]
    )

    assert result.exit_code == 0, result.output
    assert db_file.exists()
