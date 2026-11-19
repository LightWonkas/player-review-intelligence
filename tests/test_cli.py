from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from review_intel.cli import app
from review_intel.collectors.base import ReviewSource
from review_intel.db.models import App
from review_intel.domain import Store

from .fakes import FakeSource, make_raw

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


def _fake_sources(fail: bool = False) -> dict[Store, ReviewSource]:
    return {
        Store.GOOGLE_PLAY: FakeSource(
            Store.GOOGLE_PLAY, [make_raw("a1"), make_raw("a2")], fail_after=1 if fail else None
        ),
        Store.APP_STORE: FakeSource(Store.APP_STORE, [make_raw("i1")]),
    }


def _fetch_args(db_url: str, apps_config_path: Path, *extra: str) -> list[str]:
    return [
        "fetch",
        "--app",
        "critical-strike",
        "--database-url",
        db_url,
        "--apps-config",
        str(apps_config_path),
        *extra,
    ]


def test_fetch_twice_adds_no_duplicates(
    db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.build_sources", _fake_sources)
    args = _fetch_args(db_url, apps_config_path)

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert "google_play  fetched    2 -> 2 new" in first.output
    assert "app_store    fetched    1 -> 1 new" in first.output
    assert second.exit_code == 0, second.output
    assert second.output.count("0 new, 0 updated") == 2


def test_fetch_single_store(
    db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.build_sources", _fake_sources)

    result = runner.invoke(app, _fetch_args(db_url, apps_config_path, "--store", "app_store"))

    assert result.exit_code == 0, result.output
    assert "app_store" in result.output
    assert "google_play" not in result.output


def test_fetch_failure_exits_non_zero_but_saves_partial_results(
    db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.build_sources", lambda: _fake_sources(fail=True))

    result = runner.invoke(app, _fetch_args(db_url, apps_config_path))

    assert result.exit_code == 1
    assert "1 new" in result.output
    assert "ERROR: ConnectionError" in result.output


def test_fetch_unknown_app_is_a_clear_error(db_url: str, apps_config_path: Path) -> None:
    args = _fetch_args(db_url, apps_config_path)
    args[args.index("critical-strike")] = "nope"

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "unknown app 'nope'" in result.output
