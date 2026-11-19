from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from review_intel.cli import app
from review_intel.collectors.base import ReviewSource
from review_intel.db.models import App
from review_intel.domain import Sentiment, Store

from .conftest import REPO_ROOT
from .fake_anthropic import VALID_INPUT, FakeAnthropic, always, fake_client, tool_message
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


def test_classify_prints_category_counts_and_cost(
    db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.build_sources", _fake_sources)
    runner.invoke(app, _fetch_args(db_url, apps_config_path))
    fake = FakeAnthropic(always(tool_message(VALID_INPUT)))
    monkeypatch.setattr("review_intel.cli.make_client", lambda _settings: fake_client(fake))
    args = [
        "classify",
        "--app",
        "critical-strike",
        "--database-url",
        db_url,
        "--apps-config",
        str(apps_config_path),
    ]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert "Classified 3, failed 0" in first.output
    assert "est. cost $" in first.output
    assert "bug_crash" in first.output
    assert "100.0%" in first.output
    assert second.exit_code == 0, second.output
    assert "Nothing to classify" in second.output
    assert len(fake.messages.calls) == 3


def test_eval_command_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from review_intel.evals.dataset import LabeledReview, append_label, split_for
    from review_intel.evals.judge import JudgeVerdict

    from .fake_anthropic import FakeParsedResponse, with_judge

    monkeypatch.chdir(tmp_path)  # predictions/results/README are written relative to cwd
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "taxonomy.yaml").write_text(
        (REPO_ROOT / "config" / "taxonomy.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Readme\n", encoding="utf-8")
    labels_file = tmp_path / "labels.jsonl"
    for i in range(4):
        key = f"google_play:{i}"
        append_label(
            LabeledReview(
                key=key,
                store=Store.GOOGLE_PLAY,
                rating=1,
                title=None,
                body="crash",
                app_version=None,
                primary_category="bug_crash",
                categories=["bug_crash"],
                sentiment=Sentiment.NEGATIVE,
                split=split_for(key),
                labeled_at=datetime(2026, 11, 19, tzinfo=UTC),
            ),
            labels_file,
        )
    verdict = JudgeVerdict(score=5, unsupported_claims=[], reasoning="ok")
    fake = with_judge(
        FakeAnthropic(always(tool_message(VALID_INPUT))), lambda _: FakeParsedResponse(verdict)
    )
    monkeypatch.setattr("review_intel.cli.make_client", lambda _settings: fake_client(fake))

    result = runner.invoke(
        app, ["eval", "--labels-file", str(labels_file), "--judge", "2", "--write-readme"]
    )

    assert result.exit_code == 0, result.output
    assert "all  n=4   primary acc 1.000" in result.output
    assert "Summary faithfulness (claude-sonnet-5-5, n=2): mean 5.00/5" in result.output
    assert len(list((tmp_path / "evals" / "results").glob("*.json"))) == 1
    assert "<!-- eval:start -->" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_eval_without_labels_explains_what_to_do(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "--labels-file", str(tmp_path / "none.jsonl")])

    assert result.exit_code == 1
    assert "review-intel label" in result.output


def test_classify_dry_run_counts_and_estimates_without_calling_claude(
    db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.build_sources", _fake_sources)

    def no_client(_settings: object) -> None:
        raise AssertionError("dry run must not create a client")

    monkeypatch.setattr("review_intel.cli.make_client", no_client)
    runner.invoke(app, _fetch_args(db_url, apps_config_path))

    result = runner.invoke(
        app,
        [
            "classify",
            "--app",
            "critical-strike",
            "--dry-run",
            "--database-url",
            db_url,
            "--apps-config",
            str(apps_config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Would classify 3 reviews with claude-haiku-5-5" in result.output
    assert "with --batch" in result.output
