"""End-to-end CLI flow for M5: import notes, index, ask, spikes. All offline."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from review_intel.cli import app

from .fake_embedder import FakeEmbedder
from .test_ask import _client
from .test_cli import _fake_sources, _fetch_args

runner = CliRunner()


def _common(db_url: str, apps_config_path: Path) -> list[str]:
    return [
        "--app",
        "critical-strike",
        "--database-url",
        db_url,
        "--apps-config",
        str(apps_config_path),
    ]


def test_import_index_ask_and_spikes(
    tmp_path: Path, db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.build_sources", _fake_sources)
    monkeypatch.setattr("review_intel.cli.make_embedder", lambda _name: FakeEmbedder())
    common = _common(db_url, apps_config_path)
    notes = tmp_path / "notes.md"
    notes.write_text("## 1.0 (2026-10-30)\n- Fixed the shop crash\n", encoding="utf-8")

    fetched = runner.invoke(app, _fetch_args(db_url, apps_config_path))
    imported = runner.invoke(app, ["patches", "import", *common, "--file", str(notes)])
    indexed = runner.invoke(app, ["index", *common])
    again = runner.invoke(app, ["index", *common])

    assert fetched.exit_code == 0, fetched.output
    assert "Patch notes: 1 new" in imported.output
    assert "Embedded 3 reviews and 1 patch-note lines" in indexed.output
    assert "Embedded 0 reviews and 0 patch-note lines" in again.output

    client, _ = _client("The shop crash was fixed in [v1.0] [r1], see also [r777].")
    monkeypatch.setattr("review_intel.cli.make_client", lambda _settings: client)
    asked = runner.invoke(app, ["ask", "was the shop crash fixed?", *common])

    assert asked.exit_code == 0, asked.output
    assert "Sources: reviews r1 | versions 1.0" in asked.output
    assert "Removed citations not in the context: r777" in asked.output
    assert "r777]" not in asked.output.splitlines()[0]

    spikes = runner.invoke(app, ["spikes", *common])
    assert spikes.exit_code == 0, spikes.output
    assert "No spikes found." in spikes.output  # nothing is classified in this test


def test_ask_before_index_says_what_to_run(
    db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("review_intel.cli.make_embedder", lambda _name: FakeEmbedder())

    result = runner.invoke(app, ["ask", "anything?", *_common(db_url, apps_config_path)])

    assert result.exit_code == 1
    assert "review-intel index" in result.output


def test_patches_import_rejects_a_file_without_sections(
    tmp_path: Path, db_url: str, apps_config_path: Path
) -> None:
    empty = tmp_path / "notes.md"
    empty.write_text("just text\n", encoding="utf-8")

    result = runner.invoke(
        app, ["patches", "import", *_common(db_url, apps_config_path), "--file", str(empty)]
    )

    assert result.exit_code == 1
