"""End-to-end CLI flow for M5: import notes, index, ask, spikes. All offline."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from review_intel.cli import app
from review_intel.evals.judge import JudgeVerdict
from review_intel.report import ReportProse

from .fake_anthropic import (
    FakeAnthropic,
    FakeParsedResponse,
    always,
    fake_client,
    text_message,
)
from .fake_anthropic import with_judge as with_parse
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
    assert "No significant spikes" in spikes.output  # nothing is classified in this test


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


def test_report_command_writes_markdown_and_stores_it(
    tmp_path: Path, db_url: str, apps_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prose = ReportProse(summary="Quiet week.", issue_notes=[], action_items=[], caveats=[])

    def respond(params: dict[str, object]) -> FakeParsedResponse:
        if params["output_format"] is JudgeVerdict:
            return FakeParsedResponse(JudgeVerdict(score=5, unsupported_claims=[], reasoning=""))
        return FakeParsedResponse(prose)

    client = fake_client(with_parse(FakeAnthropic(always(text_message("unused"))), respond))
    monkeypatch.setattr("review_intel.cli.make_client", lambda _settings: client)
    monkeypatch.setattr("review_intel.cli.make_embedder", lambda _name: FakeEmbedder())
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    out = tmp_path / "report.md"
    common = _common(db_url, apps_config_path)

    result = runner.invoke(app, ["report", *common, "--end", "2026-11-19", "--out", str(out)])
    no_webhook = runner.invoke(app, ["report", *common, "--post-slack"])

    assert result.exit_code == 0, result.output
    assert "Judge: 5/5" in result.output
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Player review report:") and "Quiet week." in text
    assert "**2026-11-12 to 2026-11-18**" in text
    assert no_webhook.exit_code != 0
