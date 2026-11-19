import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy.orm import Session

from review_intel.report import ReportRun, gather_facts
from review_intel.slack import post_to_slack, slack_text

from .rag_data import MODEL, PROMPT, SLUG, Seeder
from .test_report import _prose

NOW = datetime(2026, 10, 15, tzinfo=UTC)


def _run(session: Session, apps_config_path: Path) -> ReportRun:
    seed = Seeder(session, apps_config_path)
    for day in (8, 9):
        seed.review("ads after every match", categories=["ads"], day=day)
    seed.review("crash", categories=["bug_crash"], day=10)
    facts = gather_facts(
        session, SLUG, "Test Game", model=MODEL, prompt_version=PROMPT, days=7, now=NOW
    )
    return ReportRun("# md", facts, _prose(facts), None, {}, [])


def test_slack_text_has_summary_issues_and_actions(
    session: Session, apps_config_path: Path
) -> None:
    text = slack_text(_run(session, apps_config_path))

    assert text.startswith("*Player review report: Test Game*")
    assert "- Ads: 2 reviews (67%), previous period 0" in text
    assert "- [P1, product] Cap ads per session." in text


def test_post_sends_json_and_raises_on_rejection() -> None:
    sent: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200 if len(sent) == 1 else 404)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    post_to_slack("https://hooks.slack.test/x", "hello", client=client)
    with pytest.raises(httpx.HTTPStatusError):
        post_to_slack("https://hooks.slack.test/x", "again", client=client)

    assert sent == [{"text": "hello"}, {"text": "again"}]
