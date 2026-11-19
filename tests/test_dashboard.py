from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from review_intel.apps_config import load_apps_config
from review_intel.dashboard.app import DashboardDeps, create_app, render_report
from review_intel.dashboard.charts import category_bars, nice_max, sentiment_columns
from review_intel.db.models import Report
from review_intel.db.session import make_session_factory
from review_intel.insights import CategoryCount, TrendPoint

from .rag_data import MODEL, PROMPT, SLUG, Seeder

NOW = datetime(2026, 10, 15, tzinfo=UTC)


@pytest.fixture
def client(engine: Engine, apps_config_path: Path) -> TestClient:
    factory = make_session_factory(engine)
    with factory() as session:
        seed = Seeder(session, apps_config_path)
        seed.review("ads ads", categories=["ads"], day=2)
        for day in (8, 9):
            seed.review("ad after every match", categories=["ads"], day=day)
        seed.review("<img src=x onerror=alert(1)>", categories=["bug_crash"], day=10)
        session.add(
            Report(
                app_slug=SLUG,
                period_start=date(2026, 10, 8),
                period_end=date(2026, 10, 15),
                markdown=(
                    "# Week 41\n\n<script>alert(1)</script>\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
                ),
            )
        )
        session.commit()
    deps = DashboardDeps(
        session_factory=factory,
        apps=load_apps_config(apps_config_path),
        classify_model=MODEL,
        prompt_version=PROMPT,
        now=lambda: NOW,
    )
    return TestClient(create_app(deps))


def test_index_redirects_to_the_first_game(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == f"/apps/{SLUG}"


def test_app_page_shows_tiles_issues_charts_and_report(client: TestClient) -> None:
    response = client.get(f"/apps/{SLUG}?days=7")

    assert response.status_code == 200
    page = response.text
    assert "Reviews, last 7 days" in page
    assert "Top issues, last 7 days" in page and "Ads" in page
    assert page.count('<svg class="chart"') == 2
    assert "<h1>Week 41</h1>" in page and "<table>" in page  # markdown tables render


def test_untrusted_text_never_becomes_markup(client: TestClient) -> None:
    page = client.get(f"/apps/{SLUG}").text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "<img src=x" not in page


def test_unknown_app_is_404(client: TestClient) -> None:
    assert client.get("/apps/nope").status_code == 404
    assert client.get("/api/apps/nope/summary").status_code == 404


def test_json_api(client: TestClient) -> None:
    issues = client.get(f"/api/apps/{SLUG}/top-issues?days=7").json()
    categories = client.get(f"/api/apps/{SLUG}/categories?days=30").json()
    sentiment = client.get(f"/api/apps/{SLUG}/sentiment?days=21&bucket=week").json()
    summary = client.get(f"/api/apps/{SLUG}/summary").json()

    assert [(i["category"], i["reviews"]) for i in issues["issues"]] == [
        ("ads", 2),
        ("bug_crash", 1),
    ]
    assert categories["classified"] == 4
    assert sentiment["bucket"] == "week"
    assert summary["current"]["reviews"] == 3
    assert client.get(f"/api/apps/{SLUG}/top-issues?days=0").status_code == 422


def test_render_report_disables_raw_html() -> None:
    html = render_report("hi <b>there</b> [x](javascript:alert(1))")

    assert "<b>" not in html
    assert 'href="javascript' not in html


def test_nice_axis_max() -> None:
    assert nice_max(0) == (1, 1)
    assert nice_max(7) == (8, 2)
    assert nice_max(103) == (150, 50)


def test_category_bars_escape_labels_and_title_every_mark() -> None:
    svg = category_bars(
        [
            CategoryCount(category="ads", reviews=8, share=0.5),
            CategoryCount(category="x<y", reviews=0, share=0),
        ],
        {"ads": "Ads"},
    )

    assert svg.count("<title>") == 2
    assert "x&lt;y" in svg and "x<y" not in svg
    assert "Ads: 8 reviews (50%)" in svg


def test_sentiment_columns_gap_and_labels() -> None:
    points = [
        TrendPoint(
            start=date(2026, 10, d), reviews=10, positive=6, neutral=1, negative=3, avg_rating=4.0
        )
        for d in (5, 12)
    ]
    svg = sentiment_columns(points)

    assert svg.count('class="mark ') == 6  # three segments per week
    assert "Week of 2026-10-12: 3 negative of 10" in svg
    assert "Oct 12" in svg
    assert category_bars([], {}) == "" and sentiment_columns([]) == ""
