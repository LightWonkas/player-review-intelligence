"""The dashboard app. Read-only: every page and API route reads through `insights`.

Bind it to localhost (the default): it has no auth because it serves public review
data on the developer's machine. Report markdown is rendered with raw HTML switched
off, and Jinja autoescapes everything else, so review text can't inject markup.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from review_intel.apps_config import AppsConfig
from review_intel.dashboard.charts import category_bars, sentiment_columns
from review_intel.db.models import Report
from review_intel.insights import (
    CategoryBreakdown,
    PeriodSummary,
    SentimentTrend,
    TopIssues,
    category_breakdown,
    period_summary,
    sentiment_trend,
    top_issues,
)
from review_intel.report import CATEGORY_LABELS

HERE = Path(__file__).parent
TREND_WEEKS = 12
_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")


@dataclass
class DashboardDeps:
    session_factory: sessionmaker[Session]
    apps: AppsConfig
    classify_model: str
    prompt_version: str
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def render_report(markdown: str) -> Markup:
    """Raw HTML in the markdown stays text; links can't use javascript: URLs."""
    # Safe to mark: the parser has html disabled, so input markup comes out escaped.
    return Markup(_markdown.render(markdown))


@dataclass(frozen=True)
class Tile:
    label: str
    value: str
    delta: str  # e.g. "+12 vs previous 7 days"
    tone: Literal["good", "bad", "neutral"]


def _tiles(summary: PeriodSummary) -> list[Tile]:
    cur, prev = summary.current, summary.previous

    def tone(change: float, up_is_good: bool | None) -> Literal["good", "bad", "neutral"]:
        if not change or up_is_good is None:
            return "neutral"
        return "good" if (change > 0) == up_is_good else "bad"

    def arrow(change: float) -> str:
        return "▲" if change > 0 else "▼" if change < 0 else "■"

    tiles = []
    review_change = cur.reviews - prev.reviews
    tiles.append(
        Tile(
            "Reviews, last 7 days",
            f"{cur.reviews:,}",
            f"{arrow(review_change)} {review_change:+d} vs previous 7 days",
            "neutral",
        )
    )
    if cur.avg_rating is not None:
        change = round(cur.avg_rating - (prev.avg_rating or cur.avg_rating), 2)
        tiles.append(
            Tile(
                "Average rating",
                f"{cur.avg_rating:.2f}",
                f"{arrow(change)} {change:+.2f} vs previous 7 days",
                tone(change, True),
            )
        )
    negative = cur.sentiment_share["negative"]
    # Compare in whole points, as shown, so "21% vs 21%" reads as no change, not "-0".
    points = round(negative * 100) - round(prev.sentiment_share["negative"] * 100)
    tiles.append(
        Tile(
            "Negative reviews",
            f"{negative * 100:.0f}%",
            f"{arrow(points)} {points:+d} points vs previous 7 days",
            tone(points, False),
        )
    )
    bugs_change = cur.bug_reports - prev.bug_reports
    tiles.append(
        Tile(
            "Bug reports",
            f"{cur.bug_reports}",
            f"{arrow(bugs_change)} {bugs_change:+d} vs previous 7 days",
            tone(bugs_change, False),
        )
    )
    return tiles


def create_app(deps: DashboardDeps) -> FastAPI:
    app = FastAPI(title="Player Review Intelligence", docs_url="/api/docs", redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    games = {game.slug: game for game in deps.apps.games}

    def game_or_404(slug: str) -> str:
        if slug not in games:
            raise HTTPException(status_code=404, detail=f"Unknown app '{slug}'")
        return games[slug].name

    def scope() -> dict[str, Any]:
        return {"model": deps.classify_model, "prompt_version": deps.prompt_version}

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url=f"/apps/{deps.apps.games[0].slug}", status_code=307)

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/apps/{slug}", response_class=HTMLResponse)
    def app_page(request: Request, slug: str, days: int = Query(30, ge=1, le=365)) -> HTMLResponse:
        name = game_or_404(slug)
        now = deps.now()
        with deps.session_factory() as session:
            summary = period_summary(session, slug, days=7, now=now, **scope())
            categories = category_breakdown(session, slug, days=days, now=now, **scope())
            trend = sentiment_trend(
                session, slug, days=7 * TREND_WEEKS, bucket="week", now=now, **scope()
            )
            issues = top_issues(session, slug, days=7, limit=5, now=now, **scope())
            report = session.scalar(
                select(Report)
                .where(Report.app_slug == slug)
                .order_by(Report.created_at.desc())
                .limit(1)
            )
        return templates.TemplateResponse(
            request,
            "app.html",
            {
                "game_name": name,
                "slug": slug,
                "games": games,
                "days": days,
                "tiles": _tiles(summary),
                "categories": categories,
                # charts.py escapes every piece of text it puts in the SVG.
                "category_chart": Markup(category_bars(categories.categories, CATEGORY_LABELS)),
                "trend": trend,
                "trend_chart": Markup(sentiment_columns(trend.points)),
                "issues": issues,
                "labels": CATEGORY_LABELS,
                "report": report,
                "report_html": render_report(report.markdown) if report else None,
                "now": now,
            },
        )

    @app.get("/api/apps/{slug}/summary")
    def api_summary(slug: str, days: int = Query(7, ge=1, le=365)) -> PeriodSummary:
        game_or_404(slug)
        with deps.session_factory() as session:
            return period_summary(session, slug, days=days, now=deps.now(), **scope())

    @app.get("/api/apps/{slug}/categories")
    def api_categories(slug: str, days: int = Query(30, ge=1, le=365)) -> CategoryBreakdown:
        game_or_404(slug)
        with deps.session_factory() as session:
            return category_breakdown(session, slug, days=days, now=deps.now(), **scope())

    @app.get("/api/apps/{slug}/sentiment")
    def api_sentiment(
        slug: str,
        days: int = Query(84, ge=1, le=365),
        bucket: Literal["day", "week"] = "week",
    ) -> SentimentTrend:
        game_or_404(slug)
        with deps.session_factory() as session:
            return sentiment_trend(
                session, slug, days=days, bucket=bucket, now=deps.now(), **scope()
            )

    @app.get("/api/apps/{slug}/top-issues")
    def api_top_issues(
        slug: str,
        days: int = Query(7, ge=1, le=365),
        limit: int = Query(5, ge=1, le=12),
    ) -> TopIssues:
        game_or_404(slug)
        with deps.session_factory() as session:
            return top_issues(session, slug, days=days, limit=limit, now=deps.now(), **scope())

    return app
