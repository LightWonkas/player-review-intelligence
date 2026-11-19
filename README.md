# Player Review Intelligence

An AI agent system that collects public app-store reviews for mobile games, classifies them with Claude, connects complaints to game updates with RAG, and exposes everything through an MCP server and a weekly report.

First target: **Critical Strike** by Vertigo Games (Google Play + App Store). Any game can be added in `config/apps.yaml`.

> Status: **M2 (review collection)**. See [`docs/SPEC.md`](docs/SPEC.md) for the full plan.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 automatically.

```bash
uv sync
cp .env.example .env
uv run review-intel fetch --app critical-strike --store all --limit 100
uv run review-intel --help
```

`fetch` creates the database on first run and is safe to repeat: already-stored reviews are
skipped, and edited reviews are updated in place. The App Store feed serves at most ~500 of the
newest reviews per country; Google Play has no such cap.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
