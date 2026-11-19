# Player Review Intelligence

An AI agent system that collects public app-store reviews for mobile games, classifies them with Claude, connects complaints to game updates with RAG, and exposes everything through an MCP server and a weekly report.

First target: **Critical Strike** by Vertigo Games (Google Play + App Store). Any game can be added in `config/apps.yaml`.

> Status: **M1 (project skeleton)**. See [`docs/SPEC.md`](docs/SPEC.md) for the full plan.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 automatically.

```bash
uv sync
cp .env.example .env
uv run review-intel db init
uv run review-intel --help
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
