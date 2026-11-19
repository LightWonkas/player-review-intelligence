# Player Review Intelligence

An AI agent system that collects public app-store reviews for mobile games, classifies them with Claude, connects complaints to game updates with RAG, and exposes everything through an MCP server and a weekly report.

First target: **Critical Strike** by Vertigo Games (Google Play + App Store). Any game can be added in `config/apps.yaml`.

> Status: **M3 (classification with Claude)**. See [`docs/SPEC.md`](docs/SPEC.md) for the full plan.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 automatically.

```bash
uv sync
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
uv run review-intel fetch --app critical-strike --store all --limit 100
uv run review-intel classify --app critical-strike --limit 100
uv run review-intel --help
```

`fetch` creates the database on first run and is safe to repeat: already-stored reviews are
skipped, and edited reviews are updated in place. The App Store feed serves at most ~500 of the
newest reviews per country; Google Play has no such cap.

## Classification

Each review is classified by `claude-haiku-5-5` through one strict tool call
(`record_classification`) into the 12 categories in [`config/taxonomy.yaml`](config/taxonomy.yaml),
plus sentiment, severity (1-5), a bug-report flag, mentioned features and a one-line summary.
Every response is validated with Pydantic; invalid output is retried once and then recorded in
`classification_failures`, never dropped. Review text is passed as escaped, delimited data so a
review cannot inject instructions.

`prompt_version` (e.g. `v1-10aea4a5`) is a hash of the prompt and tool schema, so editing either
creates a new version and old results stay comparable. `classify` only processes reviews that have
no result for the current prompt version and model.

### Cost (measured, Critical Strike, November 2026)

| Run | Reviews | Est. cost | Per 1,000 reviews |
|---|---:|---:|---:|
| Sync, 4 parallel requests | 20 | $0.0033 | ~$0.17 |
| `--batch` (Message Batches API, -50%) | 500 | $0.0390 | ~$0.08 |

The shared prefix (tool + system prompt + taxonomy, ~1,900 tokens) is prompt-cached, so after the
first request each review pays only for its own ~300 input and ~210 output tokens. Estimates use
the price table in `src/review_intel/llm/pricing.py`; check Anthropic's pricing page for billing.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
