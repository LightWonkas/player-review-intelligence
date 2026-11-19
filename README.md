# Player Review Intelligence

An AI agent system that collects public app-store reviews for mobile games, classifies them with Claude, connects complaints to game updates with RAG, and exposes everything through an MCP server and a weekly report.

Games are configured in [`config/apps.yaml`](config/apps.yaml) (Google Play + App Store); adding one needs no code changes.

> Status: **M4 (evals)**. See [`docs/SPEC.md`](docs/SPEC.md) for the full plan.

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

### Cost (measured on real reviews, November 2026)

| Run | Reviews | Est. cost | Per 1,000 reviews |
|---|---:|---:|---:|
| Sync, 4 parallel requests | 20 | $0.0033 | ~$0.17 |
| `--batch` (Message Batches API, -50%) | 500 | $0.0390 | ~$0.08 |

The shared prefix (tool + system prompt + taxonomy, ~1,900 tokens) is prompt-cached, so after the
first request each review pays only for its own ~300 input and ~210 output tokens. Estimates use
the price table in `src/review_intel/llm/pricing.py`; check Anthropic's pricing page for billing.

## Evals

100 reviews are labeled by hand (`review-intel label`), blind: the labeler never sees the model's
answer, which would bias labels toward it. The sample is 40% random (the real mix, mostly praise)
and 60% low-rated reviews with real text, where the hard categories are, with at least 25% from
the App Store. A hash of each review fixes it in a 70/30 dev/test split: prompts are tuned on dev
and the headline number is test.

`review-intel eval` reports primary-category accuracy, per-category precision/recall/F1,
macro-F1, sentiment accuracy and a confusion matrix, and compares with the previous prompt or
model. An LLM judge (`claude-sonnet-5-5`, stronger than the classifier so no model grades its own
work) scores per-review summaries for faithfulness. Predictions are cached per prompt version, so
re-running an unchanged prompt is free; a run with the judge costs about $0.08.

<!-- eval:start -->
_No eval run yet._
<!-- eval:end -->

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
