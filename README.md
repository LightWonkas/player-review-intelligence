# Player Review Intelligence

An AI agent system that collects public app-store reviews for mobile games, classifies them with Claude, connects complaints to game updates with RAG, and exposes everything through an MCP server and a weekly report.

Games are configured in [`config/apps.yaml`](config/apps.yaml) (Google Play + App Store); adding one needs no code changes.

> Status: **M5 (RAG over patch notes)**. See [`docs/SPEC.md`](docs/SPEC.md) for the full plan.

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

The reference set has 100 labeled reviews, labeled blind: the labeler never sees the model's
answer, which would bias labels toward it. The current labels were set by `claude-opus-5-5`
(recorded per item in the `labeler` field), a stronger model than the classifier, so the scores
measure agreement with a stronger reference rather than human accuracy; `review-intel label`
lets a person relabel or extend the set. The sample is 40% random (the real mix, mostly praise)
and 60% low-rated reviews with real text, where the hard categories are, with at least 25% from
the App Store. A hash of each review fixes it in a 70/30 dev/test split: prompts are tuned on dev
and the headline number is test. With 35 test reviews, one review moves accuracy by about 3
points, and categories with 1-3 examples are anecdotes rather than measurements.

`review-intel eval` reports primary-category accuracy, per-category precision/recall/F1,
macro-F1, sentiment accuracy and a confusion matrix, and compares with the previous prompt or
model. An LLM judge (`claude-sonnet-5-5`, stronger than the classifier so no model grades its own
work) scores per-review summaries for faithfulness. Predictions are cached per prompt version, so
re-running an unchanged prompt is free; a run with the judge costs about $0.08.

<!-- eval:start -->
| Date | Prompt | Model | Test acc | Test macro-F1 | Sentiment acc (all) | Judge |
|---|---|---|---:|---:|---:|---:|
| 2026-11-19 | `v1-10aea4a5` | claude-haiku-5-5 | 0.89 (n=35) | 0.70 | 0.97 | 4.65/5 |
<!-- eval:end -->

## Linking complaints to releases (RAG)

```bash
uv run review-intel patches fetch --app critical-strike   # release notes, one request
uv run review-intel index --app critical-strike           # local embeddings, free
uv run review-intel ask "When did matchmaking complaints start?" --app critical-strike
uv run review-intel spikes --app critical-strike
```

- **Patch notes:** The App Store page lists the last ~25 versions with dates and notes, which
  is the only public release history with dates. `patches fetch` reads it with one request.
  Notes for other stores can be imported from markdown (`patches import --file notes.md`, with
  sections like `## 15.008 (2026-10-16)`). Version numbers are shared across stores, so a note
  applies to reviews from either store.
- **Embeddings:** Reviews and individual patch-note lines are embedded locally with
  `all-MiniLM-L6-v2` (CPU, no API cost) and stored in `sqlite-vec` tables inside the same SQLite
  file, so search results join straight back to SQL. Re-indexing only embeds new or changed text.
  The model is English-focused; set `EMBEDDING_MODEL` to a multilingual model if many reviews are
  in other languages.
- **`ask`:** Retrieves similar reviews and patch notes, adds the notes for the versions those
  reviews were written on, and computes a per-version timeline and data coverage in SQL.
  `claude-sonnet-5-5` writes the answer, citing reviews as `[r<id>]` and releases as
  `[v<version>]`. Every citation is checked against the retrieved context, and any the model
  invents is removed and reported. Counts in answers come from the SQL timeline. A question
  costs about $0.014.
- **`spikes`:** No LLM is involved. For each store, a version's share of reviews mentioning a
  category is compared with the previous versions using a two-proportion z-test. Stores are kept
  apart because their reviewers differ, so a shift in store mix can't fake a spike. A spike is
  linked to the patch note in force, with that note's lines ranked by similarity to the
  complaints. When nothing is significant, the closest candidates are listed and marked as not
  significant.

With ~20 classified App Store reviews per version and only about five weeks of Google Play
history, the current data has no significant spikes. The closest candidate is a weapon-balance
increase on Google Play from 3% to 8% (z = 1.8). More history makes the detector useful; it
needs no changes for that.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
