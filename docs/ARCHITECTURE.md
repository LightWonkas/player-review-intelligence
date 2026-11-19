# Architecture: every module in two sentences

Reviews flow one way: **stores → collectors → SQLite → Claude classification → SQL insights →
report, dashboard, MCP**. Patch notes and embeddings sit beside that flow to link complaints to
releases. Each module below is described in two sentences: what it does, then why it exists or how
it works.

## Configuration and storage

| Module | What and why |
|---|---|
| `settings.py` | Reads env vars and `.env` into one typed `Settings` object (models, database URL, paths, secrets as `SecretStr`). Secrets never live in code or YAML, so the repo can be public. |
| `apps_config.py` | Loads and validates `config/apps.yaml`, the list of tracked games and their store listings. Adding a game is a config change; bad store ids fail at load time with a clear message. |
| `domain.py` | The shared enums `Store` and `Sentiment`. One definition keeps the database, prompts and APIs using the same values. |
| `db/models.py` | SQLAlchemy 2.0 models for apps, reviews, classifications, failures, patch notes, embeddings bookkeeping and reports. Uniqueness lives in DB constraints, and datetimes are always timezone-aware UTC. |
| `db/session.py` | Builds the engine from `DATABASE_URL` and loads `sqlite-vec` on every SQLite connection. Foreign keys are switched on per connection because SQLite ignores them otherwise. |
| `db/apps.py` | Upserts the games from the config into the `apps` table. It never deletes, so removing a game from the config can't wipe its history. |

## Collecting reviews

| Module | What and why |
|---|---|
| `collectors/base.py` | The `ReviewSource` protocol and the `RawReview` record every store collector returns. New stores plug in without touching ingestion. |
| `collectors/google_play.py` | Pages through `google-play-scraper` with continuation tokens, pacing and retries. The scraper returns naive local times, which are converted to UTC here. |
| `collectors/app_store.py` | Reads the public iTunes customer-reviews JSON feed, up to 10 pages. It handles the feed's quirks: a single review returned as a dict, and empty pages in the middle. |
| `retry.py` | Exponential backoff with jitter, plus a `Pacer` that keeps requests at least a second apart. Sleep and randomness are injectable, so tests run instantly. |
| `ingest.py` | Upserts fetched reviews idempotently: new ones are inserted, changed ones updated, the rest skipped. A store that fails mid-fetch keeps what it already saved. |

## Classifying with Claude

| Module | What and why |
|---|---|
| `llm/taxonomy.py` | Loads the 12 categories with their definitions and examples from `config/taxonomy.yaml`. The taxonomy is part of the cached system prompt, so editing it changes the prompt version. |
| `llm/prompts.py` | Builds the system prompt, the strict `record_classification` tool and the delimited, HTML-escaped user message. The prompt version is a hash of prompt and tool, so results from different prompts never mix. |
| `llm/schemas.py` | The Pydantic model that validates every tool call: allowed categories, primary listed first, length caps. The strict tool schema can't express all of that, so Pydantic enforces the rest. |
| `llm/pricing.py` | The token price table and usage totals behind every cost line the CLI prints. It also estimates a classification run before it happens (`classify --dry-run`). |
| `llm/client.py` | Creates the Anthropic client with retries. It is one small seam that tests replace with a fake. |
| `classify.py` | Classifies unclassified reviews either synchronously with a thread pool, warming the prompt cache first, or through the Message Batches API at half price. Invalid output is retried once and then recorded as a failure, never dropped. |

## Measuring quality

| Module | What and why |
|---|---|
| `evals/dataset.py` | The labeled reference set: deterministic sampling, a stable hash-based 70/30 dev/test split, and JSONL storage that includes the review text. The eval then runs from a fresh clone without the database. |
| `evals/labeling.py` | The blind, keyboard-driven labeling loop behind `review-intel label`. The labeler never sees the model's answer, so labels can't anchor on it. |
| `evals/metrics.py` | Accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix, written by hand. Each number is easy to explain, and failures get their own column instead of disappearing. |
| `evals/judge.py` | An LLM judge (Sonnet, structured output) that scores whether a summary is supported by its sources. A stronger model grades the classifier's summaries and the report's prose. |
| `evals/runner.py` | Runs the eval, caches predictions per prompt version, compares against the previous run and writes the README table. Re-running an unchanged prompt is free. |

## Linking complaints to releases

| Module | What and why |
|---|---|
| `patch_notes.py` | Fetches the App Store version history (25 versions with dates) or imports notes from markdown, and splits notes into lines. Version numbers are shared across stores, so one note serves both. |
| `rag/embedder.py` | Local `sentence-transformers` embeddings behind a small protocol, loaded lazily. Embedding is free and private, and tests use a deterministic fake. |
| `rag/index.py` | Stores review and patch-line vectors in `sqlite-vec` tables inside the same SQLite file, and re-embeds only changed text. Search results join straight back to SQL rows. |
| `rag/stats.py` | Loads classified reviews and counts them per version. It is the shared SQL source for timelines and spikes. |
| `rag/ask.py` | Answers a question from retrieved reviews and patch notes plus an SQL timeline, using Sonnet. Every `[r<id>]` and `[v<version>]` citation is checked against what was retrieved, and invented ones are removed. |
| `rag/spikes.py` | Flags categories whose share jumped in a version, per store, with a two-proportion z-test, and ranks the in-force patch-note lines by similarity to the complaints. When nothing is significant, it lists the closest candidates, labeled as such. |

## Serving answers

| Module | What and why |
|---|---|
| `insights.py` | Read-only SQL answers: top issues, version comparison, sentiment trend, search, period summary and category breakdown, as Pydantic models. MCP, the report and the dashboard all read through it, so their numbers agree. |
| `report.py` | Builds the weekly report: code computes and lays out the facts, Sonnet writes only the prose, and code checks the prose's numbers and review ids. A judge score in the footer tells readers it was checked. |
| `slack.py` | Posts a short version of the report to a Slack incoming webhook, only when asked. The webhook URL is a secret read from `.env`. |
| `dashboard/app.py` | The read-only FastAPI app: HTML page and JSON routes over `insights`. Report markdown renders with raw HTML off, and Jinja autoescapes everything else. |
| `dashboard/charts.py` | Server-side SVG bar and stacked-column charts with tooltips, so the page needs no JavaScript. Colors are CSS roles validated for color-vision deficiency in light and dark mode. |
| `mcp_server.py` | Exposes the tools, resources and prompts over MCP (stdio or streamable HTTP) with the official SDK's `MCPServer`. Bad input becomes a readable tool error, and the embedding model warms up in the background. |
| `cli.py` | The thin Typer CLI that wires settings to the library functions. Logic lives in the modules above, so tests and the MCP server call the same code. |

## Claude Code integration

| Path | What and why |
|---|---|
| `CLAUDE.md` | Commands, layout, conventions and the "run tests and evals" rule, loaded into every Claude Code session. It keeps new sessions from relearning the project. |
| `.mcp.json` | Registers `review-intel-mcp` for Claude Code sessions opened in this folder. The tools are then one approval away. |
| `.claude/skills/` | `weekly-review-report`, `add-new-game` and `improve-classifier`: repeatable workflows that state their cost before spending money. `improve-classifier` never edits the prompt without showing the diff. |
| `.claude/agents/` | `bug-analyst`, `sentiment-analyst` and `report-reviewer`: read-only subagents limited to the free MCP tools. Each returns a short, number-backed result. |
