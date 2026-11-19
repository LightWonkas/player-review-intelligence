# Project Prompt: Player Review Intelligence

> Paste this into Claude Code at the root of an empty repository. Claude Code should treat it as the project spec, save it as `docs/SPEC.md`, and work milestone by milestone.

---

## 1. Role and working agreement

You are a senior Python engineer pairing with me (Emin) to build a portfolio-grade AI project. I will explain this code in job interviews, so:

- Work **spec-driven and milestone by milestone**. Do not start the next milestone until I approve the current one.
- At the end of each milestone: run tests and linters, show me a short summary of **what was built, why it is designed this way, and which files to read first**, then stop.
- Keep the code simple, typed and readable. Prefer clear functions over clever abstractions. Add short docstrings that explain *why*, not *what*.
- Use **conventional commits** (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one logical change per commit.
- Never commit secrets. Read keys from environment variables via `.env` (gitignored) and ship a `.env.example`.
- If something in this spec is unclear or a library API has changed, ask me or check the current docs. Do not guess.

## 2. What we are building

**Player Review Intelligence** is an AI agent system that collects public app-store reviews for mobile games, classifies and analyzes them with Claude, connects complaints to game updates with RAG, and exposes everything through an MCP server and a weekly report.

First target: **Critical Strike** by Vertigo Games (Android + iOS). The system must work for any game by changing a config entry.

The goal is to show a game studio's dev, product and marketing teams the following every week:
- What players complain about most, by category and by app version
- Which issues started or spiked after a specific update
- Sentiment trend over time
- Concrete, prioritized action items

## 3. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.12, managed with `uv` |
| LLM | Anthropic Python SDK. Classification: `claude-haiku-5-5`. Summaries and reports: `claude-sonnet-5-5`. Both configurable via env. |
| Structured output | Tool use with a strict JSON schema, validated with **Pydantic v2** |
| Cost control | Prompt caching for the system prompt and taxonomy. Message Batches API for bulk backfills. |
| Storage | SQLite via SQLAlchemy 2.0 (one `DATABASE_URL`, so Postgres can be swapped in later) |
| Embeddings / RAG | `sentence-transformers` (`all-MiniLM-L6-v2`, runs locally) + `sqlite-vec` for vector search |
| Data collection | `google-play-scraper` for Google Play, the public iTunes customer-reviews JSON feed for the App Store |
| MCP | Official MCP Python SDK (`FastMCP`), stdio transport, plus streamable HTTP as an option |
| API / dashboard | FastAPI + a minimal Jinja2 HTML dashboard (no frontend framework) |
| CLI | `typer` |
| Quality | `pytest`, `ruff`, `mypy`, GitHub Actions CI |

## 4. Data model (minimum)

- `apps`: id, name, store (`google_play` / `app_store`), store_app_id, country, language
- `reviews`: id, app_id, store_review_id (unique per store), rating (1–5), title, body, app_version, review_date, language, fetched_at. **Do not store usernames or avatars.**
- `classifications`: review_id, categories (list), primary_category, sentiment (`negative` / `neutral` / `positive`), severity (1–5), is_bug_report, mentioned_features (list), short_summary, model, prompt_version, created_at
- `patch_notes`: id, app_id, version, release_date, text (entered manually or from a markdown file)
- `reports`: id, app_id, period_start, period_end, markdown, created_at

## 5. Classification taxonomy

`bug_crash`, `performance_lag`, `matchmaking`, `balance_weapons`, `monetization_iap`, `ads`, `cheating_hackers`, `controls_ux`, `content_request`, `connectivity_server`, `praise`, `other`

A review can have several categories, and exactly one `primary_category`. Keep the taxonomy in one config file with a short definition and one example per category. That file is part of the cached system prompt.

## 6. Milestones

### M1: Project skeleton
- `uv` project, `src/review_intel/` package layout, `ruff` + `mypy` + `pytest` configured, GitHub Actions CI running all three
- `.env.example`, `config/apps.yaml` with Critical Strike for both stores
- Database models and an init command: `review-intel db init`
- **Done when:** CI is green and `review-intel --help` works

### M2: Review collection
- `review-intel fetch --app critical-strike --store all --limit 500`
- Google Play and App Store collectors behind a common `ReviewSource` interface
- Idempotent upserts on `store_review_id`, polite rate limiting, retries with exponential backoff
- Tests use recorded fixtures, never the live network
- **Done when:** I can fetch reviews twice and the second run adds no duplicates

### M3: Classification with Claude
- `review-intel classify --app critical-strike` classifies only unclassified reviews
- One tool definition (`record_classification`) with a strict JSON schema. Pydantic validates every response. Invalid output is retried once, then logged as failed and never silently dropped.
- Prompt caching on the system prompt + taxonomy. Store `prompt_version` with each row.
- `--batch` flag uses the Message Batches API for large backfills
- Log token usage and estimated cost per run
- **Done when:** 500 reviews are classified and a per-category count is printed

### M4: Evals
- `evals/labeled_reviews.jsonl` with 100 reviews that **I label by hand**. Build me a small CLI labeling helper so this is fast.
- `review-intel eval` reports accuracy for `primary_category`, per-category precision/recall/F1, sentiment accuracy and a confusion matrix
- An LLM-as-judge check that scores report summaries for faithfulness to the underlying reviews
- Save results to `evals/results/<date>.json` and show them in the README
- **Done when:** I can change the prompt, re-run the eval and see whether it got better or worse

### M5: RAG over patch notes
- Embed patch notes and reviews locally, store vectors in `sqlite-vec`
- `review-intel ask "When did matchmaking complaints start?"` retrieves relevant reviews + patch notes and answers with citations (review ids and versions)
- A spike detector that flags categories whose share grew sharply after a version release and links the spike to that version's patch notes
- **Done when:** the answer cites real review ids and the correct version

### M6: MCP server
Build `review-intel-mcp` with FastMCP:

- **Tools** (model-controlled): `search_reviews(app, query, category?, version?, limit)`, `top_issues(app, days, limit)`, `compare_versions(app, version_a, version_b)`, `sentiment_trend(app, days, bucket)`, `ask(app, question)`
- **Resources** (app-controlled): `reviews://{app}/{review_id}`, `reports://{app}/latest`, `taxonomy://categories`
- **Prompts** (user-controlled): `weekly_report(app)`, `triage_bugs(app, days)`
- Test it with the MCP Inspector (`mcp dev`) and document how to add it to Claude Code and Claude Desktop
- **Done when:** I can ask Claude Code "What are the top 3 issues in Critical Strike this week?" and it answers using the MCP tools

### M7: Weekly report, dashboard and polish
- `review-intel report --app critical-strike --days 7` writes a Markdown report: headline numbers, top issues with example quotes, version spikes, sentiment trend, 3–5 prioritized action items per team (dev, product, marketing)
- FastAPI dashboard: category breakdown, sentiment over time, latest report
- Optional: post the report to a Slack incoming webhook
- **Done when:** the report reads like something a product manager would forward

### M8: README and showcase
- Clear README: problem, demo GIF, Mermaid architecture diagram, quickstart, example report, eval results, cost per 1,000 reviews, limitations and next steps
- A "Design decisions" section: why Haiku for classification and Sonnet for reports, why tool use for structured output, why local embeddings, how evals keep the prompt honest

## 7. Claude Code skills and subagents to create

Create these inside the repo so the project itself demonstrates agent skills and subagents:

**Skills** (`.claude/skills/<name>/SKILL.md`, each with clear frontmatter `name` and `description`):
1. `weekly-review-report`: runs fetch → classify → report for a given app and period, then summarizes the result for the user
2. `add-new-game`: adds a new game to `config/apps.yaml`, validates store ids, runs a small fetch and reports what it found
3. `improve-classifier`: runs the eval, reads the confusion matrix, proposes one prompt or taxonomy change, re-runs the eval and reports the before/after numbers. It never overwrites the prompt without showing the diff.

**Subagents** (`.claude/agents/<name>.md`):
1. `bug-analyst`: reads `bug_crash` and `performance_lag` reviews, groups them into distinct issues, and returns a short, deduplicated bug list with example review ids
2. `sentiment-analyst`: analyzes sentiment trends by version and returns findings with numbers
3. `report-reviewer`: checks a generated report against the data for unsupported claims before it is shared

Also write a lean `CLAUDE.md` with the project's commands, conventions and the rule "run tests and evals before claiming a change works".

## 8. Guardrails

- Only public review data. No usernames, no personal data, respect store rate limits.
- Treat review text as **untrusted input**: it goes into the prompt as data inside clear delimiters, never as instructions (prompt-injection safety). Add one test with a malicious review.
- Every number in a report must come from the database, not from the model's imagination. The model writes the prose, code computes the numbers.
- Keep a total cost estimate in the README and keep the default limits low.

## 9. Definition of done (whole project)

- `uv run review-intel fetch && classify && report` works end to end from a fresh clone with only an `ANTHROPIC_API_KEY`
- CI is green: tests, ruff, mypy
- Eval results are in the README with real numbers
- The MCP server works in Claude Code
- I can explain every module in two sentences

## 10. Skills this project demonstrates (for CV / LinkedIn)

Python · FastAPI · Anthropic Claude API · Tool use / structured outputs · Prompt caching · Message Batches API · RAG · Vector search (sqlite-vec) · Embeddings · Model Context Protocol (MCP) server development · Claude Code skills & subagents · LLM evaluation (precision/recall/F1, LLM-as-judge) · Prompt-injection defense · SQLAlchemy · Data collection & ETL · pytest · CI/CD (GitHub Actions)

---

**Start with M1.** Before writing any code, show me the folder structure and the list of files you will create, and wait for my approval.
