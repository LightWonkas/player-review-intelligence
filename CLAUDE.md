# Player Review Intelligence

AI system that collects public app-store reviews for mobile games, classifies them with Claude, links complaints to releases and reports weekly. Full spec: `docs/SPEC.md`. We build it milestone by milestone; do not start the next milestone without approval.

## Commands

```bash
uv sync                          # install
uv run review-intel --help       # CLI
uv run review-intel db init      # create tables + register games from config/apps.yaml
uv run review-intel fetch --app critical-strike --store all --limit 100
uv run python scripts/record_fixtures.py   # re-record test fixtures (live network, by hand only)
uv run review-intel classify --app critical-strike --limit 100   # costs money: Claude API
uv run review-intel classify --app critical-strike --limit 500 --batch   # half price, async
uv run review-intel label --app critical-strike   # hand-label the eval set (blind)
uv run review-intel eval --write-readme           # score vs labels (~$0.08; cached per prompt version)
uv run review-intel patches fetch --app critical-strike   # App Store version history (1 request)
uv run review-intel index --app critical-strike   # local embeddings into sqlite-vec (free)
uv run review-intel ask "When did matchmaking complaints start?" --app critical-strike   # ~$0.02, Sonnet
uv run review-intel spikes --app critical-strike  # per-store category jumps by version (no LLM)
uv run review-intel-mcp                           # MCP server (stdio); also wired up in .mcp.json
uv run review-intel report --app critical-strike --days 7   # weekly report (~$0.03, Sonnet + judge)
uv run review-intel dashboard                     # FastAPI dashboard on http://127.0.0.1:8765
npx @modelcontextprotocol/inspector --cli uv run review-intel-mcp --method tools/list
uv run pytest                    # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                      # strict type check
```

## Layout

- `src/review_intel/cli.py`: thin Typer CLI; logic lives in library modules
- `src/review_intel/db/`: SQLAlchemy models, engine setup, apps sync
- `src/review_intel/collectors/`: `ReviewSource` interface + Google Play / App Store collectors
- `src/review_intel/ingest.py`: idempotent upserts; `retry.py`: backoff + request pacing
- `src/review_intel/llm/`: taxonomy, prompt + strict tool, Pydantic schema, pricing, client
- `src/review_intel/classify.py`: sync / batch classification, failures, category counts
- `config/taxonomy.yaml`: categories; editing it changes `prompt_version` automatically
- `src/review_intel/evals/`: labeled set + dev/test split, metrics, eval runner, Sonnet judge
- `evals/`: `labeled_reviews.jsonl` (hand labels), `predictions/` (cache), `results/` (one JSON per run)
- `src/review_intel/patch_notes.py`: App Store version history + markdown notes, line chunks
- `src/review_intel/rag/`: embedder, sqlite-vec index, SQL stats, `ask` (citations checked), spikes
- `src/review_intel/insights.py`: SQL answers (top issues, version compare, trend, search) shared by MCP and reports
- `src/review_intel/report.py`: weekly report (facts from SQL, Sonnet prose, number/id checks, judge); `slack.py`
- `src/review_intel/dashboard/`: FastAPI app, Jinja2 templates, server-side SVG charts
- `src/review_intel/mcp_server.py`: MCP tools/resources/prompts (SDK 2.x `MCPServer` = FastMCP)
- `src/review_intel/apps_config.py`: validated `config/apps.yaml` (games to track)
- `src/review_intel/settings.py`: env / `.env` settings

## Conventions

- Python 3.12, fully typed, `mypy --strict` clean. Docstrings explain *why*, not *what*.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one logical change each.
- Secrets only via `.env` (gitignored). Keep `.env.example` current.
- Datetimes are timezone-aware UTC. Uniqueness is enforced by DB constraints.
- Tests never hit the network or a real LLM; use recorded fixtures. Tests use `FakeEmbedder`, never torch.
- Review text is untrusted input: pass it as delimited data, never as instructions.
- Numbers in reports come from SQL, never from the model.
- Claude calls cost money: keep default limits low and ask before large live runs.
- Haiku 5.5: no `temperature`/`top_p`; forced `tool_choice` is fine here (Sonnet/Opus 5.5 reject it).

## Rule

Run tests and evals before claiming a change works. Tune prompts on the `dev` split; report `test`.
