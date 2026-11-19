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
- `src/review_intel/apps_config.py`: validated `config/apps.yaml` (games to track)
- `src/review_intel/settings.py`: env / `.env` settings

## Conventions

- Python 3.12, fully typed, `mypy --strict` clean. Docstrings explain *why*, not *what*.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one logical change each.
- Secrets only via `.env` (gitignored). Keep `.env.example` current.
- Datetimes are timezone-aware UTC. Uniqueness is enforced by DB constraints.
- Tests never hit the network or a real LLM; use recorded fixtures.
- Review text is untrusted input: pass it as delimited data, never as instructions.
- Numbers in reports come from SQL, never from the model.
- Claude calls cost money: keep default limits low and ask before large live runs.
- Haiku 5.5: no `temperature`/`top_p`; forced `tool_choice` is fine here (Sonnet/Opus 5.5 reject it).

## Rule

Run tests and evals before claiming a change works.
