"""`review-intel` command-line interface.

The CLI is a thin layer: it reads settings, calls library functions and prints
results. Logic lives in modules that tests (and later the MCP server) call directly.
"""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import anthropic
import httpx
import typer
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from review_intel import __version__
from review_intel.apps_config import AppsConfig, GameConfig, load_apps_config
from review_intel.classify import (
    FATAL_API_ERRORS,
    Classifier,
    RunSummary,
    category_counts,
    classify_sync,
    collect_batch,
    select_unclassified,
    submit_batch,
    wait_for_batch,
)
from review_intel.collectors import build_sources
from review_intel.dashboard.app import DashboardDeps, create_app
from review_intel.db.apps import SyncResult, sync_apps
from review_intel.db.models import App
from review_intel.db.session import init_db, make_engine, make_session_factory
from review_intel.domain import Store
from review_intel.evals.dataset import (
    DEFAULT_LABELS_PATH,
    load_labels,
    sample_candidates,
)
from review_intel.evals.labeling import run_labeling
from review_intel.evals.runner import (
    evaluate,
    format_report,
    latest_judge,
    load_reports,
    predict,
    previous_report,
    render_readme_block,
    run_judge,
    save_report,
    update_readme,
)
from review_intel.ingest import fetch_and_store
from review_intel.llm.client import make_client
from review_intel.llm.pricing import UsageTotals, estimate_classification_cost
from review_intel.llm.prompts import prompt_version
from review_intel.llm.taxonomy import load_taxonomy
from review_intel.patch_notes import (
    EmptyPageError,
    NotesResult,
    fetch_app_store_history,
    load_markdown,
    store_notes,
)
from review_intel.rag.ask import answer_question, build_context, render_context
from review_intel.rag.embedder import Embedder, SentenceTransformerEmbedder
from review_intel.rag.index import IndexMissingError, index_game
from review_intel.rag.spikes import Spike, SpikeSettings, detect_spikes, near_misses
from review_intel.report import ReportError, default_report_path, generate_report, store_report
from review_intel.settings import get_settings
from review_intel.slack import post_to_slack, slack_text

app = typer.Typer(
    help="Player Review Intelligence: collect, classify and report on app-store reviews.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")

DatabaseUrlOption = Annotated[
    str | None, typer.Option(help="Override DATABASE_URL from the environment.")
]
AppsConfigOption = Annotated[
    Path | None, typer.Option(help="Override APPS_CONFIG_PATH (default config/apps.yaml).")
]


class StoreChoice(StrEnum):
    ALL = "all"
    GOOGLE_PLAY = "google_play"
    APP_STORE = "app_store"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"review-intel {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Player Review Intelligence: collect, classify and report on app-store reviews."""
    # WARNING level: retries and failures are visible, routine progress is not.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _prepare_db(
    database_url: str | None, apps_config: Path | None
) -> tuple[Engine, AppsConfig, SyncResult]:
    """Create tables and register configured games. Every command starts here, so a
    fresh clone never needs a separate `db init` step."""
    settings = get_settings()
    config = load_apps_config(apps_config or settings.apps_config_path)
    engine = make_engine(database_url or settings.database_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        sync = sync_apps(session, config)
        session.commit()
    return engine, config, sync


def _get_game(engine: Engine, config: AppsConfig, slug: str) -> GameConfig:
    try:
        return config.get_game(slug)
    except KeyError as exc:
        engine.dispose()
        raise typer.BadParameter(str(exc.args[0]), param_hint="--app") from exc


@db_app.command("init")
def db_init(database_url: DatabaseUrlOption = None, apps_config: AppsConfigOption = None) -> None:
    """Create database tables and register the games from apps.yaml. Safe to re-run."""
    engine, _, sync = _prepare_db(database_url, apps_config)
    tables = init_db(engine)
    # Hide any password so the URL is safe to print in logs and screenshots.
    safe_url = engine.url.render_as_string(hide_password=True)
    engine.dispose()

    typer.echo(f"Database ready: {safe_url}")
    typer.echo(f"Tables: {', '.join(tables)}")
    typer.echo(f"Apps: {sync.created} added, {sync.updated} updated, {sync.unchanged} unchanged")


@app.command()
def fetch(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    store: Annotated[StoreChoice, typer.Option(help="Which store to fetch.")] = StoreChoice.ALL,
    limit: Annotated[int, typer.Option(min=1, help="Max reviews per store, newest first.")] = 100,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Fetch the newest public reviews and save them. Re-running adds no duplicates."""
    engine, config, _ = _prepare_db(database_url, apps_config)
    game = _get_game(engine, config, app_slug)

    stores = list(Store) if store is StoreChoice.ALL else [Store(store.value)]
    configured = {listing.store for listing in game.listings}
    if not configured.intersection(stores):
        engine.dispose()
        raise typer.BadParameter(f"{app_slug} has no {store.value} listing", param_hint="--store")

    results = fetch_and_store(
        make_session_factory(engine), config, app_slug, stores, limit, build_sources()
    )
    engine.dispose()

    for result in results:
        up = result.upsert
        line = (
            f"{result.store.value:<12} fetched {result.fetched:>4} -> "
            f"{up.inserted} new, {up.updated} updated, {up.unchanged} unchanged"
        )
        if result.error:
            line += f"  [ERROR: {result.error}]"
        typer.echo(line)

    if any(result.error for result in results):
        raise typer.Exit(code=1)


@app.command()
def classify(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    limit: Annotated[int, typer.Option(min=1, help="Max reviews to classify, newest first.")] = 100,
    concurrency: Annotated[
        int, typer.Option(min=1, max=16, help="Parallel requests (synchronous mode).")
    ] = 4,
    batch: Annotated[
        bool, typer.Option("--batch", help="Use the Message Batches API (half price, slower).")
    ] = False,
    resume: Annotated[
        str | None, typer.Option(help="Collect an earlier batch by id instead of submitting.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Only count the reviews and estimate the cost.")
    ] = False,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Classify reviews that have no result yet for the current prompt version and model."""
    settings = get_settings()
    engine, config, _ = _prepare_db(database_url, apps_config)
    _get_game(engine, config, app_slug)
    factory = make_session_factory(engine)
    if dry_run:
        # No API call and no key needed: lets scripts and skills ask before spending.
        version = _current_prompt_version()
        with factory() as session:
            count = len(
                select_unclassified(session, app_slug, version, settings.classify_model, limit)
            )
        engine.dispose()
        sync = estimate_classification_cost(count, settings.classify_model)
        batch_cost = estimate_classification_cost(count, settings.classify_model, batch=True)
        estimate = (
            f"about ${sync:.4f} sync or ${batch_cost:.4f} with --batch"
            if sync is not None and batch_cost is not None
            else "cost unknown (model not in the price table)"
        )
        typer.echo(
            f"Would classify {count} reviews with {settings.classify_model} "
            f"(prompt {version}): {estimate}"
        )
        return
    try:
        classifier = Classifier(
            make_client(settings), settings.classify_model, load_taxonomy(settings.taxonomy_path)
        )
        typer.echo(f"Model {classifier.model} | prompt {classifier.prompt_version}")
        started = time.monotonic()
        summary = _run_classification(
            classifier, factory, app_slug, limit, concurrency, batch, resume
        )
        if summary is not None:
            # Wall-clock for the whole command, including time spent waiting on a batch.
            summary.seconds = time.monotonic() - started
    except anthropic.AuthenticationError as exc:
        engine.dispose()
        typer.echo("Anthropic rejected the API key. Set ANTHROPIC_API_KEY in .env.", err=True)
        raise typer.Exit(code=1) from exc
    except (*FATAL_API_ERRORS, anthropic.AnthropicError) as exc:
        engine.dispose()
        typer.echo(f"Anthropic API error, run stopped: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if summary is not None:
        _print_summary(summary, classifier.model)
    with factory() as session:
        _print_category_counts(session, app_slug, classifier)
    engine.dispose()
    if summary is not None and summary.failed:
        raise typer.Exit(code=1)


def _run_classification(
    classifier: Classifier,
    factory: Callable[[], Session],
    slug: str,
    limit: int,
    concurrency: int,
    batch: bool,
    resume: str | None,
) -> RunSummary | None:
    if resume is None:
        with factory() as session:
            reviews = select_unclassified(
                session, slug, classifier.prompt_version, classifier.model, limit
            )
        if not reviews:
            typer.echo("Nothing to classify: every review has a result for this prompt and model.")
            return None
        typer.echo(f"Classifying {len(reviews)} reviews ({'batch' if batch else 'sync'})...")
        if not batch:
            return classify_sync(
                factory, classifier, reviews, concurrency=concurrency, on_progress=_progress
            )
        resume = submit_batch(classifier, reviews)
        typer.echo(f"Submitted batch {resume}. If this stops, collect it later with:")
        typer.echo(f"  review-intel classify --app {slug} --resume {resume}")

    wait_for_batch(
        classifier.client,
        resume,
        on_status=lambda b: typer.echo(
            f"  batch {b.processing_status}: {b.request_counts.succeeded} succeeded, "
            f"{b.request_counts.processing} processing"
        ),
    )
    return collect_batch(factory, classifier, resume)


def _progress(done: int, total: int) -> None:
    if done % 50 == 0 or done == total:
        typer.echo(f"  {done}/{total}")


def _print_summary(summary: RunSummary, model: str) -> None:
    typer.echo("")
    typer.echo(
        f"Classified {summary.classified}, failed {summary.failed} in {summary.seconds:.0f}s"
    )
    if summary.usage.requests:
        typer.echo(f"  sync:  {summary.usage.describe(model)}")
    if summary.batch_usage.requests:
        typer.echo(f"  batch: {summary.batch_usage.describe(model, batch=True)}")
    cost = summary.cost_usd(model)
    if cost is not None:
        typer.echo(f"  total estimated cost: ${cost:.4f}")
    if summary.failed:
        typer.echo("  Failed reviews are in the classification_failures table; re-run to retry.")


def _print_category_counts(session: Session, slug: str, classifier: Classifier) -> None:
    counts = category_counts(session, slug, classifier.prompt_version, classifier.model)
    total = sum(count for _, count in counts)
    if not total:
        return
    typer.echo("")
    typer.echo(f"Primary category ({total} classified reviews, this prompt + model):")
    for category, count in counts:
        typer.echo(f"  {category:<20} {count:>5}  {count / total:6.1%}")


LabelsFileOption = Annotated[Path, typer.Option(help="Labeled eval set (JSON Lines).")]
SPARE_CANDIDATES = 40  # extra reviews offered so skipped ones can be replaced


@app.command()
def label(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    target: Annotated[int, typer.Option(min=1, help="How many labels the set should have.")] = 100,
    labels_file: LabelsFileOption = DEFAULT_LABELS_PATH,
    seed: Annotated[int, typer.Option(help="Sampling seed; keep it fixed.")] = 7,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Label reviews by hand for the eval set. Blind: the model's answers are not shown.

    Progress is saved after every review; run the command again to continue."""
    settings = get_settings()
    engine, config, _ = _prepare_db(database_url, apps_config)
    _get_game(engine, config, app_slug)
    existing = load_labels(labels_file)
    if len(existing) >= target:
        engine.dispose()
        typer.echo(f"{labels_file} already has {len(existing)} labels (target {target}).")
        return
    with make_session_factory(engine)() as session:
        candidates = sample_candidates(
            session,
            app_slug,
            target + SPARE_CANDIDATES,
            seed=seed,
            exclude={item.key for item in existing},
        )
    engine.dispose()

    typer.echo(f"{len(existing)}/{target} labeled. Keys: number = category, s = skip, q = quit.")
    result = run_labeling(
        candidates,
        load_taxonomy(settings.taxonomy_path),
        labels_file,
        already_labeled=len(existing),
        target=target,
        ask=input,
        say=typer.echo,
    )
    total = len(existing) + result.labeled
    typer.echo("")
    typer.echo(f"Saved {result.labeled} new labels ({total}/{target}) to {labels_file}.")
    if total < target and not result.quit:
        typer.echo("Ran out of candidates; re-run with a different --seed to get more.")


@app.command("eval")
def eval_command(
    labels_file: LabelsFileOption = DEFAULT_LABELS_PATH,
    judge: Annotated[
        int, typer.Option(min=0, help="Summaries to check with the LLM judge (0 = skip).")
    ] = 20,
    concurrency: Annotated[int, typer.Option(min=1, max=16)] = 4,
    write_readme: Annotated[
        bool, typer.Option("--write-readme", help="Update the eval table in README.md.")
    ] = False,
) -> None:
    """Score the classifier against the hand-labeled set and compare with the last run."""
    settings = get_settings()
    labels = load_labels(labels_file)
    if not labels:
        typer.echo(f"No labels in {labels_file}. Run `review-intel label` first.", err=True)
        raise typer.Exit(code=1)

    taxonomy = load_taxonomy(settings.taxonomy_path)
    try:
        client = make_client(settings)
        classifier = Classifier(client, settings.classify_model, taxonomy)
        predictions, classify_usage = predict(labels, classifier, concurrency=concurrency)
        judge_scores, judge_usage = None, UsageTotals()
        if judge:
            judge_scores, judge_usage = run_judge(
                client, settings.judge_model, labels, predictions, judge, concurrency=concurrency
            )
        else:
            judge_scores = latest_judge(load_reports(), classifier.prompt_version, classifier.model)
            if judge_scores is not None:
                typer.echo(
                    "Judge: reusing the last result for this prompt and model "
                    "(same cached summaries); pass --judge N to run it again."
                )
    except anthropic.AuthenticationError as exc:
        typer.echo("Anthropic rejected the API key. Set ANTHROPIC_API_KEY in .env.", err=True)
        raise typer.Exit(code=1) from exc
    except (*FATAL_API_ERRORS, anthropic.AnthropicError) as exc:
        typer.echo(f"Anthropic API error, eval stopped: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report = evaluate(
        labels,
        predictions,
        taxonomy.names,
        prompt_version=classifier.prompt_version,
        model=classifier.model,
        judge=judge_scores,
        cost_usd={
            "classification": classify_usage.cost_usd(classifier.model),
            "judge": judge_usage.cost_usd(settings.judge_model),
        },
    )
    previous = previous_report(load_reports(), report.prompt_version, report.model)
    path = save_report(report)
    typer.echo(format_report(report, previous))
    typer.echo("")
    typer.echo(f"Saved {path}")
    if write_readme:
        update_readme(Path("README.md"), render_readme_block(load_reports()))
        typer.echo("Updated the eval table in README.md")


# --- M5: patch notes, vector index, ask, spikes ------------------------------------

patches_app = typer.Typer(help="Patch notes (release notes per version).", no_args_is_help=True)
app.add_typer(patches_app, name="patches")


def make_embedder(model_name: str) -> Embedder:
    """Indirection so tests can swap in a fake without loading torch."""
    return SentenceTransformerEmbedder(model_name)


def _current_prompt_version() -> str:
    return prompt_version(load_taxonomy(get_settings().taxonomy_path))


def _store_app(session: Session, slug: str, store: Store) -> App | None:
    return session.scalar(select(App).where(App.slug == slug, App.store == store))


def _print_notes_result(result: NotesResult) -> None:
    typer.echo(
        f"Patch notes: {result.created} new, {result.updated} updated, {result.unchanged} unchanged"
    )


@patches_app.command("fetch")
def patches_fetch(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Fetch the App Store version history (~25 versions with dates and notes)."""
    engine, config, _ = _prepare_db(database_url, apps_config)
    game = _get_game(engine, config, app_slug)
    listing = next((item for item in game.listings if item.store is Store.APP_STORE), None)
    if listing is None:
        engine.dispose()
        raise typer.BadParameter(f"{app_slug} has no app_store listing", param_hint="--app")
    try:
        notes = fetch_app_store_history(listing.store_app_id, listing.country)
    except (httpx.HTTPError, EmptyPageError) as exc:
        engine.dispose()
        typer.echo(f"Could not fetch the version history: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    with make_session_factory(engine)() as session:
        store_app = _store_app(session, app_slug, Store.APP_STORE)
        assert store_app is not None  # registered by _prepare_db
        result = store_notes(session, store_app, notes)
        session.commit()
    engine.dispose()
    typer.echo(f"Fetched {len(notes)} versions ({notes[-1].version} to {notes[0].version})")
    _print_notes_result(result)


@patches_app.command("import")
def patches_import(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Markdown with '## <version> (<date>)'."),
    ],
    store: Annotated[
        StoreChoice, typer.Option(help="Store the notes belong to.")
    ] = StoreChoice.GOOGLE_PLAY,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Import hand-written notes from a markdown file."""
    if store is StoreChoice.ALL:
        raise typer.BadParameter("pick one store", param_hint="--store")
    notes = load_markdown(file)
    if not notes:
        typer.echo(f"No '## <version> (<YYYY-MM-DD>)' sections found in {file}.", err=True)
        raise typer.Exit(code=1)
    engine, config, _ = _prepare_db(database_url, apps_config)
    _get_game(engine, config, app_slug)
    with make_session_factory(engine)() as session:
        store_app = _store_app(session, app_slug, Store(store.value))
        if store_app is None:
            engine.dispose()
            raise typer.BadParameter(
                f"{app_slug} has no {store.value} listing", param_hint="--store"
            )
        result = store_notes(session, store_app, notes)
        session.commit()
    engine.dispose()
    _print_notes_result(result)


@app.command()
def index(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Embed new or changed reviews and patch-note lines locally (no API cost)."""
    settings = get_settings()
    engine, config, _ = _prepare_db(database_url, apps_config)
    _get_game(engine, config, app_slug)
    embedder = make_embedder(settings.embedding_model)
    started = time.monotonic()
    with make_session_factory(engine)() as session:
        result = index_game(session, app_slug, embedder)
        session.commit()
    engine.dispose()
    typer.echo(
        f"Embedded {result.reviews_embedded} reviews and {result.chunks_embedded} patch-note "
        f"lines with {embedder.name} ({result.unchanged} unchanged, {result.removed} removed) "
        f"in {time.monotonic() - started:.0f}s"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="A question about the game's reviews.")],
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    reviews: Annotated[int, typer.Option(min=1, max=50, help="Reviews to retrieve.")] = 12,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Print what was retrieved.")
    ] = False,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Answer a question from retrieved reviews and patch notes, with citations."""
    settings = get_settings()
    engine, config, _ = _prepare_db(database_url, apps_config)
    _get_game(engine, config, app_slug)
    embedder = make_embedder(settings.embedding_model)
    with make_session_factory(engine)() as session:
        try:
            context = build_context(
                session,
                app_slug,
                question,
                embedder,
                model=settings.classify_model,
                prompt_version=_current_prompt_version(),
                k_reviews=reviews,
            )
        except IndexMissingError as exc:
            engine.dispose()
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    engine.dispose()
    if show_context:
        typer.echo(render_context(context))
        typer.echo("")
    try:
        result = answer_question(make_client(settings), settings.report_model, context)
    except anthropic.AuthenticationError as exc:
        typer.echo("Anthropic rejected the API key. Set ANTHROPIC_API_KEY in .env.", err=True)
        raise typer.Exit(code=1) from exc
    if result.error:
        typer.echo(f"No answer: {result.error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(result.answer)
    typer.echo("")
    cited_reviews = ", ".join(f"r{i}" for i in result.cited_reviews) or "none"
    cited_versions = ", ".join(result.cited_versions) or "none"
    typer.echo(f"Sources: reviews {cited_reviews} | versions {cited_versions}")
    for note in result.notes:
        typer.echo(f"Note: {note}")
    if result.invalid_citations:
        removed = ", ".join(result.invalid_citations)
        typer.echo(f"Removed citations not in the context: {removed}", err=True)
    if result.usage is not None:
        totals = UsageTotals()
        totals.add(result.usage)
        typer.echo(f"{settings.report_model}: {totals.describe(settings.report_model)}")


@app.command()
def spikes(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    min_reviews: Annotated[
        int, typer.Option(min=2, help="Classified reviews needed on each side.")
    ] = SpikeSettings.min_reviews,
    min_increase: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share increase, e.g. 0.10 = 10 points.")
    ] = SpikeSettings.min_increase,
    link_lines: Annotated[
        bool, typer.Option(help="Rank patch-note lines by similarity (loads the embedder).")
    ] = True,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Flag categories whose share jumped in a version, linked to that version's notes."""
    settings = get_settings()
    engine, config, _ = _prepare_db(database_url, apps_config)
    _get_game(engine, config, app_slug)
    embedder = make_embedder(settings.embedding_model) if link_lines else None
    spike_settings = SpikeSettings(min_reviews=min_reviews, min_increase=min_increase)
    version = _current_prompt_version()
    with make_session_factory(engine)() as session:
        found = detect_spikes(
            session,
            app_slug,
            model=settings.classify_model,
            prompt_version=version,
            settings=spike_settings,
            embedder=embedder,
        )
        closest = (
            []
            if found
            else near_misses(
                session,
                app_slug,
                model=settings.classify_model,
                prompt_version=version,
                settings=spike_settings,
            )
        )
    engine.dispose()
    if not found:
        typer.echo(
            f"No significant spikes (needs +{spike_settings.min_increase:.0%} points, "
            f"z >= {spike_settings.min_z}, {spike_settings.min_reviews}+ reviews per side)."
        )
        if closest:
            typer.echo("Closest candidates, NOT significant:")
            for spike in closest:
                typer.echo(f"  {_describe_spike(spike)}")
        return
    for spike in found:
        typer.echo(_describe_spike(spike))
        if spike.note_version:
            typer.echo(f"  patch note in force: v{spike.note_version} ({spike.note_date})")
            for line, similarity in spike.note_lines:
                typer.echo(f"    {similarity:.2f}  {line}")
        else:
            typer.echo("  no patch note for this version")
        typer.echo("  examples: " + ", ".join(f"r{i}" for i in spike.example_review_ids))


def _describe_spike(spike: Spike) -> str:
    return (
        f"{spike.store.value} v{spike.version}: {spike.category} "
        f"{spike.before_share:.0%} -> {spike.after_share:.0%} "
        f"({spike.after_count}/{spike.after_n} vs {spike.before_count}/{spike.before_n} "
        f"in v{', v'.join(spike.baseline_versions)}; z={spike.z:.1f})"
    )


# --- M7: weekly report and dashboard ------------------------------------------------


@app.command()
def report(
    app_slug: Annotated[str, typer.Option("--app", help="Game slug from apps.yaml.")],
    days: Annotated[int, typer.Option(min=1, max=90, help="Length of the period.")] = 7,
    end: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Period ends at the start of this day (UTC)."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Markdown file to write.")] = None,
    judge: Annotated[bool, typer.Option(help="Score the prose with the LLM judge.")] = True,
    post_slack: Annotated[
        bool, typer.Option("--post-slack", help="Also post a summary to SLACK_WEBHOOK_URL.")
    ] = False,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Write the weekly report: numbers from the database, prose by Claude (~$0.04)."""
    settings = get_settings()
    if post_slack and settings.slack_webhook_url is None:
        raise typer.BadParameter("set SLACK_WEBHOOK_URL in .env first", param_hint="--post-slack")
    engine, config, _ = _prepare_db(database_url, apps_config)
    game = _get_game(engine, config, app_slug)
    now = end.replace(tzinfo=UTC) if end else None
    try:
        with make_session_factory(engine)() as session:
            run = generate_report(
                session,
                make_client(settings),
                slug=app_slug,
                game_name=game.name,
                classify_model=settings.classify_model,
                prompt_version=_current_prompt_version(),
                report_model=settings.report_model,
                judge_model=settings.judge_model if judge else None,
                days=days,
                now=now,
                embedder=make_embedder(settings.embedding_model),
            )
            row = store_report(session, run)
            session.commit()
    except anthropic.AuthenticationError as exc:
        engine.dispose()
        typer.echo("Anthropic rejected the API key. Set ANTHROPIC_API_KEY in .env.", err=True)
        raise typer.Exit(code=1) from exc
    except ReportError as exc:
        engine.dispose()
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    engine.dispose()

    path = out or Path(default_report_path(app_slug, run.facts.period_end.date()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.markdown, encoding="utf-8")
    typer.echo(f"Wrote {path} (report #{row.id}, {run.facts.summary.current.reviews} reviews)")
    if run.judge is not None and run.judge.verdict is not None:
        typer.echo(f"Judge: {run.judge.verdict.score}/5")
    for warning in run.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    for model_name, usage in run.usage.items():
        typer.echo(f"{model_name}: {usage.describe(model_name)}")
    if post_slack and settings.slack_webhook_url is not None:
        try:
            post_to_slack(settings.slack_webhook_url.get_secret_value(), slack_text(run))
        except httpx.HTTPError as exc:
            typer.echo(f"Slack post failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo("Posted the summary to Slack.")


@app.command()
def dashboard(
    host: Annotated[str, typer.Option(help="Keep 127.0.0.1 unless you mean to share it.")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    database_url: DatabaseUrlOption = None,
    apps_config: AppsConfigOption = None,
) -> None:
    """Serve the read-only dashboard (FastAPI) on http://127.0.0.1:8765."""
    import uvicorn

    settings = get_settings()
    engine, config, _ = _prepare_db(database_url, apps_config)
    deps = DashboardDeps(
        session_factory=make_session_factory(engine),
        apps=config,
        classify_model=settings.classify_model,
        prompt_version=_current_prompt_version(),
    )
    typer.echo(f"Dashboard on http://{host}:{port}")
    uvicorn.run(create_app(deps), host=host, port=port, log_level="warning")
