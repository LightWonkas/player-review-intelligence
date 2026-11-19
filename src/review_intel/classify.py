"""Classify stored reviews with Claude, one tool call per review.

Flow: pick reviews that have no classification for the current (prompt_version, model)
-> send each as one request -> validate the tool input with Pydantic -> store it.
An invalid answer is retried once; if it is still invalid the review is recorded in
`classification_failures`, never silently dropped.

Two ways to run the same requests:
- synchronous, a few in parallel: results in minutes, full price;
- Message Batches API (`--batch`): results within hours, half price, for backfills.
"""

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic.types import Message, Usage
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatch
from anthropic.types.messages.batch_create_params import Request
from pydantic import ValidationError
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from review_intel.db.models import App, Classification, ClassificationFailure, Review, utcnow
from review_intel.llm.pricing import UsageTotals
from review_intel.llm.prompts import (
    TOOL_NAME,
    ReviewInput,
    build_system_blocks,
    build_tool,
    build_user_message,
    prompt_version,
    tool_choice,
)
from review_intel.llm.schemas import ClassificationResult, parse_classification
from review_intel.llm.taxonomy import Taxonomy

logger = logging.getLogger(__name__)

MAX_TOKENS = 1024
MAX_ATTEMPTS = 2  # first try + one retry, as the spec requires
BATCH_ID_PREFIX = "review-"

# Errors that mean "fix your setup", not "this review failed": stop the whole run.
FATAL_API_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)


class InvalidOutputError(Exception):
    def __init__(self, reason: str, raw_output: str | None) -> None:
        super().__init__(reason)
        self.raw_output = raw_output


@dataclass
class ClassifyOutcome:
    review_id: int
    result: ClassificationResult | None
    error: str | None = None
    raw_output: str | None = None
    attempts: int = 0
    usages: list[Usage] = field(default_factory=list)


class Classifier:
    """Builds requests and turns responses into validated results. Stateless per review,
    so it is safe to share across worker threads."""

    def __init__(self, client: anthropic.Anthropic, model: str, taxonomy: Taxonomy) -> None:
        self.client = client
        self.model = model
        self.prompt_version = prompt_version(taxonomy)
        self._allowed = set(taxonomy.names)
        self._system = build_system_blocks(taxonomy)
        self._tool = build_tool(taxonomy)

    def request_params(self, review: ReviewInput) -> MessageCreateParamsNonStreaming:
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": self._system,
            "tools": [self._tool],
            "tool_choice": tool_choice(),
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": build_user_message(review)}],
        }

    def parse_response(self, message: Message) -> ClassificationResult:
        tool_inputs = [
            block.input
            for block in message.content
            if block.type == "tool_use" and block.name == TOOL_NAME
        ]
        if len(tool_inputs) != 1:
            raise InvalidOutputError(
                f"expected one {TOOL_NAME} call, got {len(tool_inputs)} "
                f"(stop_reason={message.stop_reason})",
                message.to_json(),
            )
        raw = tool_inputs[0]
        if not isinstance(raw, dict):
            raise InvalidOutputError("tool input is not an object", repr(raw))
        try:
            return parse_classification(raw, self._allowed)
        except ValidationError as exc:
            reason = "; ".join(err["msg"] for err in exc.errors())
            raise InvalidOutputError(f"validation failed: {reason}", repr(raw)) from exc

    def classify(self, review: ReviewInput, attempts: int = MAX_ATTEMPTS) -> ClassifyOutcome:
        outcome = ClassifyOutcome(review_id=review.review_id, result=None)
        params = self.request_params(review)
        for _ in range(attempts):
            outcome.attempts += 1
            try:
                message = self.client.messages.create(**params)
            except FATAL_API_ERRORS:
                raise
            except anthropic.APIError as exc:
                # The SDK already retried; give up on this review, keep the run going.
                outcome.error = f"API error: {type(exc).__name__}: {exc}"
                return outcome
            outcome.usages.append(message.usage)
            if message.stop_reason == "refusal":
                category = message.stop_details.category if message.stop_details else None
                outcome.error = f"refused by safety classifier (category={category})"
                return outcome  # retrying the same input would be refused again
            try:
                outcome.result = self.parse_response(message)
                outcome.error = None
                return outcome
            except InvalidOutputError as exc:
                logger.warning("review %s: invalid output: %s", review.review_id, exc)
                outcome.error, outcome.raw_output = str(exc), exc.raw_output
        return outcome


# --- Database helpers -------------------------------------------------------------


def _to_input(review: Review, app: App) -> ReviewInput:
    return ReviewInput(
        review_id=review.id,
        store=app.store,
        rating=review.rating,
        title=review.title,
        body=review.body,
        app_version=review.app_version,
    )


def select_unclassified(
    session: Session, slug: str, version: str, model: str, limit: int
) -> list[ReviewInput]:
    """Newest reviews of `slug` with no classification for this prompt version + model."""
    done = select(Classification.review_id).where(
        Classification.prompt_version == version, Classification.model == model
    )
    rows = session.execute(
        select(Review, App)
        .join(App, Review.app_id == App.id)
        .where(App.slug == slug, Review.id.not_in(done))
        .order_by(Review.review_date.desc(), Review.id)
        .limit(limit)
    ).all()
    return [_to_input(review, app) for review, app in rows]


def load_inputs(session: Session, review_ids: Iterable[int]) -> dict[int, ReviewInput]:
    rows = session.execute(
        select(Review, App)
        .join(App, Review.app_id == App.id)
        .where(Review.id.in_(list(review_ids)))
    ).all()
    return {review.id: _to_input(review, app) for review, app in rows}


def store_outcome(session: Session, classifier: Classifier, outcome: ClassifyOutcome) -> None:
    key = (
        ClassificationFailure.review_id == outcome.review_id,
        ClassificationFailure.prompt_version == classifier.prompt_version,
        ClassificationFailure.model == classifier.model,
    )
    if outcome.result is not None:
        result = outcome.result
        session.add(
            Classification(
                review_id=outcome.review_id,
                categories=result.categories,
                primary_category=result.primary_category,
                sentiment=result.sentiment,
                severity=result.severity,
                is_bug_report=result.is_bug_report,
                mentioned_features=result.mentioned_features,
                short_summary=result.short_summary,
                model=classifier.model,
                prompt_version=classifier.prompt_version,
            )
        )
        session.execute(delete(ClassificationFailure).where(*key))
        return

    failure = session.scalars(select(ClassificationFailure).where(*key)).one_or_none()
    if failure is None:
        failure = ClassificationFailure(
            review_id=outcome.review_id,
            prompt_version=classifier.prompt_version,
            model=classifier.model,
            attempts=0,
        )
        session.add(failure)
    failure.error = outcome.error or "unknown error"
    failure.raw_output = outcome.raw_output
    failure.attempts += outcome.attempts
    failure.last_attempt_at = utcnow()


# --- Runs -------------------------------------------------------------------------


@dataclass
class RunSummary:
    classified: int = 0
    failed: int = 0
    usage: UsageTotals = field(default_factory=UsageTotals)  # full-price requests
    batch_usage: UsageTotals = field(default_factory=UsageTotals)  # half-price batch requests
    seconds: float = 0.0
    batch_id: str | None = None

    def cost_usd(self, model: str) -> float | None:
        sync_cost = self.usage.cost_usd(model)
        batch_cost = self.batch_usage.cost_usd(model, batch=True)
        if sync_cost is None or batch_cost is None:
            return None
        return sync_cost + batch_cost

    def record(self, outcome: ClassifyOutcome) -> None:
        for usage in outcome.usages:
            self.usage.add(usage)
        if outcome.result is not None:
            self.classified += 1
        else:
            self.failed += 1


Progress = Callable[[int, int], None]
COMMIT_EVERY = 25


def classify_sync(
    session_factory: Callable[[], Session],
    classifier: Classifier,
    reviews: Sequence[ReviewInput],
    *,
    concurrency: int = 4,
    on_progress: Progress | None = None,
) -> RunSummary:
    """Classify with a few parallel requests, committing as results arrive so an
    interrupted run keeps its progress (a re-run then only does what is left)."""
    summary = RunSummary()
    started = time.monotonic()
    if not reviews:
        return summary

    with session_factory() as session:

        def handle(outcome: ClassifyOutcome) -> None:
            store_outcome(session, classifier, outcome)
            summary.record(outcome)
            done = summary.classified + summary.failed
            if done % COMMIT_EVERY == 0:
                session.commit()
            if on_progress:
                on_progress(done, len(reviews))

        # The first request runs alone so it writes the prompt cache; the parallel
        # requests after it then read the cache instead of all writing it at once.
        handle(classifier.classify(reviews[0]))
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(classifier.classify, review) for review in reviews[1:]]
            for future in as_completed(futures):
                handle(future.result())
        session.commit()

    summary.seconds = time.monotonic() - started
    return summary


def submit_batch(classifier: Classifier, reviews: Sequence[ReviewInput]) -> str:
    requests = [
        Request(
            custom_id=f"{BATCH_ID_PREFIX}{review.review_id}",
            params=classifier.request_params(review),
        )
        for review in reviews
    ]
    batch = classifier.client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    *,
    poll_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    on_status: Callable[[MessageBatch], None] | None = None,
) -> MessageBatch:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if on_status:
            on_status(batch)
        if batch.processing_status == "ended":
            return batch
        sleep(poll_seconds)


def collect_batch(
    session_factory: Callable[[], Session], classifier: Classifier, batch_id: str
) -> RunSummary:
    """Store batch results. Requests that errored, expired or returned invalid output
    get their one retry through the normal synchronous path."""
    summary = RunSummary(batch_id=batch_id)
    started = time.monotonic()
    outcomes: dict[int, ClassifyOutcome] = {}
    needs_retry: list[int] = []

    for item in classifier.client.messages.batches.results(batch_id):
        review_id = int(item.custom_id.removeprefix(BATCH_ID_PREFIX))
        outcome = ClassifyOutcome(review_id=review_id, result=None, attempts=1)
        result: Any = item.result
        if result.type == "succeeded":
            message: Message = result.message
            summary.batch_usage.add(message.usage)
            if message.stop_reason == "refusal":
                outcome.error = "refused by safety classifier"
            else:
                try:
                    outcome.result = classifier.parse_response(message)
                except InvalidOutputError as exc:
                    outcome.error, outcome.raw_output = str(exc), exc.raw_output
                    needs_retry.append(review_id)
        else:
            outcome.error = f"batch request {result.type}"
            needs_retry.append(review_id)
        outcomes[review_id] = outcome

    with session_factory() as session:
        inputs = load_inputs(session, needs_retry)

    for review_id in needs_retry:
        if review_id not in inputs:
            continue  # review was deleted meanwhile
        retry = classifier.classify(inputs[review_id], attempts=1)
        retry.attempts += 1
        if retry.result is None and retry.raw_output is None:
            retry.raw_output = outcomes[review_id].raw_output
        outcomes[review_id] = retry

    with session_factory() as session:
        for outcome in outcomes.values():
            store_outcome(session, classifier, outcome)
            # Only synchronous retries carry usages here; batch usage was counted above.
            summary.record(outcome)
        session.commit()

    summary.seconds = time.monotonic() - started
    return summary


def category_counts(session: Session, slug: str, version: str, model: str) -> list[tuple[str, int]]:
    rows = session.execute(
        select(Classification.primary_category, func.count())
        .join(Review, Classification.review_id == Review.id)
        .join(App, Review.app_id == App.id)
        .where(
            App.slug == slug,
            Classification.prompt_version == version,
            Classification.model == model,
        )
        .group_by(Classification.primary_category)
        .order_by(func.count().desc(), Classification.primary_category)
    ).all()
    return [(category, count) for category, count in rows]
