from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from review_intel.apps_config import load_apps_config
from review_intel.classify import (
    Classifier,
    category_counts,
    classify_sync,
    collect_batch,
    select_unclassified,
    submit_batch,
    wait_for_batch,
)
from review_intel.db.apps import sync_apps
from review_intel.db.models import App, Classification, ClassificationFailure, Review
from review_intel.db.session import make_session_factory
from review_intel.domain import Store
from review_intel.llm.prompts import TOOL_NAME
from review_intel.llm.taxonomy import load_taxonomy

from .conftest import REPO_ROOT
from .fake_anthropic import (
    VALID_INPUT,
    FakeAnthropic,
    FakeBatches,
    always,
    errored_result,
    fake_client,
    refusal_message,
    succeeded_result,
    text_message,
    tool_message,
    user_text,
)

TAXONOMY = load_taxonomy(REPO_ROOT / "config" / "taxonomy.yaml")
SLUG = "critical-strike"
VALID = tool_message(VALID_INPUT)
INVALID = tool_message(VALID_INPUT | {"primary_category": "not_a_category"})


@pytest.fixture
def factory(engine: Engine, apps_config_path: Path) -> Callable[[], Session]:
    factory = make_session_factory(engine)
    with factory() as session:
        sync_apps(session, load_apps_config(apps_config_path))
        app = session.scalars(select(App).where(App.store == Store.GOOGLE_PLAY)).one()
        base = datetime(2026, 11, 1, tzinfo=UTC)
        for i in range(5):
            session.add(
                Review(
                    app_id=app.id,
                    store_review_id=f"r{i}",
                    rating=2,
                    title=None,
                    body=f"review number {i}",
                    app_version="15.0",
                    review_date=base + timedelta(hours=i),
                )
            )
        session.commit()
    return factory


def _classifier(fake: FakeAnthropic, model: str = "claude-haiku-5-5") -> Classifier:
    return Classifier(fake_client(fake), model, TAXONOMY)


def _pending(factory: Callable[[], Session], classifier: Classifier) -> list[Any]:
    with factory() as session:
        return select_unclassified(session, SLUG, classifier.prompt_version, classifier.model, 100)


def _count(factory: Callable[[], Session], model: type[Any]) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def test_classifies_and_stores_every_review(factory: Callable[[], Session]) -> None:
    fake = FakeAnthropic(always(VALID))
    classifier = _classifier(fake)

    summary = classify_sync(factory, classifier, _pending(factory, classifier), concurrency=3)

    assert (summary.classified, summary.failed) == (5, 0)
    assert summary.usage.requests == 5
    assert _count(factory, Classification) == 5
    with factory() as session:
        row = session.scalars(select(Classification)).first()
        assert row is not None
        assert row.prompt_version == classifier.prompt_version
        assert row.mentioned_features == ["shop"]


def test_request_uses_forced_strict_tool_cached_system_and_no_sampling_params(
    factory: Callable[[], Session],
) -> None:
    fake = FakeAnthropic(always(VALID))
    classifier = _classifier(fake)

    classify_sync(factory, classifier, _pending(factory, classifier)[:1])

    params = fake.messages.calls[0]
    assert params["model"] == "claude-haiku-5-5"
    assert params["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert params["tools"][0]["strict"] is True
    assert params["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & set(params)


def test_newest_reviews_are_selected_first(factory: Callable[[], Session]) -> None:
    classifier = _classifier(FakeAnthropic(always(VALID)))

    pending = _pending(factory, classifier)

    assert [r.body for r in pending][:2] == ["review number 4", "review number 3"]


def test_second_run_finds_nothing_left(factory: Callable[[], Session]) -> None:
    classifier = _classifier(FakeAnthropic(always(VALID)))
    classify_sync(factory, classifier, _pending(factory, classifier))

    assert _pending(factory, classifier) == []


def test_other_model_or_prompt_version_counts_as_unclassified(
    factory: Callable[[], Session],
) -> None:
    classify_sync(factory, (c := _classifier(FakeAnthropic(always(VALID)))), _pending(factory, c))

    other_model = _classifier(FakeAnthropic(always(VALID)), model="claude-sonnet-5-5")

    assert len(_pending(factory, other_model)) == 5


def test_invalid_output_is_retried_once(factory: Callable[[], Session]) -> None:
    replies = iter([INVALID, VALID])
    fake = FakeAnthropic(lambda _: next(replies))
    classifier = _classifier(fake)

    summary = classify_sync(factory, classifier, _pending(factory, classifier)[:1])

    assert summary.classified == 1
    assert len(fake.messages.calls) == 2


def test_still_invalid_after_retry_is_recorded_not_dropped(
    factory: Callable[[], Session],
) -> None:
    fake = FakeAnthropic(always(INVALID))
    classifier = _classifier(fake)
    review = _pending(factory, classifier)[0]

    summary = classify_sync(factory, classifier, [review])

    assert (summary.classified, summary.failed) == (0, 1)
    assert len(fake.messages.calls) == 2
    with factory() as session:
        failure = session.scalars(select(ClassificationFailure)).one()
        assert failure.review_id == review.review_id
        assert failure.attempts == 2
        assert "unknown categories" in failure.error
        assert failure.raw_output is not None and "not_a_category" in failure.raw_output


def test_later_success_clears_the_failure(factory: Callable[[], Session]) -> None:
    review = _pending(factory, _classifier(FakeAnthropic(always(VALID))))[0]
    classify_sync(factory, _classifier(FakeAnthropic(always(INVALID))), [review])

    classify_sync(factory, _classifier(FakeAnthropic(always(VALID))), [review])

    assert _count(factory, ClassificationFailure) == 0
    assert _count(factory, Classification) == 1


@pytest.mark.parametrize("reply", [text_message("I think it's a bug."), tool_message({}, "other")])
def test_missing_tool_call_is_invalid(factory: Callable[[], Session], reply: Message) -> None:
    classifier = _classifier(FakeAnthropic(always(reply)))

    summary = classify_sync(factory, classifier, _pending(factory, classifier)[:1])

    assert summary.failed == 1


def test_refusal_is_recorded_without_retry(factory: Callable[[], Session]) -> None:
    fake = FakeAnthropic(always(refusal_message()))
    classifier = _classifier(fake)

    summary = classify_sync(factory, classifier, _pending(factory, classifier)[:1])

    assert summary.failed == 1
    assert len(fake.messages.calls) == 1
    with factory() as session:
        assert "refused" in session.scalars(select(ClassificationFailure.error)).one()


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_transient_api_error_fails_one_review_and_run_continues(
    factory: Callable[[], Session],
) -> None:
    def responder(params: dict[str, Any]) -> Message:
        if "review number 4" in user_text(params):
            raise anthropic.APIConnectionError(request=_request())
        return VALID

    classifier = _classifier(FakeAnthropic(responder))

    summary = classify_sync(factory, classifier, _pending(factory, classifier), concurrency=1)

    assert (summary.classified, summary.failed) == (4, 1)


def test_bad_api_key_stops_the_run(factory: Callable[[], Session]) -> None:
    def responder(_: dict[str, Any]) -> Message:
        response = httpx2.Response(401, request=_request())
        raise anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)

    classifier = _classifier(FakeAnthropic(responder))

    with pytest.raises(anthropic.AuthenticationError):
        classify_sync(factory, classifier, _pending(factory, classifier))


def test_category_counts(factory: Callable[[], Session]) -> None:
    classifier = _classifier(FakeAnthropic(always(VALID)))
    classify_sync(factory, classifier, _pending(factory, classifier))

    with factory() as session:
        counts = category_counts(session, SLUG, classifier.prompt_version, classifier.model)

    assert counts == [("bug_crash", 5)]


def test_batch_flow_retries_failed_items_synchronously(factory: Callable[[], Session]) -> None:
    batches = FakeBatches()
    fake = FakeAnthropic(always(VALID), batches)
    classifier = _classifier(fake)
    pending = _pending(factory, classifier)[:4]

    batch_id = submit_batch(classifier, pending)
    ids = [f"review-{r.review_id}" for r in pending]
    batches.results_data = [
        succeeded_result(ids[0], VALID),
        succeeded_result(ids[1], VALID),
        succeeded_result(ids[2], INVALID),  # invalid -> one sync retry
        errored_result(ids[3]),  # errored -> one sync retry
    ]
    statuses: list[str] = []
    wait_for_batch(
        classifier.client,
        batch_id,
        sleep=lambda _: None,
        on_status=lambda b: statuses.append(b.processing_status),
    )
    summary = collect_batch(factory, classifier, batch_id)

    assert [r["custom_id"] for r in batches.created] == ids
    assert statuses == ["in_progress", "ended"]
    assert (summary.classified, summary.failed) == (4, 0)
    assert summary.batch_usage.requests == 3
    assert summary.usage.requests == 2
    assert len(fake.messages.calls) == 2
    assert _count(factory, Classification) == 4
