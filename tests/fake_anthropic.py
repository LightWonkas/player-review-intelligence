"""An offline stand-in for the Anthropic client.

Responses are real SDK model objects (`Message`, `MessageBatch`, ...) built with
`model_validate`, so the code under test sees exactly the types it gets in production.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

import anthropic
from anthropic.types import Message
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from review_intel.llm.prompts import TOOL_NAME

VALID_INPUT: dict[str, Any] = {
    "categories": ["bug_crash"],
    "primary_category": "bug_crash",
    "sentiment": "negative",
    "severity": 4,
    "is_bug_report": True,
    "mentioned_features": ["Shop"],
    "short_summary": "The game crashes when opening the shop.",
}


def _usage(
    input_tokens: int = 120, output_tokens: int = 60, cache_read: int = 1500
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read,
    }


def _message(content: list[dict[str, Any]], stop_reason: str, **extra: Any) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _usage(),
            **extra,
        }
    )


def tool_message(tool_input: dict[str, Any], name: str = TOOL_NAME) -> Message:
    block = {"type": "tool_use", "id": "toolu_test", "name": name, "input": tool_input}
    return _message([block], "tool_use")


def text_message(text: str) -> Message:
    return _message([{"type": "text", "text": text}], "end_turn")


def refusal_message() -> Message:
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    return _message([], "refusal", stop_details=details)


Responder = Callable[[dict[str, Any]], Message]


def user_text(params: dict[str, Any]) -> str:
    content: str = params["messages"][0]["content"]
    return content


class FakeBatches:
    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.created: list[Any] = []
        self.results_data = results or []
        self.retrieve_calls = 0

    def _batch(self, status: str) -> MessageBatch:
        now = datetime(2026, 11, 19, tzinfo=UTC).isoformat()
        return MessageBatch.model_validate(
            {
                "id": "msgbatch_test",
                "type": "message_batch",
                "processing_status": status,
                "request_counts": {
                    "processing": 0 if status == "ended" else len(self.created),
                    "succeeded": 0,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0,
                },
                "created_at": now,
                "expires_at": now,
                "ended_at": now if status == "ended" else None,
                "archived_at": None,
                "cancel_initiated_at": None,
                "results_url": None,
            }
        )

    def create(self, *, requests: list[Any]) -> MessageBatch:
        self.created = list(requests)
        return self._batch("in_progress")

    def retrieve(self, batch_id: str) -> MessageBatch:
        self.retrieve_calls += 1
        return self._batch("ended" if self.retrieve_calls >= 2 else "in_progress")

    def results(self, batch_id: str) -> list[MessageBatchIndividualResponse]:
        return [MessageBatchIndividualResponse.model_validate(r) for r in self.results_data]


class FakeMessages:
    def __init__(self, responder: Responder, batches: FakeBatches) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []
        self.batches = batches

    def create(self, **params: Any) -> Message:
        self.calls.append(params)
        return self._responder(params)


class FakeAnthropic:
    def __init__(self, responder: Responder, batches: FakeBatches | None = None) -> None:
        self.messages = FakeMessages(responder, batches or FakeBatches())


def fake_client(fake: FakeAnthropic) -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, fake)


def always(message: Message) -> Responder:
    return lambda _params: message


def succeeded_result(custom_id: str, message: Message) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "result": {"type": "succeeded", "message": message.model_dump()},
    }


def errored_result(custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "result": {
            "type": "errored",
            "error": {
                "type": "error",
                "request_id": "req_test",
                "error": {"type": "api_error", "message": "overloaded"},
            },
        },
    }


class FakeParsedResponse:
    """The parts of a parsed beta message the judge reads."""

    def __init__(self, parsed_output: Any, stop_reason: str = "end_turn") -> None:
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.usage = Message.model_validate(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": _usage(input_tokens=400, output_tokens=80, cache_read=0),
            }
        ).usage


class FakeBetaMessages:
    def __init__(self, parse_responder: Callable[[dict[str, Any]], Any]) -> None:
        self._responder = parse_responder
        self.parse_calls: list[dict[str, Any]] = []

    def parse(self, **params: Any) -> Any:
        self.parse_calls.append(params)
        return self._responder(params)


class FakeBeta:
    def __init__(self, parse_responder: Callable[[dict[str, Any]], Any]) -> None:
        self.messages = FakeBetaMessages(parse_responder)


def with_judge(
    fake: FakeAnthropic, parse_responder: Callable[[dict[str, Any]], Any]
) -> FakeAnthropic:
    fake.beta = FakeBeta(parse_responder)  # type: ignore[attr-defined]
    return fake
