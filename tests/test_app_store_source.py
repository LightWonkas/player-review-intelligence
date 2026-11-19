from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from review_intel.apps_config import StoreListing
from review_intel.collectors.app_store import MAX_PAGES, AppStoreSource, feed_last_page
from review_intel.domain import Store
from review_intel.retry import Pacer, RetryPolicy

from .fakes import load_json

LISTING = StoreListing(store=Store.APP_STORE, store_app_id="1467648713")
FAST_RETRY = RetryPolicy(attempts=3, base_delay=0.0, jitter=0.0)

Handler = Callable[[httpx.Request], httpx.Response]


def _page_number(request: httpx.Request) -> int:
    return int(request.url.path.split("/page=")[1].split("/")[0])


def _feed(page: int, *, last_page: int | None = None, empty: bool = False) -> dict[str, Any]:
    """The recorded page, with unique ids per page and an optional rel="last" link."""
    data: dict[str, Any] = load_json("app_store_page.json")
    feed = data["feed"]
    feed["link"] = [link for link in feed["link"] if link["attributes"]["rel"] != "last"]
    if last_page is not None:
        href = f"https://itunes.apple.com/us/rss/customerreviews/page={last_page}/id=1/json"
        feed["link"].append({"attributes": {"rel": "last", "href": href}})
    if empty:
        del feed["entry"]
    else:
        for entry in feed["entry"]:
            entry["id"]["label"] += f"-p{page}"
    return data


def _serve(last_page: int | None, empty_pages: frozenset[int] = frozenset()) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        page = _page_number(request)
        return httpx.Response(200, json=_feed(page, last_page=last_page, empty=page in empty_pages))

    return handler


def _source(handler: Handler, requests: list[httpx.Request]) -> AppStoreSource:
    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(recording))
    return AppStoreSource(
        client=client, pacer=Pacer(0.0), retry_policy=FAST_RETRY, sleep=lambda _: None
    )


def test_maps_recorded_feed() -> None:
    entries = load_json("app_store_page.json")["feed"]["entry"]

    reviews = list(_source(_serve(last_page=1), []).fetch(LISTING, limit=100))

    assert [r.store_review_id for r in reviews] == [f"{e['id']['label']}-p1" for e in entries]
    first = reviews[0]
    assert first.rating == int(entries[0]["im:rating"]["label"])
    assert first.title == entries[0]["title"]["label"]
    assert first.app_version == entries[0]["im:version"]["label"]
    # "-07:00" in the feed becomes the same instant in UTC.
    expected = datetime.fromisoformat(entries[0]["updated"]["label"]).astimezone(UTC)
    assert first.review_date == expected
    assert first.review_date.tzinfo is UTC


def test_recorded_feed_declares_its_last_page() -> None:
    assert feed_last_page(load_json("app_store_page.json")) == MAX_PAGES


def test_requests_the_documented_feed_url() -> None:
    requests: list[httpx.Request] = []

    list(_source(_serve(last_page=1), requests).fetch(LISTING, limit=100))

    assert [str(r.url) for r in requests] == [
        "https://itunes.apple.com/us/rss/customerreviews/page=1/id=1467648713/sortby=mostrecent/json"
    ]


def test_stops_at_the_feeds_last_page() -> None:
    requests: list[httpx.Request] = []

    reviews = list(_source(_serve(last_page=2), requests).fetch(LISTING, limit=100))

    assert len(requests) == 2
    assert len(reviews) == 6


def test_empty_page_in_the_middle_is_skipped_not_treated_as_the_end() -> None:
    # Seen live: page 8 of 10 was empty while pages 9 and 10 had reviews.
    requests: list[httpx.Request] = []
    handler = _serve(last_page=3, empty_pages=frozenset({2}))

    reviews = list(_source(handler, requests).fetch(LISTING, limit=100))

    assert len(requests) == 3
    assert [r.store_review_id[-2:] for r in reviews] == ["p1"] * 3 + ["p3"] * 3


def test_no_reviews_at_all_makes_one_request() -> None:
    requests: list[httpx.Request] = []
    handler = _serve(last_page=None, empty_pages=frozenset({1}))

    reviews = list(_source(handler, requests).fetch(LISTING, limit=100))

    assert reviews == []
    assert len(requests) == 1


def test_stops_at_limit_without_requesting_more_pages() -> None:
    requests: list[httpx.Request] = []

    reviews = list(_source(_serve(last_page=10), requests).fetch(LISTING, limit=2))

    assert len(reviews) == 2
    assert len(requests) == 1


def test_never_requests_more_than_apple_serves() -> None:
    requests: list[httpx.Request] = []

    reviews = list(_source(_serve(last_page=99), requests).fetch(LISTING, limit=10_000))

    assert len(requests) == MAX_PAGES
    assert len(reviews) == 3 * MAX_PAGES


def test_single_entry_page_is_an_object_not_a_list() -> None:
    page = _feed(1, last_page=1)
    page["feed"]["entry"] = page["feed"]["entry"][0]

    reviews = list(_source(lambda _: httpx.Response(200, json=page), []).fetch(LISTING, 100))

    assert len(reviews) == 1


def test_entries_without_rating_are_skipped() -> None:
    page = _feed(1, last_page=1)
    page["feed"]["entry"].insert(0, {"id": {"label": "app"}, "title": {"label": "The app"}})

    reviews = list(_source(lambda _: httpx.Response(200, json=page), []).fetch(LISTING, 100))

    assert len(reviews) == 3


def test_retries_server_errors() -> None:
    requests: list[httpx.Request] = []
    ok = _serve(last_page=2)

    def flaky(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503) if len(requests) == 1 else ok(request)

    reviews = list(_source(flaky, requests).fetch(LISTING, limit=100))

    assert len(reviews) == 6
    assert len(requests) == 3  # 503, retry of page 1, page 2


def test_does_not_retry_not_found() -> None:
    requests: list[httpx.Request] = []

    with pytest.raises(httpx.HTTPStatusError):
        list(_source(lambda _: httpx.Response(404), requests).fetch(LISTING, limit=100))

    assert len(requests) == 1
