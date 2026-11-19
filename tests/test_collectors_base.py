from dataclasses import fields
from datetime import UTC, datetime

import pytest

from review_intel.collectors.base import RawReview


def test_raw_review_has_no_personal_data_fields() -> None:
    # Guardrail: if someone adds an author/username field, this test fails.
    names = {field.name for field in fields(RawReview)}

    assert not names & {"author", "user", "user_name", "username", "avatar", "user_image"}


def _raw(rating: int = 3, review_date: datetime | None = None) -> RawReview:
    return RawReview(
        store_review_id="r1",
        rating=rating,
        title=None,
        body="x",
        app_version=None,
        review_date=review_date or datetime(2026, 1, 1, tzinfo=UTC),
        language=None,
    )


@pytest.mark.parametrize("rating", [0, 6])
def test_rating_must_be_1_to_5(rating: int) -> None:
    with pytest.raises(ValueError, match="rating"):
        _raw(rating=rating)


def test_review_date_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _raw(review_date=datetime(2026, 1, 1))
