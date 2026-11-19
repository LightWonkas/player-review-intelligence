from typing import Any

import pytest
from pydantic import ValidationError

from review_intel.llm.schemas import parse_classification
from review_intel.llm.taxonomy import Taxonomy, load_taxonomy

from .conftest import REPO_ROOT
from .fake_anthropic import VALID_INPUT

SPEC_CATEGORIES = [
    "bug_crash",
    "performance_lag",
    "matchmaking",
    "balance_weapons",
    "monetization_iap",
    "ads",
    "cheating_hackers",
    "controls_ux",
    "content_request",
    "connectivity_server",
    "praise",
    "other",
]
ALLOWED = set(SPEC_CATEGORIES)


def test_repo_taxonomy_matches_the_spec() -> None:
    taxonomy = load_taxonomy(REPO_ROOT / "config" / "taxonomy.yaml")

    assert taxonomy.names == SPEC_CATEGORIES
    assert all(c.definition and c.example for c in taxonomy.categories)


def _category(name: str) -> dict[str, str]:
    return {"name": name, "definition": "d", "example": "e"}


def test_taxonomy_rejects_duplicates() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Taxonomy.model_validate(
            {"categories": [_category("a"), _category("a"), _category("other")]}
        )


def test_taxonomy_requires_other_fallback() -> None:
    with pytest.raises(ValidationError, match="other"):
        Taxonomy.model_validate({"categories": [_category("a"), _category("b")]})


def _parse(**overrides: Any) -> Any:
    return parse_classification(VALID_INPUT | overrides, ALLOWED)


def test_valid_output_parses() -> None:
    result = _parse()

    assert result.primary_category == "bug_crash"
    assert result.severity == 4


def test_primary_is_added_to_categories_and_duplicates_removed() -> None:
    result = _parse(categories=["ads", "ads"], primary_category="bug_crash")

    assert result.categories == ["bug_crash", "ads"]


def test_features_are_normalised() -> None:
    result = _parse(mentioned_features=[" Shop ", "shop", "", "AK-47"])

    assert result.mentioned_features == ["shop", "ak-47"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"primary_category": "not_a_category"},
        {"categories": ["bug_crash", "made_up"]},
        {"categories": ["ads", "praise", "other", "matchmaking", "controls_ux"]},
        {"severity": 6},
        {"sentiment": "angry"},
        {"short_summary": ""},
        {"short_summary": "x" * 500},
        {"unexpected": "field"},
    ],
)
def test_invalid_outputs_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _parse(**overrides)
