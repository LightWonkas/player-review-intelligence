"""The shape we accept back from the model.

The tool is `strict`, so the API already guarantees types and enum values. Pydantic
still validates every response because strict mode cannot express rules like
"primary_category must be one of categories" or length limits, and because the
eval and the database should never depend on a guarantee we did not check ourselves.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from review_intel.domain import Sentiment

MAX_CATEGORIES = 4
MAX_FEATURES = 5
MAX_SUMMARY_CHARS = 240


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: list[str] = Field(min_length=1)
    primary_category: str
    sentiment: Sentiment
    severity: int = Field(ge=1, le=5)
    is_bug_report: bool
    mentioned_features: list[str] = Field(default_factory=list)
    short_summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)

    @field_validator("categories", "primary_category")
    @classmethod
    def _known_categories(cls, value: Any, info: ValidationInfo) -> Any:
        allowed: set[str] | None = (info.context or {}).get("allowed_categories")
        if allowed is None:
            return value
        names = value if isinstance(value, list) else [value]
        unknown = [name for name in names if name not in allowed]
        if unknown:
            raise ValueError(f"unknown categories: {unknown}")
        return value

    @field_validator("mentioned_features")
    @classmethod
    def _clean_features(cls, value: list[str]) -> list[str]:
        cleaned = [feature.strip().lower() for feature in value if feature.strip()]
        return list(dict.fromkeys(cleaned))[:MAX_FEATURES]

    @field_validator("short_summary")
    @classmethod
    def _strip_summary(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _normalise_categories(self) -> Self:
        # Harmless slips are fixed rather than rejected: a primary category the model
        # forgot to repeat in the list, or duplicates. Anything ambiguous still fails.
        ordered = list(dict.fromkeys([self.primary_category, *self.categories]))
        if len(ordered) > MAX_CATEGORIES:
            raise ValueError(f"at most {MAX_CATEGORIES} categories, got {len(ordered)}")
        object.__setattr__(self, "categories", ordered)
        return self


def parse_classification(raw: dict[str, Any], allowed_categories: set[str]) -> ClassificationResult:
    return ClassificationResult.model_validate(
        raw, context={"allowed_categories": allowed_categories}
    )
