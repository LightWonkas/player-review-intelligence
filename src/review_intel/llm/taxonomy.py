"""Load and validate `config/taxonomy.yaml`, the list of review categories.

The taxonomy lives in config, not code, so product people can refine definitions
without touching Python, and the eval (M4) shows whether a change helped.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_TAXONOMY_PATH = Path("config/taxonomy.yaml")
FALLBACK_CATEGORY = "other"


class Category(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    definition: str = Field(min_length=1)
    example: str = Field(min_length=1)


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: list[Category] = Field(min_length=2)

    @model_validator(mode="after")
    def _names_unique_and_fallback_present(self) -> Self:
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("category names must be unique")
        if FALLBACK_CATEGORY not in names:
            raise ValueError(f"taxonomy must include a {FALLBACK_CATEGORY!r} category")
        return self

    @property
    def names(self) -> list[str]:
        return [category.name for category in self.categories]


def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    with path.open(encoding="utf-8") as handle:
        return Taxonomy.model_validate(yaml.safe_load(handle))
