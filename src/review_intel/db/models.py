"""SQLAlchemy 2.0 models for apps, reviews, classifications, patch notes and reports.

Design notes:
- An `App` row is one game on one store. `slug` groups a game's listings, so
  `--app critical-strike` covers both Android and iOS.
- Uniqueness lives in the database, not in Python: re-fetching or re-classifying
  can never create duplicates, even if the calling code has a bug.
- No usernames or avatars are stored, by design (see docs/SPEC.md, guardrails).
"""

from datetime import UTC, date, datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Dialect,
    Enum,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from review_intel.domain import Sentiment, Store


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Always hand back timezone-aware UTC datetimes.

    SQLite has no timezone type and returns naive datetimes. Mixing naive and aware
    values later (e.g. "reviews since this release") raises TypeError, so we
    normalise to UTC on write and re-attach UTC on read.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime given; pass a timezone-aware datetime")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Store enum *values* ("google_play") as plain strings, portable across databases."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        length=20,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )


class Base(DeclarativeBase):
    pass


class App(Base):
    __tablename__ = "apps"
    __table_args__ = (UniqueConstraint("slug", "store", name="uq_apps_slug_store"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    store: Mapped[Store] = mapped_column(_str_enum(Store, "store"))
    store_app_id: Mapped[str] = mapped_column(String(255))
    country: Mapped[str] = mapped_column(String(2))
    language: Mapped[str] = mapped_column(String(2))

    reviews: Mapped[list["Review"]] = relationship(
        back_populates="app", cascade="all, delete-orphan", passive_deletes=True
    )


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("app_id", "store_review_id", name="uq_reviews_app_store_review_id"),
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_reviews_rating_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    app_id: Mapped[int] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"), index=True)
    store_review_id: Mapped[str] = mapped_column(String(255))
    rating: Mapped[int]
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    app_version: Mapped[str | None] = mapped_column(String(50), index=True)
    review_date: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    language: Mapped[str | None] = mapped_column(String(10))
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    app: Mapped[App] = relationship(back_populates="reviews")
    classifications: Mapped[list["Classification"]] = relationship(
        back_populates="review", cascade="all, delete-orphan", passive_deletes=True
    )


class Classification(Base):
    """One model's verdict on one review under one prompt version.

    Unique per (review, prompt_version, model) so an eval can compare prompt v1 with
    v2, or Haiku with Sonnet, side by side instead of a new run overwriting the old one.
    """

    __tablename__ = "classifications"
    __table_args__ = (
        UniqueConstraint(
            "review_id", "prompt_version", "model", name="uq_classifications_review_prompt_model"
        ),
        CheckConstraint("severity BETWEEN 1 AND 5", name="ck_classifications_severity_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    categories: Mapped[list[str]] = mapped_column(JSON)
    primary_category: Mapped[str] = mapped_column(String(40), index=True)
    sentiment: Mapped[Sentiment] = mapped_column(_str_enum(Sentiment, "sentiment"))
    severity: Mapped[int]
    is_bug_report: Mapped[bool]
    mentioned_features: Mapped[list[str]] = mapped_column(JSON, default=list)
    short_summary: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    review: Mapped[Review] = relationship(back_populates="classifications")


class ClassificationFailure(Base):
    """A review the model could not classify validly, kept so nothing is silently dropped.

    One row per (review, prompt_version, model); `attempts` grows on every failed run
    and the row is deleted once a later run succeeds.
    """

    __tablename__ = "classification_failures"
    __table_args__ = (
        UniqueConstraint(
            "review_id", "prompt_version", "model", name="uq_failures_review_prompt_model"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    prompt_version: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(100))
    error: Mapped[str] = mapped_column(Text)
    raw_output: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(default=0)
    last_attempt_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class PatchNote(Base):
    """Release notes for one version on one store (Android and iOS often ship on different days)."""

    __tablename__ = "patch_notes"
    __table_args__ = (UniqueConstraint("app_id", "version", name="uq_patch_notes_app_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    app_id: Mapped[int] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"), index=True)
    version: Mapped[str] = mapped_column(String(50))
    release_date: Mapped[date] = mapped_column(Date)
    text: Mapped[str] = mapped_column(Text)


class Report(Base):
    """A generated weekly report.

    Keyed by game `app_slug` rather than a single store's `app_id`, because one
    report covers all of a game's stores.
    """

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    app_slug: Mapped[str] = mapped_column(String(64), index=True)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    markdown: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
