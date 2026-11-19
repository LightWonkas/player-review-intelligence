"""Small shared vocabulary used by config, database and (later) the LLM layer.

Kept in one dependency-free module so the DB models and the YAML config can both
import it without importing each other.
"""

from enum import StrEnum


class Store(StrEnum):
    GOOGLE_PLAY = "google_play"
    APP_STORE = "app_store"


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
