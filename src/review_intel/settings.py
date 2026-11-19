"""Runtime settings read from environment variables and `.env`.

Secrets and per-machine values live in the environment, never in code or YAML,
so the repo can be public and every command reads config the same way.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional until M3: `db init` and `fetch` must work without a key.
    anthropic_api_key: SecretStr | None = None
    database_url: str = "sqlite:///data/review_intel.db"
    classify_model: str = "claude-haiku-5-5"
    report_model: str = "claude-sonnet-5-5"
    apps_config_path: Path = Path("config/apps.yaml")
    taxonomy_path: Path = Path("config/taxonomy.yaml")


def get_settings() -> Settings:
    """Build settings fresh on each call so tests can change env vars without cache tricks."""
    return Settings()
