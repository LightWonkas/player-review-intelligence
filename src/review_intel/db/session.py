"""Engine and session setup from a single `DATABASE_URL`.

Everything goes through `make_engine`, so switching SQLite for Postgres is a
config change, not a code change.
"""

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, make_url
from sqlalchemy.orm import Session, sessionmaker

from review_intel.db.models import Base


def make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return create_engine(url)

    # Create the folder for a file database so a fresh clone can run `db init` directly.
    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url)

    # SQLite ignores foreign keys unless asked per connection; without this,
    # ON DELETE CASCADE silently does nothing.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False lets CLI code read objects after committing without extra queries.
    return sessionmaker(engine, expire_on_commit=False)


def init_db(engine: Engine) -> list[str]:
    """Create any missing tables and return all table names. Safe to run repeatedly."""
    Base.metadata.create_all(engine)
    return sorted(Base.metadata.tables)
