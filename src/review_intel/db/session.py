"""Engine and session setup from a single `DATABASE_URL`.

Everything goes through `make_engine`, so switching SQLite for Postgres is a
config change, not a code change.
"""

from pathlib import Path
from typing import Any

import sqlite_vec
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

    @event.listens_for(engine, "connect")
    def _configure_connection(dbapi_connection: Any, _record: Any) -> None:
        # SQLite ignores foreign keys unless asked per connection; without this,
        # ON DELETE CASCADE silently does nothing.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        # sqlite-vec adds the vec0 virtual tables used for RAG. Loading is switched
        # back off right away so SQL can't load other extensions later.
        dbapi_connection.enable_load_extension(True)
        sqlite_vec.load(dbapi_connection)
        dbapi_connection.enable_load_extension(False)

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False lets CLI code read objects after committing without extra queries.
    return sessionmaker(engine, expire_on_commit=False)


def init_db(engine: Engine) -> list[str]:
    """Create any missing tables and return all table names. Safe to run repeatedly."""
    Base.metadata.create_all(engine)
    return sorted(Base.metadata.tables)
