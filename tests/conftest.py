from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from review_intel.db.session import init_db, make_engine, make_session_factory

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def apps_config_path() -> Path:
    return REPO_ROOT / "config" / "apps.yaml"


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    # A real file DB per test: closest to production and isolated between tests.
    return f"sqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
def engine(db_url: str) -> Iterator[Engine]:
    engine = make_engine(db_url)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with make_session_factory(engine)() as session:
        yield session
