from __future__ import annotations

import os

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from rav_ai.config import DEFAULT_DB_URL, ROOT


def db_url() -> str:
    return os.environ.get("RAV_DATABASE_URL", DEFAULT_DB_URL)


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or db_url(), pool_pre_ping=True)


def make_session(engine: Engine) -> Session:
    return sessionmaker(engine, expire_on_commit=False)()


def migrate(url: str | None = None) -> None:
    """Apply Alembic migrations to head."""
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", (url or db_url()).replace("%", "%%"))
    command.upgrade(cfg, "head")
