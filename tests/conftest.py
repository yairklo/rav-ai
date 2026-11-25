"""Fixtures: synthetic export + a throw-away Postgres DB (docker compose up -d; else skip)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from export_fixture import build_export
from rav_ai.config import LicensePolicy
from rav_ai.db.session import make_engine, migrate

ADMIN_URL = "postgresql+psycopg://rav:rav@localhost:54329/postgres"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    r = tmp_path / "raw"
    build_export(r)
    return r


@pytest.fixture
def policy() -> LicensePolicy:
    return LicensePolicy.load()


@pytest.fixture
def engine():
    name = f"rav_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as e:  # noqa: BLE001
        msg = f"Postgres not available (docker compose up -d): {e}"
        if os.environ.get("REQUIRE_DB") == "1":  # CI: a missing DB is a failure, not a skip
            pytest.fail(msg)
        pytest.skip(msg)
    url = f"postgresql+psycopg://rav:rav@localhost:54329/{name}"
    migrate(url)
    eng = make_engine(url)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()
