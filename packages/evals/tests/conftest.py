"""Postgres fixture for integration tests (``docker compose up -d`` in the repo root).

Without a reachable database the tests skip - unless ``REQUIRE_DB=1`` (CI), where they fail.
"""

from __future__ import annotations

import os
import uuid

import pytest
from rav_ai.db.models import Author, Segment, SegmentText, Snapshot, Version, Work
from rav_ai.db.session import make_engine, migrate
from rav_ai.normalize import normalize_a, normalize_b
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

ADMIN_URL = os.environ.get(
    "RAV_TEST_ADMIN_URL", "postgresql+psycopg://rav:rav@localhost:54329/postgres"
)
DB_URL_TEMPLATE = ADMIN_URL.rsplit("/", 1)[0] + "/{}"

# (ref, version key) -> raw text, with niqqud and Sefaria markup, as stored verbatim
RAW_SHEMA = "<b>שְׁמַע</b> יִשְׂרָאֵל, ה׳ אֱלֹהֵינוּ ה׳ אֶחָד."
RAW_EN = "Hear, O Israel."
RAW_NOIDX = "טקסט שאסור לאנדקס אותו"
RAW_SECONDARY = "נוסח אחר של הפסוק לגמרי"


def _fail_or_skip(msg: str) -> None:
    if os.environ.get("REQUIRE_DB") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture
def pg_engine():
    name = f"rav_evals_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(
        ADMIN_URL, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 5}
    )
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as e:  # noqa: BLE001
        admin.dispose()
        _fail_or_skip(f"Postgres not available (docker compose up -d): {e}")
    url = DB_URL_TEMPLATE.format(name)
    try:
        migrate(url)
    except Exception:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
        raise
    eng = make_engine(url)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


def _version(work_id, snap_id, lang, title, *, can_index, primary, priority=None) -> Version:
    return Version(
        work_id=work_id,
        source="sefaria",
        language=lang,
        version_title=title,
        license="CC0" if can_index else "unknown",
        can_index=can_index,
        can_display_public=can_index,
        can_train=can_index,
        is_primary=primary,
        priority=priority,
        first_snapshot_id=snap_id,
        last_snapshot_id=snap_id,
    )


def _segment(work_id, ref) -> Segment:
    return Segment(
        ref=ref,
        work_id=work_id,
        node_path=[],
        section_path=[],
        indices=[],
        section_names=[],
        sort_key=ref,
    )


def _text(seg_id, ver_id, raw) -> SegmentText:
    a, b = normalize_a(raw), normalize_b(raw)
    return SegmentText(
        segment_id=seg_id,
        version_id=ver_id,
        text_raw=raw,
        raw_hash=uuid.uuid4().hex,  # not under test here
        text_a=a.text,
        text_b=b.text,
        offsets_a=a.offsets.to_bytes(),
        offsets_b=b.offsets.to_bytes(),
    )


@pytest.fixture
def corpus(pg_engine):
    """Tiny corpus: Shema 1 (primary he + en), Noidx 1 (only non-indexable), Multi 1
    (secondary he with higher priority than nothing, plus the primary)."""
    with Session(pg_engine) as s:
        snap = Snapshot(source="sefaria", export_label="fixture", status="complete")
        author = Author(slug="a", name="A")
        s.add_all([snap, author])
        s.flush()
        works = {}
        for title in ("Shema", "Noidx", "Multi"):
            w = Work(
                title=title,
                category="Tanakh",
                category_path=["Tanakh"],
                section_names=["Verse"],
                schema={},
                author_id=author.id,
            )
            s.add(w)
            works[title] = w
        s.flush()
        he = _version(works["Shema"].id, snap.id, "he", "Primary", can_index=True, primary=True)
        en = _version(works["Shema"].id, snap.id, "en", "English", can_index=True, primary=False)
        noidx = _version(works["Noidx"].id, snap.id, "he", "NoIdx", can_index=False, primary=True)
        m_primary = _version(
            works["Multi"].id, snap.id, "he", "MPrimary", can_index=True, primary=True
        )
        m_other = _version(
            works["Multi"].id, snap.id, "he", "MOther", can_index=True, primary=False, priority=99
        )
        s.add_all([he, en, noidx, m_primary, m_other])
        segs = {r: _segment(works[r.split()[0]].id, r) for r in ("Shema 1", "Noidx 1", "Multi 1")}
        s.add_all(segs.values())
        s.flush()
        s.add_all(
            [
                _text(segs["Shema 1"].id, he.id, RAW_SHEMA),
                _text(segs["Shema 1"].id, en.id, RAW_EN),
                _text(segs["Noidx 1"].id, noidx.id, RAW_NOIDX),
                _text(segs["Multi 1"].id, m_primary.id, RAW_SHEMA),
                _text(segs["Multi 1"].id, m_other.id, RAW_SECONDARY),
            ]
        )
        s.commit()
    return pg_engine
