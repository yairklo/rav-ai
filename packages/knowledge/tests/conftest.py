"""Postgres fixture + a synthetic Gemara corpus (no copyrighted text: invented sentences)."""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest
from rav_ai.db.models import (
    Author,
    Edge,
    KlSource,
    Segment,
    SegmentText,
    Snapshot,
    Version,
    Work,
)
from rav_ai.db.session import make_engine, migrate
from rav_ai.normalize import normalize_a, normalize_b
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

ADMIN_URL = os.environ.get(
    "RAV_TEST_ADMIN_URL", "postgresql+psycopg://rav:rav@localhost:54329/postgres"
)
DB_URL_TEMPLATE = ADMIN_URL.rsplit("/", 1)[0] + "/{}"

PAGES = ["21a", "21b", "22a", "22b", "23a"]
SHABBAT = {
    "21a:1": "מתני׳ אמר רב חייב ושמואל אמר פטור",
    "21a:2": 'מתיבי רבי אושעיא חייב ור"ל אמר פטור ע"כ',
    "21b:1": "אמר רבא בעיא היא תיקו",
    "21b:2": "והלכתא כרב בכל מקום",
    "22a:1": "אמר רב ששת טעם אחר הוא",
    "22a:2": 'ת"ש ברייתא אחת פטור',
    "22b:1": "אמר אביי קושיא על כך",
    "22b:2": "אמר רבא לא קשיא כאן וכאן",
    "23a:1": "פתיחה חדשה בענין אחר",
    "23a:2": "ואמר רב נחמן סיום הדברים",
}
RASHI = {"21a:1": "פירוש רש״י על הפתיחה"}
STEIN = {"21a:1": "ביאור שטיינזלץ על הפתיחה"}
SA = {"1:1": 'כתב המחבר דין ראשון וכתב ע"כ ויש אומרים ט"ס ואינו מחוור וז"ל'}


def _fail_or_skip(msg: str) -> None:
    if os.environ.get("REQUIRE_DB") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture
def pg_engine():
    name = f"rav_kl_test_{uuid.uuid4().hex[:8]}"
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
    eng.test_url = url  # type: ignore[attr-defined]
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


def make_text(seg_id: int, ver_id: int, raw: str) -> SegmentText:
    a, b = normalize_a(raw), normalize_b(raw)
    return SegmentText(
        segment_id=seg_id,
        version_id=ver_id,
        text_raw=raw,
        raw_hash=hashlib.sha256(raw.encode()).hexdigest(),
        text_a=a.text,
        text_b=b.text,
        offsets_a=a.offsets.to_bytes(),
        offsets_b=b.offsets.to_bytes(),
    )


def _sort_key(label: str) -> str:
    page, _, rest = label.partition(":")
    if page[-1] in "ab":
        return f"{int(page[:-1]):04d}{page[-1]}{int(rest):03d}"
    return f"{int(page):04d}a{int(rest):03d}"


class World:
    """Handles to the synthetic corpus."""

    def __init__(self, engine):
        self.engine = engine
        self.work: dict[str, int] = {}
        self.seg: dict[str, int] = {}
        self.ver: dict[str, int] = {}


def add_work(
    s,
    snap,
    author,
    title,
    category,
    path,
    texts,
    *,
    can_index=True,
    can_display=True,
    can_train=True,
    source=None,
    lic="PUBLIC_COMMERCIAL",
    prefix=None,
):
    w = Work(
        title=title,
        category=category,
        category_path=path,
        section_names=["Daf", "Line"],
        schema={},
        author_id=author.id,
    )
    s.add(w)
    s.flush()
    v = Version(
        work_id=w.id,
        source="sefaria",
        language="he",
        version_title=f"{title} (test)",
        license="CC0",
        can_index=can_index,
        can_display_public=can_display,
        can_train=can_train,
        is_primary=True,
        first_snapshot_id=snap.id,
        last_snapshot_id=snap.id,
    )
    s.add(v)
    s.flush()
    segs = {}
    for label, raw in texts.items():
        ref = f"{prefix or title} {label}"
        parent = ref.rsplit(":", 1)[0]
        seg = Segment(
            ref=ref,
            work_id=w.id,
            node_path=[],
            section_path=[label.split(":")[0], label.split(":")[1]],
            indices=[],
            section_names=[],
            sort_key=_sort_key(label),
            parent_ref=parent,
        )
        s.add(seg)
        s.flush()
        s.add(make_text(seg.id, v.id, raw))
        segs[label] = seg.id
    if source:
        tier, role = source
        s.add(KlSource(work_id=w.id, tier=tier, role=role, license_tier=lic, added_by="test"))
    s.flush()
    return w, v, segs


@pytest.fixture
def world(pg_engine) -> World:
    wd = World(pg_engine)
    with Session(pg_engine) as s:
        snap = Snapshot(source="sefaria", export_label="fixture", status="complete")
        amoraim = Author(slug="gemara", name="Gemara", period="amoraim")
        rishon = Author(slug="rashi", name="Rashi", period="rishonim")
        s.add_all([snap, amoraim, rishon])
        s.flush()
        w, v, segs = add_work(
            s,
            snap,
            amoraim,
            "Shabbat",
            "Talmud",
            ["Talmud", "Bavli"],
            SHABBAT,
            source=("canonical", "primary"),
        )
        wd.work["Shabbat"], wd.ver["Shabbat"] = w.id, v.id
        wd.seg.update({f"Shabbat {k}": i for k, i in segs.items()})
        w, v, segs = add_work(
            s,
            snap,
            rishon,
            "Rashi on Shabbat",
            "Talmud",
            ["Talmud", "Bavli", "Rishonim"],
            RASHI,
            source=("rishonim", "lens"),
        )
        wd.work["Rashi"] = w.id
        rashi_seg = segs["21a:1"]
        w, v, segs = add_work(
            s, snap, rishon, "Steinsaltz on Shabbat", "Talmud", ["Talmud", "Bavli"], STEIN
        )
        wd.work["Stein"] = w.id
        stein_seg = segs["21a:1"]
        w, v, segs = add_work(
            s,
            snap,
            rishon,
            "Shulchan Arukh, Orach Chayim",
            "Halakhah",
            ["Halakhah", "Shulchan Arukh"],
            SA,
            source=("codes", "primary"),
        )
        wd.work["SA"] = w.id
        wd.seg["SA 1:1"] = segs["1:1"]
        s.add_all(
            [
                Edge(
                    segment_a_id=wd.seg["Shabbat 21a:1"],
                    segment_b_id=rashi_seg,
                    edge_type="commentary",
                    origin="sefaria",
                    snapshot_id=snap.id,
                ),
                Edge(
                    segment_a_id=wd.seg["Shabbat 21a:1"],
                    segment_b_id=stein_seg,
                    edge_type="commentary",
                    origin="sefaria",
                    snapshot_id=snap.id,
                ),
            ]
        )
        s.commit()
    return wd
