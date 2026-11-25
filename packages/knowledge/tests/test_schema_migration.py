"""Migration 0005: upgrade -> downgrade -> upgrade, and the constraints that matter."""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from rav_ai.config import ROOT
from rav_ai.db.models import KlEvidence, KlItem, KlReception, KlWitness
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

KL_TABLES = {
    "kl_jobs", "kl_tasks", "kl_items", "kl_item_segments", "kl_evidence", "kl_links",
    "kl_relations", "kl_sources", "kl_witnesses", "kl_reception", "kl_coverage", "kl_overrides",
    "kl_entities", "kl_work_aliases", "kl_reviews",
}  # fmt: skip


def _cfg(engine) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", engine.test_url.replace("%", "%%"))
    return cfg


def _state(engine):
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    ann = {c["name"] for c in insp.get_columns("annotations")}
    runs = {c["name"] for c in insp.get_columns("runs")}
    return tables, ann, runs


def test_upgrade_downgrade_upgrade(pg_engine):
    tables, ann, runs = _state(pg_engine)
    assert KL_TABLES <= tables and "knowledge_items" in runs
    assert {"status", "is_current", "method", "method_version", "context_domain", "context",
            "entity_id", "job_id", "created_at"} <= ann  # fmt: skip
    cfg = _cfg(pg_engine)
    command.downgrade(cfg, "0004")
    pg_engine.dispose()
    tables, ann, runs = _state(pg_engine)
    assert not (KL_TABLES & tables) and "knowledge_items" not in runs and "method" not in ann
    with pg_engine.connect() as c:  # the view went with them
        assert c.execute(text("SELECT to_regclass('kl_conflicts')")).scalar() is None
    command.upgrade(cfg, "head")
    pg_engine.dispose()
    tables, ann, runs = _state(pg_engine)
    assert KL_TABLES <= tables and "knowledge_items" in runs and "method" in ann
    with pg_engine.connect() as c:
        assert c.execute(text("SELECT to_regclass('kl_conflicts')")).scalar() is not None


def test_indexes(pg_engine):
    with pg_engine.connect() as c:
        defs = {r[0]: r[1] for r in c.execute(text("SELECT indexname, indexdef FROM pg_indexes"))}
    assert (
        "WHERE is_current" in defs["uq_kl_items_current"]
        and "UNIQUE" in defs["uq_kl_items_current"]
    )
    assert (
        "gin" in defs["ix_kl_items_search"].lower() and "to_tsvector" in defs["ix_kl_items_search"]
    )
    assert "WHERE is_current" in defs["uq_annotations_current_span"]
    for name in ("ix_kl_evidence_segment", "ix_kl_item_segments_segment", "ix_kl_relations_from",
                 "ix_kl_relations_to"):  # fmt: skip
        assert name in defs


def _item(key, version=1, h="h", current=False, **kw):
    return KlItem(kind="issue", item_key=key, version=version, scope_key="s", body={},
                  input_hash=h, is_current=current, **kw)  # fmt: skip


def test_item_uniqueness(pg_engine):
    with Session(pg_engine) as s:
        s.add(_item("k", 1, "a", True))
        s.commit()
    for bad in (_item("k", 1, "b"), _item("k", 2, "a"), _item("k", 3, "c", True)):
        with Session(pg_engine) as s:
            s.add(bad)
            with pytest.raises(IntegrityError):
                s.commit()
    with Session(pg_engine) as s:
        s.add(_item("k", 2, "b", False))  # a non-current version is fine
        s.commit()


def test_enum_checks_reception_witness_and_evidence(pg_engine):
    with Session(pg_engine) as s:
        it = _item("k2", 1, "x", True)
        s.add(it)
        s.flush()
        s.add(KlReception(target_item_id=it.id, target_claim="c", context="lemaase",
                          weight_label="rejected", policy_version="p1"))  # fmt: skip
        s.add(KlWitness(siglum="Vat.487", kind="manuscript", name_he="כ״י ותיקן 487"))
        s.commit()
        it_id = it.id
    with Session(pg_engine) as s:  # `rejected` only in the lemaase context
        s.add(KlReception(target_item_id=it_id, target_claim="c", context="iyun",
                          weight_label="rejected", policy_version="p1"))  # fmt: skip
        with pytest.raises(IntegrityError):
            s.commit()
    with Session(pg_engine) as s:  # unique per (target, claim, context, policy)
        s.add(KlReception(target_item_id=it_id, target_claim="c", context="lemaase",
                          weight_label="primary", policy_version="p1"))  # fmt: skip
        with pytest.raises(IntegrityError):
            s.commit()
    with Session(pg_engine) as s:
        s.add(KlEvidence(item_id=it_id, quote="q", role="bogus"))
        with pytest.raises(IntegrityError):
            s.commit()
    with Session(pg_engine) as s:
        s.add(_item("k3", 1, "y", status="weird"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_coverage_and_sources_enums(pg_engine, world):
    from rav_ai.db.models import KlCoverage, KlSource

    with Session(pg_engine) as s:
        s.add(
            KlCoverage(scope_key="Shabbat 21a", work_id=world.work["Rashi"], status="not_ingested")
        )
        s.commit()
        s.add(KlCoverage(scope_key="Shabbat 21b", work_id=world.work["Rashi"], status="maybe"))
        with pytest.raises(IntegrityError):
            s.commit()
    with Session(pg_engine) as s:
        s.add(KlSource(work_id=world.work["Stein"], tier="lamdanut", role="lens",
                       license_tier="MADE_UP"))  # fmt: skip
        with pytest.raises(IntegrityError):
            s.commit()
