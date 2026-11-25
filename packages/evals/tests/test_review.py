"""The review tool: three queues over TestClient, ratings as files, status transitions and
overrides, empty queue without items, and 127.0.0.1 only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from rav_evals.config import FIXTURES_DIR, load_config
from rav_evals.pairs import write_jsonl
from rav_evals.ratings import RatingStore
from rav_evals.review.app import ReviewContext, create_app, server_config


@pytest.fixture
def ctx(tmp_path):
    fx = tmp_path / "runs"
    shutil.copytree(FIXTURES_DIR, fx)
    pairs = write_jsonl(
        [
            {
                "pair_id": "p1",
                "unit_kind": "segment_range",
                "a_ref": "A 1",
                "b_ref": "B 1",
                "source_bucket": "explicit",
                "swapped": True,
                "dup_of": None,
                "ratings": [],
            }
        ],
        tmp_path / "pairs.jsonl",
    )
    return ReviewContext(
        rater_id="yair",
        store=RatingStore(tmp_path / "ratings"),
        cfg=load_config(),
        fixtures_dir=fx,
        pairs_path=pairs,
        reports_dir=tmp_path / "reports",
        texts={"A 1": "טקסט א", "B 1": "טקסט ב"},
    )


def test_index_and_answers_queue_save_ratings(ctx):
    c = TestClient(create_app(ctx))
    assert c.get("/").status_code == 200
    page = c.get("/answers")
    assert page.status_code == 200 and 'dir="rtl"' in page.text
    run_id = ctx.answers()[0].run.fixture_id
    r = c.post("/api/answers/claim", json={"run_id": run_id, "claim_index": 0, "verdict": "ok"})
    assert r.status_code == 200
    bad = c.post(
        "/api/answers/claim", json={"run_id": run_id, "claim_index": 0, "verdict": "wrong"}
    )
    assert bad.status_code == 422  # error type required
    r = c.post(
        "/api/answers/claim",
        json={
            "run_id": run_id,
            "claim_index": 0,
            "verdict": "wrong",
            "error_type": "wrong_attribution",
            "subtype": "wrong_holder",
            "note": "n",
        },
    )
    assert r.status_code == 200
    exp = r.json()["exported"]
    assert exp["fixtures"] and exp["reports"]
    assert Path(exp["reports"][0]).name == "errors-seed-v0.md"
    rc = json.loads(Path(exp["fixtures"][0]).read_text(encoding="utf-8"))
    assert rc["kind"] == "rejected_claim" and rc["error_type"] == "wrong_attribution"
    assert (
        c.post("/api/answers/grade", json={"run_id": run_id, "grade": "minor"}).status_code == 200
    )
    lines = ctx.store.path("yair", "answers").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(json.loads(x)["rater_id"] == "yair" for x in lines)
    assert c.get("/api/progress").json()["answers"]["graded"] == 1


def test_pairs_queue(ctx):
    c = TestClient(create_app(ctx))
    page = c.get("/pairs")
    assert page.status_code == 200 and "טקסט ב" in page.text
    assert (
        c.post("/api/pairs/rate", json={"pair_id": "p1", "label": "depends_on"}).status_code == 422
    )
    r = c.post(
        "/api/pairs/rate",
        json={"pair_id": "p1", "label": "depends_on", "direction": "ba", "confidence": 3},
    )
    assert r.status_code == 200 and r.json()["saved"]["direction"] == "ba"
    r = c.post("/api/pairs/rate", json={"pair_id": "p1", "label": "same_as", "direction": "ab"})
    assert r.json()["saved"]["direction"] is None  # symmetric label has no direction
    assert (
        c.post("/api/pairs/rate", json={"pair_id": "nope", "label": "same_as"}).status_code == 404
    )


def test_items_queue_empty_without_database(ctx):
    c = TestClient(create_app(ctx))
    r = c.get("/items")
    assert r.status_code == 200 and "ריק" in r.text


def test_items_queue_empty_with_database_but_no_items(ctx, pg_engine):
    ctx.engine = pg_engine
    r = TestClient(create_app(ctx)).get("/items")
    assert r.status_code == 200 and "ריק" in r.text


def _seed_items(engine):
    from rav_ai.db.models import KlEvidence, KlItem, KlItemSegment, KlJob, Segment, SegmentText

    with Session(engine) as s:
        seg = s.scalars(select(Segment).where(Segment.ref == "Gem 1:1")).one()
        st = s.scalars(select(SegmentText).where(SegmentText.segment_id == seg.id)).first()
        job = KlJob(stage="s2", scope_key="Gem 1", status="promoted")
        s.add(job)
        s.flush()
        ids = []
        for n, (conf, cr) in enumerate(
            [(0.9, {"ok": True}), (0.3, {"ok": True}), (0.8, {"ok": True, "queue_priority": True})]
        ):
            it = KlItem(
                kind="sugya_summary",
                item_key=f"sum:{n}",
                scope_key="Gem 1",
                title_he=f"סיכום {n}",
                input_hash=f"h{n}",
                job_id=job.id,
                confidence=conf,
                check_report=cr,
                body={
                    "steps": [
                        {"id": "s1", "move": "statement", "text_he": "פתיחה"},
                        {"id": "s2", "move": "question", "responds_to": "s1", "text_he": "קושיה"},
                    ]
                },
            )
            s.add(it)
            s.flush()
            s.add(KlItemSegment(item_id=it.id, segment_id=seg.id, role="anchor"))
            s.add(
                KlEvidence(
                    item_id=it.id,
                    claim_id="s1",
                    segment_id=seg.id,
                    version_id=st.version_id,
                    span_start=0,
                    span_end=4,
                    quote="טקסט",
                    locate_status="located",
                )
            )
            ids.append(it.id)
        s.commit()
    return ids


def test_items_queue_order_and_transitions(ctx, kl_corpus):
    from rav_ai.db.models import KlItem, KlOverride, KlReview

    ids = _seed_items(kl_corpus)
    ctx.engine = kl_corpus
    c = TestClient(create_app(ctx))
    page = c.get("/items")
    assert page.status_code == 200 and "<mark" in page.text and "סיכום 2" in page.text  # priority
    r = c.post(f"/api/items/{ids[2]}/action", json={"action": "approve"})
    assert r.status_code == 200 and r.json()["status"] == "verified"
    r = c.post(f"/api/items/{ids[1]}/action", json={"action": "reject", "note": "גבול שגוי"})
    assert r.json()["status"] == "rejected"
    r = c.post(f"/api/items/{ids[0]}/action", json={"action": "fix", "claim_id": "s2"})
    assert r.status_code == 200 and r.json()["hidden_claims"] == ["s2"] and r.json()["applied"]
    with Session(kl_corpus) as s:
        acts = sorted(o.action for o in s.scalars(select(KlOverride)))
        assert acts == ["approve", "edit", "reject"]
        assert sorted(v.verdict for v in s.scalars(select(KlReview))) == [
            "approve",
            "edit",
            "reject",
        ]
        assert s.get(KlItem, ids[1]).is_current is False
        cur = s.scalars(
            select(KlItem).where(KlItem.item_key == "sum:0", KlItem.is_current.is_(True))
        ).one()
        assert cur.body["hidden_claims"] == ["s2"] and cur.prompt_version == "human"
        assert cur.status == "verified"
    diff = TestClient(create_app(ctx)).get("/items?diff=1")
    assert diff.status_code in (200,)
    assert len(ctx.store.path("yair", "items").read_text(encoding="utf-8").splitlines()) == 3
    assert c.post("/api/items/999999/action", json={"action": "approve"}).status_code == 404


def test_server_listens_on_localhost_only(ctx):
    app = create_app(ctx)
    cfg = server_config(app)
    assert cfg.host == "127.0.0.1"
    for host in ("0.0.0.0", "::", "192.168.1.5", "localhost"):
        with pytest.raises(ValueError):
            server_config(app, host)
