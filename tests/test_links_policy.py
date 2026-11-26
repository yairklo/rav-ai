import pytest
from sqlalchemy import text

from export_fixture import BER, MB, SA
from rav_ai import links_policy as lp
from rav_ai.ingest.links import ingest_links
from rav_ai.ingest.sefaria import finish_snapshot, ingest_all, start_snapshot


def test_every_sefaria_origin_is_a_candidate_with_explicit_kinds():
    mk = lambda t, o="sefaria": {"origin": o, "edge_type": t, "confidence": None}  # noqa: E731
    assert lp.is_candidate(mk("")) and lp.is_candidate(mk("anything new"))
    assert lp.candidate_kind(mk("mesorat hashas")) == lp.KIND_MESORAT_HASHAS
    assert lp.candidate_kind(mk("  Mesorat HaShas ")) == lp.KIND_MESORAT_HASHAS
    for t in ("ein mishpat / ner mitsvah", "ein mishpat"):
        assert lp.candidate_kind(mk(t)) == lp.KIND_EIN_MISHPAT
    for t in ("commentary", "targum", "super commentary", "super_commentary"):
        assert lp.candidate_kind(mk(t)) in lp.NAVIGATION_KINDS
    assert lp.candidate_kind(mk("reference")) == lp.KIND_OTHER


def test_fail_closed_everything_but_human_is_a_candidate():
    for origin in ("sefaria", "heuristic", "llm", "unknown", None, "", "Sefaria"):
        assert lp.is_candidate(origin) is True, origin  # a bare origin string / None
        edge = {"origin": origin, "edge_type": "ein mishpat / ner mitsvah"}
        assert lp.is_candidate(edge) is True, origin
        assert lp.candidate_kind(edge) is not None, origin
    assert lp.is_candidate({"edge_type": "x"}) is True  # no origin key at all
    assert lp.is_candidate("human") is False
    assert lp.is_candidate({"origin": "human", "edge_type": "mesorat hashas"}) is False
    assert lp.candidate_kind({"origin": "human", "edge_type": "mesorat hashas"}) is None


def test_non_sefaria_candidates_get_auto_link_kind_and_label():
    for o in ("heuristic", "llm", "unknown", None, "Sefaria"):
        assert lp.candidate_kind({"origin": o, "edge_type": "mesorat hashas"}) == lp.KIND_AUTO
    assert lp.candidate_label("sefaria") == lp.LABEL_SEFARIA
    assert lp.candidate_label({"origin": "sefaria"}) == lp.LABEL_SEFARIA
    for o in ("heuristic", "llm", None, "", "weird"):
        assert lp.candidate_label(o) == lp.LABEL_AUTO
    assert "לא ראיה" in lp.LABEL_SEFARIA and "לא ראיה" in lp.LABEL_AUTO


def test_object_with_origin_and_unsupported_types():
    from types import SimpleNamespace

    obj = SimpleNamespace(origin="heuristic", edge_type="commentary")
    assert lp.is_candidate(obj) is True
    assert lp.candidate_kind(obj) == lp.KIND_AUTO
    assert lp.candidate_kind(SimpleNamespace(origin="sefaria", edge_type="targum")) == (
        lp.KIND_NAVIGATION
    )
    for bad in (1, 3.5, [], object(), {"origin": 7}):
        with pytest.raises(TypeError):
            lp.is_candidate(bad)


def _import(engine, raw_dir, policy, extra_rows):
    f = raw_dir / "links" / "links0.csv"
    f.write_text(f.read_text(encoding="utf-8") + extra_rows, encoding="utf-8")
    snap = start_snapshot(engine, raw_dir)
    stats = ingest_all(engine, raw_dir, policy, snap.id, log=lambda *_: None)
    stats["links"] = ingest_links(engine, raw_dir, snap.id, log=lambda *_: None)
    finish_snapshot(engine, snap.id, stats)
    return snap.id


EXTRA = (
    f'"{SA} 1:2","{BER} 2a:1",mesorat hashas,"{SA}","{BER}",h,h\n'
    f'"{SA} 1:2","{MB} 1:1",ein mishpat / ner mitsvah,"{SA}","{MB}",h,h\n'
)


def test_candidate_links_on_seed_and_idempotent_reingest(engine, raw_dir, policy):
    sid = _import(engine, raw_dir, policy, EXTRA)
    with engine.connect() as c:
        seg = c.execute(text("SELECT id FROM segments WHERE ref = :r"), {"r": f"{SA} 1:2"})
        sa12 = seg.scalar_one()
        links = lp.candidate_links(c, [sa12])
        kinds = {link.candidate_kind for link in links}
        assert {lp.KIND_MESORAT_HASHAS, lp.KIND_EIN_MISHPAT} <= kinds
        assert all(link.candidate for link in links)
        only = lp.candidate_links(c, [sa12], kinds={lp.KIND_MESORAT_HASHAS})
        assert {link.edge_type for link in only} == {"mesorat hashas"}
        assert lp.candidate_links(c, []) == []
        assert {link.origin for link in links} == {"sefaria"}
        n_edges = c.execute(text("SELECT count(*) FROM edges")).scalar_one()
        conf = c.execute(text("SELECT count(*) FROM edges WHERE confidence IS NOT NULL"))
        assert conf.scalar_one() == 0
    # a heuristic / llm / unknown-origin edge is returned too (fail-closed), a human one is not
    with engine.begin() as c:
        a, b = c.execute(text("SELECT id FROM segments ORDER BY id LIMIT 2")).scalars().all()
        for origin in ("heuristic", "llm", "weird", "human"):
            c.execute(
                text("""INSERT INTO edges (segment_a_id, segment_b_id, edge_type, origin, a_span,
                        b_span, range_collapsed, snapshot_id)
                        VALUES (:a, :b, 'x', :o, 1, 1, false, :s)"""),
                {"a": a, "b": b, "o": origin, "s": sid},
            )
    with engine.connect() as c:
        got = {link.origin: link for link in lp.candidate_links(c, [a, b])}
        assert {"heuristic", "llm", "weird"} <= set(got) and "human" not in got
        assert all(got[o].candidate and got[o].candidate_kind == lp.KIND_AUTO for o in ("llm",))
    with engine.begin() as c:
        c.execute(text("DELETE FROM edges WHERE origin IN ('heuristic','llm','weird','human')"))
    # re-ingest the same links into the same snapshot: no duplicates
    ingest_links(engine, raw_dir, sid, log=lambda *_: None)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM edges")).scalar_one() == n_edges
        dup = c.execute(
            text("""SELECT count(*) FROM (SELECT 1 FROM edges
                    GROUP BY segment_a_id, segment_b_id, edge_type HAVING count(*) > 1) x""")
        )
        assert dup.scalar_one() == 0
