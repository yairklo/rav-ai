from sqlalchemy import text

from export_fixture import BER, MB, SA
from rav_ai import links_policy as lp
from rav_ai.ingest.links import ingest_links
from rav_ai.ingest.sefaria import finish_snapshot, ingest_all, start_snapshot


def test_every_sefaria_origin_is_a_candidate_with_explicit_kinds():
    mk = lambda t, o="sefaria": {"origin": o, "edge_type": t, "confidence": None}  # noqa: E731
    assert lp.is_candidate(mk("")) and lp.is_candidate(mk("anything new"))
    assert not lp.is_candidate(mk("mesorat hashas", "llm"))
    assert lp.candidate_kind(mk("mesorat hashas")) == lp.KIND_MESORAT_HASHAS
    for t in ("ein mishpat / ner mitsvah", "ein mishpat"):
        assert lp.candidate_kind(mk(t)) == lp.KIND_EIN_MISHPAT
    for t in ("commentary", "targum", "super commentary", "super_commentary"):
        assert lp.candidate_kind(mk(t)) in lp.NAVIGATION_KINDS
    assert lp.candidate_kind(mk("reference")) == lp.KIND_OTHER
    assert lp.candidate_kind(mk("commentary", "heuristic")) is None


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
        n_edges = c.execute(text("SELECT count(*) FROM edges")).scalar_one()
        conf = c.execute(text("SELECT count(*) FROM edges WHERE confidence IS NOT NULL"))
        assert conf.scalar_one() == 0
    # re-ingest the same links into the same snapshot: no duplicates
    ingest_links(engine, raw_dir, sid, log=lambda *_: None)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM edges")).scalar_one() == n_edges
        dup = c.execute(
            text("""SELECT count(*) FROM (SELECT 1 FROM edges
                    GROUP BY segment_a_id, segment_b_id, edge_type HAVING count(*) > 1) x""")
        )
        assert dup.scalar_one() == 0
