"""PostgresPairSource on a tiny corpus: explicit (candidate) links, FTS neighbours in other
works, and same-work hard negatives outside the slice; linked pairs never count as neighbours."""

from __future__ import annotations

from rav_ai.db.models import Author, Edge, Segment, Snapshot, Work
from sqlalchemy.orm import Session

from conftest import _text, _version
from rav_evals.pairs import PostgresPairSource, sample_pairs

SHARED = "מדליקין שמנים פתילות בחנוכה"


def _seed(engine):
    with Session(engine) as s:
        snap = Snapshot(source="sefaria", export_label="pairs", status="complete")
        au = Author(slug="p", name="P")
        s.add_all([snap, au])
        s.flush()
        ids = {}
        for title in ("Tract", "Code"):
            w = Work(
                title=title,
                category="Talmud",
                category_path=["Talmud"],
                section_names=["Daf"],
                schema={},
                author_id=au.id,
            )
            s.add(w)
            s.flush()
            v = _version(w.id, snap.id, "he", title, can_index=True, primary=True)
            s.add(v)
            s.flush()
            for page, n, txt in [
                ("2a", 1, SHARED),
                ("2a", 2, "טקסט אחר לגמרי בלי חפיפה"),
                ("5b", 1, SHARED + " נוספות"),
            ]:
                ref = f"{title} {page}:{n}"
                seg = Segment(
                    ref=ref,
                    work_id=w.id,
                    node_path=[],
                    section_path=[page, str(n)],
                    indices=[],
                    section_names=[],
                    sort_key=f"{page}{n}",
                    parent_ref=f"{title} {page}",
                )
                s.add(seg)
                s.flush()
                s.add(_text(seg.id, v.id, txt))
                ids[ref] = seg.id
        s.add(
            Edge(
                segment_a_id=ids["Code 2a:2"],
                segment_b_id=ids["Tract 2a:2"],
                edge_type="mesorat hashas",
                origin="sefaria",
                snapshot_id=snap.id,
            )
        )
        # a linked lexical neighbour: must not be offered as a neighbour
        s.add(
            Edge(
                segment_a_id=ids["Tract 2a:1"],
                segment_b_id=ids["Code 5b:1"],
                edge_type="",
                origin="sefaria",
                snapshot_id=snap.id,
            )
        )
        s.commit()


def test_pg_pair_source(pg_engine):
    _seed(pg_engine)
    src = PostgresPairSource(pg_engine, "Tract 2a-3a", ["mesorat hashas"], min_shared_terms=3)
    assert [(c.a_ref, c.b_ref) for c in src.explicit()] == [("Tract 2a:2", "Code 2a:2")]
    nb = {(c.a_ref, c.b_ref) for c in src.neighbors()}
    assert nb == {("Tract 2a:1", "Code 2a:1")}  # Code 5b:1 is linked -> excluded
    hn = {(c.a_ref, c.b_ref) for c in src.hard_negatives()}
    assert hn == {("Tract 2a:1", "Tract 5b:1")}  # same work, outside the slice
    res = sample_pairs(src, 3, seed=1, duplicate_fraction=0)
    assert res.bucket_counts() == {"explicit": 1, "neighbor": 1, "hard_negative": 1}
