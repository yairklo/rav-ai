import json

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from conftest import BER, HEB1, MB, RASHI, SA, build_export
from rav_ai.config import ROOT
from rav_ai.db.models import Version
from rav_ai.ingest.links import ingest_links
from rav_ai.ingest.sefaria import finish_snapshot, ingest_all, start_snapshot
from rav_ai.license_report import fetch_rows, markdown, summarize, write_csv
from rav_ai.lookup import get_texts, neighbors


def run_import(engine, raw_dir, policy):
    snap = start_snapshot(engine, raw_dir)
    stats = ingest_all(engine, raw_dir, policy, snap.id, log=lambda *_: None)
    stats["links"] = ingest_links(engine, raw_dir, snap.id, log=lambda *_: None)
    finish_snapshot(engine, snap.id, stats)
    return snap.id, stats


def test_canonical_ref_gives_exact_raw_text_and_hash(engine, raw_dir, policy):
    run_import(engine, raw_dir, policy)
    with Session(engine) as s:
        rows = get_texts(s, f"{SA} 1:1")
        he = [r for r in rows if r.version_title == "Torat Emet"]
        assert he[0].text_raw == HEB1  # byte-exact: niqqud, markup, double space, newline
        assert rows[0].is_primary
        h = s.execute(
            text("""SELECT raw_hash, text_a, text_b FROM segment_text t
              JOIN segments g ON g.id=t.segment_id JOIN versions v ON v.id=t.version_id
              WHERE g.ref=:r AND v.version_title='Torat Emet'"""),
            {"r": f"{SA} 1:1"},
        ).one()
        import hashlib

        assert h[0] == hashlib.sha256(HEB1.encode()).hexdigest()
        assert h[1] == "יתגבר כארי לעמד בבקר"
        assert get_texts(s, f"{SA} 2:1") == []  # empty source segment is not created
        assert get_texts(s, f"{SA} 2:2")[0].text_raw == "סעיף ג בסימן ב"
        seg = s.execute(
            text("SELECT section_path, indices, parent_ref FROM segments WHERE ref=:r"),
            {"r": f"{BER} 2b:1"},
        ).one()
        assert seg == (["2b", "1"], [3, 0], f"{BER} 2b")  # Talmud daf addressing


def test_license_flags_stored_per_version_and_nothing_filtered_at_ingest(engine, raw_dir, policy):
    run_import(engine, raw_dir, policy)
    with Session(engine) as s:
        flags = {
            v.version_title: (v.license, v.can_index, v.can_display_public, v.can_train)
            for v in s.scalars(select(Version))
        }
    assert flags["Torat Emet"] == ("Public Domain", True, True, True)
    assert flags["NC Edition"] == ("CC-BY-NC", True, False, False)
    assert flags["Restricted"] == ("Copyrighted", False, False, False)
    assert flags["No License"] == (None, False, False, False)  # fail closed
    assert flags["MB"] == ("CC-BY-SA", True, True, True)
    assert "Eng" not in flags  # English is out of scope (Hebrew/Aramaic only)
    with Session(engine) as s:  # ...but restricted texts ARE stored (gates act later)
        assert get_texts(s, f"{SA} 1:1")[-1].text_raw
        assert len(get_texts(s, f"{SA} 1:1")) == 4


def test_display_gate_public_vs_personal(engine, raw_dir, policy, monkeypatch):
    run_import(engine, raw_dir, policy)
    with Session(engine) as s:
        rows = {r.version_title: r for r in get_texts(s, f"{SA} 1:1")}
    assert rows["Restricted"].shown("personal").startswith("[withheld")  # can_index False
    assert rows["NC Edition"].shown("personal") == "נ״ק א"
    assert rows["NC Edition"].shown("public").startswith("[withheld")
    assert rows["Torat Emet"].shown("public") == HEB1
    monkeypatch.setenv("RAV_DEPLOYMENT_MODE", "public")
    assert rows["NC Edition"].shown().startswith("[withheld")


def test_export_gate(engine, raw_dir, policy, tmp_path):
    from rav_ai.cli.export import main

    run_import(engine, raw_dir, policy)
    out = tmp_path / "train.jsonl"
    main(["--purpose", "train", "--out", str(out), "--db", engine.url.render_as_string(False)])
    versions = {
        json.loads(line)["version"] for line in out.read_text(encoding="utf-8").splitlines()
    }
    assert versions == {"Torat Emet", "MB", "Vilna", "Rashi"}  # no NC / Restricted / unlicensed


def test_edges_resolution_and_bidirectionality(engine, raw_dir, policy):
    _, stats = run_import(engine, raw_dir, policy)
    ls = stats["links"]
    assert ls["links_total"] == 7
    assert ls["unresolved_by_reason"] == {
        "ref_not_found": 1,
        "no_hebrew_version": 1,
        "book_not_in_export": 1,
    }
    assert ls["links_unresolved"] == 3
    with Session(engine) as s:

        def linked(ref):
            return {
                (o.ref, o.edge_type)
                for side in neighbors(s, ref)[0].values()
                for byp in side.values()
                for items in byp.values()
                for o in items
            }

        assert (f"{MB} 1:1", "commentary") in linked(f"{SA} 1:1")
        assert (f"{SA} 1:1", "commentary") in linked(f"{MB} 1:1")  # both directions
        # range <-> range expands to pairs; queryable from any member
        assert {r for r, _ in linked(f"{BER} 2a:2")} >= {f"{SA} 1:1", f"{SA} 1:2"}
        assert (f"{BER} 2a:1", "commentary") in linked(f"{RASHI} 2a:1:1")
        # section-level endpoints (Berakhot 2a <-> MB siman 1) expand to their segments
        assert (f"{MB} 1:2", "") in linked(f"{BER} 2a:1")
        # link types are kept verbatim
        types = {t for (t,) in s.execute(text("SELECT DISTINCT edge_type FROM edges"))}
        assert types == {"commentary", "", "quotation"} - {"quotation"}  # quotation was unresolved


def test_layers_split_by_period(engine, raw_dir, policy):
    run_import(engine, raw_dir, policy)
    with Session(engine) as s:
        out, period = neighbors(s, f"{BER} 2a:1")
        assert period == "talmud"
        above = {o.ref for byp in out["above"].values() for it in byp.values() for o in it}
        assert f"{RASHI} 2a:1:1" in above  # rishonim sit above the Gemara
        assert any(
            o.period == "acharonim"
            for byp in out["above"].values()
            for it in byp.values()
            for o in it
        )
        out, period = neighbors(s, f"{RASHI} 2a:1:1")
        below = {o.ref for byp in out["below"].values() for it in byp.values() for o in it}
        assert f"{BER} 2a:1" in below


def test_snapshot_and_hash_diff(engine, raw_dir, policy):
    sid1, _ = run_import(engine, raw_dir, policy)
    # Re-import of the identical export: nothing re-processed, no changes
    sid2 = start_snapshot(engine, raw_dir)  # first snapshot is 'complete' -> new one opens
    assert sid2.id != sid1
    stats = ingest_all(engine, raw_dir, policy, sid2.id, log=lambda *_: None)
    assert stats.get("versions_processed", 0) == 0 and stats["versions_unchanged_skipped"] > 0
    finish_snapshot(engine, sid2.id, stats)

    # New export: one segment changed, one added, one removed
    build_export(raw_dir, sa_second="סעיף ב — תוקן")
    p = raw_dir / f"json/Halakhah/Shulchan Arukh/{SA}/Hebrew/Torat Emet.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["text"] = [[HEB1, "סעיף ב — תוקן", "סעיף חדש"]]  # 1:1 same, 1:2 changed, 1:3 added
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")  # 2:2 removed
    (raw_dir / "last_export.txt").write_text("2026-02-01T00:00:00", encoding="utf-8")
    sid3 = start_snapshot(engine, raw_dir)
    st3 = ingest_all(engine, raw_dir, policy, sid3.id, log=lambda *_: None)
    assert st3["errors"] == [], st3
    with engine.connect() as c:
        ch = c.execute(
            text("""SELECT change, ref FROM snapshot_changes WHERE snapshot_id=:s
                               ORDER BY ref"""),
            {"s": sid3.id},
        ).all()
    assert sorted(ch) == sorted(
        [("added", f"{SA} 1:3"), ("changed", f"{SA} 1:2"), ("removed", f"{SA} 2:2")]
    )
    with Session(engine) as s:
        assert get_texts(s, f"{SA} 1:2")[0].text_raw == "סעיף ב — תוקן"
        assert get_texts(s, f"{SA} 1:1")[0].text_raw == HEB1  # unchanged text untouched
        assert get_texts(s, f"{SA} 2:2") == [] or all(
            r.version_title != "Torat Emet" for r in get_texts(s, f"{SA} 2:2")
        )


def test_license_report_is_per_version(engine, raw_dir, policy, tmp_path):
    run_import(engine, raw_dir, policy)
    rows = fetch_rows(engine)
    assert len(rows) == 7  # 4 SA + MB + Berakhot + Rashi Hebrew versions
    summ = summarize(rows)
    assert summ["can_train"]["cannot_train"]["versions"] == 3  # NC, Restricted, none
    assert summ["license"]["CC-BY-NC"]["books"] == 1
    out = tmp_path / "l.csv"
    write_csv(rows, summ, out)
    last = out.read_text(encoding="utf-8-sig").strip().splitlines()[-1]
    assert last.startswith("#,total,all")
    assert "By can_train" in markdown(summ)


def test_alembic_creates_everything_including_pgvector(engine):
    from rav_ai.db.models import Base

    with engine.connect() as c:
        tables = {
            r[0]
            for r in c.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
        }
        ext = c.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar()
    assert set(Base.metadata.tables) <= tables
    assert {"quote_spans", "annotations", "cards", "edges", "segment_text"} <= tables
    assert ext == 1


def test_policy_file_defaults():
    from rav_ai.config import LicensePolicy

    p = LicensePolicy.load(ROOT / "config" / "license_policy.toml")
    assert not p.evaluate("CC-BY-NC").can_train and not p.evaluate("CC-BY-NC").can_display_public
    assert not p.evaluate("Copyrighted").can_index
    assert p.evaluate("Public Domain").can_train and p.evaluate("cc0").can_display_public
    assert p.evaluate(None) == p.evaluate("something-new")  # fail closed
