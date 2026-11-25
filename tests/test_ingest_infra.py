import json

from sqlalchemy import text

from export_fixture import BER, MB, RASHI, SA
from rav_ai.config import LicensePolicy
from rav_ai.ingest.links import ingest_links
from rav_ai.ingest.manifest import write_manifest
from rav_ai.ingest.order import LAYER_NAMES, layer_of
from rav_ai.ingest.report import build
from rav_ai.ingest.sefaria import finish_snapshot, ingest_all, start_snapshot

quiet = lambda *_: None  # noqa: E731


def run(engine, raw_dir, policy):
    snap = start_snapshot(engine, raw_dir)
    stats = ingest_all(engine, raw_dir, policy, snap.id, log=quiet)
    stats["links"] = ingest_links(engine, raw_dir, snap.id, log=quiet)
    finish_snapshot(engine, snap.id, stats)
    return snap.id, stats


def counts(engine):
    with engine.connect() as c:
        return {
            t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar()
            for t in (
                "works",
                "versions",
                "segments",
                "segment_text",
                "edges",
                "unresolved_links",
                "authors",
            )
        }


def test_rerun_is_idempotent(engine, raw_dir, policy):
    run(engine, raw_dir, policy)
    first = counts(engine)
    _, stats = run(engine, raw_dir, policy)  # same export, new snapshot
    assert counts(engine) == first  # no duplicate segments/texts/edges/unresolved rows
    assert stats.get("versions_processed", 0) == 0
    # an interrupted run resumes the same snapshot instead of opening a new one
    s1 = start_snapshot(engine, raw_dir)
    s2 = start_snapshot(engine, raw_dir)
    assert s1.id == s2.id


def test_unresolved_links_record_books(engine, raw_dir, policy):
    run(engine, raw_dir, policy)
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT book_a, book_b, reason FROM unresolved_links ORDER BY reason")
        ).all()
    assert set(rows) == {
        (SA, "English Only Book", "no_hebrew_version"),
        (SA, "Mystery Book", "book_not_in_export"),
        (BER, SA, "ref_not_found"),
        (SA, f"{BER}, Appendix", "ref_not_found"),  # node of a work: not "book_not_in_export"
    }


def test_manifest_has_checksums(raw_dir, tmp_path):
    out = write_manifest(raw_dir, tmp_path / "snap", log=quiet)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["export_label"] == "2026-01-01T00:00:00"
    paths = {f["path"]: f for f in doc["files"]}
    assert all(len(f["sha256"]) == 64 for f in doc["files"])
    assert not any(p.endswith("merged.json") for p in paths) and "links/links0.csv" in paths


def test_layer_order():
    assert layer_of("Shabbat", ["Talmud", "Bavli", "Seder Moed"]) == 0
    assert layer_of("Mishnah Shabbat", ["Mishnah", "Seder Moed"]) == 0
    names = [
        LAYER_NAMES[layer_of(t, ["X"])]
        for t in (
            "Rashi on Shabbat",
            "Tosafot on Shabbat",
            "Mishneh Torah, Foo",
            "Chiddushei HaRamban on Shabbat",
            "Chiddushei HaRashba on Shabbat",
            "Chiddushei HaRitva on Shabbat",
            "Chiddushei HaRan on Shabbat",
            "Rosh on Shabbat",
            "Tur, Orach Chayim",
            "Beit Yosef, Orach Chayim",
            "Shulchan Arukh, Orach Chayim",
            "Mishnah Berurah",
            "Genesis",
        )
    ]
    assert names == [
        "rashi",
        "tosafot",
        "rambam",
        "ramban",
        "rashba",
        "ritva",
        "ran",
        "rosh",
        "tur+beit_yosef",
        "tur+beit_yosef",
        "shulchan_arukh+rema+nosei_keilim",
        "shulchan_arukh+rema+nosei_keilim",
        "rest",
    ]


def test_ingest_order_is_layered(engine, raw_dir, policy):
    order = []
    ingest_all(
        engine, raw_dir, policy, start_snapshot(engine, raw_dir).id, log=lambda m: order.append(m)
    )
    with engine.connect() as c:
        titles = [t for (t,) in c.execute(text("SELECT title FROM works ORDER BY id"))]
    assert titles == [BER, RASHI, SA, MB]  # Gemara, Rashi, SA, then SA commentary


def test_report(engine, raw_dir, policy, tmp_path, monkeypatch):
    import tiktoken

    class Enc:
        def encode_ordinary_batch(self, texts, num_threads=1):
            return [t.split() for t in texts]

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: Enc())
    _, stats = run(engine, raw_dir, policy)
    out = build(engine, LicensePolicy.load(), stats, tmp_path, log=quiet)
    assert out["books"] == 4
    md = (tmp_path / "ingest_report.md").read_text(encoding="utf-8")
    assert "resolved:" in md and "no_hebrew_version" in md and "`No License`" not in md
    header = (tmp_path / "books.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "versions_can_train" in header and "links_resolved_pct" in header
    import csv

    with open(tmp_path / "books.csv", encoding="utf-8-sig", newline="") as f:
        rows = {r["book"]: r for r in csv.DictReader(f)}
    assert "CC-BY-NC:1" in rows[SA]["licenses"] and rows[SA]["versions_can_train"] == "1"
    assert rows[SA]["tokens_top_version"] and rows[BER]["segments"] == "3"


def test_renormalize_updates_derived_forms_only(engine, raw_dir, policy):
    run(engine, raw_dir, policy)
    with engine.begin() as c:  # simulate an older normalizer's output
        c.execute(text("UPDATE segment_text SET text_a='OLD', text_b='OLD'"))
    snap = start_snapshot(engine, raw_dir)
    ingest_all(engine, raw_dir, policy, snap.id, log=quiet)  # plain re-run keeps the stale forms
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM segment_text WHERE text_a='OLD'")).scalar() > 0
    stats = ingest_all(engine, raw_dir, policy, snap.id, log=quiet, renormalize=True)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM segment_text WHERE text_a='OLD'")).scalar() == 0
        assert (
            c.execute(text("SELECT count(*) FROM snapshot_changes WHERE change='changed'")).scalar()
            == 0
        )  # raw hashes unchanged => no diff entries
    assert stats["changed"] == 0


def test_bavli_primary_is_wikisource_and_gaps_reported(
    engine, raw_dir, policy, tmp_path, monkeypatch
):
    import tiktoken

    class Enc:
        def encode_ordinary_batch(self, texts, num_threads=1):
            return [t.split() for t in texts]

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: Enc())
    _, stats = run(engine, raw_dir, policy)
    with engine.connect() as c:
        prim = dict(
            c.execute(
                text("""SELECT v.version_title, v.is_primary FROM versions v
              JOIN works w ON w.id=v.work_id WHERE w.title=:t"""),
                {"t": BER},
            ).all()
        )
        # a page that has both versions: the displayable text comes first
        first = c.execute(
            text("""SELECT v.version_title, v.can_display_public FROM segment_text t
              JOIN versions v ON v.id=t.version_id JOIN segments s ON s.id=t.segment_id
              WHERE s.ref=:r ORDER BY v.is_primary DESC, v.priority DESC NULLS LAST LIMIT 1"""),
            {"r": f"{BER} 2a:1"},
        ).one()
    assert prim["Wikisource Talmud Bavli"] is True
    assert prim["William Davidson Edition - Aramaic"] is False and prim["Vilna"] is False
    assert first == ("Wikisource Talmud Bavli", True)
    from rav_ai.cli.export import FLAG  # noqa: F401  (William Davidson stays, flags intact)

    with engine.connect() as c:
        wd = c.execute(
            text("""SELECT can_display_public, can_train, license FROM versions
              WHERE version_title LIKE 'William Davidson%'""")
        ).one()
        assert wd == (False, False, "CC-BY-NC")
    build(engine, LicensePolicy.load(), stats, tmp_path, log=quiet)
    md = (tmp_path / "ingest_report.md").read_text(encoding="utf-8")
    assert (
        "without a text in the primary" in md and "**Berakhot** (1): 2b" in md
    )  # 2b: no Wikisource
    # idempotent: re-applying the rules changes nothing
    from rav_ai.ingest.sefaria import apply_primary_rules

    apply_primary_rules(engine)
    with engine.connect() as c:
        assert (
            c.execute(
                text(
                    "SELECT count(*) FROM versions WHERE is_primary AND work_id="
                    "(SELECT id FROM works WHERE title=:t)"
                ),
                {"t": BER},
            ).scalar()
            == 1
        )


def test_unresolved_book_columns_hold_long_titles(engine):
    long_title = "Commentary on " + "x" * 400  # > 256, within real-export max (307) headroom
    with engine.begin() as c:
        sid = c.execute(
            text(
                "INSERT INTO snapshots (source, export_label, status, taken_at) "
                "VALUES ('sefaria', 't', 'running', now()) RETURNING id"
            )
        ).scalar_one()
        c.execute(
            text(
                "INSERT INTO unresolved_links (snapshot_id, citation_a, citation_b, edge_type,"
                " reason, side, book_a, book_b) VALUES (:s, 'a', 'b', '', 'ref_not_found',"
                " 'both', :t, :t)"
            ),
            {"s": sid, "t": long_title},
        )
        got = c.execute(text("SELECT book_a, book_b FROM unresolved_links")).one()
    assert got == (long_title, long_title)


def test_migrations_downgrade_upgrade_roundtrip(engine):
    from alembic import command
    from alembic.config import Config

    from rav_ai.config import ROOT

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", engine.url.render_as_string(hide_password=False))
    command.downgrade(cfg, "-1")
    command.upgrade(cfg, "head")
    with engine.connect() as c:
        n = c.execute(
            text(
                "SELECT character_maximum_length FROM information_schema.columns "
                "WHERE table_name='unresolved_links' AND column_name='book_a'"
            )
        ).scalar()
    assert n == 512


def test_work_prefix_picks_longest_parent():
    from rav_ai.ingest.links import work_prefix

    works = {"Tur": 1, "Tur HaArokh": 2, "Beit Yosef": 3}
    assert work_prefix("Tur, Orach Chayim", works) == "Tur"
    assert work_prefix("Tur HaArokh, Orach Chayim", works) == "Tur HaArokh"
    assert work_prefix("Turim", works) is None
