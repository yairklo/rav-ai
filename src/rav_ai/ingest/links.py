"""Ingest Sefaria links as `edges` between segments (set-based resolution in PostgreSQL).

Pipeline: COPY every link row into a staging table -> resolve exact single-segment citations
with one SQL join -> resolve the rest (sections, ranges, complex nodes) per work in Python via
WorkIndex -> expand to segment pairs in SQL. Unresolvable links go to `unresolved_links`.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from rav_ai.sefaria.linkrefs import WorkIndex

Log = Callable[[str], None]
_LINK_FILE_RE = re.compile(r"^links\d+\.csv$")  # excludes links_by_book*.csv aggregates
MAX_IDS = 3000  # stored ids per endpoint
EXPAND_LIMIT = 3000  # max segment pairs generated for one link; beyond that: collapsed edge


def link_files(raw_dir: Path) -> list[Path]:
    d = raw_dir / "links"
    return sorted(
        (p for p in d.glob("links*.csv") if _LINK_FILE_RE.match(p.name)),
        key=lambda p: int(p.stem.removeprefix("links")),
    )


def iter_rows(raw_dir: Path) -> Iterator[tuple]:
    for path in link_files(raw_dir):
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Citation 1,Citation 2,Conection Type,Text 1,Text 2,...
            for n, row in enumerate(reader, start=2):
                if len(row) >= 5:
                    yield row[0], row[1], row[2], row[3], row[4], path.name, n


def toc_titles(raw_dir: Path) -> set[str]:
    p = raw_dir / "table_of_contents.json"
    if not p.exists():
        return set()
    out: set[str] = set()

    def rec(node):
        if isinstance(node, list):
            for n in node:
                rec(n)
        elif isinstance(node, dict):
            if "title" in node and "contents" not in node:
                out.add(node["title"])
            rec(node.get("contents", []))

    rec(json.loads(p.read_text(encoding="utf-8")))
    return out


def ingest_links(engine: Engine, raw_dir: Path, snap_id: int, log: Log = print) -> dict:
    stats: dict = {}
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("DROP TABLE IF EXISTS link_stage, link_exact, endpoint_res, lf")
        cur.execute("""CREATE UNLOGGED TABLE link_stage (
            id bigserial PRIMARY KEY, c1 text, c2 text, ltype text, t1 text, t2 text,
            file text, line int)""")
        n = 0
        with cur.copy("COPY link_stage (c1,c2,ltype,t1,t2,file,line) FROM STDIN") as cp:
            for row in iter_rows(raw_dir):
                cp.write_row(row)
                n += 1
        raw.commit()
        stats["links_total"] = n
        log(f"staged {n:,} links")

        cur.execute("DELETE FROM edges WHERE origin='sefaria' AND snapshot_id=%s", (snap_id,))
        cur.execute("DELETE FROM unresolved_links WHERE snapshot_id=%s", (snap_id,))
        cur.execute("""CREATE UNLOGGED TABLE link_exact AS
            SELECT l.id lid, s1.id a, s2.id b FROM link_stage l
            JOIN segments s1 ON s1.ref = l.c1 JOIN segments s2 ON s2.ref = l.c2""")
        cur.execute("CREATE INDEX ON link_exact(lid)")
        cur.execute(f"""INSERT INTO edges (segment_a_id, segment_b_id, edge_type, origin,
              a_span, b_span, range_collapsed, citation_a, citation_b, snapshot_id)
            SELECT DISTINCT ON (e.a, e.b, l.ltype) e.a, e.b, l.ltype, 'sefaria', 1, 1, false,
                   l.c1, l.c2, {int(snap_id)}
            FROM link_exact e JOIN link_stage l ON l.id = e.lid WHERE e.a <> e.b""")
        cur.execute("SELECT count(*) FROM link_exact")
        stats["links_exact"] = cur.fetchone()[0]
        raw.commit()
        log(f"exact single-segment links: {stats['links_exact']:,}")

        cur.execute("""SELECT c, t FROM (
            SELECT l.c1 c, l.t1 t FROM link_stage l
              WHERE NOT EXISTS (SELECT 1 FROM link_exact e WHERE e.lid=l.id)
            UNION
            SELECT l.c2, l.t2 FROM link_stage l
              WHERE NOT EXISTS (SELECT 1 FROM link_exact e WHERE e.lid=l.id)) x""")
        by_book: dict[str, list[str]] = {}
        for c, t in cur.fetchall():
            by_book.setdefault(t, []).append(c)
        log(f"{sum(map(len, by_book.values())):,} non-trivial endpoints in {len(by_book)} books")

        cur.execute("SELECT title, id FROM works")
        works = dict(cur.fetchall())
        known = toc_titles(raw_dir)
        cur.execute("""CREATE UNLOGGED TABLE endpoint_res (
            c text, t text, ids bigint[], n int, reason text)""")
        reader = engine.raw_connection()  # separate connection: a COPY blocks its own session
        with cur.copy("COPY endpoint_res (c,t,ids,n,reason) FROM STDIN") as cp:
            for title, cits in by_book.items():
                wid = works.get(title)
                if wid is None:
                    reason = "no_hebrew_version" if title in known else "book_not_in_export"
                    for c in cits:
                        cp.write_row((c, title, None, 0, reason))
                    continue
                cur2 = reader.cursor()
                cur2.execute(
                    """SELECT ref, node_index, indices, node_path, section_path, id
                                FROM segments WHERE work_id=%s""",
                    (wid,),
                )
                idx = WorkIndex.build(title, cur2.fetchall())
                cur2.close()
                for c in cits:
                    ids = idx.resolve(c)
                    if ids is None:
                        cp.write_row((c, title, None, 0, "unparseable_citation"))
                    elif not ids:
                        cp.write_row((c, title, None, 0, "ref_not_found"))
                    else:
                        cp.write_row((c, title, ids[:MAX_IDS], len(ids), None))
        reader.close()
        cur.execute("CREATE INDEX ON endpoint_res (c, t)")
        raw.commit()
        log("endpoints resolved; expanding edges")

        cur.execute("""
        CREATE TEMP TABLE lf AS
        SELECT l.id, l.c1, l.c2, l.ltype, l.t1, l.t2, r1.ids ids1, r1.n n1, r1.reason rs1,
               r2.ids ids2, r2.n n2, r2.reason rs2
        FROM link_stage l
        JOIN endpoint_res r1 ON r1.c=l.c1 AND r1.t=l.t1
        JOIN endpoint_res r2 ON r2.c=l.c2 AND r2.t=l.t2
        WHERE NOT EXISTS (SELECT 1 FROM link_exact e WHERE e.lid=l.id)""")
        cur.execute(f"""INSERT INTO edges (segment_a_id, segment_b_id, edge_type, origin,
              a_span, b_span, range_collapsed, citation_a, citation_b, snapshot_id)
            SELECT DISTINCT ON (a, b, ltype) a, b, ltype, 'sefaria', n1, n2, collapsed,
                   c1, c2, {int(snap_id)}
            FROM (
              SELECT x AS a, y AS b, ltype, n1, n2, false AS collapsed, c1, c2
              FROM lf, LATERAL unnest(ids1) x, LATERAL unnest(ids2) y
              WHERE ids1 IS NOT NULL AND ids2 IS NOT NULL AND n1::bigint*n2 <= {EXPAND_LIMIT}
              UNION ALL
              SELECT ids1[1], ids2[1], ltype, n1, n2, true, c1, c2 FROM lf
              WHERE ids1 IS NOT NULL AND ids2 IS NOT NULL AND n1::bigint*n2 > {EXPAND_LIMIT}
            ) q WHERE a <> b""")
        cur.execute(f"""INSERT INTO unresolved_links
              (snapshot_id, citation_a, citation_b, edge_type, reason, side, book_a, book_b)
            SELECT {int(snap_id)}, c1, c2, ltype, COALESCE(rs1, rs2),
                   CASE WHEN rs1 IS NOT NULL AND rs2 IS NOT NULL THEN 'both'
                        WHEN rs1 IS NOT NULL THEN 'a' ELSE 'b' END, t1, t2
            FROM lf WHERE rs1 IS NOT NULL OR rs2 IS NOT NULL""")
        cur.execute("SELECT count(*) FROM unresolved_links WHERE snapshot_id=%s", (snap_id,))
        stats["links_unresolved"] = cur.fetchone()[0]
        cur.execute(
            "SELECT reason, count(*) FROM unresolved_links WHERE snapshot_id=%s "
            "GROUP BY 1 ORDER BY 2 DESC",
            (snap_id,),
        )
        stats["unresolved_by_reason"] = dict(cur.fetchall())
        cur.execute(
            "SELECT count(*) FROM edges WHERE snapshot_id=%s AND origin='sefaria'", (snap_id,)
        )
        stats["edges"] = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM edges WHERE snapshot_id=%s AND range_collapsed", (snap_id,)
        )
        stats["edges_range_collapsed"] = cur.fetchone()[0]

        # diff vs previous snapshot's edges, then drop the old generation
        cur.execute(
            """SELECT max(snapshot_id) FROM edges WHERE origin='sefaria' AND snapshot_id<>%s""",
            (snap_id,),
        )
        prev = cur.fetchone()[0]
        if prev:
            cur.execute(
                """SELECT count(*) FROM (SELECT segment_a_id,segment_b_id,edge_type FROM edges
                WHERE snapshot_id=%s EXCEPT SELECT segment_a_id,segment_b_id,edge_type FROM edges
                WHERE snapshot_id=%s) x""",
                (snap_id, prev),
            )
            stats["edges_added_vs_prev"] = cur.fetchone()[0]
            cur.execute(
                """SELECT count(*) FROM (SELECT segment_a_id,segment_b_id,edge_type FROM edges
                WHERE snapshot_id=%s EXCEPT SELECT segment_a_id,segment_b_id,edge_type FROM edges
                WHERE snapshot_id=%s) x""",
                (prev, snap_id),
            )
            stats["edges_removed_vs_prev"] = cur.fetchone()[0]
        cur.execute("DELETE FROM edges WHERE origin='sefaria' AND snapshot_id<>%s", (snap_id,))
        cur.execute("DELETE FROM unresolved_links WHERE snapshot_id<>%s", (snap_id,))
        cur.execute("DROP TABLE IF EXISTS link_exact, endpoint_res, link_stage, lf")
        raw.commit()
    finally:
        raw.close()
    stats["resolved_pct"] = round(
        100 * (stats["links_total"] - stats["links_unresolved"]) / max(stats["links_total"], 1), 3
    )
    return stats


def summarize_reasons(stats: dict) -> Counter:
    return Counter(stats.get("unresolved_by_reason", {}))


_ = text  # imported for callers composing SQL
