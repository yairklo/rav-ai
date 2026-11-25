"""Ingest report: per book segments, versions, licenses, link resolution, tokens (tiktoken)."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from rav_ai.config import LicensePolicy

COLUMNS = [
    "book", "category", "period", "segments", "versions", "licenses", "versions_can_index",
    "versions_can_display_public", "versions_can_train", "links_resolved", "links_unresolved",
    "links_resolved_pct", "words_top_version", "tokens_top_version", "tokens_per_word",
]  # fmt: skip

BOOKS_SQL = text("""
SELECT w.id, w.title, w.category, COALESCE(w.period,'unknown'),
  (SELECT count(*) FROM segments s WHERE s.work_id = w.id),
  count(v.id), COALESCE(sum(v.can_index::int),0), COALESCE(sum(v.can_display_public::int),0),
  COALESCE(sum(v.can_train::int),0)
FROM works w LEFT JOIN versions v ON v.work_id = w.id GROUP BY w.id ORDER BY w.category, w.title""")

LICENSES_SQL = text("""SELECT work_id, COALESCE(NULLIF(license,''),'(none)'), count(*)
                       FROM versions GROUP BY 1,2 ORDER BY 1,3 DESC""")

TOP_VERSION_SQL = text("""
SELECT DISTINCT ON (work_id) work_id, id, word_count FROM versions
ORDER BY work_id, is_primary DESC, word_count DESC, id""")

RESOLVED_SQL = text("""
SELECT w.title, count(DISTINCT (e.citation_a, e.citation_b, e.edge_type)) FROM (
  SELECT segment_a_id sid, citation_a, citation_b, edge_type FROM edges
  UNION ALL SELECT segment_b_id, citation_a, citation_b, edge_type FROM edges) e
JOIN segments s ON s.id = e.sid JOIN works w ON w.id = s.work_id GROUP BY w.title""")

UNRESOLVED_SQL = text("""
SELECT b, count(*) FROM (SELECT book_a b FROM unresolved_links
  UNION ALL SELECT book_b FROM unresolved_links WHERE book_b IS DISTINCT FROM book_a) x
GROUP BY b""")


def token_counts(engine: Engine, top_versions: dict[int, int], encoding: str, log=print):
    """work_id -> (words, tokens) over the text_b of each work's top version."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    version_to_work = {v: w for w, v in top_versions.items()}
    words: Counter = Counter()
    toks: Counter = Counter()
    batch: list[tuple[int, str]] = []

    def flush():
        if not batch:
            return
        ids = enc.encode_ordinary_batch([t for _, t in batch], num_threads=8)
        for (w, t), tk in zip(batch, ids, strict=True):
            words[w] += len(t.split())
            toks[w] += len(tk)
        batch.clear()

    q = text("SELECT version_id, text_b FROM segment_text WHERE version_id = ANY(:v)")
    with engine.connect().execution_options(stream_results=True, yield_per=5000) as c:
        for n, (vid, tb) in enumerate(c.execute(q, {"v": list(top_versions.values())}), 1):
            if tb:
                batch.append((version_to_work[vid], tb))
            if len(batch) >= 4000:
                flush()
            if n % 500_000 == 0:
                log(f"  tokenized {n:,} segments")
    flush()
    return words, toks


def in_scope_pct(lk: dict) -> str:
    by = lk.get("unresolved_by_reason", {})
    scope = (
        lk.get("links_total", 0) - by.get("book_not_in_export", 0) - by.get("no_hebrew_version", 0)
    )
    bad = (
        lk.get("links_unresolved", 0)
        - by.get("book_not_in_export", 0)
        - by.get("no_hebrew_version", 0)
    )
    return f"{100 * (scope - bad) / scope:.3f}" if scope else "n/a"


def build(
    engine: Engine,
    policy: LicensePolicy,
    snapshot_stats: dict | None,
    out_dir: Path,
    encoding: str = "o200k_base",
    log=print,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with engine.connect() as c:
        books = c.execute(BOOKS_SQL).all()
        lic_by_work: dict[int, list[str]] = defaultdict(list)
        for wid, lic, n in c.execute(LICENSES_SQL):
            lic_by_work[wid].append(f"{lic}:{n}")
        top = {w: (v, wc) for w, v, wc in c.execute(TOP_VERSION_SQL)}
        resolved = dict(c.execute(RESOLVED_SQL).all())
        unresolved = dict(c.execute(UNRESOLVED_SQL).all())
        lic_rows = c.execute(
            text("""SELECT COALESCE(NULLIF(license,''),'(none)'), count(*),
              sum(word_count) FROM versions GROUP BY 1 ORDER BY 3 DESC""")
        ).all()
        reasons = c.execute(
            text("""SELECT reason, count(*) FROM unresolved_links
              GROUP BY 1 ORDER BY 2 DESC""")
        ).all()
        samples = {}
        for reason, _ in reasons:
            samples[reason] = c.execute(
                text("""SELECT citation_a, citation_b, edge_type, side
                FROM unresolved_links WHERE reason=:r ORDER BY id LIMIT 10"""),
                {"r": reason},
            ).all()
        top_books = c.execute(
            text("""SELECT reason, COALESCE(CASE WHEN side='b' THEN book_b
              ELSE book_a END,'?'), count(*) FROM unresolved_links GROUP BY 1,2
              ORDER BY 3 DESC LIMIT 25""")
        ).all()
    with engine.connect() as c:
        rnf_top = c.execute(
            text("""SELECT k, count(*) FROM (
              SELECT CASE WHEN book_a ~ 'Zohar' OR book_b ~ 'Zohar' THEN 'Zohar (any)'
                WHEN book_a ~ '^Rav Hirsch on Torah' OR book_b ~ '^Rav Hirsch on Torah'
                  THEN 'Rav Hirsch on Torah'
                ELSE CASE WHEN side='b' THEN book_b ELSE book_a END END k
              FROM unresolved_links WHERE reason='ref_not_found') u
              GROUP BY 1 ORDER BY 2 DESC LIMIT 20""")
        ).all()
        rnf_gap = c.execute(
            text("""SELECT count(*) FILTER (WHERE book_a ~ 'Zohar' OR book_b ~ 'Zohar'
                OR book_a ~ '^Rav Hirsch on Torah' OR book_b ~ '^Rav Hirsch on Torah'),
              count(*) FROM unresolved_links WHERE reason='ref_not_found'""")
        ).one()
        ritva = c.execute(
            text("SELECT count(*) FROM works WHERE title = 'Ritva on Shabbat'")
        ).scalar()
    with engine.connect() as c:
        nowiki = c.execute(
            text("""
            SELECT w.title, s.section_path->>0 AS daf, count(*) FROM works w
            JOIN segments s ON s.work_id = w.id JOIN segment_text t ON t.segment_id = s.id
            JOIN versions v ON v.id = t.version_id
            WHERE w.category_path ? 'Bavli' AND w.dependence IS NULL
            GROUP BY w.title, daf, s.node_index HAVING NOT bool_or(v.is_primary)
            ORDER BY w.title, s.node_index, min(s.sort_key)""")
        ).all()
        bavli_pages = c.execute(
            text("""
            SELECT count(*) FROM (SELECT 1 FROM works w JOIN segments s ON s.work_id = w.id
            WHERE w.category_path ? 'Bavli' AND w.dependence IS NULL
            GROUP BY w.title, s.section_path->>0) x""")
        ).scalar()
    words, toks = token_counts(engine, {w: v for w, (v, _) in top.items()}, encoding, log)

    rows = []
    for wid, title, cat, period, nseg, nver, ci, cd, ct in books:
        r_ok, r_bad = resolved.get(title, 0), unresolved.get(title, 0)
        pct = f"{100 * r_ok / (r_ok + r_bad):.2f}" if r_ok + r_bad else ""
        w, t = words.get(wid, 0), toks.get(wid, 0)
        rows.append(
            [
                title,
                cat,
                period,
                nseg,
                nver,
                "; ".join(lic_by_work.get(wid, [])),
                ci,
                cd,
                ct,
                r_ok,
                r_bad,
                pct,
                w,
                t,
                f"{t / w:.2f}" if w else "",
            ]
        )
    csv_path = out_dir / "books.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(COLUMNS)
        wr.writerows(rows)

    # license problems: non-empty license strings the policy does not recognize, and missing ones
    problems = []
    for lic, n, wc in lic_rows:
        key = policy.match(None if lic == "(none)" else lic)
        if key == "unknown":
            problems.append((lic, n, wc or 0))

    by_period: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        a = by_period[r[2]]
        a[0] += r[12]
        a[1] += r[13]
        a[2] += 1
    lk = (snapshot_stats or {}).get("links", {})
    md = [
        "# Ingest report",
        "",
        f"Books: {len(rows):,}; versions: {sum(r[4] for r in rows):,}; "
        f"segments: {sum(r[3] for r in rows):,}",
        "",
        "## Link resolution",
        "",
    ]
    if lk:
        md += [
            f"- links total: {lk.get('links_total', 0):,}; unresolved: "
            f"{lk.get('links_unresolved', 0):,}; **resolved: {lk.get('resolved_pct')}%**",
            "- **resolved among links whose both books are ingested (Hebrew): "
            f"{in_scope_pct(lk)}%** "
            "(excludes book_not_in_export / no_hebrew_version: books outside the "
            "Hebrew export)",
            f"- exact single-segment links: {lk.get('links_exact', 0):,}; edges: "
            f"{lk.get('edges', 0):,} (range-collapsed: {lk.get('edges_range_collapsed', 0):,})",
            "",
        ]
    md += ["| reason | links |", "|---|---:|", *[f"| {r} | {n:,} |" for r, n in reasons], ""]
    md += [
        "### Unresolved: top books by reason",
        "",
        "| reason | book | links |",
        "|---|---|---:|",
        *[f"| {r} | {b} | {n:,} |" for r, b, n in top_books],
        "",
    ]
    gap, rnf_all = rnf_gap
    md += [
        "### `ref_not_found` by book (top 20)",
        "",
        f"Export gaps (Hebrew text absent or numbered differently from the links): "
        f"Zohar + Rav Hirsch = {gap:,} of {rnf_all:,}. The rest ({rnf_all - gap:,}) is not "
        "individually classified: partial Hebrew texts vs. resolver failures.",
        "",
        "| book | links |",
        "|---|---:|",
        *[f"| {k} | {n:,} |" for k, n in rnf_top],
        "",
    ]
    if not ritva:
        md += ["**Known export gap:** `Ritva on Shabbat` is not in the Sefaria export.", ""]
    md += ["### Unresolved: examples (citation A | citation B | type | side)", ""]
    for reason, ex in samples.items():
        md.append(f"**{reason}**")
        md += [f"- `{a}` | `{b}` | {t or '(untyped)'} | {s}" for a, b, t, s in ex]
        md.append("")
    md += [
        "## Licenses (per version)",
        "",
        "| license | versions | words |",
        "|---|---:|---:|",
        *[f"| {lic} | {n:,} | {wc or 0:,} |" for lic, n, wc in lic_rows],
        "",
    ]
    md += ["### License problems (fail-closed => can_index=false)", ""]
    md += [f"- `{lic}`: {n} versions, {wc:,} words" for lic, n, wc in problems] or ["- none"]
    md += [
        "",
        "## Bavli pages without a text in the primary (Wikisource) version",
        "",
        f"{len(nowiki):,} of {bavli_pages:,} pages (these fall back to other versions, which "
        "may be non-displayable).",
        "",
    ]
    by_tr: dict[str, list[str]] = defaultdict(list)
    for tr, daf, _ in nowiki:
        by_tr[tr].append(daf)
    for tr, dafim in by_tr.items():
        md.append(
            f"- **{tr}** ({len(dafim)}): {', '.join(dafim[:25])}"
            + (" ..." if len(dafim) > 25 else "")
        )
    md += [
        "",
        f"## Tokens ({encoding}, top version per book, form B)",
        "",
        "| period | books | words | tokens | tokens/word |",
        "|---|---:|---:|---:|---:|",
    ]
    tw = tt = 0
    for period, (w, t, nb) in sorted(by_period.items(), key=lambda kv: -kv[1][1]):
        md.append(
            f"| {period} | {nb} | {w:,} | {t:,} | {t / w:.2f} |"
            if w
            else f"| {period} | {nb} | 0 | 0 | |"
        )
        tw, tt = tw + w, tt + t
    md.append(f"| **all** | {len(rows)} | {tw:,} | {tt:,} | {tt / max(tw, 1):.2f} |")
    (out_dir / "ingest_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"books": len(rows), "csv": str(csv_path), "words": tw, "tokens": tt}
