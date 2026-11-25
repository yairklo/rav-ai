"""Per-version license report generated from the database (versions + works + authors)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

COLUMNS = [
    "book",
    "category",
    "period",
    "author",
    "language",
    "version_title",
    "source_url",
    "license",
    "words",
    "can_index",
    "can_display_public",
    "can_train",
]

QUERY = text("""
SELECT w.title, w.category, COALESCE(w.period,'unknown'), COALESCE(a.name,''), v.language,
       v.version_title, COALESCE(v.source_url,''), COALESCE(v.license,'(none)'), v.word_count,
       v.can_index, v.can_display_public, v.can_train
FROM versions v JOIN works w ON w.id=v.work_id LEFT JOIN authors a ON a.id=w.author_id
ORDER BY w.category, w.title, v.language, v.version_title""")


def fetch_rows(engine: Engine) -> list[tuple]:
    with engine.connect() as c:
        return [tuple(r) for r in c.execute(QUERY)]


def summarize(rows: list[tuple]) -> dict:
    i = {c: n for n, c in enumerate(COLUMNS)}

    def agg(keyf, filt=lambda r: True):
        d = defaultdict(lambda: [set(), 0, 0])
        for r in rows:
            if filt(r):
                a = d[keyf(r)]
                a[0].add(r[i["book"]])
                a[1] += 1
                a[2] += r[i["words"]]
        return {
            k: {"books": len(v[0]), "versions": v[1], "words": v[2]}
            for k, v in sorted(d.items(), key=lambda kv: -kv[1][2])
        }

    out = {
        "license": agg(lambda r: r[i["license"]]),
        "category": agg(lambda r: r[i["category"]]),
        "period": agg(lambda r: r[i["period"]]),
        "can_train": agg(lambda r: "can_train" if r[i["can_train"]] else "cannot_train"),
        "can_index": agg(lambda r: "can_index" if r[i["can_index"]] else "cannot_index"),
        "can_display_public": agg(
            lambda r: "can_display_public" if r[i["can_display_public"]] else "no_public_display"
        ),
        "total": agg(lambda r: "all"),
    }
    out["license_x_category"] = {
        lic: agg(lambda r: r[i["category"]], lambda r, lic=lic: r[i["license"]] == lic)
        for lic in out["license"]
    }
    return out


def write_csv(rows: list[tuple], summary: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([str(x).lower() if isinstance(x, bool) else x for x in r])
        w.writerow([])
        w.writerow(["# SUMMARY", "dimension", "value", "books", "versions", "words"])
        for dim in (
            "can_train",
            "can_index",
            "can_display_public",
            "license",
            "period",
            "category",
            "total",
        ):
            for k, v in summary[dim].items():
                w.writerow(["#", dim, k, v["books"], v["versions"], v["words"]])


def markdown(summary: dict) -> str:
    t = summary["total"].get("all", {"books": 0, "versions": 0, "words": 0})
    lines = [
        "# License summary (per version)",
        "",
        f"Total: {t['books']} books, {t['versions']} versions, {t['words']:,} words "
        "(words = whitespace tokens of the search-normalized text)",
        "",
    ]
    for dim in ("can_train", "can_index", "can_display_public", "license", "period", "category"):
        lines += [
            f"## By {dim}",
            "",
            f"| {dim} | books | versions | words |",
            "|---|---:|---:|---:|",
        ]
        for k, v in summary[dim].items():
            lines.append(f"| {k} | {v['books']} | {v['versions']} | {v['words']:,} |")
        lines.append("")
    return "\n".join(lines)
