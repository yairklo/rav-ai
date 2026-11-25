"""Tokens-per-word for Hebrew/Aramaic with common tokenizers (tiktoken), overall and by period.

Words are whitespace tokens of the search-normalized text (text_b). A stratified sample
(--per-group segments per period x language-class) is tokenized; both text_b and text_a are
measured. 'aramaic-heavy' = base texts of Talmud/Targum/Zohar (heuristic by category); all else
is classed 'hebrew'. Writes reports/token_stats.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import tiktoken
from sqlalchemy import text

from rav_ai.config import ROOT
from rav_ai.db.session import db_url, make_engine

CLASS_SQL = """CASE WHEN w.dependence IS NULL AND (w.category IN ('Talmud','Targum')
                 OR w.title ILIKE 'Zohar%%') THEN 'aramaic-heavy' ELSE 'hebrew' END"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-group", type=int, default=20000)
    ap.add_argument("--encodings", nargs="+", default=["o200k_base", "cl100k_base"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "token_stats.md"))
    args = ap.parse_args(argv)
    encs = {n: tiktoken.get_encoding(n) for n in args.encodings}
    engine = make_engine(args.db or db_url())
    q = text(f"""
        SELECT period, cls, text_a, text_b FROM (
          SELECT COALESCE(w.period,'unknown') period, {CLASS_SQL} cls, t.text_a, t.text_b,
                 row_number() OVER (PARTITION BY COALESCE(w.period,'unknown'), {CLASS_SQL}
                                    ORDER BY md5(t.id::text)) rn
          FROM segment_text t JOIN versions v ON v.id=t.version_id
          JOIN segments s ON s.id=t.segment_id JOIN works w ON w.id=s.work_id
          WHERE v.is_primary AND length(t.text_b) > 0) x
        WHERE rn <= :n""")
    acc: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    with engine.connect() as c:
        for period, cls, ta, tb in c.execute(q, {"n": args.per_group}):
            words = len(tb.split())
            for key in ((period, cls), ("ALL", cls), (period, "ALL"), ("ALL", "ALL")):
                a = acc[key]
                a["segments"] += 1
                a["words"] += words
                for name, enc in encs.items():
                    a[f"{name}/b"] += len(enc.encode(tb, disallowed_special=()))
                    a[f"{name}/a"] += len(enc.encode(ta, disallowed_special=()))
    lines = [
        "# Tokens per Hebrew/Aramaic word",
        "",
        f"Sample: up to {args.per_group:,} primary-version segments per (period, class). "
        "Words = whitespace tokens of the search-normalized text.",
        "",
    ]
    head = (
        ["period", "class", "segments", "words"]
        + [f"{n} tok/word (b)" for n in encs]
        + [f"{n} tok/word (a)" for n in encs]
    )
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (period, cls), a in sorted(acc.items(), key=lambda kv: (kv[0][0] != "ALL", kv[0])):
        w = max(a["words"], 1)
        row = [period, cls, f"{a['segments']:,}", f"{a['words']:,}"]
        row += [f"{a[f'{n}/b'] / w:.2f}" for n in encs] + [f"{a[f'{n}/a'] / w:.2f}" for n in encs]
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines) + "\n"
    open(args.out, "w", encoding="utf-8").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
