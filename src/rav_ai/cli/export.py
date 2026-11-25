"""Export segment texts as JSONL, enforcing license flags (the export gate).

  rav-export --purpose train --out out.jsonl [--form raw|a|b]
Only versions whose flag for the purpose is true are written.
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import text

from rav_ai.db.session import db_url, make_engine

FLAG = {"train": "can_train", "index": "can_index", "display": "can_display_public"}
COL = {"raw": "text_raw", "a": "text_a", "b": "text_b"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--purpose", required=True, choices=FLAG)
    ap.add_argument("--form", default="raw", choices=COL)
    ap.add_argument("--out", default="-")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    flag, col = FLAG[args.purpose], COL[args.form]
    q = text(f"""SELECT s.ref, w.title, v.language, v.version_title, v.license, t.{col}
                 FROM segment_text t JOIN versions v ON v.id=t.version_id
                 JOIN segments s ON s.id=t.segment_id JOIN works w ON w.id=s.work_id
                 WHERE v.{flag} ORDER BY w.id, s.node_index, s.sort_key""")
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")  # noqa: SIM115
    n = 0
    with make_engine(args.db or db_url()).connect().execution_options(stream_results=True) as c:
        for ref, work, lang, vt, lic, body in c.execute(q):
            out.write(
                json.dumps(
                    {
                        "ref": ref,
                        "work": work,
                        "language": lang,
                        "version": vt,
                        "license": lic,
                        "text": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    if out is not sys.stdout:
        out.close()
    print(f"exported {n} rows for purpose={args.purpose}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
