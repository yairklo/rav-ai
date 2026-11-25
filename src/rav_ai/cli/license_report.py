"""Write reports/licenses.csv (one row per VERSION) and reports/license_summary.md.

Reads the database, so run `rav-ingest sefaria` first. can_index / can_display_public /
can_train come from config/license_policy.toml (stored on each version). The CSV ends with
'#'-prefixed summary rows (by flag, license, period, category).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rav_ai.config import ROOT
from rav_ai.db.session import db_url, make_engine
from rav_ai.license_report import fetch_rows, markdown, summarize, write_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "reports" / "licenses.csv"))
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    rows = fetch_rows(make_engine(args.db or db_url()))
    summary = summarize(rows)
    out = Path(args.out)
    write_csv(rows, summary, out)
    md = markdown(summary)
    out.with_name("license_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out} ({len(rows)} version rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
