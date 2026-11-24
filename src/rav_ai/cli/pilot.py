"""Build and report the Hilkhot Chanukah pilot slice -> reports/pilot_hanukkah.md."""

from __future__ import annotations

import argparse
import sys

from rav_ai.config import ROOT
from rav_ai.db.session import db_url, make_engine, make_session
from rav_ai.pilot import build, render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "pilot_hanukkah.md"))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = make_session(make_engine(args.db or db_url()))
    md = render(build(s))
    open(args.out, "w", encoding="utf-8").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
