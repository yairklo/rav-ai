"""Ingest the Sefaria export (Hebrew + Aramaic) into PostgreSQL as a dated snapshot.

rav-ingest sefaria                 # all downloaded Hebrew versions + links (resumable)
rav-ingest sefaria --book Berakhot --skip-links
rav-ingest links                   # (re)build edges for the latest snapshot
rav-ingest recompute-flags         # re-apply config/license_policy.toml to stored versions
rav-ingest diff [--snapshot N]     # what changed in a snapshot vs. before
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import func, select

from rav_ai.config import DATA_DIR, RAW_DIR, LicensePolicy
from rav_ai.db.models import Snapshot, SnapshotChange
from rav_ai.db.session import db_url, make_engine, migrate
from rav_ai.ingest.links import ingest_links
from rav_ai.ingest.manifest import write_manifest
from rav_ai.ingest.sefaria import (
    apply_primary_rules,
    finish_snapshot,
    ingest_all,
    recompute_flags,
    start_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--db", default=None, help="SQLAlchemy URL (default: docker-compose Postgres)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sefaria")
    s.add_argument("--book", action="append", default=[])
    s.add_argument("--skip-links", action="store_true")
    s.add_argument(
        "--renormalize",
        action="store_true",
        help="recompute forms A/B/offsets of existing segments in place",
    )
    sub.add_parser("links")
    sub.add_parser("recompute-flags")
    rp = sub.add_parser("report", help="write reports/ingest_report.md and reports/books.csv")
    rp.add_argument("--out-dir", default=None)
    rp.add_argument("--encoding", default="o200k_base")
    d = sub.add_parser("diff")
    d.add_argument("--snapshot", type=int)
    args = ap.parse_args(argv)

    url = args.db or db_url()
    migrate(url)
    engine = make_engine(url)
    raw = Path(args.raw_dir)
    policy = LicensePolicy.load()

    if args.cmd == "sefaria":
        snap = start_snapshot(engine, raw)
        print(f"snapshot {snap.id} (export {snap.export_label})")
        write_manifest(raw, DATA_DIR / "snapshots")
        stats = ingest_all(
            engine, raw, policy, snap.id, set(args.book) or None, renormalize=args.renormalize
        )
        if not args.skip_links and not args.book:
            stats["links"] = ingest_links(engine, raw, snap.id)
        finish_snapshot(engine, snap.id, stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if stats.get("errors"):
            return 1
    elif args.cmd == "links":
        with engine.connect() as c:
            sid = c.execute(select(func.max(Snapshot.id))).scalar_one()
        stats = ingest_links(engine, raw, sid)
        finish_snapshot(engine, sid, {"links": stats})
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.cmd == "report":
        from rav_ai.config import ROOT
        from rav_ai.ingest.report import build

        with engine.connect() as c:
            sid = c.execute(
                select(func.max(Snapshot.id)).where(Snapshot.status == "complete")
            ).scalar()
            st = c.execute(select(Snapshot.stats).where(Snapshot.id == sid)).scalar() if sid else {}
        out = build(engine, policy, st, Path(args.out_dir or ROOT / "reports"), args.encoding)
        print(json.dumps(out, indent=1))
    elif args.cmd == "recompute-flags":
        print(f"updated {recompute_flags(engine, policy)} versions")
        print(f"primary rules touched {apply_primary_rules(engine)} versions")
    elif args.cmd == "diff":
        with engine.connect() as c:
            sid = args.snapshot or c.execute(select(func.max(Snapshot.id))).scalar_one()
            rows = c.execute(
                select(SnapshotChange.change, func.count())
                .where(SnapshotChange.snapshot_id == sid)
                .group_by(SnapshotChange.change)
            ).all()
            snap = c.execute(select(Snapshot.stats).where(Snapshot.id == sid)).scalar()
        print(f"snapshot {sid}: {dict(rows)}")
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
