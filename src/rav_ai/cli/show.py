"""Show a canonical ref: its text, license flags, and every layer above/below it.

rav-show "Shabbat 21b:5"
RAV_DEPLOYMENT_MODE=public rav-show "Shabbat 21b:5"     # enforces can_display_public
"""

from __future__ import annotations

import argparse
import sys

from rav_ai.db.session import db_url, make_engine, make_session
from rav_ai.lookup import PERIOD_RANK, deployment_mode, get_texts, neighbors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("--db", default=None)
    ap.add_argument("--all-versions", action="store_true")
    ap.add_argument("--max-links", type=int, default=5, help="max refs shown per type/period")
    ap.add_argument("--chars", type=int, default=160)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    session = make_session(make_engine(args.db or db_url()))
    texts = get_texts(session, args.ref)
    if not texts:
        print(f"no segment with ref {args.ref!r}", file=sys.stderr)
        return 1
    print(f"== {args.ref}   (deployment_mode={deployment_mode()})")
    for t in texts if args.all_versions else texts[:1]:
        print(f"[{t.language}] {t.version_title}\n{t.shown()}\n")
    print("-- License (per version)")
    for t in texts:
        print(
            f"  {t.version_title}: {t.license or '(none)'} | index={t.can_index} "
            f"public={t.can_display_public} train={t.can_train}"
        )

    res = neighbors(session, args.ref)
    out, my_period = res
    print(f"\n-- Layers around this ref (this ref: {texts[0].work}, period={my_period})")
    for side, title in (
        ("above", "ABOVE (later layers citing/commenting)"),
        ("same", "SAME LAYER"),
        ("below", "BELOW (earlier sources)"),
    ):
        groups = out[side]
        total = sum(len(v) for g in groups.values() for v in g.values())
        print(f"\n### {title}: {total}")
        for etype, byp in sorted(groups.items(), key=lambda kv: -sum(map(len, kv[1].values()))):
            n = sum(len(v) for v in byp.values())
            print(f"  [{etype}] {n}")
            for period, items in sorted(byp.items(), key=lambda kv: PERIOD_RANK.get(kv[0], 5)):
                print(f"    <{period}> {len(items)}")
                for it in items[: args.max_links]:
                    print(f"      {it.ref}  {it.snippet[: args.chars].replace(chr(10), ' ')}")
                if len(items) > args.max_links:
                    print(f"      ... {len(items) - args.max_links} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
