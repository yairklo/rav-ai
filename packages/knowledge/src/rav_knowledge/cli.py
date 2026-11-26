"""rav-kl: estimate | run | status | promote | invalidate | report | sources | conflicts."""

from __future__ import annotations

import argparse
import json
import sys

from rav_ai.db.session import make_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import IMPLEMENTED_STAGES, STAGES, load_config
from .estimate import estimate, format_estimate
from .runner import PromoteRefused, invalidate, promote, report, run_stage, status
from .sources import seed_lens_records
from .stages import DAG


def _stages(arg: str) -> list[str]:
    out = [x.strip() for x in arg.split(",") if x.strip()]
    bad = [x for x in out if x not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s): {bad}; known: {STAGES}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rav-kl", description=__doc__)
    ap.add_argument("--db", help="database URL (default RAV_DATABASE_URL / local docker)")
    ap.add_argument("--config", help="path to knowledge.toml")
    ap.add_argument("--mode", choices=["personal", "public"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("estimate", help="token / cost estimate (no LLM calls)")
    p.add_argument("scope")
    p.add_argument("--stages", default="s0b,s1,s2")
    p = sub.add_parser("run", help="run a stage over a scope")
    p.add_argument("stage")
    p.add_argument("scope")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--budget-usd", type=float, help="api backend only")
    p.add_argument(
        "--limit", type=int, metavar="N", help="at most N new tasks in this run, then stop"
    )
    sub.add_parser("status")
    p = sub.add_parser(
        "promote", help="make a job's output current (rollback = promote an old job)"
    )
    p.add_argument("--job", type=int, required=True)
    p.add_argument("--force", metavar="NOTE", help="override the gate; the note is recorded")
    p = sub.add_parser("invalidate", help="re-anchor evidence after text changes; stale if lost")
    p.add_argument("--segments", help="comma-separated segment ids (default: all)")
    p = sub.add_parser("report", help="grounding metrics -> reports/kl/")
    p.add_argument("--job", type=int, required=True)
    sub.add_parser("seed-lenses", help="register Asif / Rav Asher Weiss / VBM / HebrewBooks rows")
    p = sub.add_parser("register-work", help="add / update a kl_sources row (fail-closed registry)")
    p.add_argument("title")
    p.add_argument("--tier", required=True)
    p.add_argument(
        "--role", required=True, choices=["primary", "lens", "witness", "reference_only"]
    )
    p.add_argument(
        "--license-tier",
        required=True,
        choices=["PUBLIC_COMMERCIAL", "NON_COMMERCIAL_ONLY", "PERSONAL_ONLY_RESTRICTED"],
    )
    p.add_argument("--authority-areas", help="comma-separated, from the closed list")
    p.add_argument("--weight-tier", choices=["core", "major", "extended", "long_tail"])
    p = sub.add_parser(
        "seed-sources", help="register works from config/kl_sources_seed.toml (license re-checked)"
    )
    p.add_argument("--file", help="seed file (default config/kl_sources_seed.toml)")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("conflicts", help="human decisions that a later output disagreed with")
    a = ap.parse_args(argv)

    from pathlib import Path

    overrides = {"mode": a.mode} if a.mode else {}
    cfg = load_config(Path(a.config) if a.config else None, **overrides)
    engine = make_engine(a.db)
    if a.cmd == "estimate":
        with Session(engine) as s:
            print(format_estimate(estimate(s, a.scope, _stages(a.stages), cfg)))
    elif a.cmd == "run":
        from .backends import StopRun, make_llm

        if a.limit is not None and a.limit < 1:
            raise SystemExit("--limit must be >= 1")
        llm = make_llm(cfg)
        if cfg.backend != "api":
            if a.budget_usd is not None:
                print(f"note: --budget-usd is ignored with backend={cfg.backend}", file=sys.stderr)
            if a.stage != "s0" and not a.dry_run:
                try:
                    llm.ensure_logged_in()
                except StopRun as e:
                    print(f"error: {e}", file=sys.stderr)
                    return 2
        r = run_stage(
            engine,
            a.stage,
            a.scope,
            cfg,
            llm,
            budget_usd=a.budget_usd,
            dry_run=a.dry_run,
            limit=a.limit,
        )
        print(json.dumps(r.__dict__, ensure_ascii=False, indent=2, default=str))
        if r.stopped:
            print(f"stopped: {r.stopped}", file=sys.stderr)
            print(f"resume with: {r.resume}", file=sys.stderr)
        if r.failed:
            return 1
        return 3 if r.stopped else 0
    elif a.cmd == "status":
        for row in status(engine):
            print(json.dumps(row, ensure_ascii=False))
    elif a.cmd == "promote":
        try:
            print(
                json.dumps(
                    promote(engine, a.job, force_note=a.force, cfg=cfg), indent=2, default=str
                )
            )
        except PromoteRefused as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
    elif a.cmd == "invalidate":
        ids = [int(x) for x in a.segments.split(",")] if a.segments else None
        print(json.dumps(invalidate(engine, ids)))
    elif a.cmd == "report":
        print(json.dumps(report(engine, a.job, cfg=cfg), ensure_ascii=False, indent=2))
    elif a.cmd == "seed-lenses":
        with Session(engine) as s:
            print(f"added {seed_lens_records(s)} kl_sources rows")
            s.commit()
    elif a.cmd == "register-work":
        from .sources import register_work

        with Session(engine) as s:
            src = register_work(
                s,
                a.title,
                tier=a.tier,
                role=a.role,
                license_tier=a.license_tier,
                weight_tier=a.weight_tier,
                authority_areas=a.authority_areas.split(",") if a.authority_areas else None,
                added_by="cli",
            )
            s.commit()
            print(f"registered work_id={src.work_id} tier={src.tier} role={src.role}")
    elif a.cmd == "seed-sources":
        from .sources import seed_sources

        with Session(engine) as s:
            res = seed_sources(s, Path(a.file) if a.file else None, dry_run=a.dry_run)
            if not a.dry_run:
                s.commit()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 1 if res["refused"] else 0
    elif a.cmd == "conflicts":
        with engine.connect() as c:
            for row in c.execute(text("SELECT * FROM kl_conflicts ORDER BY override_id")):
                print(dict(row._mapping))
    return 0


_ = (IMPLEMENTED_STAGES, DAG)

if __name__ == "__main__":
    raise SystemExit(main())
