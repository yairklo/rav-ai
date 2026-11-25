"""CLI.

Legacy: ``rav-evals QUESTIONS.yaml --answer-fn pkg.mod:func --sources sources.json``.

KL-3 subcommands:
  run            live run over the golden set (manual; needs an API key and a database)
  regress        recorded fixtures -> verifier + deterministic metrics vs the baseline (CI)
  compare        compare 2-3 run bundles
  golden-status  questions per category / split, and every TODO still pending
  pairs-sample   deterministic balanced sample of pairs for the gold
  pairs-kappa    Cohen's kappa (two raters) / self agreement (one rater)
  attribution    misattribution rate, Wilson CI and the gate, from answer ratings
  review         the rating tool (127.0.0.1 only)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

from .config import (
    BASELINE_PATH,
    FIXTURES_DIR,
    GOLDEN_DIR,
    KL_GOLDEN_DIR,
    RATINGS_DIR,
    REPORTS_DIR,
    load_config,
)
from .golden import load_golden, load_goldens
from .runner import run_eval
from .verifier import InMemorySourceStore

SUBCOMMANDS = (
    "run",
    "regress",
    "compare",
    "golden-status",
    "pairs-sample",
    "pairs-kappa",
    "attribution",
    "review",
)


def default_golden_paths() -> list[Path]:
    return sorted(GOLDEN_DIR.glob("*.yaml")) + sorted(KL_GOLDEN_DIR.glob("*.yaml"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        return legacy_main(argv)
    cmd, rest = argv[0], argv[1:]
    return {
        "run": cmd_run,
        "regress": cmd_regress,
        "compare": cmd_compare,
        "golden-status": cmd_golden_status,
        "pairs-sample": cmd_pairs_sample,
        "pairs-kappa": cmd_pairs_kappa,
        "attribution": cmd_attribution,
        "review": cmd_review,
    }[cmd](rest)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _db_url(given: str | None) -> str:
    if given:
        return given
    from rav_ai.db.session import db_url

    return db_url()


# ---- run (live) -----------------------------------------------------------------------------


def cmd_run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rav-evals run")
    p.add_argument("--knowledge", choices=["off", "base", "base+lens"], default="off")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--golden", nargs="*", help="golden YAML files (default: all)")
    p.add_argument("--database-url", help="default: the rav_ai database URL")
    p.add_argument("--model")
    p.add_argument("--out-dir", help="default: reports/evals/run-<knowledge>-<split>")
    p.add_argument("--record-fixtures", help="also save every result as a regression fixture")
    a = p.parse_args(argv)
    from .live import run_live

    golden = load_goldens(a.golden or default_golden_paths(), a.split)
    slug = f"run-{a.knowledge.replace('+', '_')}-{a.split}"
    out = Path(a.out_dir or REPORTS_DIR / "evals" / slug)
    bundle = run_live(
        golden,
        _db_url(a.database_url),
        a.knowledge,
        out,
        model=a.model,
        record_fixtures=Path(a.record_fixtures) if a.record_fixtures else None,
    )
    m = bundle["metrics"]
    keys = (
        "questions",
        "by_status",
        "verifier_pass_rate",
        "V5",
        "V6",
        "V7",
        "accepted_false_error",
        "cost_usd",
    )
    print(_dump({k: m[k] for k in keys}))
    print(f"bundle: {out / 'bundle.json'}")
    return 0


# ---- regress --------------------------------------------------------------------------------


def cmd_regress(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rav-evals regress")
    p.add_argument("--fixtures", default=str(FIXTURES_DIR))
    p.add_argument("--baseline", default=str(BASELINE_PATH))
    p.add_argument("--update", action="store_true", help="write the current metrics as baseline")
    p.add_argument("--out", help="write the metrics as a bundle (for compare)")
    a = p.parse_args(argv)
    from .kl_metrics import baseline_from, regress

    res = regress(Path(a.fixtures), Path(a.baseline))
    if a.update:
        Path(a.baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(a.baseline).write_text(_dump(baseline_from(res["metrics"])) + "\n", encoding="utf-8")
        print(f"baseline written: {a.baseline}")
        res = regress(Path(a.fixtures), Path(a.baseline))
    if a.out:
        bundle = {"label": "regress", "knowledge": "recorded", "metrics": res["metrics"]}
        Path(a.out).write_text(_dump(bundle), encoding="utf-8")
    print(f"{res['fixtures']} recorded runs, {res['rejected_fixtures']} rejected claims")
    for f in res["failures"]:
        print(f"FAIL {f}")
    print("PASS" if res["passed"] else "FAILED")
    return 0 if res["passed"] else 1


# ---- compare --------------------------------------------------------------------------------


def cmd_compare(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rav-evals compare")
    p.add_argument("bundles", nargs="+", help="2-3 run dirs or bundle.json files")
    p.add_argument("--out-dir", default=str(REPORTS_DIR / "evals"))
    a = p.parse_args(argv)
    if not 2 <= len(a.bundles) <= 3:
        p.error("give two or three bundles")
    from .compare import compare, load_bundle, to_markdown, write_compare

    c = compare([load_bundle(Path(b)) for b in a.bundles])
    jp, mp = write_compare(c, Path(a.out_dir))
    print(to_markdown(c))
    print(f"reports: {jp}, {mp}")
    return 0


# ---- golden ---------------------------------------------------------------------------------


def golden_status(paths: list[Path]) -> dict:
    allq = load_goldens(paths, split=None)
    table: dict[str, dict[str, int]] = {}
    for (cat, split), n in sorted(
        Counter((q.category.value, q.split) for q in allq.questions).items()
    ):
        table.setdefault(cat, {"dev": 0, "heldout": 0})[split] = n
    pending = allq.pending_report()
    return {
        "questions": len(allq.questions),
        "by_category": table,
        "pending": len(pending),
        "pending_fields": pending,
    }


def cmd_golden_status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rav-evals golden-status")
    p.add_argument("--golden", nargs="*")
    a = p.parse_args(argv)
    print(_dump(golden_status([Path(x) for x in a.golden] if a.golden else default_golden_paths())))
    return 0


# ---- pairs ----------------------------------------------------------------------------------


def cmd_pairs_sample(argv: list[str]) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(prog="rav-evals pairs-sample")
    p.add_argument("--n", type=int, default=cfg.pairs_n)
    p.add_argument("--seed", type=int, default=cfg.pairs_seed)
    p.add_argument("--scope", default=cfg.pairs_scope)
    p.add_argument("--database-url")
    p.add_argument("--out", default=str(KL_GOLDEN_DIR / "pairs.draft.jsonl"))
    p.add_argument("--force", action="store_true", help="overwrite a file that has ratings")
    a = p.parse_args(argv)
    from sqlalchemy import create_engine

    from .pairs import PostgresPairSource, read_jsonl, sample_pairs, write_jsonl

    out = Path(a.out)
    if not a.force and any(r.get("ratings") for r in read_jsonl(out)):
        print(f"{out} already has ratings; refusing to overwrite (use --force)")
        return 1
    src = PostgresPairSource(
        create_engine(_db_url(a.database_url)),
        a.scope,
        cfg.explicit_edge_types,
        pool_limit=cfg.neighbor_pool_limit,
        min_shared_terms=cfg.min_shared_terms,
    )
    res = sample_pairs(
        src, a.n, a.seed, duplicate_fraction=cfg.duplicate_fraction, curated=cfg.curated_pairs
    )
    write_jsonl(res.records, out)
    print(_dump({**res.summary(), "out": str(out), "seed": a.seed, "scope": a.scope}))
    return 0


def cmd_pairs_kappa(argv: list[str]) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(prog="rav-evals pairs-kappa")
    p.add_argument("--pairs", default=str(KL_GOLDEN_DIR / "pairs.draft.jsonl"))
    p.add_argument("--ratings-dir", default=str(RATINGS_DIR))
    p.add_argument("--raters", nargs=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPORTS_DIR / "kl"))
    a = p.parse_args(argv)
    from .pairs import kappa_markdown, pairs_kappa, read_jsonl
    from .ratings import RatingStore

    rep = pairs_kappa(
        read_jsonl(Path(a.pairs)),
        RatingStore(Path(a.ratings_dir)).events("pairs"),
        raters=tuple(a.raters) if a.raters else None,
        seed=a.seed,
        self_min=cfg.self_agreement_min,
        recheck_days=cfg.self_recheck_days,
    )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs-kappa.json").write_text(_dump(rep), encoding="utf-8")
    (out / "pairs-kappa.md").write_text(kappa_markdown(rep), encoding="utf-8")
    print(kappa_markdown(rep))
    return 0


# ---- attribution ------------------------------------------------------------------------------


def cmd_attribution(argv: list[str]) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(prog="rav-evals attribution")
    p.add_argument("--ratings-dir", default=str(RATINGS_DIR))
    p.add_argument("--fixtures", default=str(FIXTURES_DIR))
    p.add_argument("--sample", type=int, help="print a stratified sample of N units to rate")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    from .attribution import attribution_units, misattribution_report, stratified_sample
    from .kl_metrics import load_fixtures
    from .ratings import RatingStore

    if a.sample:
        runs, _ = load_fixtures(Path(a.fixtures))
        units = [
            u
            for rr in runs
            for u in attribution_units(rr.fixture_id, rr.question.category.value, rr.output.answer)
        ]
        print(_dump([u.__dict__ for u in stratified_sample(units, a.sample, a.seed)]))
        return 0
    store = RatingStore(Path(a.ratings_dir))
    ev = [e for e in store.events("answers") if e.get("kind") == "claim"]
    rep = misattribution_report(ev, cfg.misattribution_max_rate, cfg.misattribution_ci_level)
    print(_dump(rep))
    return 0 if rep["verdict"] != "FAIL" else 1


# ---- review ---------------------------------------------------------------------------------


def cmd_review(argv: list[str]) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(prog="rav-evals review")
    p.add_argument("--rater", default="yair", help="rater_id (file name of the ratings)")
    p.add_argument("--port", type=int, default=cfg.review_port)
    p.add_argument("--database-url", help="default: the rav_ai database; 'none' = no database")
    p.add_argument("--pairs", default=str(KL_GOLDEN_DIR / "pairs.draft.jsonl"))
    p.add_argument("--fixtures", default=str(FIXTURES_DIR))
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    import webbrowser

    from .ratings import RatingStore, check_rater
    from .review.app import ReviewContext, serve

    engine = None
    if a.database_url != "none":
        from sqlalchemy import create_engine

        try:
            engine = create_engine(_db_url(a.database_url))
            with engine.connect():
                pass
        except Exception as e:  # noqa: BLE001 - the answers and pairs queues work without it
            print(f"no database ({e}); the KL items queue is empty")
            engine = None
    ctx = ReviewContext(
        rater_id=check_rater(a.rater),
        store=RatingStore(),
        cfg=cfg,
        fixtures_dir=Path(a.fixtures),
        pairs_path=Path(a.pairs),
        engine=engine,
    )
    serve(ctx, a.port, opener=None if a.no_browser else webbrowser.open)
    return 0


# ---- legacy ---------------------------------------------------------------------------------


def legacy_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rav-evals", epilog="subcommands: " + ", ".join(SUBCOMMANDS))
    p.add_argument("questions", help="golden set YAML")
    p.add_argument("--answer-fn", required=True, help="module:callable(GoldenQuestion)->RunOutput")
    p.add_argument("--sources", help='JSON file {"ref": "text", ...} (fake store, for tests)')
    p.add_argument(
        "--database-url",
        help="verify against the corpus in Postgres and log every result to the runs table",
    )
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--out-dir", default="eval-out")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path.cwd()))
    mod_name, _, fn_name = args.answer_fn.partition(":")
    answer_fn = getattr(importlib.import_module(mod_name), fn_name)
    if bool(args.sources) == bool(args.database_url):
        p.error("give exactly one of --sources and --database-url")
    engine = None
    knowledge = None
    if args.database_url:
        from sqlalchemy import create_engine

        from .kl_lookup import PostgresKnowledgeLookup
        from .pg_store import PostgresSourceStore

        engine = create_engine(args.database_url)
        store = PostgresSourceStore(engine)
        knowledge = PostgresKnowledgeLookup(engine)
    else:
        store = InMemorySourceStore(json.loads(Path(args.sources).read_text(encoding="utf-8")))

    golden = load_golden(args.questions, args.split)
    report = run_eval(golden, answer_fn, store, knowledge=knowledge)
    if engine is not None:
        from sqlalchemy.orm import Session

        from .models import record_report

        with Session(engine) as session:
            record_report(session, golden, report)
    jp, mp = report.write(args.out_dir)
    s = report.summary()
    print(
        f"{s['passed']}/{s['questions']} passed, {s['pending']} pending judge, "
        f"{s['failed']} failed, {s['errors']} errors; reports: {jp}, {mp}"
    )
    return 1 if s["failed"] or s["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
