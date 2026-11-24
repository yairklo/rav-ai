"""CLI: ``rav-evals QUESTIONS.yaml --answer-fn pkg.mod:func --sources sources.json``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .golden import load_golden
from .runner import run_eval
from .verifier import InMemorySourceStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rav-evals")
    p.add_argument("questions", help="golden set YAML")
    p.add_argument("--answer-fn", required=True, help="module:callable(GoldenQuestion)->RunOutput")
    p.add_argument("--sources", help='JSON file {"ref": "text", ...} (fake store, for tests)')
    p.add_argument(
        "--database-url",
        help="verify against the corpus in Postgres and log every result to the runs table",
    )
    p.add_argument("--out-dir", default="eval-out")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path.cwd()))
    mod_name, _, fn_name = args.answer_fn.partition(":")
    answer_fn = getattr(importlib.import_module(mod_name), fn_name)
    if bool(args.sources) == bool(args.database_url):
        p.error("give exactly one of --sources and --database-url")
    engine = None
    if args.database_url:
        from sqlalchemy import create_engine

        from .pg_store import PostgresSourceStore

        engine = create_engine(args.database_url)
        store = PostgresSourceStore(engine)
    else:
        store = InMemorySourceStore(json.loads(Path(args.sources).read_text(encoding="utf-8")))

    golden = load_golden(args.questions)
    report = run_eval(golden, answer_fn, store)
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
