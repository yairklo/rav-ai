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
    p.add_argument("--sources", required=True, help='JSON file {"ref": "text", ...} (fake store)')
    p.add_argument("--out-dir", default="eval-out")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path.cwd()))
    mod_name, _, fn_name = args.answer_fn.partition(":")
    answer_fn = getattr(importlib.import_module(mod_name), fn_name)
    store = InMemorySourceStore(json.loads(Path(args.sources).read_text(encoding="utf-8")))

    report = run_eval(load_golden(args.questions), answer_fn, store)
    jp, mp = report.write(args.out_dir)
    s = report.summary()
    print(f"{s['passed']}/{s['questions']} passed; reports: {jp}, {mp}")
    return 0 if s["passed"] == s["questions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
