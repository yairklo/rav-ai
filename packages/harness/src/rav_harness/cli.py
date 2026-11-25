"""CLI: ``rav-harness ask "question"`` and ``rav-harness eval golden.yaml``."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from rav_ai.db.session import make_engine, make_session
from rav_evals.golden import load_golden

from .config import KNOWLEDGE_MODES, load_config
from .integration import build_harness, run_golden
from .llm import LiteLLMClient


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rav-harness")
    p.add_argument("--model", help="LiteLLM model id (default: config/harness.toml)")
    p.add_argument("--no-record", action="store_true", help="do not write to the runs table")
    p.add_argument(
        "--knowledge",
        choices=KNOWLEDGE_MODES,
        help="knowledge layer: off | base | base+lens (default: config/harness.toml)",
    )
    p.add_argument("--mode", choices=("personal", "public"), help="deployment mode (default: env)")
    p.add_argument(
        "--allow-personal-only",
        action="store_true",
        help="let PERSONAL_ONLY_RESTRICTED knowledge sources in (personal mode only)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    e = sub.add_parser("eval")
    e.add_argument("golden")
    e.add_argument("--out-dir", default="eval-out")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.model:
        cfg = cfg.with_model(args.model)
    if args.knowledge:
        cfg = cfg.with_knowledge(args.knowledge)
    if args.allow_personal_only:
        cfg = replace(cfg, allow_personal_only=True)
    session = make_session(make_engine())
    llm = LiteLLMClient()
    if args.cmd == "ask":
        res = build_harness(session, llm, cfg, record=not args.no_record, mode=args.mode).run(
            args.question
        )
        out = res.answer.model_dump(mode="json") if res.answer else {"error": res.error}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print(
            f"steps={res.steps} tokens={res.input_tokens}+{res.output_tokens} "
            f"cost={res.cost_usd} run={res.run_id}"
        )
        return 0 if res.answer else 1
    report = run_golden(
        session, load_golden(args.golden), llm, cfg, record=not args.no_record, mode=args.mode
    )
    jp, mp = report.write(args.out_dir)
    s = report.summary()
    print(f"{s['passed']}/{s['questions']} passed; reports: {jp}, {mp}")
    return 0 if s["passed"] == s["questions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
