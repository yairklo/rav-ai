"""Cost estimate before any run: token counts (provider count_tokens if a key exists, else
tiktoken ~ approximation) from ONE version of the text, cost per page and in total."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rav_ai.db.models import KlItem
from sqlalchemy import select
from sqlalchemy.orm import Session
from textnorm import find_acronyms, rank_candidates

from .config import IMPLEMENTED_STAGES, KLConfig
from .domain import domain_for
from .llm_io import count_tokens, format_line
from .prompts import load_prompt
from .stages import S2, STAGES, RunContext
from .store import load_input, load_scope


@dataclass
class Estimate:
    scope: str
    method: str = ""
    pages: list[dict] = field(default_factory=list)
    stages: dict[str, dict] = field(default_factory=dict)
    total_cost_usd: float | None = 0.0
    notes: list[str] = field(default_factory=list)


def estimate(s: Session, scope_key: str, stages: list[str], cfg: KLConfig) -> Estimate:
    scope = load_scope(s, scope_key)
    est = Estimate(scope_key)
    methods: set[str] = set()
    page_tokens: dict[str, int] = {}
    for p in scope.pages:
        inp = load_input(s, p.segment_ids, cfg, with_aids=True)
        text = "\n".join(format_line(x, cfg.anchor_format) for x in inp.segs)
        n, m = count_tokens(text, cfg.model_for("s1"))
        methods.add(m)
        aid_n = (
            count_tokens("\n".join(format_line(a) for a in inp.aids), None)[0] if inp.aids else 0
        )
        page_tokens[p.label] = n
        est.pages.append(
            {
                "page": p.label,
                "segments": len(inp.segs),
                "tokens": n,
                "aid_tokens": aid_n,
                "excluded": dict(inp.excluded),
            }
        )
    est.method = "/".join(sorted(methods))
    work = None
    total = 0.0
    for st in stages:
        if st == "s0":
            est.stages[st] = {
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "note": "deterministic",
            }
            continue
        if st not in IMPLEMENTED_STAGES:
            est.notes.append(f"{st}: reserved, not implemented yet")
            continue
        model = cfg.model_for(st)
        prompt = load_prompt(st)
        overhead = count_tokens(prompt.text, model)[0]
        if st == "s1":
            rc = RunContext(cfg, None, scope, st, model, prompt)
            specs = STAGES["s1"].plan(rc, s)
            tin = sum(x.est_in for x in specs)
            tout = sum(x.est_out for x in specs)
        elif st == "s2":
            sugyot = list(
                s.scalars(
                    select(KlItem).where(
                        KlItem.kind == "sugya",
                        KlItem.is_current,
                        KlItem.start_segment_id.in_(scope.segment_ids),
                    )
                )
            )
            if sugyot:
                rc = RunContext(cfg, None, scope, st, model, prompt)
                specs = S2().plan(rc, s)
                tin, tout = sum(x.est_in for x in specs), sum(x.est_out for x in specs)
            else:
                body = sum(page_tokens.values()) + sum(p["aid_tokens"] for p in est.pages)
                n_sg = max(1, math.ceil(len(scope.pages) / 1.5))
                tin = body + overhead * n_sg
                tout = int(tin * cfg.out_ratio.get("s2", 0.35))
                est.notes.append("s2: no sugyot yet; assumed ~1.5 pages per sugya")
        else:  # s0b: ambiguous acronyms found in memory
            work = work or None
            occ = 0
            for p in scope.pages:
                inp = load_input(s, p.segment_ids, cfg)
                for x in inp.segs:
                    wk = x.info.work_title
                    dom = domain_for(scope.category, None)
                    for c in find_acronyms(x.text.text_raw):
                        if len(rank_candidates(c.norm, dom)) >= 2:
                            occ += 1
                _ = wk
            n_pages = len(scope.pages)
            tin = overhead * n_pages + occ * 90
            tout = int(occ * 8 + n_pages * 20)
            est.notes.append(f"s0b: {occ} ambiguous acronym occurrences")
        cost = cfg.cost(model, tin, tout)
        est.stages[st] = {"model": model, "tokens_in": tin, "tokens_out": tout, "cost_usd": cost}
        if cost is None:
            est.total_cost_usd = None
        elif est.total_cost_usd is not None:
            total += cost
    if est.total_cost_usd is not None:
        est.total_cost_usd = total
        base = sum(page_tokens.values()) or 1
        for p in est.pages:
            share = page_tokens[p["page"]] / base
            p["cost_usd"] = round(total * share, 4)
    return est


def format_estimate(e: Estimate) -> str:
    lines = [f"scope: {e.scope}   token count: {e.method}"]
    lines.append(f"{'page':8} {'segs':>5} {'tokens':>8} {'aid tok':>8} {'cost $':>8}")
    for p in e.pages:
        lines.append(
            f"{p['page']:8} {p['segments']:>5} {p['tokens']:>8} {p['aid_tokens']:>8} "
            f"{p.get('cost_usd', float('nan')):>8.4f}"
        )
    lines.append("")
    for st, d in e.stages.items():
        c = d["cost_usd"]
        lines.append(
            f"{st:4} {d.get('model', '-'):28} in={d['tokens_in']:>9,} out={d['tokens_out']:>8,} "
            f"cost=${c:.4f}"
            if c is not None
            else f"{st:4} cost=unknown (no price)"
        )
    tot = e.total_cost_usd
    lines.append(f"TOTAL: ${tot:.4f}" if tot is not None else "TOTAL: unknown")
    lines += [f"note: {n}" for n in e.notes]
    return "\n".join(lines)
