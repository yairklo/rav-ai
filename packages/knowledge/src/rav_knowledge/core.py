"""Generate -> validate (K1) -> locate evidence -> K-checks -> one retry -> accept/reject."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from rav_ai.db.models import Annotation
from rav_harness.llm import LLM
from sqlalchemy import select
from sqlalchemy.orm import Session

from .checks import CheckContext, CheckReport, EvRec, confidence, run_checks
from .config import KLConfig
from .evidence import locate
from .llm_io import call_tool
from .schemas import TOOL_MODELS, iter_evidence
from .store import InputSeg, LoadedInput


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, i: int, o: int, c: float | None) -> None:
        self.input_tokens += i
        self.output_tokens += o
        self.cost_usd += c or 0.0
        self.calls += 1


@dataclass
class GenResult:
    ok: bool
    model: BaseModel | None
    body: dict[str, Any] | None
    evidence: list[EvRec] = field(default_factory=list)
    report: CheckReport = field(default_factory=CheckReport)
    confidence: float | None = None
    attempts: int = 0
    error: str | None = None
    raw: dict[str, Any] | None = None


def markers_for(s: Session, inp: LoadedInput) -> dict[str, set[str]]:
    """S0 discourse/error markers per segment label (for K14)."""
    by_id = {x.info.segment_id: x.info.label for x in inp.segs}
    out: dict[str, set[str]] = {}
    rows = s.scalars(
        select(Annotation).where(
            Annotation.segment_id.in_(list(by_id)),
            Annotation.kind == "discourse_marker",
            Annotation.is_current,
        )
    )
    for a in rows:
        out.setdefault(by_id[a.segment_id], set()).add(a.chosen or "")
    return out


def build_records(
    model: BaseModel,
    inp: LoadedInput,
    *,
    min_words: int = 2,
    min_chars: int = 6,
) -> list[EvRec]:
    by_label = inp.by_label()
    recs = []
    for path, ev, claim_text in iter_evidence(model):
        x: InputSeg | None = by_label.get(ev.seg)
        loc = locate(
            ev.quote,
            x.text if x else None,
            w_start=ev.w_start,
            min_words=min_words,
            min_chars=min_chars,
        )
        recs.append(EvRec(path, claim_text, ev, loc, x.info if x else None))
    return recs


def generate(
    s: Session,
    llm: LLM,
    cfg: KLConfig,
    *,
    model_id: str,
    messages_fn,  # (feedback|None) -> messages
    kind: str,
    inp: LoadedInput,
    usage: Usage,
    extra_ctx: dict[str, Any] | None = None,
    max_attempts: int = 2,
) -> GenResult:
    """The shared loop of every LLM stage. ``kind`` selects the submit_<kind> tool."""
    feedback: str | None = None
    last = GenResult(False, None, None)
    markers = markers_for(s, inp)
    seg_order = {x.info.label: i for i, x in enumerate(inp.segs)}
    for attempt in range(1, max_attempts + 1):
        res = call_tool(llm, cfg, model_id, messages_fn(feedback), kind)
        usage.add(res.input_tokens, res.output_tokens, res.cost_usd)
        last = GenResult(False, None, None, attempts=attempt, raw=res.args)
        if res.args is None:
            last.error = feedback = f"K1: {res.error}"
            continue
        tool_model = TOOL_MODELS[kind]
        try:
            parsed = tool_model.model_validate(res.args)
        except Exception as e:  # noqa: BLE001
            last.error = feedback = f"K1: schema: {str(e)[:600]}"
            continue
        ctx = CheckContext(
            kind=kind,
            model=parsed,
            evidence=build_records(
                parsed, inp, min_words=cfg.quote_min_words, min_chars=cfg.quote_min_chars
            ),
            markers=markers,
            seg_order=seg_order,
            window_labels=set(seg_order),
            k3_max_fraction=cfg.k3_not_found_max_fraction,
            issue_similarity_warn=cfg.issue_similarity_warn,
            step_max_words=cfg.step_max_words,
            **(extra_ctx or {}),
        )
        report = run_checks(ctx)
        last = GenResult(
            report.ok,
            parsed,
            parsed.model_dump(mode="json"),
            ctx.evidence,
            report,
            confidence(report, ctx),
            attempt,
            None if report.ok else report.feedback(),
            res.args,
        )
        if report.ok:
            return last
        feedback = report.feedback()
    return last
