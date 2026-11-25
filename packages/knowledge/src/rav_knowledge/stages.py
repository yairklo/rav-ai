"""Pipeline stages. Each stage plans idempotent tasks and executes one task in a transaction.

S0   deterministic: acronyms (ranked candidates by context), citations, discourse / quote /
     error markers -> annotations (each with method + method_version)
S0b  fast model, constrained choice among S0 candidates, per occurrence
S1   sugya boundaries on a sliding window of pages, merged at scope level
S2   basic sugya_summary from the Gemara only (Rashi / Steinsaltz = reading aid, never evidence)
S3, S3L, S4, S4G, S5: reserved in the DAG, implemented in a later PR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rav_ai.db.models import (
    Annotation,
    KlItem,
    KlItemSegment,
    KlTask,
    Segment,
    Work,
)
from rav_ai.normalize import OffsetMap
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from textnorm import find_acronyms, find_citations, normalize, rank_candidates

from .checks import coverage_gaps
from .config import KLConfig
from .core import GenResult, Usage, generate
from .domain import domain_for
from .licensing import item_permission
from .llm_io import build_messages, count_tokens
from .prompts import Prompt, load_prompt
from .schemas import SCHEMA_VERSION, schema_version_for
from .store import (
    LoadedInput,
    Scope,
    add_evidence,
    add_link,
    add_segments,
    current_item,
    load_input,
    sha,
    uses_steinsaltz,
    write_item,
)

DAG: dict[str, tuple[str, ...]] = {
    "s0": (),
    "s0b": ("s0",),
    "s1": ("s0",),
    "s2": ("s1",),
    "s3": ("s2",),
    "s3l": ("s3",),
    "s4": ("s3",),
    "s4g": ("s3",),
    "s5": ("s4", "s4g"),
}
S0_METHOD_VERSION = "s0.1"


class TaskFailed(Exception):
    def __init__(self, msg: str, usage: Usage | None = None):
        super().__init__(msg)
        self.usage = usage


@dataclass
class TaskSpec:
    key: str
    input_hash: str
    payload: dict[str, Any] = field(default_factory=dict)
    est_in: int = 0
    est_out: int = 0


@dataclass
class Outcome:
    item_ids: list[int] = field(default_factory=list)
    annotation_ids: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    result: dict | None = None
    stats: dict[str, int] = field(default_factory=dict)


@dataclass
class RunContext:
    cfg: KLConfig
    llm: Any
    scope: Scope
    stage: str
    model: str | None
    prompt: Prompt | None
    job_id: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    prior_results: dict[str, dict] = field(default_factory=dict)  # task_key -> result (this job)

    def bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def note_excluded(self, inp: LoadedInput) -> None:
        for why, n in inp.excluded.items():
            self.bump(f"excluded.{why}", n)


def _est(rc: RunContext, messages: list[dict]) -> tuple[int, int]:
    n, _ = count_tokens("\n".join(m["content"] for m in messages), rc.model)
    return n, int(n * rc.cfg.out_ratio.get(rc.stage, 0.2))


# ---- S0 ---------------------------------------------------------------------------------

_DISCOURSE = {
    "matni": ("מתני'", "מתניתין"),
    "gemara": "גמ'",
    "mitivei": "מתיבי",
    "tash": ('ת"ש', "תא שמע"),
    "itivei": "איתיביה",
    "teiku": "תיקו",
    "hilkheta": "והלכתא",
}
_QUOTE = {"vezal": 'וז"ל', "akhal": 'עכ"ל'}
_ERROR = {
    "tas": 'ט"ס',
    "tzl": 'צ"ל',
    "lg": 'ל"ג',
    "eino_mechuvar": "ואינו מחוור",
    "lo_nahagu": "ולא נהגו כן",
}
MARKER_KINDS = (
    ("discourse_marker", _DISCOURSE),
    ("quote_marker", _QUOTE),
    ("error_marker", _ERROR),
)


def _marker_regexes() -> list[tuple[str, str, re.Pattern]]:
    out = []
    for kind, table in MARKER_KINDS:
        for name, pats in table.items():
            for pat in (pats,) if isinstance(pats, str) else pats:
                norm = normalize(pat, "A").text
                out.append((kind, name, re.compile(rf"(?<![א-ת]){re.escape(norm)}(?![א-ת])")))
    return out


_MARKERS = _marker_regexes()


def find_markers(text_a: str, offsets_a: bytes) -> list[tuple[str, str, int, int]]:
    """(kind, name, raw_start, raw_end) of every marker in the level-A text."""
    om = OffsetMap.from_bytes(offsets_a)
    out = []
    for kind, name, rx in _MARKERS:
        for m in rx.finditer(text_a):
            rs, re_ = om.to_raw(m.start(), m.end())
            out.append((kind, name, rs, re_))
    return sorted(out, key=lambda x: (x[2], x[3], x[1]))


class S0:
    name = "s0"

    def plan(self, rc: RunContext, s: Session) -> list[TaskSpec]:
        specs = []
        for page in rc.scope.pages:
            inp = load_input(s, page.segment_ids, rc.cfg)
            rc.note_excluded(inp)
            if not inp.segs:
                continue
            specs.append(
                TaskSpec(
                    f"s0:{page.label}",
                    sha("s0", S0_METHOD_VERSION, inp.raw_hashes, rc.cfg.mode),
                    {"segment_ids": page.segment_ids},
                )
            )
        return specs

    def execute(self, rc: RunContext, s: Session, spec: TaskSpec) -> Outcome:
        out = Outcome()
        inp = load_input(s, spec.payload["segment_ids"], rc.cfg)
        work = s.get(Work, rc.scope.work_id)
        domain = domain_for(work.category, work.category_path)
        ids = [x.info.segment_id for x in inp.segs]
        # a new method version retires the old rows of the same methods
        s.execute(
            update(Annotation)
            .where(
                Annotation.segment_id.in_(ids),
                Annotation.is_current,
                Annotation.method.like("s0.%"),
                Annotation.method_version != S0_METHOD_VERSION,
            )
            .values(is_current=False)
        )
        rows: list[dict] = []

        def row(x, kind, a, b, method, cands, chosen, conf=None, ctx=None):
            rows.append(
                dict(
                    segment_id=x.info.segment_id,
                    version_id=x.text.version_id,
                    span_start=a,
                    span_end=b,
                    kind=kind,
                    candidates=cands,
                    chosen=chosen,
                    confidence=conf,
                    origin="rule",
                    raw_hash=x.text.raw_hash,
                    status="generated",
                    is_current=True,
                    method=method,
                    method_version=S0_METHOD_VERSION,
                    context_domain=domain,
                    context=ctx,
                    job_id=rc.job_id,
                )
            )

        for x in inp.segs:
            raw = x.text.text_raw
            for c in find_acronyms(raw):
                ranked = rank_candidates(c.norm, domain)
                cands = [
                    {
                        "text": e.text,
                        "domain": e.domain,
                        "source": e.source,
                        "prior": e.prior.get(domain),
                    }
                    for e in ranked
                ]
                row(
                    x,
                    "abbrev",
                    c.span[0],
                    c.span[1],
                    "s0.acronym",
                    cands,
                    None,
                    ctx={"norm": c.norm, "kind": c.kind, "could_be_numeral": c.could_be_numeral},
                )
            for c in find_citations(raw, context=rc.scope.work_title):
                row(
                    x,
                    "citation",
                    c.span[0],
                    c.span[1],
                    "s0.citation",
                    [{"id": c.id, "kind": c.kind}],
                    c.id,
                    c.confidence,
                )
            for kind, name, a, b in find_markers(x.text.text_a, x.text.offsets_a):
                row(x, kind, a, b, f"s0.{kind}", [{"marker": name}], name)
        if rows:
            stmt = (
                pg_insert(Annotation)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["segment_id", "version_id", "span_start", "span_end", "kind"],
                    index_where=Annotation.is_current,
                )
                .returning(Annotation.id)
            )
            out.annotation_ids = list(s.scalars(stmt))
        out.stats["annotations"] = len(rows)
        return out


# ---- S0b --------------------------------------------------------------------------------


class S0b:
    name = "s0b"

    def _pending(self, s: Session, ids: list[int]) -> list[Annotation]:
        """Ambiguous abbreviations (>= 2 candidates) without a current LLM choice."""
        done = {
            (a.segment_id, a.span_start, a.span_end)
            for a in s.scalars(
                select(Annotation).where(
                    Annotation.segment_id.in_(ids),
                    Annotation.kind == "abbrev_choice",
                    Annotation.is_current,
                )
            )
        }
        anns = s.scalars(
            select(Annotation)
            .where(
                Annotation.segment_id.in_(ids),
                Annotation.kind == "abbrev",
                Annotation.is_current,
            )
            .order_by(Annotation.segment_id, Annotation.span_start)
        )
        return [
            a
            for a in anns
            if len(a.candidates or []) >= 2 and (a.segment_id, a.span_start, a.span_end) not in done
        ]

    def _lines(self, s: Session, anns: list[Annotation], inp: LoadedInput) -> str:
        by_seg = {x.info.segment_id: x for x in inp.segs}
        lines = []
        for a in anns:
            x = by_seg[a.segment_id]
            raw = x.text.text_raw
            ctx = raw[max(0, a.span_start - 60) : a.span_end + 60]
            cands = "; ".join(f"{i}={c['text']}" for i, c in enumerate(a.candidates))
            lines.append(f"id={a.id} ⟨{raw[a.span_start : a.span_end]}⟩ …{ctx}…  מועמדים: {cands}")
        return "\n".join(lines)

    def plan(self, rc: RunContext, s: Session) -> list[TaskSpec]:
        specs = []
        for page in rc.scope.pages:
            inp = load_input(s, page.segment_ids, rc.cfg)
            rc.note_excluded(inp)
            anns = self._pending(s, [x.info.segment_id for x in inp.segs])
            if not anns:
                continue
            msgs = build_messages(rc.prompt, [], rc.cfg, extra=self._lines(s, anns, inp))
            ein, eout = _est(rc, msgs)
            specs.append(
                TaskSpec(
                    f"s0b:{page.label}",
                    sha(
                        "s0b",
                        SCHEMA_VERSION,
                        rc.prompt.hash,
                        rc.model,
                        rc.cfg.params(),
                        [(a.id, a.raw_hash, a.candidates) for a in anns],
                    ),
                    {"segment_ids": page.segment_ids},
                    ein,
                    eout,
                )
            )
        return specs

    def execute(self, rc: RunContext, s: Session, spec: TaskSpec) -> Outcome:
        out = Outcome()
        inp = load_input(s, spec.payload["segment_ids"], rc.cfg)
        anns = {a.id: a for a in self._pending(s, [x.info.segment_id for x in inp.segs])}
        extra = self._lines(s, list(anns.values()), inp)
        from .llm_io import call_tool

        res = call_tool(
            rc.llm,
            rc.cfg,
            rc.model,
            build_messages(rc.prompt, [], rc.cfg, extra=extra),
            "abbrev_choice",
        )
        out.usage.add(res.input_tokens, res.output_tokens, res.cost_usd)
        if res.args is None:
            raise TaskFailed(str(res.error), out.usage)
        n_cur = 0
        for ch in res.args.get("choices", []):
            a = anns.get(ch.get("id"))
            idx = ch.get("choice")
            if a is None or not isinstance(idx, int) or not 0 <= idx < len(a.candidates):
                out.stats["ignored_choices"] = out.stats.get("ignored_choices", 0) + 1
                continue
            has_current = s.scalars(
                select(Annotation.id).where(
                    Annotation.segment_id == a.segment_id,
                    Annotation.span_start == a.span_start,
                    Annotation.span_end == a.span_end,
                    Annotation.kind == "abbrev_choice",
                    Annotation.is_current,
                )
            ).first()
            new = Annotation(
                segment_id=a.segment_id,
                version_id=a.version_id,
                span_start=a.span_start,
                span_end=a.span_end,
                kind="abbrev_choice",
                candidates=a.candidates,
                chosen=a.candidates[idx]["text"],
                confidence=None,
                origin="llm",
                raw_hash=a.raw_hash,
                status="generated",
                is_current=has_current is None,
                method="s0b.llm_choice",
                method_version=rc.prompt.version,
                context_domain=a.context_domain,
                context={"model": rc.model, "choice": idx},
                job_id=rc.job_id,
            )
            s.add(new)
            s.flush()
            out.annotation_ids.append(new.id)
            n_cur += 1
        out.stats["choices"] = n_cur
        return out


# ---- S1 ---------------------------------------------------------------------------------


def windows(pages: list, size: int, overlap: int) -> list[list]:
    step = max(1, size - overlap)
    out, i = [], 0
    while True:
        out.append(pages[i : i + size])
        if i + size >= len(pages):
            return out
        i += step


class S1:
    name = "s1"

    def plan(self, rc: RunContext, s: Session) -> list[TaskSpec]:
        specs: list[TaskSpec] = []
        wins = windows(rc.scope.pages, rc.cfg.window_amudim, rc.cfg.window_overlap)
        for i, w in enumerate(wins):
            ids = [x for p in w for x in p.segment_ids]
            inp = load_input(s, ids, rc.cfg)
            rc.note_excluded(inp)
            if not inp.segs:
                continue
            msgs = build_messages(rc.prompt, inp.segs, rc.cfg)
            ein, eout = _est(rc, msgs)
            last = i == len(wins) - 1
            specs.append(
                TaskSpec(
                    f"w:{i:03d}",
                    sha(
                        "s1",
                        SCHEMA_VERSION,
                        rc.prompt.hash,
                        rc.model,
                        rc.cfg.params(),
                        inp.raw_hashes,
                    ),
                    {"segment_ids": ids, "window": i, "last": last},
                    ein,
                    eout,
                )
            )
        if specs:
            specs.append(
                TaskSpec("merge", sha("s1.merge", [x.input_hash for x in specs]), {"merge": True})
            )
        return specs

    def execute(self, rc: RunContext, s: Session, spec: TaskSpec) -> Outcome:
        if spec.payload.get("merge"):
            return self._merge(rc, s, spec)
        out = Outcome()
        inp = load_input(s, spec.payload["segment_ids"], rc.cfg)
        gen = generate(
            s,
            rc.llm,
            rc.cfg,
            model_id=rc.model,
            messages_fn=lambda fb: build_messages(rc.prompt, inp.segs, rc.cfg, feedback=fb),
            kind="sugya",
            inp=inp,
            usage=out.usage,
        )
        if not gen.ok:
            raise TaskFailed(f"boundaries rejected: {gen.error}", out.usage)
        order = {x.info.label: x for x in inp.segs}
        gidx = {sid: i for i, sid in enumerate(rc.scope.segment_ids)}
        first_idx, last_idx = gidx[inp.segs[0].info.segment_id], gidx[inp.segs[-1].info.segment_id]
        win, last = spec.payload["window"], spec.payload["last"]
        derive = (
            rc.cfg.window_overlap > 0
        )  # with an overlap the edge is certain; else trust the LLM
        sugyot = []
        for sg in gen.body["sugyot"]:
            a, b = order[sg["start_seg"]], order[sg["end_seg"]]
            si, ei = gidx[a.info.segment_id], gidx[b.info.segment_id]
            sugyot.append(
                {
                    "title_he": sg["title_he"],
                    "start_id": a.info.segment_id,
                    "end_id": b.info.segment_id,
                    "start_ref": a.info.ref,
                    "end_ref": b.info.ref,
                    "start_idx": si,
                    "end_idx": ei,
                    "open_start": win > 0
                    and (bool(sg.get("open_start")) or (derive and si == first_idx)),
                    "open_end": not last
                    and (bool(sg.get("open_end")) or (derive and ei == last_idx)),
                }
            )
        perm = item_permission([(x.version, x.source) for x in inp.segs])
        out.result = {
            "window": win,
            "last": last,
            "idx": [gidx[x.info.segment_id] for x in inp.segs],
            "sugyot": sugyot,
            "version_ids": inp.version_ids,
            "can_display_public": perm.can_display_public,
            "can_train": perm.can_train,
        }
        return out

    def _merge(self, rc: RunContext, s: Session, spec: TaskSpec) -> Outcome:
        out = Outcome()
        results = [
            r for k, r in sorted(rc.prior_results.items()) if k.startswith("w:") and r is not None
        ]
        spans, universe = merge_windows(results)
        ids = rc.scope.segment_ids
        holes, overlaps = coverage_gaps([(x["start_idx"], x["end_idx"]) for x in spans], universe)
        findings: dict[int, list[str]] = {}  # span index -> messages

        def flag(pred, msg: str) -> None:
            for j, x in enumerate(spans):
                if pred(x):
                    findings.setdefault(j, []).append(msg)

        for a, b in _ranges(holes):
            msg = f"hole: segments {ids[a]}..{ids[b]} belong to no sugya"
            flag(lambda x, a=a, b=b: x["end_idx"] == a - 1 or x["start_idx"] == b + 1, msg)
        for a, b in _ranges(overlaps):
            msg = f"overlap: segments {ids[a]}..{ids[b]} belong to more than one sugya"
            flag(lambda x, a=a, b=b: x["start_idx"] <= b and x["end_idx"] >= a, msg)
        versions = sorted({v for r in results for v in r["version_ids"]})
        display = all(r["can_display_public"] for r in results)
        train = all(r["can_train"] for r in results)
        for j, sg in enumerate(spans):
            seg_ids = ids[sg["start_idx"] : sg["end_idx"] + 1]
            report = None
            if j in findings or sg.get("disagree"):
                report = {
                    "ok": j not in findings,
                    "findings": [
                        {"code": "K8", "message": m, "path": "", "level": "error"}
                        for m in findings.get(j, [])
                    ]
                    + (
                        [
                            {
                                "code": "K8",
                                "message": "boundary disagreement between windows",
                                "path": "",
                                "level": "warn",
                            }
                        ]
                        if sg.get("disagree")
                        else []
                    ),
                    "downgrades": [],
                    "queue_priority": True,
                }
            item = write_item(
                s,
                kind="sugya",
                item_key=f"sugya:{sg['start_ref']}",
                scope_key=rc.scope.key,
                body={k: sg[k] for k in ("title_he", "start_ref", "end_ref")},
                input_hash=sha(spec.input_hash, sg["start_ref"], sg["end_ref"], sg["end_idx"]),
                job_id=rc.job_id,
                model=rc.model,
                prompt_version=rc.prompt.version,
                prompt_hash=rc.prompt.hash,
                schema_version=SCHEMA_VERSION,
                work_id=rc.scope.work_id,
                start_segment_id=sg["start_id"],
                end_segment_id=sg["end_id"],
                title_he=sg["title_he"],
                confidence=1.0 if report is None else 0.3,
                source_version_ids=versions,
                can_display_public=display,
                can_train=train,
                check_report=report,
                search_text_he=sg["title_he"],
            )
            add_segments(
                s, item, [(sid, "anchor" if i == 0 else "member") for i, sid in enumerate(seg_ids)]
            )
            out.item_ids.append(item.id)
        out.stats["sugyot"] = len(spans)
        if holes:
            out.stats["coverage_holes"] = len(holes)
        if overlaps:
            out.stats["coverage_overlaps"] = len(overlaps)
        return out


def _ranges(idx: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in sorted(idx):
        if out and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def merge_windows(results: list[dict]) -> tuple[list[dict], list[int]]:
    """Stitch per-window sugyot into scope-level sugyot (pure function, window order).

    A sugya cut by the right edge of a window (``open_end``) is carried to the next window; a
    sugya of that window flagged ``open_start`` continues it (the carried title and start win,
    the end comes from the continuation, and it may itself stay open). A sugya seen again in the
    overlap is dropped. Returns (sugyot, universe of segment indices that were loaded)."""
    spans: list[dict] = []
    carry: dict | None = None
    covered_to = -1
    universe = sorted({i for r in results for i in r.get("idx", [])})

    def finalize(sg: dict) -> None:
        nonlocal covered_to
        spans.append(sg)
        covered_to = max(covered_to, sg["end_idx"])

    for r in results:
        first_idx = min(r.get("idx") or [0])
        for sg in sorted(r["sugyot"], key=lambda x: x["start_idx"]):
            sg = dict(sg)
            if (
                carry is not None
                and sg["open_start"]
                and carry["start_idx"] <= sg["end_idx"] < carry["end_idx"]
                and sg["start_idx"] in (carry["start_idx"], first_idx)
            ):
                # the next window says the carried sugya ends here: shorten it, close it and let
                # whatever starts after be a new sugya; the disagreement goes to the queue
                for k in ("end_id", "end_ref", "end_idx"):
                    carry[k] = sg[k]
                carry["open_end"] = False
                carry["disagree"] = True
                finalize(carry)
                carry = None
                continue
            seen_to = max(covered_to, carry["end_idx"] if carry else -1)
            if sg["end_idx"] <= seen_to:
                continue  # seen again in the overlap; it must not close the carried sugya
            if carry is not None and (
                sg["start_idx"] <= carry["end_idx"]
                or (sg["open_start"] and sg["start_idx"] == carry["end_idx"] + 1)
            ):  # the continuation covers the tail of the carry
                for k in ("end_id", "end_ref", "end_idx"):
                    carry[k] = sg[k]
                carry["open_end"] = sg["open_end"]
                if not carry["open_end"]:
                    finalize(carry)
                    carry = None
                continue
            if carry is not None:  # nothing continued it: it ends where the window ended
                finalize(carry)
                carry = None
            if sg["end_idx"] <= covered_to:
                continue
            if sg["open_end"]:
                carry = sg
            else:
                finalize(sg)
    if carry is not None:
        finalize(carry)
    return spans, universe


# ---- S2 ---------------------------------------------------------------------------------


def _summary_text(body: dict) -> str:
    return " ".join([body.get("title_he", "")] + [st.get("text_he", "") for st in body["steps"]])


class S2:
    name = "s2"

    def _sugyot(self, s: Session, rc: RunContext) -> list[KlItem]:
        ids = rc.scope.segment_ids
        return list(
            s.scalars(
                select(KlItem)
                .where(
                    KlItem.kind == "sugya",
                    KlItem.is_current,
                    KlItem.start_segment_id.in_(ids),
                    KlItem.status != "rejected",
                )
                .order_by(KlItem.start_segment_id)
            )
        )

    def _segment_ids(self, s: Session, item: KlItem) -> list[int]:
        return list(
            s.scalars(
                select(KlItemSegment.segment_id)
                .where(KlItemSegment.item_id == item.id)
                .order_by(KlItemSegment.ord)
            )
        )

    def plan(self, rc: RunContext, s: Session) -> list[TaskSpec]:
        specs = []
        sugyot = self._sugyot(s, rc)
        if not sugyot:
            rc.stats["note"] = "no current sugyot in scope: run s1 (and promote) first"
        for sg in sugyot:
            inp = load_input(s, self._segment_ids(s, sg), rc.cfg, with_aids=True)
            rc.note_excluded(inp)
            if not inp.segs:
                continue
            msgs = build_messages(
                rc.prompt,
                inp.segs,
                rc.cfg,
                aids=inp.aids,
                prior=[(sg.item_key, sg.version, sg.title_he or "")],
            )
            ein, eout = _est(rc, msgs)
            specs.append(
                TaskSpec(
                    f"s2:{sg.item_key}",
                    sha(
                        "s2",
                        SCHEMA_VERSION,
                        rc.prompt.hash,
                        rc.model,
                        rc.cfg.params(),
                        inp.raw_hashes,
                        (sg.item_key, sg.version),
                    ),
                    {"sugya_id": sg.id},
                    ein,
                    eout,
                )
            )
        return specs

    def _run_model(self, rc, s, sg, inp, model_id, usage: Usage) -> GenResult:
        return generate(
            s,
            rc.llm,
            rc.cfg,
            model_id=model_id,
            messages_fn=lambda fb: build_messages(
                rc.prompt,
                inp.segs,
                rc.cfg,
                aids=inp.aids,
                prior=[(sg.item_key, sg.version, sg.title_he or "")],
                feedback=fb,
            ),
            kind="sugya_summary",
            inp=inp,
            usage=usage,
        )

    def _store(self, rc, s, sg, inp, spec, gen: GenResult, model_id, *, shadow=False) -> KlItem:
        perm = item_permission(
            [(x.version, x.source) for x in inp.segs + inp.aids],
            steinsaltz_aid=uses_steinsaltz(inp, rc.cfg),
        )
        body = gen.body if gen.body is not None else {"raw": gen.raw, "error": gen.error}
        title = (gen.body or {}).get("title_he") or sg.title_he
        item = write_item(
            s,
            kind="sugya_summary",
            item_key=f"summary:{sg.item_key}",
            scope_key=rc.scope.key,
            body=body,
            input_hash=sha(spec.input_hash, model_id),
            job_id=rc.job_id,
            model=model_id,
            prompt_version=rc.prompt.version,
            prompt_hash=rc.prompt.hash,
            schema_version=schema_version_for("sugya_summary"),
            status="generated" if gen.ok else "rejected",
            work_id=sg.work_id,
            start_segment_id=sg.start_segment_id,
            end_segment_id=sg.end_segment_id,
            title_he=title,
            confidence=gen.confidence,
            source_version_ids=inp.version_ids,
            can_display_public=perm.can_display_public,
            can_train=perm.can_train,
            check_report={
                **gen.report.to_json(),
                "attempts": gen.attempts,
                "error": gen.error,
                "shadow": shadow,
            },
            rejected_by=None if gen.ok else "auto",
            search_text_he=_summary_text(body) if gen.ok else None,
            shadow=shadow,
        )
        add_segments(
            s,
            item,
            [(x.info.segment_id, "member") for x in inp.segs]
            + [(a.info.segment_id, "commentary") for a in inp.aids],
        )
        add_evidence(
            s,
            item.id,
            [
                dict(
                    claim_id=r.path,
                    segment_id=r.loc.segment_id,
                    version_id=r.loc.version_id,
                    span_start=r.loc.span_start,
                    span_end=r.loc.span_end,
                    quote=r.ev.quote,
                    quote_prefix=r.loc.quote_prefix,
                    quote_suffix=r.loc.quote_suffix,
                    raw_hash=r.loc.raw_hash,
                    role=r.ev.role,
                    locate_status=r.loc.status,
                    entail_status=gen.report.entail.get(r.path),
                )
                for r in gen.evidence
            ],
        )
        add_link(s, item.id, sg.id, "summarizes")
        add_link(s, item.id, sg.id, "depends_on")
        return item

    def execute(self, rc: RunContext, s: Session, spec: TaskSpec) -> Outcome:
        out = Outcome()
        sg = s.get(KlItem, spec.payload["sugya_id"])
        inp = load_input(s, self._segment_ids(s, sg), rc.cfg, with_aids=True)
        gen = self._run_model(rc, s, sg, inp, rc.model, out.usage)
        item = self._store(rc, s, sg, inp, spec, gen, rc.model)
        out.item_ids.append(item.id)
        out.stats["rejected" if not gen.ok else "accepted"] = 1
        # second model on a deterministic ~fraction of tasks: shadow output, never current
        frac = rc.cfg.second_model_fraction
        if rc.cfg.second_model and int(sha(spec.key)[:8], 16) % 1000 < frac * 1000:
            g2 = self._run_model(rc, s, sg, inp, rc.cfg.second_model, out.usage)
            it2 = self._store(rc, s, sg, inp, spec, g2, rc.cfg.second_model, shadow=True)
            out.item_ids.append(it2.id)
            out.stats["second_model_tasks"] = 1
            out.stats["second_model_agree_ok"] = int(g2.ok == gen.ok)
        return out


STAGES: dict[str, Any] = {"s0": S0(), "s0b": S0b(), "s1": S1(), "s2": S2()}


def stage_prompt(stage: str) -> Prompt | None:
    if stage == "s0":
        return None
    return load_prompt(stage)


__all__ = ["DAG", "STAGES", "Outcome", "RunContext", "TaskFailed", "TaskSpec", "current_item"]
_ = (KlTask, Segment)
