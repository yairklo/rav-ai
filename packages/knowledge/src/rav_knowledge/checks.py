"""Deterministic K-checks over a submitted item. Pure functions over ``CheckContext``.

K1 schema  K2 evidence present  K3 located  K4 domain  K5 attribution by tier  K6 entailment
(Protocol + stub)  K7 coverage (warn)  K8 sugya boundaries  K9 chronology  K10 source_claim
holder  K11 source_claim target  K12 chakira side  K13 move graph  K14 S0 marker contradiction
(warn, goes to the top of the review queue)  K15 position needs an issue  KM machloket sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from .evidence import Located
from .schemas import (
    CRITICAL_ROLES,
    Chakira,
    Ev,
    GirsaVariant,
    Holder,
    Issue,
    Machloket,
    OpenQuestion,
    Position,
    SourceClaim,
    Step,
    Sugya,
    SugyaBoundaries,
    SugyaSummary,
    iter_evidence,
    validate_body,
)

PERIOD_RANK = {
    "tannaim": 1,
    "amoraim": 2,
    "savoraim": 3,
    "geonim": 4,
    "rishonim": 5,
    "acharonim": 6,
    "modern": 7,
}
NON_HUMAN_HOLDERS = {"system", "model", "llm", "ai", "claude", "unknown", "מערכת", "מודל"}


@dataclass
class SegInfo:
    label: str  # local label, "S7"
    segment_id: int
    ref: str
    work_id: int | None = None
    work_title: str | None = None
    author_slug: str | None = None
    tier: str | None = None
    source_role: str | None = None  # primary | lens | witness | reference_only | None (unknown)
    period: str | None = None
    order: int = 0  # position in the input window


@dataclass
class EvRec:
    path: str
    claim_text: str
    ev: Ev
    loc: Located
    seg: SegInfo | None = None


class EntailmentChecker(Protocol):
    def check(self, claim_text: str, quote: str) -> str:
        """'entailed' | 'not_entailed' | 'unchecked'"""
        ...


class StubEntailment:
    """K6 placeholder: nothing is checked yet; entail_status stays 'unchecked'."""

    def check(self, claim_text: str, quote: str) -> str:
        return "unchecked"


@dataclass
class Finding:
    code: str
    message: str
    path: str = ""
    level: str = "error"  # error | warn


@dataclass
class CheckReport:
    findings: list[Finding] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)  # paths downgraded to reading (K5)
    entail: dict[str, str] = field(default_factory=dict)  # evidence path -> status
    queue_priority: bool = False  # K14

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> list[str]:
        return sorted({f.code for f in self.findings})

    def feedback(self) -> str:
        return "\n".join(f"- {f.code}: {f.message} ({f.path})" for f in self.errors)

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": [vars(f) for f in self.findings],
            "downgrades": self.downgrades,
            "queue_priority": self.queue_priority,
        }

    def add(self, code: str, msg: str, path: str = "", level: str = "error") -> None:
        self.findings.append(Finding(code, msg, path, level))


@dataclass
class CheckContext:
    kind: str
    model: BaseModel
    evidence: list[EvRec] = field(default_factory=list)
    markers: dict[str, set[str]] = field(default_factory=dict)  # seg label -> S0 marker names
    known_issues: dict[str, str] = field(default_factory=dict)  # issue key -> canonical question
    position_issue: dict[str, str] = field(default_factory=dict)  # position key -> issue key
    coverage: dict[int, str] = field(default_factory=dict)  # lens work_id -> coverage status
    eras: dict[str, str] = field(default_factory=dict)  # entity slug / label -> period
    seg_order: dict[str, int] = field(default_factory=dict)  # seg label -> doc order
    window_labels: set[str] = field(default_factory=set)
    entailment: EntailmentChecker = field(default_factory=StubEntailment)
    k3_max_fraction: float = 0.10
    # K3 strictness is carried by the roles: step / position / holder_evidence default to a
    # critical role in schemas.py
    issue_similarity_warn: float = 0.6
    step_max_words: int = 25


# ---- helpers ---------------------------------------------------------------------------


def _words(s: str) -> set[str]:
    return set(re.findall(r"[א-תa-z0-9]+", s.lower()))


def similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _ev_for(ctx: CheckContext, prefix: str) -> list[EvRec]:
    return [e for e in ctx.evidence if e.path == prefix or e.path.startswith(prefix + "[")]


def _era(ctx: CheckContext, h: Holder | None) -> int | None:
    if h is None:
        return None
    for k in (h.entity_slug, h.label):
        if k and k in ctx.eras:
            return PERIOD_RANK.get(ctx.eras[k])
    if h.tier in ("rishonim", "acharonim"):
        return PERIOD_RANK[h.tier]
    return None


def _same_holder(a: Holder | None, b: Holder | None) -> bool:
    if a is None or b is None:
        return False
    if a.entity_slug and b.entity_slug:
        return a.entity_slug == b.entity_slug
    return a.label.strip() == b.label.strip()


# ---- individual checks -----------------------------------------------------------------


def k2_has_evidence(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if isinstance(m, SugyaSummary):
        for i, st in enumerate(m.steps):
            if not st.evidence:
                r.add("K2", f"step {st.id} has no evidence", f"steps[{i}]")
        return
    if isinstance(m, (Sugya, SugyaBoundaries, Machloket)):
        return
    if isinstance(m, Chakira):
        return  # K12 is stricter
    if isinstance(m, SourceClaim):
        return  # K10/K11
    if isinstance(m, (Issue, Position, OpenQuestion, GirsaVariant)) or hasattr(m, "evidence"):
        evs = [e for e in ctx.evidence]
        if not evs:
            r.add("K2", "item has no evidence")


def k3_located(ctx: CheckContext, r: CheckReport) -> None:
    scored = [e for e in ctx.evidence if e.loc.status != "out_of_scope"]
    if not scored:
        return
    bad = [e for e in scored if e.loc.status == "not_found"]
    for e in bad:
        if ev_role_critical(e):
            r.add("K3", f"role {e.ev.role} quote not found: {e.ev.quote[:40]!r}", e.path)
    frac = len(bad) / len(scored)
    if frac > ctx.k3_max_fraction:
        r.add("K3", f"{len(bad)}/{len(scored)} quotes not found (> {ctx.k3_max_fraction:.0%})")


def ev_role_critical(e: EvRec) -> bool:
    return e.ev.role in CRITICAL_ROLES


def k4_domain(ctx: CheckContext, r: CheckReport) -> None:
    for e in ctx.evidence:
        if e.loc.status == "out_of_scope":
            r.add("K4", f"evidence outside the provided input: seg {e.ev.seg!r}", e.path)


def _matches_source(name: str | None, seg: SegInfo) -> bool:
    n = (name or "").strip()
    return bool(n) and n in (seg.work_title, seg.author_slug)


def _lens_holder(seg: SegInfo) -> Holder:
    return Holder(
        label=seg.author_slug or seg.work_title or "lens",
        entity_slug=seg.author_slug,
        work=seg.work_title,
        tier=seg.tier,
    )


def k5_attribution(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, Position):
        return
    ev = [e for e in ctx.evidence if e.loc.status in ("located", "ambiguous") and e.seg]
    if not ev:
        return  # K2/K3 report it
    h = m.holder

    def holder_owns(e: EvRec) -> bool:
        return bool(
            e.seg
            and (
                (h.work and e.seg.work_title == h.work)
                or (h.entity_slug and e.seg.author_slug == h.entity_slug)
            )
        )

    non_lens = [e for e in ev if e.seg and e.seg.source_role not in ("lens", None)]
    if not non_lens:
        if any(holder_owns(e) for e in ev):
            return  # the holder is the lens author himself: his own method, not a reading
        # evidence only from a lens (or from an unregistered source: fail closed): the claim
        # becomes a reading of the lens author, not the holder's own view
        lens = next((e.seg for e in ev if e.seg and e.seg.source_role == "lens"), None)
        m.reading_only = True
        m.attribution_grade = "inferred_by_source"
        if lens is not None:
            m.read_by = _lens_holder(lens)
        r.downgrades.append("position")
        r.add("K5", "evidence only from a lens source: downgraded to reading", "", "warn")
        return
    if h.attribution == "reported":
        if not h.reported_by:
            r.add("K5", "reported attribution without reported_by", "holder")
            return
        by = [
            e
            for e in ev
            if e.ev.role == "attributes" and e.seg and _matches_source(h.reported_by, e.seg)
        ]
        if not by:
            r.add(
                "K5",
                "attribution=reported needs located evidence with role=attributes from the text "
                "or author named in reported_by",
                "holder",
            )
        return
    if not any(holder_owns(e) for e in non_lens):
        r.add(
            "K5",
            "direct attribution needs evidence in the holder's own text "
            "(or attribution=reported with reported_by)",
            "holder",
        )


def k6_entailment(ctx: CheckContext, r: CheckReport) -> None:
    for e in ctx.evidence:
        if e.loc.status not in ("located", "ambiguous"):
            continue
        st = ctx.entailment.check(e.claim_text, e.ev.quote)
        r.entail[e.path] = st
        if st == "not_entailed":
            r.add("K6", "claim not entailed by its quote", e.path)


def k7_coverage(ctx: CheckContext, r: CheckReport) -> None:
    if ctx.kind not in ("issue", "position", "machloket", "open_question", "chakira"):
        return
    missing = [w for w, st in ctx.coverage.items() if st in ("not_ingested", "unknown")]
    if missing:
        r.add(
            "K7",
            f"{len(missing)} lens work(s) not ingested/unknown: absence of a view is not proven",
            level="warn",
        )


def k8_boundaries(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, SugyaBoundaries):
        return
    spans: list[tuple[int, int, str]] = []
    for i, s in enumerate(m.sugyot):
        a, b = ctx.seg_order.get(s.start_seg), ctx.seg_order.get(s.end_seg)
        if a is None or b is None:
            r.add("K8", f"boundary outside the window: {s.start_seg}-{s.end_seg}", f"sugyot[{i}]")
            continue
        if a > b:
            r.add("K8", "start after end", f"sugyot[{i}]")
            continue
        spans.append((a, b, s.title_he))
    spans.sort()
    for (_a1, b1, t1), (a2, _b2, t2) in zip(spans, spans[1:], strict=False):
        if a2 <= b1:
            r.add("K8", f"overlapping sugyot: {t1!r} / {t2!r}")
    if not m.sugyot:
        r.add("K8", "no sugyot")


def coverage_gaps(spans: list[tuple[int, int]], universe: list[int]) -> tuple[list[int], list[int]]:
    """Every index of ``universe`` must lie in exactly one span (inclusive ends).

    Returns (holes, overlaps): the indices that lie in no span / in more than one."""
    holes, overlaps = [], []
    for i in universe:
        n = sum(1 for a, b in spans if a <= i <= b)
        if n == 0:
            holes.append(i)
        elif n > 1:
            overlaps.append(i)
    return holes, overlaps


def _rel_error(ctx: CheckContext, r: CheckReport, rel: str, frm: Holder | None, to: Holder | None):
    a, b = _era(ctx, frm), _era(ctx, to)
    if a is None or b is None:
        return
    if rel in ("disputes", "rules_like", "objects_to") and b > a:
        r.add("K9", f"{rel} points to a later authority", level="error")
    if rel == "claims_error_in" and a < b:
        r.add("K9", "claims_error_in only from a later (or same-generation) authority")


def k9_chronology(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if isinstance(m, Position):
        for x in m.relations:
            _rel_error(ctx, r, x.rel, m.holder, x.target)
    elif isinstance(m, SourceClaim):
        _rel_error(ctx, r, "claims_error_in", m.holder, m.target.holder)


def k10_source_claim_holder(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, SourceClaim):
        return
    h = m.holder
    if not h.label.strip() or h.label.strip().lower() in NON_HUMAN_HOLDERS:
        r.add("K10", "source_claim holder must be a human", "holder")
    ev = _ev_for(ctx, "holder_evidence")
    if not ev or not any(e.loc.status in ("located", "ambiguous") for e in ev):
        r.add("K10", "holder_evidence missing or not located", "holder_evidence")
    if _same_holder(h, m.target.holder):
        r.add("K10", "holder equals the target's holder", "holder")


def k11_source_claim_target(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, SourceClaim):
        return
    t = m.target
    if t.target_status == "unresolved":
        return  # the target is not in the corpus: stored as such, never forced
    ev = _ev_for(ctx, "target.evidence")
    if not ev or ev[0].loc.status not in ("located", "ambiguous"):
        alt = (m.proposed_alternative or "").strip()
        if alt and t.quote.strip() == alt:
            r.add("K11", "corrected text written in place of the source text", "target")
        r.add("K11", "target not located (use target_status=unresolved if not in corpus)", "target")


def k12_chakira(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, Chakira):
        return
    if len(m.sides) < 2:
        r.add("K12", "a chakira needs at least two sides")
    for i, s in enumerate(m.sides):
        ev = _ev_for(ctx, f"sides[{i}].proposer_evidence")
        if not ev or not any(e.loc.status in ("located", "ambiguous") for e in ev):
            r.add("K12", "side without located proposer_evidence", f"sides[{i}]")
        if not s.interprets.strip():
            r.add("K12", "side without interprets", f"sides[{i}]")
        for a in s.aligned:
            if s.asserted_by is None or _same_holder(s.asserted_by, a):
                r.add("K12", "aligned holder needs an asserted_by other than itself", f"sides[{i}]")


def k13_move_graph(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, SugyaSummary):
        return
    idx: dict[str, int] = {}
    for i, st in enumerate(m.steps):
        if st.id in idx:
            r.add("K13", f"duplicate step id {st.id}", f"steps[{i}]")
        idx[st.id] = i
    attacked: set[str] = set()
    for i, st in enumerate(m.steps):
        if i > 0 and not st.responds_to:
            r.add("K13", f"step {st.id} has no responds_to", f"steps[{i}]")
        if st.responds_to:
            j = idx.get(st.responds_to)
            if j is None or j >= i:
                r.add(
                    "K13", f"responds_to {st.responds_to!r} is not an earlier step", f"steps[{i}]"
                )
                continue
            tgt: Step = m.steps[j]
            if st.subtype == "teyuvta" and tgt.move != "objection":
                r.add("K13", "teyuvta must close an objection", f"steps[{i}]")
            if st.relation == "attacks":
                attacked.add(tgt.id)
    for i, st in enumerate(m.steps):
        if st.standing == "refuted" and st.id not in attacked:
            r.add("K13", f"{st.id} is refuted but no step attacks it", f"steps[{i}]")


_MARKER_EXPECT: dict[str, Any] = {
    "mitivei": lambda st: st.move == "objection",
    "itivei": lambda st: st.move == "objection",
    "tash": lambda st: st.move in ("objection", "support", "answer", "statement"),
    "teiku": lambda st: st.standing == "open" or st.subtype == "teiku",
    "hilkheta": lambda st: st.move == "conclusion" or st.subtype == "hilkheta",
}


def k14_markers(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, SugyaSummary):
        return
    by_path: dict[int, set[str]] = {}
    for e in ctx.evidence:
        mt = re.match(r"steps\[(\d+)\]", e.path)
        if mt:
            by_path.setdefault(int(mt.group(1)), set()).update(ctx.markers.get(e.ev.seg, ()))
    for i, st in enumerate(m.steps):
        for mk in by_path.get(i, ()):
            ok = _MARKER_EXPECT.get(mk)
            if ok is not None and not ok(st):
                r.add("K14", f"marker {mk} contradicts move={st.move}", f"steps[{i}]", "warn")
                r.queue_priority = True


def k15_position_issue(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if isinstance(m, Position):
        if not m.issue.strip():
            r.add("K15", "position without issue", "issue")
        elif ctx.known_issues and m.issue not in ctx.known_issues:
            r.add("K15", f"position refers to unknown issue {m.issue!r}", "issue")
    elif isinstance(m, Issue):
        for key, q in ctx.known_issues.items():
            if similarity(m.canonical_q_he, q) >= ctx.issue_similarity_warn:
                r.add("K15", f"issue is similar to existing {key}", "", "warn")
                break


def k_machloket(ctx: CheckContext, r: CheckReport) -> None:
    m = ctx.model
    if not isinstance(m, Machloket):
        return
    if len(m.sides) < 2 or any(not s for s in m.sides):
        r.add("KM", "machloket needs two non-empty sides")
    for side in m.sides:
        for pid in side:
            if ctx.position_issue.get(pid) != m.issue_id:
                r.add("KM", f"position {pid} does not belong to issue {m.issue_id}")


ALL_CHECKS = (
    k2_has_evidence,
    k3_located,
    k4_domain,
    k5_attribution,
    k6_entailment,
    k7_coverage,
    k8_boundaries,
    k9_chronology,
    k10_source_claim_holder,
    k11_source_claim_target,
    k12_chakira,
    k13_move_graph,
    k14_markers,
    k15_position_issue,
    k_machloket,
)


def run_checks(ctx: CheckContext) -> CheckReport:
    r = CheckReport()
    for chk in ALL_CHECKS:
        chk(ctx, r)
    return r


def collect_evidence(model: BaseModel) -> list[tuple[str, Ev, str]]:
    return iter_evidence(model)


def confidence(report: CheckReport, ctx: CheckContext) -> float:
    """Derived from the checks, never self-reported by the model."""
    n = len(ctx.evidence)
    nf = sum(1 for e in ctx.evidence if e.loc.status in ("not_found", "out_of_scope"))
    amb = sum(1 for e in ctx.evidence if e.loc.status == "ambiguous")
    c = 1.0 - 0.1 * len(report.warnings) - (0.5 * nf / n if n else 0.0) - (0.05 * amb)
    if not report.ok:
        c = min(c, 0.3)
    return round(max(0.0, min(1.0, c)), 3)


__all__ = [
    "ALL_CHECKS",
    "CheckContext",
    "CheckReport",
    "EntailmentChecker",
    "EvRec",
    "Located",
    "SegInfo",
    "StubEntailment",
    "collect_evidence",
    "coverage_gaps",
    "confidence",
    "run_checks",
    "similarity",
    "validate_body",
]
