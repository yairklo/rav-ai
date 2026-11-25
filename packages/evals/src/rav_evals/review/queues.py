"""Data behind the three review queues (no web code here, so it is testable on its own).

(a) answers: recorded runs, a claim at a time, next to the text of the sources read.
(b) pairs: two texts side by side, a label 1-6.
(c) knowledge-layer items: the sugya screen (raw text with evidence spans highlighted, the step
    tree), approve / reject / fix written to kl_reviews + kl_overrides.
"""

from __future__ import annotations

import difflib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import EvalsConfig
from ..kl_lookup import InMemoryKnowledgeLookup
from ..kl_metrics import RecordedRun, RejectedClaim, evaluate_recorded, load_fixtures
from ..pairs import read_jsonl
from ..ratings import RatingStore
from ..schema import Answer
from ..verifier import InMemorySourceStore, verify_answer

# ---- (a) answers --------------------------------------------------------------------------


@dataclass
class AnswerEntry:
    run: RecordedRun
    status: str  # verifier/auto status of the recorded run
    auto_codes: list[str]


def answer_queue(fixtures_dir: Path, cfg: EvalsConfig, seed: int = 0) -> list[AnswerEntry]:
    """Stratified by category (round-robin, up to ``answers_sample_per_category`` each),
    questions that failed automatically first."""
    runs, _ = load_fixtures(fixtures_dir)
    entries = []
    for rr in runs:
        res, _s = evaluate_recorded(rr, cfg)
        codes = res.verification.failure_codes() if res.verification else []
        entries.append(AnswerEntry(rr, res.status, codes))
    rng = random.Random(seed)
    by_cat: dict[str, list[AnswerEntry]] = defaultdict(list)
    for e in sorted(entries, key=lambda e: e.run.fixture_id):
        by_cat[e.run.question.category.value].append(e)
    for v in by_cat.values():
        rng.shuffle(v)
        v.sort(key=lambda e: e.status != "failed")  # stable: failed first
        del v[cfg.answers_sample_per_category :]
    out: list[AnswerEntry] = []
    cats = sorted(by_cat)
    while any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c]:
                out.append(by_cat[c].pop(0))
    out.sort(key=lambda e: e.status != "failed")
    return out


def rejected_from_rating(
    entry: AnswerEntry, rating: dict[str, Any], cfg: EvalsConfig
) -> RejectedClaim:
    rr = entry.run
    i = int(rating["claim_index"])
    claim = rr.output.answer.claims[i]
    rep = verify_answer(
        Answer(claims=[claim], disclaimer="x"),
        rr.output.retrieved_refs,
        InMemorySourceStore(rr.sources),
        knowledge=None if rr.knowledge is None else InMemoryKnowledgeLookup(rr.knowledge),
        markers=cfg.markers,
    )
    return RejectedClaim(
        fixture_id=f"rejected-{rr.fixture_id}-c{i}",
        run_fixture=rr.fixture_id,
        claim_index=i,
        claim=claim,
        error_type=rating.get("error_type") or "other",
        subtype=rating.get("subtype"),
        note=rating.get("note", ""),
        rater_id=rating["rater_id"],
        prompt_version=rr.output.prompt_version,
        auto_caught=not rep.passed,
    )


def export_errors(
    entries: list[AnswerEntry],
    store: RatingStore,
    fixtures_dir: Path,
    reports_dir: Path,
    cfg: EvalsConfig,
) -> dict[str, Any]:
    """Rejected / fixed claims -> reports/kl/errors-<prompt_version>.md and one regression
    fixture per claim (``rejected-*.json`` next to the recorded runs)."""
    by_id = {e.run.fixture_id: e for e in entries}
    latest = store.latest("answers", "unit_id")
    groups: dict[str, list[tuple[RejectedClaim, AnswerEntry]]] = defaultdict(list)
    written = []
    for (_rater, _unit), r in sorted(latest.items()):
        if r.get("kind") != "claim" or r.get("verdict") not in ("wrong", "fix"):
            continue
        e = by_id.get(r["run_id"])
        if e is None:
            continue
        rc = rejected_from_rating(e, r, cfg)
        p = fixtures_dir / f"{rc.fixture_id}.json"
        p.write_text(rc.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(p)
        groups[rc.prompt_version or "unknown"].append((rc, e))
    reports = []
    out_dir = reports_dir / "kl"
    for pv, items in sorted(groups.items()):
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Rejected claims - prompt_version {pv}", ""]
        by_type: dict[str, list] = defaultdict(list)
        for rc, e in items:
            by_type[rc.error_type].append((rc, e))
        for et, xs in sorted(by_type.items()):
            lines += [f"## {et} ({len(xs)})", ""]
            for rc, e in xs:
                sub = f" / {rc.subtype}" if rc.subtype else ""
                caught = "caught by verifier" if rc.auto_caught else "NOT caught automatically"
                lines.append(
                    f"- `{e.run.fixture_id}` claim {rc.claim_index}{sub} ({caught}): "
                    f"{rc.claim.text}" + (f" - {rc.note}" if rc.note else "")
                )
            lines.append("")
        p = out_dir / f"errors-{pv}.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        reports.append(p)
    return {"fixtures": [str(p) for p in written], "reports": [str(p) for p in reports]}


# ---- (b) pairs ----------------------------------------------------------------------------


def pair_queue(pairs_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(pairs_path)


# ---- (c) knowledge-layer items --------------------------------------------------------------


@dataclass
class ItemView:
    item: Any
    segments: list[dict[str, Any]] = field(default_factory=list)  # ref, html-ready parts
    tree: list[dict[str, Any]] = field(default_factory=list)  # nested steps
    claims: list[dict[str, Any]] = field(default_factory=list)  # flat, keyboard order
    diff: str = ""
    previous_job: int | None = None


REVIEW_KINDS = ("sugya", "sugya_summary")


def item_queue(s: Session, cfg: EvalsConfig, seed: int = 0) -> list[int]:
    """Current, not yet reviewed sugya / sugya_summary items, in review order: windows that
    disagreed (queue_priority), then K-check contradictions, then low confidence, then a
    seeded random sample of high-confidence items (to estimate the unseen error rate)."""
    from rav_ai.db.models import KlItem

    items = list(
        s.scalars(
            select(KlItem)
            .where(
                KlItem.is_current.is_(True),
                KlItem.status == "generated",
                KlItem.kind.in_(REVIEW_KINDS),
            )
            .order_by(KlItem.id)
        )
    )

    def rank(i) -> tuple:
        cr = i.check_report or {}
        if cr.get("queue_priority"):
            return (0, i.confidence or 0.0, i.id)
        if not cr.get("ok", True):
            return (1, i.confidence or 0.0, i.id)
        if i.confidence is None or i.confidence < cfg.kl_low_conf:
            return (2, i.confidence or 0.0, i.id)
        return (3, 0.0, i.id)

    ranked = sorted(items, key=rank)
    head = [i for i in ranked if rank(i)[0] < 3]
    high = [i for i in ranked if rank(i)[0] == 3]
    rng = random.Random(seed)
    sample = rng.sample(high, min(cfg.kl_high_conf_sample, len(high)))
    return [i.id for i in head] + sorted(i.id for i in sample)


def _raw_text(s: Session, segment_id: int, version_ids: list[int] | None):
    from rav_ai.db.models import Segment, SegmentText, Version

    q = (
        select(SegmentText, Segment.ref)
        .join(Segment, Segment.id == SegmentText.segment_id)
        .join(Version, Version.id == SegmentText.version_id)
        .where(SegmentText.segment_id == segment_id)
        .order_by(Version.is_primary.desc(), Version.id)
    )
    rows = s.execute(q).all()
    for st, ref in rows:
        if not version_ids or st.version_id in version_ids:
            return st, ref
    return rows[0] if rows else (None, None)


def _highlight(raw: str, spans: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    """Split raw text into parts, marking evidence spans (non-overlapping, first wins)."""
    parts, pos = [], 0
    for a, b, claim in sorted(spans):
        if a < pos or b <= a or a > len(raw):
            continue
        if a > pos:
            parts.append({"text": raw[pos:a], "mark": None})
        parts.append({"text": raw[a:b], "mark": claim})
        pos = b
    if pos < len(raw):
        parts.append({"text": raw[pos:], "mark": None})
    return parts


def _tree(steps: list[dict[str, Any]], hidden: set[str]) -> list[dict[str, Any]]:
    nodes = {
        str(st.get("id", i)): {
            **st,
            "id": str(st.get("id", i)),
            "children": [],
            "hidden": str(st.get("id", i)) in hidden,
        }
        for i, st in enumerate(steps)
    }
    roots = []
    for n in nodes.values():
        parent = n.get("responds_to")
        if parent is not None and str(parent) in nodes and str(parent) != n["id"]:
            nodes[str(parent)]["children"].append(n)
        else:
            roots.append(n)
    return roots


def item_view(s: Session, item_id: int) -> ItemView | None:
    from rav_ai.db.models import KlEvidence, KlItem, KlItemSegment

    it = s.get(KlItem, item_id)
    if it is None:
        return None
    v = ItemView(item=it)
    ev = list(s.scalars(select(KlEvidence).where(KlEvidence.item_id == it.id)))
    seg_ids = list(
        s.scalars(
            select(KlItemSegment.segment_id)
            .where(KlItemSegment.item_id == it.id)
            .order_by(KlItemSegment.ord, KlItemSegment.segment_id)
        )
    )
    for sid in seg_ids:
        st, ref = _raw_text(s, sid, it.source_version_ids)
        if st is None:
            continue
        spans = [
            (e.span_start, e.span_end, e.claim_id or "")
            for e in ev
            if e.segment_id == sid
            and e.span_start is not None
            and e.span_end is not None
            and e.locate_status == "located"
        ]
        v.segments.append({"ref": ref, "parts": _highlight(st.text_raw, spans)})
    body = it.body or {}
    hidden = {str(x) for x in body.get("hidden_claims", [])}
    steps = body.get("steps") or []
    v.tree = _tree(steps, hidden)
    v.claims = [
        {
            "id": str(st.get("id", i)),
            "text": st.get("text_he", ""),
            "move": st.get("move", ""),
            "hidden": str(st.get("id", i)) in hidden,
        }
        for i, st in enumerate(steps)
    ]
    if not v.claims:  # a sugya item: the item itself is the one claim (its boundaries)
        v.claims = [
            {"id": "", "text": it.title_he or it.item_key, "move": it.kind, "hidden": False}
        ]
    prev = s.scalars(
        select(KlItem)
        .where(KlItem.item_key == it.item_key, KlItem.version < it.version)
        .order_by(KlItem.version.desc())
    ).first()
    if prev is not None:
        v.previous_job = prev.job_id
        a = json.dumps(prev.body, ensure_ascii=False, indent=1, sort_keys=True).splitlines()
        b = json.dumps(it.body, ensure_ascii=False, indent=1, sort_keys=True).splitlines()
        v.diff = "\n".join(
            difflib.unified_diff(
                a,
                b,
                f"v{prev.version} (job {prev.job_id})",
                f"v{it.version} (job {it.job_id})",
                lineterm="",
            )
        )
    return v


def item_action(
    s: Session, item_id: int, action: str, reviewer: str, *, claim_id: str = "", note: str = ""
) -> dict[str, Any]:
    """approve -> verified; reject -> rejected (human, no longer current); fix -> hide the
    claim. Each is a kl_reviews row + a kl_overrides row (so a re-run cannot undo it)."""
    from rav_ai.db.models import KlItem, KlOverride, KlReview

    it = s.get(KlItem, item_id)
    if it is None:
        raise KeyError(item_id)
    result: dict[str, Any] = {"item_id": it.id, "action": action}
    if action == "approve":
        s.add(
            KlOverride(
                target_key=it.item_key,
                action="approve",
                corrected={"input_hash": it.input_hash},
                note=note,
                reviewer=reviewer,
            )
        )
        s.add(KlReview(item_id=it.id, reviewer=reviewer, verdict="approve", note=note))
        it.status = "verified"
    elif action == "reject":
        s.add(KlOverride(target_key=it.item_key, action="reject", note=note, reviewer=reviewer))
        s.add(KlReview(item_id=it.id, reviewer=reviewer, verdict="reject", note=note))
        it.status, it.rejected_by, it.is_current = "rejected", "human", False
    elif action == "fix":
        hidden = sorted({*map(str, (it.body or {}).get("hidden_claims", [])), str(claim_id)})
        o = KlOverride(
            target_key=it.item_key,
            action="edit",
            corrected={"hidden_claims": hidden},
            note=note,
            reviewer=reviewer,
        )
        s.add(o)
        s.add(
            KlReview(
                item_id=it.id,
                reviewer=reviewer,
                verdict="edit",
                note=f"hide claim {claim_id}" + (f": {note}" if note else ""),
            )
        )
        s.flush()
        result["hidden_claims"] = hidden
        result["applied"] = _apply_edit(s, it)
    else:
        raise ValueError(f"unknown action {action!r}")
    s.commit()
    result["status"] = it.status
    return result


def _apply_edit(s: Session, it) -> bool:
    """Apply the edit now through the knowledge layer's own override logic (a new verified
    human version, made current). Without rav_knowledge installed the override still stands
    and is applied on the next run."""
    try:
        from rav_knowledge.runner import apply_overrides
    except ImportError:  # pragma: no cover - rav_knowledge is in the workspace
        return False
    apply_overrides(s, [it.id])
    return True
