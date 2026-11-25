"""Data behind the three review queues (no web code here, so it is testable on its own).

(a) answers: recorded runs, a claim at a time, next to the text of the sources read.
(b) pairs: two texts side by side, a label 1-6.
(c) knowledge-layer items: the sugya screen (raw text with evidence spans highlighted, the step
    tree), approve / reject / fix written to kl_reviews + kl_overrides.
"""

from __future__ import annotations

import difflib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import EvalsConfig
from ..kl_metrics import (
    RecordedRun,
    RejectedClaim,
    claim_codes,
    evaluate_recorded,
    load_fixtures,
)
from ..pairs import read_jsonl
from ..ratings import RatingStore

# ---- (a) answers --------------------------------------------------------------------------


@dataclass
class AnswerEntry:
    run: RecordedRun
    status: str  # verifier/auto status of the recorded run
    auto_codes: list[str]


def _entries(
    fixtures_dir: Path, cfg: EvalsConfig, *, include_synthetic: bool, include_heldout: bool
) -> list[AnswerEntry]:
    runs, _ = load_fixtures(fixtures_dir)
    out = []
    for rr in runs:
        if (rr.synthetic and not include_synthetic) or (
            rr.split == "heldout" and not include_heldout
        ):
            continue
        res, _s = evaluate_recorded(rr, cfg)
        codes = res.verification.failure_codes() if res.verification else []
        out.append(AnswerEntry(rr, res.status, codes))
    return out


def answer_queue(
    fixtures_dir: Path,
    cfg: EvalsConfig,
    seed: int = 0,
    *,
    include_synthetic: bool = False,
    include_heldout: bool = False,
) -> list[AnswerEntry]:
    """Stratified by category (round-robin, up to ``answers_sample_per_category`` each),
    questions that failed automatically first. Synthetic seed runs and heldout runs are left
    out by default (heldout must never feed prompt tuning)."""
    entries = _entries(
        fixtures_dir, cfg, include_synthetic=include_synthetic, include_heldout=include_heldout
    )
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
    codes = claim_codes(claim, rr, cfg)
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
        auto_caught=bool(codes),
        caught_codes=codes,
    )


def export_errors(
    fixtures_dir: Path,
    store: RatingStore,
    reports_dir: Path,
    cfg: EvalsConfig,
) -> dict[str, Any]:
    """Rejected / fixed claims -> ``reports/kl/errors-<prompt_version>.md`` (the counter-examples
    that may feed the next prompt) and one regression fixture per claim (``rejected-*.json``).

    Heldout runs go to a separate ``errors-heldout-<pv>.md`` that must never enter a prompt,
    and get no fixture; synthetic seed runs are skipped entirely."""
    entries = _entries(fixtures_dir, cfg, include_synthetic=True, include_heldout=True)
    by_id = {e.run.fixture_id: e for e in entries}
    latest = store.latest("answers", "unit_id")
    groups: dict[tuple[str, str], list[tuple[RejectedClaim, AnswerEntry]]] = defaultdict(list)
    written, skipped_synthetic = [], 0
    for (_rater, _unit), r in sorted(latest.items()):
        if r.get("kind") != "claim" or r.get("verdict") not in ("wrong", "fix"):
            continue
        e = by_id.get(r["run_id"])
        if e is None:
            continue
        if e.run.synthetic:
            skipped_synthetic += 1
            continue
        rc = rejected_from_rating(e, r, cfg)
        if e.run.split != "heldout":
            p = fixtures_dir / f"{rc.fixture_id}.json"
            p.write_text(rc.model_dump_json(indent=2) + "\n", encoding="utf-8")
            written.append(p)
        groups[(e.run.split or "dev", rc.prompt_version or "unknown")].append((rc, e))
    reports = []
    out_dir = reports_dir / "kl"
    for (split, pv), items in sorted(groups.items()):
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"# Rejected claims - prompt_version {pv}"
        if split == "heldout":
            title += " - HELDOUT: never use in a prompt"
        lines = [title, ""]
        by_type: dict[str, list] = defaultdict(list)
        for rc, e in items:
            by_type[rc.error_type].append((rc, e))
        for et, xs in sorted(by_type.items()):
            lines += [f"## {et} ({len(xs)})", ""]
            for rc, e in xs:
                sub = f" / {rc.subtype}" if rc.subtype else ""
                caught = (
                    f"caught by {', '.join(rc.caught_codes)}"
                    if rc.caught_codes
                    else "NOT caught automatically"
                )
                lines.append(
                    f"- `{e.run.fixture_id}` claim {rc.claim_index}{sub} ({caught}): "
                    f"{rc.claim.text}" + (f" - {rc.note}" if rc.note else "")
                )
            lines.append("")
        name = f"errors-heldout-{pv}.md" if split == "heldout" else f"errors-{pv}.md"
        p = out_dir / name
        p.write_text("\n".join(lines), encoding="utf-8")
        reports.append(p)
    return {
        "fixtures": [str(p) for p in written],
        "reports": [str(p) for p in reports],
        "skipped_synthetic": skipped_synthetic,
    }


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


BUCKETS = (
    "warnings",  # 0 K14/K15 warnings, queue_priority
    "model_disagreement",  # 1 a shadow second-model output that differs
    "weighty_relation",  # 2 rules_like / disputes / psak_not_accepted / girsa_error
    "low_confidence",  # 3 confidence < kl_low_conf
    "used_in_reviewed_run",  # 4 relied on by a run reviewed in the answers queue
    "random",  # 5 a seeded random fraction of the rest
)


def ordered_items(
    s: Session, cfg: EvalsConfig, seed: int = 0, reviewed_item_ids: Iterable[int] = ()
) -> list[tuple[int, str]]:
    """Current, generated items in review order (design §7.4 / §11 KL-3), as (id, bucket).
    An item is listed once, in its first bucket; a bucket without data is just empty."""
    from rav_ai.db.models import KlItem, KlRelation

    items = [
        i
        for i in s.scalars(
            select(KlItem)
            .where(KlItem.is_current.is_(True), KlItem.status == "generated")
            .order_by(KlItem.id)
        )
        if not (i.check_report or {}).get("shadow")
    ]
    keys = [i.item_key for i in items]
    shadows: dict[str, list] = defaultdict(list)
    if keys:
        for sh in s.scalars(select(KlItem).where(KlItem.item_key.in_(keys))):
            if (sh.check_report or {}).get("shadow"):
                shadows[sh.item_key].append(sh.body)
    weighty = set(cfg.kl_weighty)
    ids = [i.id for i in items]
    weighty_ids: set[int] = set()
    if ids:
        for f, t, rel in s.execute(
            select(KlRelation.from_item_id, KlRelation.to_item_id, KlRelation.rel).where(
                (KlRelation.from_item_id.in_(ids)) | (KlRelation.to_item_id.in_(ids))
            )
        ):
            if rel in weighty:
                weighty_ids.update((f, t))
    reviewed = set(reviewed_item_ids)
    warn_codes = set(cfg.kl_warning_codes)

    def bucket(i) -> int | None:
        cr = i.check_report or {}
        codes = {f.get("code") for f in cr.get("findings", []) if isinstance(f, dict)}
        if cr.get("queue_priority") or codes & warn_codes:
            return 0
        if any(b != i.body for b in shadows.get(i.item_key, [])):
            return 1
        body = i.body or {}
        if i.id in weighty_ids or body.get("claim_kind") in weighty or body.get("kind") in weighty:
            return 2
        if i.confidence is None or i.confidence < cfg.kl_low_conf:
            return 3
        if i.id in reviewed:
            return 4
        return None

    ranked = [(bucket(i), i) for i in items]
    head = sorted(
        (
            (b, i.confidence if i.confidence is not None else -1.0, i.id)
            for b, i in ranked
            if b is not None
        )
    )
    rest = sorted(i.id for b, i in ranked if b is None)
    k = math.ceil(cfg.kl_random_fraction * len(rest)) if rest else 0
    sample = sorted(random.Random(seed).sample(rest, k))
    return [(i, BUCKETS[b]) for b, _c, i in head] + [(i, BUCKETS[5]) for i in sample]


def item_queue(
    s: Session, cfg: EvalsConfig, seed: int = 0, reviewed_item_ids: Iterable[int] = ()
) -> list[int]:
    return [i for i, _b in ordered_items(s, cfg, seed, reviewed_item_ids)]


def reviewed_run_item_ids(fixtures_dir: Path, store: RatingStore) -> set[int]:
    """kl item ids that runs reviewed in the answers queue relied on (runs.knowledge_items)."""
    rated = {e["run_id"] for e in store.events("answers") if e.get("run_id")}
    if not rated:
        return set()
    runs, _ = load_fixtures(fixtures_dir)
    out: set[int] = set()
    for rr in runs:
        if rr.fixture_id in rated:
            for it in rr.output.knowledge_items:
                if isinstance(it, dict) and it.get("tool") != "_meta" and it.get("item_id"):
                    out.add(int(it["item_id"]))
    return out


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


def _hidden_so_far(s: Session, it) -> set[str]:
    from rav_ai.db.models import KlOverride

    hidden = {str(x) for x in (it.body or {}).get("hidden_claims", [])}
    for o in s.scalars(
        select(KlOverride).where(KlOverride.target_key == it.item_key, KlOverride.action == "edit")
    ):
        hidden |= {str(x) for x in (o.corrected or {}).get("hidden_claims", [])}
    return hidden


def item_action(
    s: Session,
    item_id: int,
    action: str,
    reviewer: str,
    *,
    claim_id: str = "",
    note: str = "",
    error_type: str | None = None,
) -> dict[str, Any]:
    """approve -> verified; reject -> rejected (human, no longer current); fix -> the claim is
    hidden (an ``edit`` override with ``hidden_claims``), the item stays ``generated`` and
    current - hiding one claim does not approve the others. Each action writes kl_reviews and
    kl_overrides (so a re-run cannot undo it). The harness must read ``hidden_claims`` (KL-2)."""
    from rav_ai.db.models import KlItem, KlOverride, KlReview

    it = s.get(KlItem, item_id)
    if it is None:
        raise KeyError(item_id)
    tag = f"error_type={error_type}" if error_type else ""
    rnote = "; ".join(x for x in (tag, note) if x)
    result: dict[str, Any] = {"item_id": it.id, "action": action, "error_type": error_type}
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
        s.add(KlReview(item_id=it.id, reviewer=reviewer, verdict="approve", note=rnote or None))
        it.status = "verified"
    elif action == "reject":
        s.add(KlOverride(target_key=it.item_key, action="reject", note=rnote, reviewer=reviewer))
        s.add(KlReview(item_id=it.id, reviewer=reviewer, verdict="reject", note=rnote or None))
        it.status, it.rejected_by, it.is_current = "rejected", "human", False
    elif action == "fix":
        if not claim_id:
            raise ValueError("fix needs the claim_id to hide")
        hidden = sorted(_hidden_so_far(s, it) | {str(claim_id)})
        s.add(
            KlOverride(
                target_key=it.item_key,
                action="edit",
                corrected={"hidden_claims": hidden},
                note=rnote,
                reviewer=reviewer,
            )
        )
        s.add(
            KlReview(
                item_id=it.id,
                reviewer=reviewer,
                verdict="edit",
                note="; ".join(x for x in (f"hide claim {claim_id}", rnote) if x),
            )
        )
        result["hidden_claims"] = hidden
    else:
        raise ValueError(f"unknown action {action!r}")
    s.commit()
    result["status"] = it.status
    return result
