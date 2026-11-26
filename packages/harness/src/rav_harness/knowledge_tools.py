"""Knowledge-layer tools (KL-2): a deterministic, telegraphic map over the corpus.

The text is the authority; the layer is a map. Nothing returned here is citable and nothing
here adds to ``retrieved_refs``: every citation must be read with ``get_segment`` and passes the
verifier. Reads rav_ai.db.models only (no dependency on rav_knowledge).
"""

from __future__ import annotations

import inspect
import re
import tomllib
from collections import defaultdict
from typing import Any

import sqlalchemy as sa
from rav_ai.config import ROOT
from rav_ai.db.models import (
    Annotation,
    Edge,
    KlCoverage,
    KlEvidence,
    KlItem,
    KlItemSegment,
    KlJob,
    KlLink,
    KlOverride,
    KlReception,
    KlRelation,
    KlSource,
    Segment,
    SegmentText,
    Work,
)
from rav_ai.links_policy import candidate_label, is_candidate
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from textnorm import normalize

from .config import HarnessConfig
from .scope import (
    ScopeGate,
    extract_refs,
    hebrew_number,
    is_practical,
    key_covers_ref,
    ref_in_scope,
    scope_key_in_scope,
)
from .validators import KnowledgeContext

KNOWN_GAPS_PATH = ROOT / "config" / "known_gaps.toml"

ORIGIN_RANK = {"llm": 0, "judge": 1, "human": 2}
NO_SOURCE = "אין מקור"
PERSONAL = "PERSONAL_ONLY_RESTRICTED"
# items from these tiers (or from a role=lens source) are the "lens" layer: hidden in base
SECONDARY_TIERS = ("modern_secondary", "sugya_compendium", "lamdanut")
STEP_WORDS, POSITION_WORDS = 25, 20
PSAK_RELS = (
    "rules_like",
    "derived_from",
    "takes_precedence_over",
    "disputes",
    "objects_to",
    "qualifies",
    "aligns_with_side",
)
TIER_LEVEL = {
    "canonical": 1,
    "rishonim": 2,
    "codes": 3,
    "nosei_kelim": 3,
    "acharonim": 4,
    "later_responsa": 4,
}
STRUCTURAL_LINKS = (
    "part_of",
    "summarizes",
    "sub_issue_of",
    "position_on",
    "about_issue",
    "raised_in",
)
KNOWLEDGE_TOOL_NAMES = (
    "coverage",
    "get_sugya",
    "get_shitot",
    "get_psak_chain",
    "get_chakirot",
    "find_issues",
    "explain_abbrev",
    "get_links",
    "get_method_principles",
)
LENS_ONLY_TOOLS = ("get_chakirot", "get_method_principles")


def clip(text: str | None, n: int) -> str:
    words = (text or "").split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_S = {"type": "string"}
KL_SPECS: dict[str, dict] = {
    s["function"]["name"]: s
    for s in [
        _spec(
            "coverage",
            "Run FIRST. Is the question/ref in the ingested pilot scope (in_scope|partial|"
            "out_of_scope) and which core books were not checked. The corpus's silence is "
            "not 'no machloket'.",
            {"ref_or_query": _S},
            ["ref_or_query"],
        ),
        _spec(
            "get_sugya",
            "Map of a sugya at a ref: telegraphic steps (<=25 words) with status/evidence refs.",
            {"ref": _S, "with_readings": {"type": "boolean"}},
            ["ref"],
        ),
        _spec(
            "get_shitot",
            "Issues (nidonim), positions (<=20 words) with reception per context, sub-issues, "
            "machloket and source_claims (with their target). By ref, issue_key or query.",
            {"ref": _S, "issue_key": _S, "query": _S},
            [],
        ),
        _spec(
            "get_psak_chain",
            "Chain of poskim around a siman/ref, grouped by level, with relation kind, "
            "attribution and evidence origin.",
            {"siman_or_ref": _S},
            ["siman_or_ref"],
        ),
        _spec(
            "get_chakirot",
            "Chakira matrix (method x side) for verified chakirot; empty cell = 'אין מקור'.",
            {"ref": _S, "query": _S},
            [],
        ),
        _spec(
            "find_issues",
            "Find issues (nidonim) by free-text query.",
            {"query": _S, "limit": {"type": "integer"}},
            ["query"],
        ),
        _spec(
            "explain_abbrev",
            "How an abbreviation token in a segment was expanded (candidates, choice).",
            {"ref": _S, "token": _S},
            ["ref", "token"],
        ),
        _spec(
            "get_links",
            "Sefaria links at a ref (Ein Mishpat, Masoret HaShas, ...): CANDIDATES, not evidence.",
            {"ref": _S, "kinds": {"type": "array", "items": _S}},
            ["ref"],
        ),
        _spec(
            "get_method_principles",
            "Methodological lens (R. Asher Weiss): how to approach a sugya. Not evidence, not a "
            "position. Mention only as an attributed lens.",
            {"query": _S},
            [],
        ),
    ]
}


class KnowledgeTools:
    def __init__(self, session: Session, cfg: HarnessConfig, mode: str):
        self.session, self.cfg, self.mode = session, cfg, mode
        if cfg.phrases is None:
            raise ValueError("knowledge tools need [knowledge.phrases] in config/harness.toml")
        self.phrases = cfg.phrases
        self.ctx = KnowledgeContext(
            session=session, phrases=cfg.phrases, mode=mode, knowledge=cfg.knowledge
        )
        self._origin_cache: dict[int, str] = {}

    # ---- public --------------------------------------------------------------------------

    def specs(self) -> list[dict[str, Any]]:
        lens = self.cfg.knowledge == "base+lens"
        return [KL_SPECS[n] for n in KNOWLEDGE_TOOL_NAMES if lens or n not in LENS_ONLY_TOOLS]

    def has(self, name: str) -> bool:
        return name in KNOWLEDGE_TOOL_NAMES

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in LENS_ONLY_TOOLS and self.cfg.knowledge != "base+lens":
            return self._env(name, error=f"{name} is not available in knowledge=base")
        fn = getattr(self, name)
        try:
            inspect.signature(fn).bind(**arguments)
        except TypeError as e:  # wrong/missing arguments from the model
            return {"error": f"bad arguments for {name}: {e}"}
        return fn(**arguments)

    def _env(self, tool: str, **payload: Any) -> dict[str, Any]:
        return {
            "layer": "knowledge",
            "notice": self.phrases.notice,
            "tool": tool,
            "knowledge": self.cfg.knowledge,
            **payload,
        }

    # ---- filters and views ---------------------------------------------------------------

    def _vis(self) -> list:
        c = [
            KlItem.is_current,
            KlItem.stale.is_(False),
            KlItem.status.in_(("verified", "generated")),
            or_(
                KlItem.confidence >= self.cfg.min_confidence,
                and_(KlItem.confidence.is_(None), KlItem.status == "verified"),
            ),
        ]
        if self.mode == "public":
            c.append(KlItem.can_display_public)
        if not (self.mode == "personal" and self.cfg.allow_personal_only):
            c.append(or_(KlSource.license_tier.is_(None), KlSource.license_tier != PERSONAL))
        if self.cfg.knowledge != "base+lens":
            c.append(
                or_(
                    KlSource.work_id.is_(None),
                    and_(KlSource.role != "lens", KlSource.tier.notin_(SECONDARY_TIERS)),
                )
            )
        return c

    def _select(self, *where) -> sa.Select:
        return (
            select(KlItem, KlSource)
            .outerjoin(KlSource, KlSource.work_id == KlItem.work_id)
            .where(*self._vis(), *where)
        )

    def _items(self, *where, limit: int | None = None) -> list[tuple[KlItem, KlSource | None]]:
        stmt = self._select(*where).order_by(KlItem.id)
        if limit:
            stmt = stmt.limit(limit)
        return [(i, s) for i, s in self.session.execute(stmt).all()]

    def _segment_ids(self, ref: str) -> list[int]:
        return list(
            self.session.scalars(
                select(Segment.id).where(or_(Segment.ref == ref, Segment.parent_ref == ref))
            )
        )

    def _items_at(self, ref: str, kinds: tuple[str, ...]) -> list[tuple[KlItem, KlSource | None]]:
        seg_ids = self._segment_ids(ref)
        if not seg_ids:
            return []
        member = select(KlItemSegment.item_id).where(KlItemSegment.segment_id.in_(seg_ids))
        return self._items(
            KlItem.kind.in_(kinds),
            or_(KlItem.id.in_(member), KlItem.start_segment_id.in_(seg_ids)),
        )

    def _search(self, query: str, kind: str, limit: int) -> list[tuple[KlItem, KlSource | None]]:
        toks = []
        for t in normalize(query, "B").text.split():
            if re.fullmatch(r"\w{2,}", t) and t not in toks:
                toks.append(t)
        if not toks:
            return []
        tsq = sa.func.to_tsquery("simple", " | ".join(toks[:12]))
        tsv = sa.func.to_tsvector("simple", sa.func.coalesce(KlItem.search_text_he, ""))
        rank = sa.func.ts_rank(tsv, tsq)
        rows = self.session.execute(
            self._select(KlItem.kind == kind, tsv.op("@@")(tsq))
            .order_by(rank.desc(), KlItem.id)
            .limit(max(1, min(limit, 25)))
        ).all()
        return [(i, s) for i, s in rows]

    def _evidence_refs(self, item_ids: list[int]) -> dict[int, list[tuple[str, str | None]]]:
        out: dict[int, list[tuple[str, str | None]]] = defaultdict(list)
        if not item_ids:
            return out
        rows = self.session.execute(
            select(KlEvidence.item_id, Segment.ref, KlEvidence.claim_id)
            .join(Segment, Segment.id == KlEvidence.segment_id)
            .where(KlEvidence.item_id.in_(item_ids), KlEvidence.locate_status == "located")
            .order_by(KlEvidence.id)
        ).all()
        for iid, ref, claim in rows:
            if (ref, claim) not in out[iid]:
                out[iid].append((ref, claim))
        return out

    def _promoted(self, items: list[KlItem]) -> dict[int, int | None]:
        ids = {i.job_id for i in items if i.job_id}
        promoted = (
            set(
                self.session.scalars(
                    select(KlJob.id).where(KlJob.id.in_(ids), KlJob.status == "promoted")
                )
            )
            if ids
            else set()
        )
        return {i.id: (i.job_id if i.job_id in promoted else None) for i in items}

    def _origin_of(self, *item_ids: int) -> str:
        """Weakest evidence_origin of the given ends (llm < judge < human)."""
        return min((self._origin(i) for i in item_ids), key=ORIGIN_RANK.__getitem__)

    def _origin(self, item_id: int) -> str:
        """evidence_origin of an item: llm | judge | human. The schema has no column for it,
        so it is derived: human reviewer/override > judge check > llm."""
        if item_id in self._origin_cache:
            return self._origin_cache[item_id]
        it = self.session.get(KlItem, item_id)
        origin = "llm"
        if it is not None:
            overridden = self.session.scalars(
                select(KlOverride.id).where(
                    KlOverride.target_key == it.item_key,
                    KlOverride.action.in_(("approve", "edit")),
                )
            ).first()
            if it.prompt_version == "human" or overridden:
                origin = "human"
            elif (it.check_report or {}).get("judge"):
                origin = "judge"
        self._origin_cache[item_id] = origin
        return origin

    def _views(
        self, rows: list[tuple[KlItem, KlSource | None]], tool: str
    ) -> dict[int, dict[str, Any]]:
        evs = self._evidence_refs([i.id for i, _ in rows])
        promoted = self._promoted([i for i, _ in rows])
        out: dict[int, dict[str, Any]] = {}
        for it, src in rows:
            refs = list(dict.fromkeys(r for r, _ in evs.get(it.id, [])))
            tier = src.tier if src else None
            out[it.id] = {
                "item_id": it.id,
                "key": it.item_key,
                "kind": it.kind,
                "version": it.version,
                "tier": tier,
                "status": it.status,
                "confidence": it.confidence,
                "evidence_refs": refs[:12],
                "promoted_job": promoted[it.id],
            }
            self.ctx.note_item(
                it.id, it.item_key, it.version, tool, tier, src.role if src else None
            )
            self.ctx.layer_refs.update(refs)
        return out

    # ---- relations -----------------------------------------------------------------------

    def _relation_evidence(self, r: KlRelation) -> list[str]:
        ids = [i for i in (r.evidence_ids or []) if isinstance(i, int)]
        if not ids:
            return []
        refs = self.session.scalars(
            select(Segment.ref)
            .join(KlEvidence, KlEvidence.segment_id == Segment.id)
            .where(KlEvidence.id.in_(ids))
            .order_by(KlEvidence.id)
        ).all()
        refs = list(dict.fromkeys(refs))
        self.ctx.layer_refs.update(refs)
        return refs

    def _admit_ends(self, ends: set[int], visible: set[int], tool: str) -> set[int]:
        """``visible`` plus those of ``ends`` that pass the filter (an end that is filtered out
        must not leak through a relation or link)."""
        others = ends - visible
        if not others:
            return visible
        rows = self._items(KlItem.id.in_(others))
        self._views(rows, tool)
        return visible | {i.id for i, _ in rows}

    def _relations_of(
        self, ids: set[int], visible: set[int], tool: str = "get_shitot"
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        rels = self.session.scalars(
            select(KlRelation)
            .where(
                KlRelation.is_current,
                or_(KlRelation.from_item_id.in_(ids), KlRelation.to_item_id.in_(ids)),
            )
            .order_by(KlRelation.id)
        ).all()
        visible = self._admit_ends(
            {x for r in rels for x in (r.from_item_id, r.to_item_id)}, visible, tool
        )
        out = []
        for r in rels:
            if r.from_item_id not in visible or r.to_item_id not in visible:
                continue  # an end that is filtered out must not leak through a relation
            origin = self._origin_of(r.from_item_id, r.to_item_id)
            ev = self._relation_evidence(r)
            view = {
                "table": "kl_relations",
                "id": r.id,
                "rel": r.rel,
                "from_item": r.from_item_id,
                "to_item": r.to_item_id,
                "attribution": r.attribution,
                "stance": r.stance,
                "evidence_origin": origin,
                "candidate": origin == "sefaria",
                "evidence_refs": ev,
            }
            self.ctx.relations_seen[("kl_relations", r.id)] = {**view, "endpoints": ev}
            out.append(view)
        return out

    def _link_views(
        self, ids: set[int], visible: set[int], tool: str = "get_shitot"
    ) -> list[dict[str, Any]]:
        """Non-structural kl_links (depends_on, discusses, continues) touching ``ids``, shown with
        their kind and evidence and explicitly *not* the same nidon."""
        if not ids:
            return []
        links = self.session.scalars(
            select(KlLink)
            .where(
                KlLink.rel.notin_(STRUCTURAL_LINKS),
                or_(KlLink.from_item_id.in_(ids), KlLink.to_item_id.in_(ids)),
            )
            .order_by(KlLink.id)
        ).all()
        visible = self._admit_ends(
            {x for lk in links for x in (lk.from_item_id, lk.to_item_id)}, visible, tool
        )
        out = []
        for lk in links:
            if lk.from_item_id not in visible or lk.to_item_id not in visible:
                continue
            ev: list[str] = []
            for e in (lk.from_item_id, lk.to_item_id):
                for ref, _ in self._evidence_refs([e]).get(e, []):
                    if ref not in ev:
                        ev.append(ref)
            self.ctx.layer_refs.update(ev)
            origin = self._origin_of(lk.from_item_id, lk.to_item_id)
            view = {
                "table": "kl_links",
                "id": lk.id,
                "rel": lk.rel,
                "from_item": lk.from_item_id,
                "to_item": lk.to_item_id,
                "attribution": None,
                "stance": None,
                "evidence_origin": origin,
                "candidate": origin == "sefaria",
                "evidence_refs": ev[:12],
                "same_nidon": False,
            }
            self.ctx.relations_seen[("kl_links", lk.id)] = {**view, "endpoints": ev}
            out.append(view)
        return out

    # ---- tools ---------------------------------------------------------------------------

    def find_issues(self, query: str, limit: int = 10) -> dict[str, Any]:
        rows = self._search(query, "issue", limit)
        views = self._views(rows, "find_issues")
        items = []
        for it, _ in rows:
            v = views[it.id]
            v["scope_key"] = it.scope_key
            v["question"] = clip((it.body or {}).get("canonical_q_he") or it.title_he, 20)
            items.append(v)
        return self._env("find_issues", query=query, items=items)

    def get_sugya(self, ref: str, with_readings: bool = False) -> dict[str, Any]:
        rows = self._items_at(ref, ("sugya",))
        vs = self._views(rows, "get_sugya")
        readings_ok = with_readings and self.cfg.knowledge == "base+lens"
        out = []
        for sg, _ in rows:
            v = vs[sg.id]
            v["title"] = clip(sg.title_he, 12)
            sums = self._items(
                KlItem.kind == "sugya_summary",
                KlItem.id.in_(
                    select(KlLink.from_item_id).where(
                        KlLink.to_item_id == sg.id, KlLink.rel == "summarizes"
                    )
                ),
            )
            sv = self._views(sums, "get_sugya")
            v["summaries"] = []
            for sm, _s in sums:
                evs = self._evidence_refs([sm.id]).get(sm.id, [])
                steps, groups = [], []
                for i, st in enumerate((sm.body or {}).get("steps", [])):
                    pref = f"steps[{i}]."
                    step = {
                        "id": st.get("id"),
                        "move": st.get("move"),
                        "subtype": st.get("subtype"),
                        "responds_to": st.get("responds_to"),
                        "standing": st.get("standing"),
                        "text": clip(st.get("text_he"), STEP_WORDS),
                        "evidence_refs": list(
                            dict.fromkeys(r for r, c in evs if c and c.startswith(pref))
                        ),
                    }
                    rds = st.get("readings") or []
                    if rds and readings_ok:
                        step["readings"] = [
                            {
                                "holder": (r.get("holder") or {}).get("label"),
                                "text": clip(r.get("text_he"), STEP_WORDS),
                            }
                            for r in rds
                        ]
                        if len(rds) > 1:
                            groups.append(
                                [
                                    (r.get("holder") or {}).get("label")
                                    or clip(r.get("text_he"), 4)
                                    for r in rds
                                ]
                            )
                    elif rds:
                        step["readings_hidden"] = len(rds)
                    steps.append(step)
                self.ctx.reading_groups[sm.id] = groups
                v["summaries"].append({**sv[sm.id], "steps": steps})
            out.append(v)
        extra = {}
        if with_readings and not readings_ok:
            extra["readings_note"] = "readings are available only with knowledge=base+lens"
        return self._env("get_sugya", ref=ref, items=out, **extra)

    def _reception(self, ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        out: dict[int, list[dict[str, Any]]] = defaultdict(list)
        if not ids:
            return out
        rows = self.session.scalars(
            select(KlReception)
            .where(KlReception.target_item_id.in_(ids))
            .order_by(KlReception.computed_at.desc(), KlReception.id.desc())
        ).all()
        seen = set()
        for r in rows:
            k = (r.target_item_id, r.target_claim, r.context)
            if k in seen:
                continue
            seen.add(k)
            basis = r.basis or {}
            out[r.target_item_id].append(
                {
                    "context": r.context,
                    "weight_label": r.weight_label,
                    "basis": basis,
                    "unusual_interpretation": bool(basis.get("unusual_interpretation")),
                }
            )
        return out

    def _issues_for(self, ref, issue_key, query) -> list[tuple[KlItem, KlSource | None]]:
        if issue_key:
            return self._items(KlItem.item_key == issue_key, KlItem.kind == "issue")
        if query:
            return self._search(query, "issue", 5)
        if ref:
            rows = {i.id: (i, s) for i, s in self._items_at(ref, ("issue",))}
            pos = [p.id for p, _ in self._items_at(ref, ("position", "open_question"))]
            if pos:  # issues reached through positions anchored at the ref
                issue_ids = select(KlLink.to_item_id).where(
                    KlLink.from_item_id.in_(pos), KlLink.rel.in_(("position_on", "about_issue"))
                )
                for i, s in self._items(KlItem.kind == "issue", KlItem.id.in_(issue_ids)):
                    rows[i.id] = (i, s)
            return list(rows.values())
        return []

    def get_shitot(
        self, ref: str | None = None, issue_key: str | None = None, query: str | None = None
    ) -> dict[str, Any]:
        issues = self._issues_for(ref, issue_key, query)[:5]
        issue_views = self._views(issues, "get_shitot")
        visible: set[int] = set(issue_views)
        blocks = []
        for iss, _ in issues:
            links = self.session.execute(
                select(KlLink.from_item_id, KlLink.rel).where(
                    KlLink.to_item_id == iss.id,
                    KlLink.rel.in_(("position_on", "about_issue", "sub_issue_of")),
                )
            ).all()
            pos_ids = [f for f, r in links if r in ("position_on", "about_issue")]
            sub_ids = [f for f, r in links if r == "sub_issue_of"]
            pos_rows = self._items(
                KlItem.kind == "position",
                or_(KlItem.id.in_(pos_ids), KlItem.body["issue"].astext == iss.item_key),
            )
            sub_rows = self._items(KlItem.id.in_(sub_ids), KlItem.kind == "issue")
            mk_rows = self._items(
                KlItem.kind == "machloket", KlItem.body["issue_id"].astext == iss.item_key
            )
            oq_rows = self._items(
                KlItem.kind == "open_question", KlItem.body["issue"].astext == iss.item_key
            )
            views = [self._views(r, "get_shitot") for r in (pos_rows, sub_rows, mk_rows, oq_rows)]
            for d in views:
                visible |= set(d)
            blocks.append((iss, pos_rows, sub_rows, mk_rows, oq_rows, *views))
        ids = set(visible)
        rec = self._reception(sorted(ids))
        sc_rows = (
            self._items(
                KlItem.kind == "source_claim",
                or_(
                    KlItem.id.in_(select(KlLink.from_item_id).where(KlLink.to_item_id.in_(ids))),
                    KlItem.id.in_(
                        select(KlRelation.from_item_id).where(
                            KlRelation.is_current, KlRelation.to_item_id.in_(ids)
                        )
                    ),
                ),
            )
            if ids
            else []
        )
        scv = self._views(sc_rows, "get_shitot")
        visible |= set(scv)
        rels = self._relations_of(ids | set(scv), visible)
        link_views = self._link_views(ids, visible)
        result = []
        for iss, pos_rows, sub_rows, mk_rows, oq_rows, pv, sv, mv, ov in blocks:
            positions = []
            for p, _ in pos_rows:
                b = p.body or {}
                holder = b.get("holder") or {}
                receptions = rec.get(p.id, [])
                unusual = any(r["unusual_interpretation"] for r in receptions) or bool(
                    b.get("unusual_interpretation")
                )
                if unusual:
                    self.ctx.unusual_items.add(p.id)
                if b.get("normative_level") == "documented_practice":
                    self.ctx.documented_practice[p.id] = set(pv[p.id]["evidence_refs"])
                positions.append(
                    {
                        **pv[p.id],
                        "holder": holder.get("label"),
                        "known_via": holder.get("known_via"),
                        "view": clip(b.get("view_he"), POSITION_WORDS),
                        "side_label": b.get("side_label"),
                        "normative_level": b.get("normative_level"),
                        "reception": receptions,
                        "unusual_interpretation": unusual,
                        "relations": [r for r in rels if p.id in (r["from_item"], r["to_item"])],
                    }
                )
            result.append(
                {
                    **issue_views[iss.id],
                    "question": clip((iss.body or {}).get("canonical_q_he"), 20),
                    "positions": positions,
                    "sub_issues": [
                        {**sv[s.id], "question": clip((s.body or {}).get("canonical_q_he"), 20)}
                        for s, _ in sub_rows
                    ],
                    "machloket": [
                        {
                            **mv[m.id],
                            "sides": (m.body or {}).get("sides"),
                            "crux": clip((m.body or {}).get("crux_he"), 20),
                        }
                        for m, _ in mk_rows
                    ],
                    "open_questions": [
                        {
                            **ov[q.id],
                            "q_kind": (q.body or {}).get("kind"),
                            "text": clip((q.body or {}).get("text_he"), 20),
                        }
                        for q, _ in oq_rows
                    ],
                    "related_issues": [
                        lk for lk in link_views if iss.id in (lk["from_item"], lk["to_item"])
                    ],
                }
            )
        source_claims = []
        for sc, _ in sc_rows:
            b = sc.body or {}
            t = b.get("target") or {}
            source_claims.append(
                {
                    **scv[sc.id],
                    "claim_kind": b.get("claim_kind"),
                    "holder": (b.get("holder") or {}).get("label"),
                    "target": {  # shown together with the claim, never hidden behind it
                        "work": t.get("work"),
                        "ref": t.get("ref"),
                        "quote": clip(t.get("quote"), 20),
                        "target_status": t.get("target_status"),
                    },
                    "proposed_alternative": clip(b.get("proposed_alternative"), 20),
                    "responses": len(b.get("responses") or []),
                }
            )
        return self._env("get_shitot", issues=result, source_claims=source_claims, relations=rels)

    def get_psak_chain(self, siman_or_ref: str) -> dict[str, Any]:
        text = siman_or_ref.strip()
        n = int(text) if text.isdigit() else hebrew_number(text)
        if n:
            refs = [f"{t}, Orach Chayim {n}" for t in ("Shulchan Arukh", "Tur", "Beit Yosef")]
        else:
            refs = extract_refs(text) or [text]
        start: dict[int, tuple[KlItem, KlSource | None]] = {}
        for r in refs:
            for i, s in self._items_at(
                r, ("position", "issue", "open_question", "source_claim", "commentary_note")
            ):
                start[i.id] = (i, s)
        if not start:
            return self._env("get_psak_chain", ref=siman_or_ref, levels=[], relations=[])
        depth = {
            r[0]: r[1]
            for r in self.session.execute(
                sa.text(
                    """
                    WITH RECURSIVE chain(item_id, depth, path) AS (
                      SELECT id, 0, ARRAY[id] FROM kl_items WHERE id = ANY(:start)
                      UNION ALL
                      SELECT o.other, c.depth + 1, c.path || o.other
                      FROM chain c
                      JOIN kl_relations r ON r.is_current AND r.rel = ANY(:rels)
                        AND (r.from_item_id = c.item_id OR r.to_item_id = c.item_id)
                      CROSS JOIN LATERAL (SELECT CASE WHEN r.from_item_id = c.item_id
                        THEN r.to_item_id ELSE r.from_item_id END AS other) o
                      WHERE c.depth < 6 AND NOT (o.other = ANY(c.path))
                    )
                    SELECT item_id, MIN(depth) FROM chain GROUP BY item_id
                    """
                ),
                {"start": list(start), "rels": list(PSAK_RELS)},
            )
        }
        rows = self._items(KlItem.id.in_(list(depth)))
        views = self._views(rows, "get_psak_chain")
        by_level: dict[int, list[dict]] = defaultdict(list)
        names: dict[int, str] = {}
        for it, src in rows:
            lvl = TIER_LEVEL.get(src.tier, 9) if src else 9
            names[lvl] = src.tier if src else "unknown"
            b = it.body or {}
            by_level[lvl].append(
                {
                    **views[it.id],
                    "depth": depth[it.id],
                    "holder": (b.get("holder") or {}).get("label"),
                    "view": clip(b.get("view_he") or b.get("text_he"), POSITION_WORDS),
                }
            )
        rels = [
            r
            for r in self._relations_of(set(views), set(views), "get_psak_chain")
            if r["rel"] in PSAK_RELS
        ]
        levels = [
            {"level": lvl, "tier": names[lvl], "items": by_level[lvl]} for lvl in sorted(by_level)
        ]
        return self._env("get_psak_chain", ref=siman_or_ref, levels=levels, relations=rels)

    def get_chakirot(self, ref: str | None = None, query: str | None = None) -> dict[str, Any]:
        rows = (
            self._items_at(ref, ("chakira",))
            if ref
            else self._search(query, "chakira", 10)
            if query
            else []
        )
        views = self._views(rows, "get_chakirot")
        matrices, retrieval_only = [], []
        for it, _ in rows:
            b = it.body or {}
            status = b.get("cluster_status") or (it.check_report or {}).get("cluster_status")
            if status != "verified":  # unverified clusters are for retrieval only
                retrieval_only.append(
                    {**views[it.id], "question": clip(b.get("question_he"), 20), "structure": None}
                )
                continue
            evs = self._evidence_refs([it.id]).get(it.id, [])
            sides = b.get("sides") or []
            by_holder: dict[str, dict[int, dict]] = defaultdict(dict)
            for si, side in enumerate(sides):
                srefs = list(dict.fromkeys(r for r, c in evs if c and c.startswith(f"sides[{si}]")))
                prop = (side.get("proposer") or {}).get("label")
                if prop:
                    by_holder[prop][si] = {"role": "proposer", "evidence_refs": srefs}
                for a in side.get("aligned") or []:
                    if a.get("label"):
                        by_holder[a["label"]][si] = {
                            "role": "aligned",
                            "evidence_refs": srefs,
                            "asserted_by": (side.get("asserted_by") or {}).get("label"),
                        }
            matrices.append(
                {
                    **views[it.id],
                    "question": clip(b.get("question_he"), 20),
                    "sides": [clip(s.get("label_he"), 12) for s in sides],
                    "matrix": [
                        {
                            "method": holder,
                            "cells": [cells.get(i, NO_SOURCE) for i in range(len(sides))],
                        }
                        for holder, cells in by_holder.items()
                    ],
                }
            )
        return self._env(
            "get_chakirot", ref=ref, query=query, items=matrices, retrieval_only=retrieval_only
        )

    def explain_abbrev(self, ref: str, token: str) -> dict[str, Any]:
        want = normalize(token, "B").text.replace(" ", "")
        seg_ids = self._segment_ids(ref)
        out = []
        if seg_ids:
            rows = self.session.execute(
                select(Annotation, SegmentText.text_raw, Segment.ref)
                .join(Segment, Segment.id == Annotation.segment_id)
                .join(
                    SegmentText,
                    and_(
                        SegmentText.segment_id == Annotation.segment_id,
                        SegmentText.version_id == Annotation.version_id,
                    ),
                )
                .where(
                    Annotation.segment_id.in_(seg_ids),
                    Annotation.kind == "abbrev",
                    Annotation.is_current,
                )
                .order_by(Annotation.id)
            ).all()
            for a, raw, sref in rows:
                span = raw[a.span_start : a.span_end]
                if normalize(span, "B").text.replace(" ", "") != want:
                    continue
                out.append(
                    {
                        "ref": sref,
                        "token": span,
                        "chosen": a.chosen,
                        "candidates": [c.get("text") for c in (a.candidates or [])][:6],
                        "origin": a.origin,
                        "method": a.method,
                        "status": a.status,
                        "confidence": a.confidence,
                    }
                )
        return self._env("explain_abbrev", ref=ref, token=token, items=out)

    def get_links(self, ref: str, kinds: list[str] | None = None) -> dict[str, Any]:
        seg_ids = self._segment_ids(ref)
        if not seg_ids:
            return self._env("get_links", ref=ref, items=[])
        q = select(Edge).where(or_(Edge.segment_a_id.in_(seg_ids), Edge.segment_b_id.in_(seg_ids)))
        if kinds:
            q = q.where(Edge.edge_type.in_([("" if k == "untyped" else k) for k in kinds]))
        edges = self.session.scalars(q.order_by(Edge.id).limit(40)).all()
        ends = {e.segment_a_id for e in edges} | {e.segment_b_id for e in edges}
        refs = dict(
            self.session.execute(select(Segment.id, Segment.ref).where(Segment.id.in_(ends))).all()
        )
        items = []
        for e in edges:
            ra, rb = refs.get(e.segment_a_id), refs.get(e.segment_b_id)
            cand = is_candidate(e.origin)
            origin = e.origin or "unknown"
            view = {
                "table": "edges",
                "id": e.id,
                "rel": e.edge_type or "untyped",
                "edge_type": e.edge_type or "untyped",
                "ref_a": ra,
                "ref_b": rb,
                "citation_a": e.citation_a,
                "citation_b": e.citation_b,
                "candidate": cand,
                "evidence_origin": origin,
                "evidence_refs": [],
                "label": candidate_label(e.origin) if cand else None,
            }
            self.ctx.relations_seen[("edges", e.id)] = {
                **view,
                "endpoints": [r for r in (ra, rb) if r],
            }
            items.append(view)
        return self._env("get_links", ref=ref, items=items)

    def get_method_principles(self, query: str | None = None) -> dict[str, Any]:
        # PERSONAL_ONLY_RESTRICTED: base+lens, personal mode and allow_personal_only; else []
        if not (
            self.cfg.knowledge == "base+lens"
            and self.mode == "personal"
            and self.cfg.allow_personal_only
        ):
            return self._env("get_method_principles", items=[])
        rows = (
            self._search(query, "method_principle", 10)
            if query
            else self._items(KlItem.kind == "method_principle", limit=10)
        )
        views = self._views(rows, "get_method_principles")
        items = [
            {
                **views[it.id],
                "principle": clip((it.body or {}).get("principle_he"), POSITION_WORDS),
                "source": ((it.body or {}).get("source") or {}).get("label"),
                "role": "lens: דרך גישה, לא ראיה ולא עמדה",
            }
            for it, _ in rows
        ]
        return self._env("get_method_principles", query=query, items=items)

    # ---- coverage and the scope gate -----------------------------------------------------

    def _known_gaps(self, refs: list[str]) -> list[str]:
        """config/known_gaps.toml (ingest PR 3): external gaps, `[[gap]] work, scope, ...`. A gap
        applies when its scope key covers a ref, or "whole work" of a commentary on the ref's
        tractate ("Ritva on Shabbat" for "Shabbat 22b")."""
        if not KNOWN_GAPS_PATH.exists():
            return []
        with open(KNOWN_GAPS_PATH, "rb") as f:
            gaps = tomllib.load(f).get("gap", [])
        out = []
        for g in gaps:
            scope = str(g.get("scope", ""))
            work = g.get("work")
            if not work:
                continue
            if scope.startswith("whole work"):
                hit = any(work.endswith(" on " + r.rsplit(" ", 1)[0]) for r in refs)
            else:
                key = scope.split(" (")[0]
                hit = any(key_covers_ref(key, r) for r in refs)
            if hit:
                out.append(work)
        return out

    def _unchecked_core(
        self, refs: list[str], scope_keys: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        """(titles, title -> name shown in the answer text: he_title when there is one)."""
        rows = self.session.execute(
            select(KlCoverage.scope_key, Work.title, Work.he_title)
            .join(Work, Work.id == KlCoverage.work_id)
            .join(KlSource, KlSource.work_id == KlCoverage.work_id)
            .where(
                KlSource.weight_tier == "core",
                KlSource.role != "reference_only",
                KlCoverage.status.in_(("not_ingested", "unknown")),
            )
            .order_by(Work.title)
        ).all()
        hit = [r for r in rows if r[0] in scope_keys or any(key_covers_ref(r[0], x) for x in refs)]
        display = {title: (he or title) for _, title, he in hit}
        titles = [title for _, title, _ in hit]
        for work in self._known_gaps(refs):
            titles.append(work)
            display.setdefault(work, work)
        titles = list(dict.fromkeys(titles))
        return titles, {t: display[t] for t in titles}

    def _confirmed_scope_keys(self, issue_keys: list[str], refs: list[str]) -> list[str]:
        """Scope keys of in-scope issues, plus the kl scope keys that cover an in-scope ref;
        only keys that are themselves inside the pilot scope."""
        rules = self.cfg.scope
        found = set(issue_keys)
        if refs:
            all_keys = self.session.scalars(
                select(KlItem.scope_key).where(KlItem.is_current).distinct()
            ).all()
            found |= {k for k in all_keys if any(key_covers_ref(k, r) for r in refs)}
        return sorted(k for k in found if scope_key_in_scope(k, rules))

    def _gate(self, text: str) -> ScopeGate:
        rules = self.cfg.scope
        refs = extract_refs(text)
        issues = self.find_issues(text, 5)["items"] if text.strip() else []
        gate = ScopeGate(
            "out_of_scope",
            refs=refs,
            issues=[{"key": i["key"], "scope_key": i["scope_key"]} for i in issues],
            practical=is_practical(text, self.phrases.practical),
        )
        gate.refs_in_scope = [r for r in refs if ref_in_scope(r, rules)]
        gate.refs_out_of_scope = [r for r in refs if r not in gate.refs_in_scope]
        issues_in = [i for i in gate.issues if scope_key_in_scope(i["scope_key"], rules)]
        if refs:
            if gate.refs_in_scope and not gate.refs_out_of_scope:
                gate.status = "in_scope"
            elif gate.refs_in_scope:
                gate.status = "partial"
                gate.reasons.append("חלק מהמקורות שבשאלה מחוץ לתחום")
        elif issues_in:
            gate.status = "in_scope" if len(issues_in) == len(gate.issues) else "partial"
        if gate.status == "out_of_scope":
            gate.reasons.append("אין בשאלה מקור בתחום הפיילוט, ולא נמצא נידון בתחום")
            return gate
        keys = [i["scope_key"] for i in issues_in]
        gate.scope_keys = self._confirmed_scope_keys(keys, gate.refs_in_scope)
        gate.unchecked_core_books, gate.unchecked_display = self._unchecked_core(
            gate.refs_in_scope, keys
        )
        if gate.unchecked_core_books and gate.status == "in_scope":
            gate.status = "partial"
            gate.reasons.append("ספרי core שלא נבדקו")
        return gate

    def coverage(self, ref_or_query: str) -> dict[str, Any]:
        gate = self._gate(ref_or_query)
        return self._env("coverage", **gate.as_dict())

    def run_scope_gate(self, question: str) -> ScopeGate:
        gate = self._gate(question)
        self.ctx.set_gate(gate)
        return gate
