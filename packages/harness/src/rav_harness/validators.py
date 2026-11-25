"""Deterministic knowledge-layer checks on a submitted Answer (KL-2, checks 4a-4j).

Every failure is one message that starts with a stable ``[CODE]`` and tells the model what to do.
No LLM here. The context (what the knowledge tools returned in this run) is filled by
``KnowledgeTools``; with no context (knowledge=off) nothing here runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rav_ai.db.models import KlItem, KlSource, KlWitness, Segment
from rav_evals.schema import Answer, Claim, ClaimType
from sqlalchemy import select
from sqlalchemy.orm import Session

from .scope import ScopeGate

# ---- rejection codes ----------------------------------------------------------------------
KL_UNREAD_EVIDENCE_REF = "KL_UNREAD_EVIDENCE_REF"  # 4a
KL_LENS_ROLE = "KL_LENS_ROLE"  # 4b
KL_LENS_ONLY_RISHON = "KL_LENS_ONLY_RISHON"  # 4b
KL_RELATION_FIELDS = "KL_RELATION_FIELDS"  # 4c
KL_RELATION_UNKNOWN = "KL_RELATION_UNKNOWN"  # 4c
KL_RELATION_MISMATCH = "KL_RELATION_MISMATCH"  # 4c
KL_CANDIDATE_ONLY = "KL_CANDIDATE_ONLY"  # 4c
KL_GIRSA_WITNESS = "KL_GIRSA_WITNESS"  # 4d
KL_COVERAGE_BOOKS = "KL_COVERAGE_BOOKS"  # 4e
KL_SILENCE_IS_NOT_AGREEMENT = "KL_SILENCE_IS_NOT_AGREEMENT"  # 4e
KL_SCOPE_MISMATCH = "KL_SCOPE_MISMATCH"  # 4f
KL_OUT_OF_SCOPE_CONCLUSION = "KL_OUT_OF_SCOPE_CONCLUSION"  # 4f
KL_SCOPE_NOTICE = "KL_SCOPE_NOTICE"  # 4f
KL_UNUSUAL_FLAG = "KL_UNUSUAL_FLAG"  # 4g
KL_MARAN_INSTRUCTIONS = "KL_MARAN_INSTRUCTIONS"  # 4h
KL_SA_PSAK = "KL_SA_PSAK"  # 4i
KL_SA_PRACTICAL = "KL_SA_PRACTICAL"  # 4i
KL_READINGS = "KL_READINGS"  # 4j
KL_UNKNOWN_ITEM = "KL_UNKNOWN_ITEM"

ALL_CODES = (
    KL_UNREAD_EVIDENCE_REF,
    KL_LENS_ROLE,
    KL_LENS_ONLY_RISHON,
    KL_RELATION_FIELDS,
    KL_RELATION_UNKNOWN,
    KL_RELATION_MISMATCH,
    KL_CANDIDATE_ONLY,
    KL_GIRSA_WITNESS,
    KL_COVERAGE_BOOKS,
    KL_SILENCE_IS_NOT_AGREEMENT,
    KL_SCOPE_MISMATCH,
    KL_OUT_OF_SCOPE_CONCLUSION,
    KL_SCOPE_NOTICE,
    KL_UNUSUAL_FLAG,
    KL_MARAN_INSTRUCTIONS,
    KL_SA_PSAK,
    KL_SA_PRACTICAL,
    KL_READINGS,
    KL_UNKNOWN_ITEM,
)

ORIGINS = ("sefaria", "llm", "judge", "human", "heuristic")
UNCHECKED_PHRASE = "ספרים מרכזיים שלא נבדקו כאן:"
OUT_OF_SCOPE_PHRASE = "מחוץ למה שנקלט ונבדק"
PARTIAL_PHRASE = "כיסוי חלקי"
UNUSUAL_PHRASE = "פירוש חריג"
NO_DISPUTANT_PHRASE = "לא נמצא חולק בקורפוס שנבדק"
MORE_READINGS_PHRASE = "יש הסברים נוספים"
CANDIDATE_MSG = 'קישור עין משפט/מסורת הש"ס הוא מועמד, לא ראיה'


def _n(s: str) -> str:
    return (s or "").replace("״", '"').replace("׳", "'").replace("”", '"').replace("’", "'")


_GIRSA_RE = re.compile(
    r'טעות סופר|צ"ל|ליתא|גרס[הא]|גירס[הא]|מגי[הע]|הגה[הת]|שגוי|טעות ב(?:דפוס|נוסח|גרסה)'
    r'|(?:ה?דפוס|ה?נוסח|ה?כ"י)\s+(?:טועה|מוטעה)'
)
_GIRSA_FORBIDDEN_RE = re.compile(
    r"(?:ה?דפוס|ה?נוסח)\s+(?:שגוי|טועה|מוטעה)|טעות ב(?:דפוס|נוסח)|טעות סופר ב(?:דפוס|נוסח)"
)
_HOLDS_RE = re.compile(r'סובר|סבור|סוברים|ס"ל|פוסק|שיטתו|מכריע|דעתו|לשיטת')
_LENS_MARK_RE = re.compile(r"לפי הבנת|הבנת|לפי פירוש|לפי ביאור|מבאר|מפרש|קורא|לפי דברי|מבין")
_MARAN_RE = re.compile(r"(?:קיבלנו|קבלנו|נוהגים לפי)\s+(?:את\s+)?הוראות\s+מרן|הוראות מרן")
_SA_PSAK_RE = re.compile(
    r'(?:ה?שו"ע|השולחן ערוך|מרן)\s+(?:פסק|פוסק)[^.]{0,80}?ולכן'
    r'|נדחה מפני (?:ה?שו"ע|השולחן ערוך|מרן)'
    r'|(?:מאחר|כיוון|היות) ש(?:ה?שו"ע|מרן) פסק'
    r'|הלכה כ(?:שו"ע|שולחן ערוך) ולכן'
)
_SILENCE_RE = re.compile(r"אין (?:מחלוקת|חולק)|לא נחלקו|דעה אחת בלבד|כולם מסכימים")
_DISPUTANT_RE = re.compile(r"חולק|חולקים|חלוק")


@dataclass
class KnowledgeContext:
    """What the knowledge tools returned during one run (the validator's only input besides the
    answer). Layer refs are evidence pointers: seen, never read."""

    session: Session
    mode: str = "personal"
    knowledge: str = "base"
    layer_refs: set[str] = field(default_factory=set)
    items_seen: dict[int, dict[str, Any]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)  # runs.knowledge_items
    relations_seen: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    reading_groups: dict[int, list[list[str]]] = field(default_factory=dict)
    unusual_items: set[int] = field(default_factory=set)
    documented_practice: dict[int, set[str]] = field(default_factory=dict)
    gate: ScopeGate | None = None
    _role_cache: dict[str, str | None] = field(default_factory=dict)

    def note_item(
        self, item_id: int, key: str, version: int, tool: str, tier: str | None, role: str | None
    ) -> None:
        self.items_seen[item_id] = {"key": key, "tier": tier, "role": role}
        rec = {"item_id": item_id, "version": version, "tool": tool, "tier": tier}
        if rec not in self.records:
            self.records.append(rec)

    def set_gate(self, gate: ScopeGate) -> None:
        self.gate = gate

    def source_role(self, ref: str) -> str | None:
        """kl_sources.role of the work a ref belongs to (None: unregistered)."""
        if ref not in self._role_cache:
            self._role_cache[ref] = self.session.scalars(
                select(KlSource.role)
                .join(Segment, Segment.work_id == KlSource.work_id)
                .where(Segment.ref == ref)
                .limit(1)
            ).first()
        return self._role_cache[ref]

    def witness_ok(self, witness_id: int) -> bool:
        """The witness exists and a current girsa_variant item points at its siglum."""
        w = self.session.get(KlWitness, witness_id)
        if w is None:
            return False
        return (
            self.session.scalars(
                select(KlItem.id).where(
                    KlItem.kind == "girsa_variant",
                    KlItem.is_current,
                    KlItem.status != "rejected",
                    KlItem.body["witness_siglum"].astext == w.siglum,
                )
            ).first()
            is not None
        )


def _claim_text(c: Claim) -> str:
    return _n(c.text)


def _fmt(code: str, msg: str) -> str:
    return f"[{code}] {msg}"


def validate_knowledge(answer: Answer, retrieved: set[str], ctx: KnowledgeContext) -> list[str]:
    problems: list[str] = []
    all_text = _n("\n".join([*(c.text for c in answer.claims), answer.disclaimer]))

    for i, c in enumerate(answer.claims):
        text = _claim_text(c)
        refs = {cit.ref for cit in c.citations}

        # 4a: a ref the model only saw as an evidence_ref of the layer was never read
        for ref in sorted(refs):
            if ref in ctx.layer_refs and ref not in retrieved:
                problems.append(
                    _fmt(
                        KL_UNREAD_EVIDENCE_REF,
                        f"claim {i} cites {ref!r}, which appeared only as an evidence_ref of the "
                        "knowledge layer; read it with get_segment before citing",
                    )
                )

        for item_id in c.kl_item_ids:
            if item_id not in ctx.items_seen:
                problems.append(
                    _fmt(
                        KL_UNKNOWN_ITEM,
                        f"claim {i}: kl_item_ids {item_id} was not returned by a tool",
                    )
                )

        # 4b: lens works are secondary and attributed
        lens_cits = [cit for cit in c.citations if ctx.source_role(cit.ref) == "lens"]
        for cit in lens_cits:
            if cit.source_role != "secondary_lens" or not (cit.attribution or "").strip():
                problems.append(
                    _fmt(
                        KL_LENS_ROLE,
                        f"claim {i}: {cit.ref!r} is a lens work; set source_role=secondary_lens "
                        'and attribution (e.g. "לפי הבנת X ברש"י")',
                    )
                )
        if (
            c.type == ClaimType.INTERPRETATION
            and c.citations
            and all(cit.source_role == "secondary_lens" for cit in c.citations)
            and _HOLDS_RE.search(text)
            and not _LENS_MARK_RE.search(text)
        ):
            problems.append(
                _fmt(
                    KL_LENS_ONLY_RISHON,
                    f"claim {i} attributes a position to the Rishon on the strength of a lens "
                    "work alone: קרא את דברי הראשון עצמו ב-get_segment, או כתוב "
                    '"הלשון מתיישבת עם ..." / "לפי הבנת X ..."',
                )
            )

        # 4c: a relation is shown with its kind and evidence; Sefaria links are candidates
        seen_rels = []
        for rr in c.relations:
            if not rr.rel.strip() or rr.evidence_origin not in ORIGINS:
                problems.append(
                    _fmt(
                        KL_RELATION_FIELDS,
                        f"claim {i}: relation {rr.table}:{rr.id} needs rel and evidence_origin "
                        f"(one of {', '.join(ORIGINS)})",
                    )
                )
                continue
            seen = ctx.relations_seen.get((rr.table, rr.id))
            if seen is None:
                problems.append(
                    _fmt(
                        KL_RELATION_UNKNOWN,
                        f"claim {i}: relation {rr.table}:{rr.id} was not returned by a tool",
                    )
                )
                continue
            if (
                seen["rel"] != rr.rel
                or seen["evidence_origin"] != rr.evidence_origin
                or bool(seen["candidate"]) != rr.candidate
            ):
                problems.append(
                    _fmt(
                        KL_RELATION_MISMATCH,
                        f"claim {i}: relation {rr.table}:{rr.id} is rel={seen['rel']}, "
                        f"evidence_origin={seen['evidence_origin']}, candidate={seen['candidate']} "
                        "in the tool result; copy it as returned",
                    )
                )
                continue
            seen_rels.append(seen)
        if (
            c.relations
            and len(seen_rels) == len(c.relations)
            and all(r["candidate"] for r in seen_rels)
        ):
            ends = {e for r in seen_rels for e in r["endpoints"]}
            missing = sorted(e for e in ends if e not in refs or e not in retrieved)
            if not ends or missing:
                problems.append(
                    _fmt(
                        KL_CANDIDATE_ONLY,
                        f"claim {i}: {CANDIDATE_MSG}. Read both ends with get_segment and cite "
                        f"them (missing: {', '.join(missing) or 'endpoints unknown'})",
                    )
                )

        # 4d: textual-witness claims need a witness, or an attributed interpretation
        if c.type != ClaimType.QUOTE and _GIRSA_RE.search(text):
            if any(
                cit.witness_id is not None and ctx.witness_ok(cit.witness_id) for cit in c.citations
            ):
                pass
            elif any(cit.witness_id is not None for cit in c.citations):
                problems.append(
                    _fmt(
                        KL_GIRSA_WITNESS,
                        f"claim {i}: witness_id does not exist in kl_witnesses or is not tied to a "
                        "girsa_variant item",
                    )
                )
            elif (
                c.type == ClaimType.INTERPRETATION
                and any((cit.attribution or "").strip() for cit in c.citations)
                and not _GIRSA_FORBIDDEN_RE.search(text)
            ):
                pass  # "הב"ח מגיה...": a source's claim, in the claimant's name, with a citation
            else:
                problems.append(
                    _fmt(
                        KL_GIRSA_WITNESS,
                        f"claim {i} asserts a girsa/scribal error without a witness_id. Say it in "
                        'the claimant\'s name as an interpretation with their citation ("הב"ח '
                        'מגיה ..."), or cite a registered witness ("בכ"י X ליתא, לפי Y"). '
                        'Never "הדפוס שגוי"',
                    )
                )

        # 4g: unusual interpretation must be marked
        if set(c.kl_item_ids) & ctx.unusual_items and (
            "unusual_interpretation" not in c.flags or UNUSUAL_PHRASE not in text
        ):
            problems.append(
                _fmt(
                    KL_UNUSUAL_FLAG,
                    f'claim {i} rests on an unusual interpretation: add flags=["unusual_'
                    f'interpretation"] and mark the text "{UNUSUAL_PHRASE}"',
                )
            )

        # 4h: "we received Maran's instructions" needs documented practice
        if _MARAN_RE.search(text):
            ok = False
            for item_id in c.kl_item_ids:
                if item_id in ctx.documented_practice and refs & ctx.documented_practice[item_id]:
                    ok = True
            if not ok:
                problems.append(
                    _fmt(
                        KL_MARAN_INSTRUCTIONS,
                        f"claim {i} says instructions were received from Maran without a "
                        "documented_practice item of this nidon: put its item id in kl_item_ids "
                        "and cite its evidence (after get_segment)",
                    )
                )

        # 4i: the Shulchan Arukh is not a verdict-by-fiat
        if _SA_PSAK_RE.search(text):
            problems.append(
                _fmt(
                    KL_SA_PSAK,
                    f"claim {i} treats the Shulchan Arukh's ruling as settling the matter; "
                    "present the Mechaber, the Rema and those who disagree, with their reasoning",
                )
            )

        # 4j: several readings of one position
        for item_id in c.kl_item_ids:
            for group in ctx.reading_groups.get(item_id, []):
                if MORE_READINGS_PHRASE not in text and not all(_n(h) in text for h in group if h):
                    problems.append(
                        _fmt(
                            KL_READINGS,
                            f"claim {i}: the item has several readings ({'; '.join(group)}): "
                            f'show all of them or write "{MORE_READINGS_PHRASE}"',
                        )
                    )
                    break

    gate = ctx.gate
    if gate is not None:
        # 4f
        if answer.scope != gate.status:
            problems.append(
                _fmt(
                    KL_SCOPE_MISMATCH,
                    f"set scope={gate.status!r} (the scope gate's result); got {answer.scope!r}",
                )
            )
        if gate.status == "out_of_scope":
            if any(
                c.type in (ClaimType.SYSTEM_CONCLUSION, ClaimType.CHIDUSH) for c in answer.claims
            ):
                problems.append(
                    _fmt(
                        KL_OUT_OF_SCOPE_CONCLUSION,
                        "the question is out of scope: no system_conclusion and no chidush; say "
                        "what is missing (you may point to sources actually read)",
                    )
                )
            if OUT_OF_SCOPE_PHRASE not in all_text:
                problems.append(
                    _fmt(
                        KL_SCOPE_NOTICE,
                        f'say that the question is {OUT_OF_SCOPE_PHRASE} ("{OUT_OF_SCOPE_PHRASE}") '
                        "and what is missing",
                    )
                )
        elif gate.status == "partial" and PARTIAL_PHRASE not in all_text:
            problems.append(
                _fmt(
                    KL_SCOPE_NOTICE,
                    f'partial coverage: include a coverage notice with "{PARTIAL_PHRASE}" and '
                    "what was not checked",
                )
            )
        # 4e
        if gate.unchecked_core_books:
            missing_books = [
                b for b in gate.unchecked_core_books if b not in answer.unchecked_core_books
            ]
            if missing_books:
                problems.append(
                    _fmt(
                        KL_COVERAGE_BOOKS,
                        "unchecked_core_books must list: " + "; ".join(gate.unchecked_core_books),
                    )
                )
            if UNCHECKED_PHRASE not in all_text or any(
                _n(b) not in all_text for b in gate.unchecked_core_books
            ):
                problems.append(
                    _fmt(
                        KL_COVERAGE_BOOKS,
                        f'the text must include "{UNCHECKED_PHRASE} '
                        + "; ".join(gate.unchecked_core_books)
                        + '"',
                    )
                )
            for i, c in enumerate(answer.claims):
                t = _claim_text(c)
                if _SILENCE_RE.search(t) and "בקורפוס שנבדק" not in t:
                    problems.append(
                        _fmt(
                            KL_SILENCE_IS_NOT_AGREEMENT,
                            f'claim {i}: the corpus\'s silence is not "no machloket" while core '
                            'books were not checked; write "לא נמצא ... בקורפוס שנבדק"',
                        )
                    )
        # 4i (practical question in scope)
        if gate.practical and gate.status in ("in_scope", "partial"):
            have = (
                "המחבר" in all_text
                and ('הרמ"א' in all_text or 'רמ"א' in all_text)
                and (_DISPUTANT_RE.search(all_text) or NO_DISPUTANT_PHRASE in all_text)
            )
            if not have:
                problems.append(
                    _fmt(
                        KL_SA_PRACTICAL,
                        "a practical in-scope question needs the Mechaber (המחבר), the Rema "
                        f'(הרמ"א) and the dissenters, or "{NO_DISPUTANT_PHRASE}"',
                    )
                )
    return problems
