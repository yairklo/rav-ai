"""Pydantic schemas of the knowledge-layer items (what the LLM submits, per kind).

Offsets are never part of these schemas: the model quotes, the code locates (evidence.py).
`seg` is the local label of an input segment ("S7"), mapped to a segment_id by the code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SCHEMA_VERSION = 1
# position and Holder (and every kind that embeds a Holder) are at version 2 (KL-1-fix-2)
_HOLDER_KINDS = {
    "position",
    "open_question",
    "chakira",
    "source_claim",
    "commentary_note",
    "method_principle",
    "sugya_summary",
}


def schema_version_for(kind: str) -> int:
    return 2 if kind in _HOLDER_KINDS else SCHEMA_VERSION


EvRole = Literal[
    "states",
    "supports",
    "attributes",
    "adopts",
    "disputes",
    "interprets",
    "proposes",
    "claims_error",
    "witness",
]
CRITICAL_ROLES = ("states", "attributes", "proposes", "claims_error")

Move = Literal["statement", "question", "objection", "answer", "support", "conclusion", "aside"]
Subtype = Literal[
    "baraita",
    "memra",
    "teyuvta",
    "ukimta",
    "teiku",
    "hilkheta",
    "kal_vachomer",
    "story",
    "dichuy",
    "hava_amina",
    "maskana",
    "shinuya",
]
StepRelation = Literal["attacks", "defends", "answers", "supports", "refines"]
Standing = Literal["accepted", "refuted", "qualified", "open"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Ev(_Base):
    """A pointer into the text: the model supplies a quote, never offsets."""

    seg: str
    quote: str
    role: EvRole = "supports"
    w_start: int | None = None  # word index of the quote start (anchor_format = numbered)


def default_role(evs: list[Ev], role: str) -> None:
    """Evidence whose role the model did not state gets the claim-bearing default (K3 is strict
    about these roles), never the lenient generic 'supports'."""
    for e in evs:
        if "role" not in e.model_fields_set:
            e.role = role  # type: ignore[assignment]


class QuotedIn(_Base):
    work: str
    evidence_ids: list[int | str] = []


class Holder(_Base):
    label: str
    entity_slug: str | None = None
    work: str | None = None
    tier: str | None = None
    attribution: Literal["direct", "reported"] = "direct"
    reported_by: str | None = None
    known_via: Literal["direct", "quoted_only"] | None = "direct"
    quoted_in: list[QuotedIn] | None = None

    @model_validator(mode="after")
    def _quoted_only(self) -> Holder:
        if self.known_via == "quoted_only" and not self.quoted_in:
            raise ValueError("known_via=quoted_only requires a non-empty quoted_in")
        return self


class Reading(_Base):
    text_he: str
    holder: Holder | None = None
    evidence: list[Ev] = []


class Step(_Base):
    id: str
    move: Move
    subtype: Subtype | None = None
    responds_to: str | None = None
    relation: StepRelation | None = None
    raises_issue: str | None = None
    standing: Standing = "accepted"
    speaker: str | None = None  # "ambiguous" for Rabba/Raba when the text does not tell
    text_he: str
    evidence: list[Ev] = []
    readings: list[Reading] = []

    @field_validator("text_he")
    @classmethod
    def _short(cls, v: str) -> str:
        if len(v.split()) > 25:
            raise ValueError("text_he longer than 25 words")
        return v

    @model_validator(mode="after")
    def _roles(self) -> Step:
        default_role(self.evidence, "states")
        return self


class Sugya(_Base):
    title_he: str
    start_seg: str
    end_seg: str
    # the sugya continues before the first / after the last segment of the window (S1 merge
    # carries it over to the neighbouring window)
    open_start: bool = False
    open_end: bool = False
    notes: str | None = None


class SugyaBoundaries(_Base):
    """S1 output for one window: the sugyot that start or end in it."""

    sugyot: list[Sugya]
    notes: str | None = None


class SugyaSummary(_Base):
    title_he: str
    steps: list[Step]
    notes: str | None = None
    readings: list[Reading] = []  # empty in the Gemara-only pass


class Issue(_Base):
    canonical_q_he: str
    parent_issue: str | None = None
    anchors: list[str] = []
    raised_in: list[str] = []  # sugya item keys
    evidence: list[Ev] = []
    notes: str | None = None


class RelationRef(_Base):
    rel: Literal["objects_to", "disputes", "rules_like", "claims_error_in", "answers", "supports"]
    target: Holder
    evidence: list[Ev] = []


class Condition(_Base):
    text_he: str
    evidence: list[Ev] = []


class Leniency(_Base):
    text_he: str
    evidence: list[Ev] = []


class Position(_Base):
    issue: str
    holder: Holder
    view_he: str
    side_label: str | None = None  # positions with the same label form one side of a machloket
    evidence: list[Ev] = []
    reason: str | None = None
    normative_level: str | None = None
    lechatchila_bediavad: Literal["lechatchila", "bediavad", "both"] | None = None
    conditions: list[Condition] = []
    community: str | None = None
    depends_on_girsa: str | None = None
    is_practical_ruling: bool = False
    relations: list[RelationRef] = []
    reading_only: bool = False  # set by check K5: evidence only from a lens source
    read_by: Holder | None = None  # reading_only: the lens author who reads this view
    attribution_grade: Literal["inferred_by_source"] | None = None  # set with reading_only
    # KL-1-fix-2 (all optional; a field that is set needs evidence)
    meakev: Literal["yes", "no", "disputed"] | None = None
    obligation_type: (
        Literal["din", "takana", "minhag_mevorach", "minhag", "minhag_taut", "midat_chasidut"]
        | None
    ) = None
    leniency_reason: Leniency | None = None
    instruction_scope: Literal["individual", "public"] | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _roles_and_fix2(self) -> Position:
        default_role(self.evidence, "states")
        if self.leniency_reason is not None:
            default_role(self.leniency_reason.evidence, "states")
        set_fields = [
            n
            for n in ("meakev", "obligation_type", "instruction_scope")
            if getattr(self, n) is not None
        ]
        if set_fields and not self.evidence:
            raise ValueError(f"{', '.join(set_fields)} set without evidence")
        if self.leniency_reason is not None and not self.leniency_reason.evidence:
            raise ValueError("leniency_reason set without evidence")
        return self


class OpenQuestion(_Base):
    issue: str
    kind: Literal["tzarich_iyun", "teiku", "unanswered_kushya", "lo_nitbaer"]
    holder: Holder | None = None
    text_he: str = ""
    evidence: list[Ev] = []


class ChakiraSide(_Base):
    label_he: str
    proposer: Holder
    proposer_evidence: list[Ev] = []
    interprets: str = ""
    nafka_minot: list[str] = []
    aligned: list[Holder] = []  # who is aligned with this side, asserted_by someone else
    asserted_by: Holder | None = None


class Chakira(_Base):
    issue: str | None = None
    question_he: str
    sides: list[ChakiraSide]


class SourceClaimTarget(_Base):
    work: str | None = None
    ref: str | None = None
    quote: str = ""
    evidence: Ev | None = None
    target_status: Literal["located", "unresolved"] = "located"  # unresolved: not in the corpus
    holder: Holder | None = None


class SourceClaim(_Base):
    claim_kind: Literal["girsa_error", "misquotation", "hasaga", "psak_not_accepted"]
    target: SourceClaimTarget
    holder: Holder
    holder_evidence: list[Ev] = []
    proposed_alternative: str | None = None  # kept apart from the source text, never replacing it
    consequence: str | None = None
    responses: list[Reading] = []

    @model_validator(mode="after")
    def _roles(self) -> SourceClaim:
        default_role(self.holder_evidence, "attributes")
        return self


class GirsaVariant(_Base):
    base: Ev
    variant_text: str
    witness_kind: Literal[
        "rishon_quote", "dikdukei_sofrim", "manuscript", "printed_edition", "quote_mismatch"
    ]
    witness_siglum: str | None = None
    evidence: list[Ev] = []


class Machloket(_Base):
    issue_id: str  # item_key of the issue
    sides: list[list[str]]  # position item_keys per side
    crux_he: str | None = None
    sugya_ids: list[str] = []
    machloket_kind: Literal["gemara", "rishonim", "poskim"] = "gemara"


class CommentaryNote(_Base):
    sugya: str
    holder: Holder
    text_he: str
    evidence: list[Ev] = []


class MethodPrinciple(_Base):
    principle_he: str
    examples: list[str] = []
    source: Holder | None = None
    evidence: list[Ev] = []


class FreeCard(_Base):
    """topic_card / mishnah_card: shape fixed in a later PR; for now a titled text."""

    title_he: str
    text_he: str = ""
    evidence: list[Ev] = []


class AbbrevChoice(_Base):
    id: int  # annotation id
    choice: int | None = None  # index into the candidates, None = none fits / unclear


class AbbrevChoices(_Base):
    choices: list[AbbrevChoice]
    notes: str | None = None


KIND_MODELS: dict[str, type[BaseModel]] = {
    "sugya": Sugya,
    "sugya_summary": SugyaSummary,
    "commentary_note": CommentaryNote,
    "issue": Issue,
    "position": Position,
    "machloket": Machloket,
    "open_question": OpenQuestion,
    "chakira": Chakira,
    "source_claim": SourceClaim,
    "girsa_variant": GirsaVariant,
    "method_principle": MethodPrinciple,
    "topic_card": FreeCard,
    "mishnah_card": FreeCard,
}

# what the LLM submits per tool (submit_<name>)
TOOL_MODELS: dict[str, type[BaseModel]] = {
    **KIND_MODELS,
    "sugya": SugyaBoundaries,
    "abbrev_choice": AbbrevChoices,
}


def validate_body(kind: str, raw: dict[str, Any]) -> tuple[BaseModel | None, list[str]]:
    """K1: schema validation. Returns (model, errors)."""
    model = KIND_MODELS.get(kind)
    if model is None:
        return None, [f"unknown kind {kind!r}"]
    try:
        return model.model_validate(raw), []
    except Exception as e:  # noqa: BLE001 - pydantic ValidationError
        return None, [str(e)]


def iter_evidence(obj: Any, path: str = "") -> list[tuple[str, Ev, str]]:
    """All Ev inside a model/dict as (claim_path, Ev, claim_text), in document order.

    claim_path is a stable JSON-path-like id of the enclosing claim; claim_text the nearest
    textual field of that claim (for entailment checks).
    """
    out: list[tuple[str, Ev, str]] = []

    def walk(o: Any, p: str, text: str) -> None:
        if isinstance(o, Ev):
            out.append((p, o, text))
        elif isinstance(o, BaseModel):
            t = text
            for name in ("text_he", "view_he", "question_he", "canonical_q_he", "label_he"):
                v = getattr(o, name, None)
                if isinstance(v, str) and v:
                    t = v
                    break
            for name in type(o).model_fields:
                walk(getattr(o, name), f"{p}.{name}" if p else name, t)
        elif isinstance(o, list):
            for i, x in enumerate(o):
                walk(x, f"{p}[{i}]", text)

    walk(obj, path, "")
    return out
