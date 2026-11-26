"""Golden question set format (YAML) and loader."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PLACEHOLDER_PREFIX = "TODO"


def is_placeholder(value: str) -> bool:
    return value.strip().upper().startswith(PLACEHOLDER_PREFIX)


class Category(StrEnum):
    DIRECT_SOURCE = "direct_source"  # מקור ישיר
    DISPUTE = "dispute"  # מחלוקת
    MODERN = "modern"  # שאלה מודרנית
    FALSE_PREMISE = "false_premise"  # הנחה שגויה
    OUT_OF_CORPUS = "out_of_corpus"  # מחוץ לקורפוס
    # KL-3 (knowledge layer)
    NOVEL = "novel"  # יישום חדש שאין עליו מקור ישיר
    SUGYA_FLOW = "sugya_flow"  # מהלך הסוגיה
    DISPUTE_CHAIN = "dispute_chain"  # שרשרת מחלוקת גמרא -> ראשונים -> פוסקים
    TZDADIM = "tzdadim"  # צדדים / חקירה
    GIRSA_DEPENDENT = "girsa_dependent"  # דין שתלוי בגרסה
    PSAK_NOT_ACCEPTED = "psak_not_accepted"  # פסק שלא נתקבל
    OUT_OF_SCOPE = "out_of_scope"  # מחוץ לפרוסה (נבדק מול Answer.scope)
    SNIF = "snif"  # פוסק מצרף שיטה / גרסה שנדחתה
    SYCOPHANCY = "sycophancy"  # הנחה טעונה / לחץ


class Side(BaseModel):
    """One side of a dispute that a good answer must mention."""

    label: str
    terms: list[str] = Field(
        default_factory=list,
        description="Any of these terms appearing in the answer counts as mentioning the side. "
        "Empty or TODO = not yet filled in; the check is skipped.",
    )


RubricKind = Literal[
    "cites_ref",  # params: ref
    "mentions_any",  # params: terms
    "mentions_all",  # params: terms
    "mentions_none",  # params: terms
    "has_claim_type",  # params: claim_types (any-of)
    "verifier_passes",  # no params
    "llm",  # judged by an LLM judge; params unused
    "novel_protocol",  # system_conclusion/chidush flagged + yesod cited + sides + hedged
    "scope_is",  # params: scope (Answer.scope must equal it)
    "has_flag",  # params: flags (any-of, on some claim), e.g. [snif]
    "no_unbacked_confirmation",  # no confirm-marker claim without a citation (sycophancy)
]


class RubricItem(BaseModel):
    """A binary (pass/fail) check."""

    id: str
    description: str
    kind: RubricKind
    ref: str | None = None
    terms: list[str] = Field(default_factory=list)
    claim_types: list[str] = Field(
        default_factory=list, description="has_claim_type: any of these claim types suffices."
    )
    scope: str | None = None
    flags: list[str] = Field(default_factory=list)
    gate: bool = Field(
        default=True,
        description="False = reported as a metric only (e.g. the LLM sycophancy judge).",
    )


class SycophancySpec(BaseModel):
    """(a) loaded_premise: the question asserts something false ("this is a scribal error,
    right?", "Rashi holds X, right?" with X reversed). (b) pressure: the harness is single-turn,
    so the question quotes the previous answer and then pushes back."""

    kind: Literal["loaded_premise", "pressure"]
    premise: Literal["false_error", "reversed_view", "other"] = "other"
    prior_answer: str = ""  # pressure: the previous answer, quoted into the question
    prior_answer_refs: list[str] = Field(default_factory=list)  # refs the prior answer cited


class GoldenQuestion(BaseModel):
    id: str
    question: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"]
    required_refs: list[str] = Field(default_factory=list)
    required_sides: list[Side] = Field(default_factory=list)
    rubric: list[RubricItem] = Field(default_factory=list)
    status: Literal["draft", "reviewed"] = "draft"
    notes: str = ""
    split: Literal["dev", "heldout"] = Field(
        default="dev",
        description="heldout is never used to tune prompts and runs only with --split heldout.",
    )
    scope_keys: list[str] = Field(default_factory=list, description="kl scope(s) for V8")
    sycophancy: SycophancySpec | None = None

    def todo_fields(self) -> list[str]:
        """Paths of every value still a TODO placeholder (not yet marked by a human)."""
        out: list[str] = []
        if is_placeholder(self.question) or "TODO" in self.question:
            out.append("question")
        out += [
            f"required_refs[{i}]" for i, r in enumerate(self.required_refs) if is_placeholder(r)
        ]
        for i, sd in enumerate(self.required_sides):
            if is_placeholder(sd.label) or not sd.terms or any(is_placeholder(t) for t in sd.terms):
                out.append(f"required_sides[{i}]")
        for it in self.rubric:
            vals = ([it.ref] if it.kind == "cites_ref" else []) + list(it.terms)
            if any(v is None or is_placeholder(v) for v in vals):
                out.append(f"rubric[{it.id}]")
        if self.sycophancy:
            if "TODO" in self.sycophancy.prior_answer:
                out.append("sycophancy.prior_answer")
            out += [
                f"sycophancy.prior_answer_refs[{i}]"
                for i, r in enumerate(self.sycophancy.prior_answer_refs)
                if is_placeholder(r)
            ]
        return out

    @property
    def pending(self) -> bool:
        """True while any value is still a TODO: reported as pending, never as passed."""
        return bool(self.todo_fields())


class GoldenSet(BaseModel):
    version: str
    status: Literal["draft", "reviewed"] = "draft"
    description: str = ""
    questions: list[GoldenQuestion]

    def select_split(self, split: Literal["dev", "heldout"]) -> GoldenSet:
        return self.model_copy(
            update={"questions": [q for q in self.questions if q.split == split]}
        )

    def pending_report(self) -> dict[str, list[str]]:
        return {q.id: q.todo_fields() for q in self.questions if q.pending}


def load_golden(path: str | Path, split: Literal["dev", "heldout"] | None = "dev") -> GoldenSet:
    """Loads the ``dev`` split by default: heldout is never loaded by accident (both splits
    only with an explicit ``split=None``, e.g. golden-status)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    g = GoldenSet.model_validate(data)
    return g.select_split(split) if split else g


def load_goldens(
    paths: list[str | Path], split: Literal["dev", "heldout"] | None = "dev"
) -> GoldenSet:
    """Several golden files merged into one set (version = joined versions)."""
    sets = [load_golden(p, split) for p in paths]
    return GoldenSet(
        version="+".join(s.version for s in sets),
        status="reviewed" if all(s.status == "reviewed" for s in sets) else "draft",
        description="; ".join(s.description for s in sets if s.description),
        questions=[q for s in sets for q in s.questions],
    )
