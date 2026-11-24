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


class GoldenSet(BaseModel):
    version: str
    status: Literal["draft", "reviewed"] = "draft"
    description: str = ""
    questions: list[GoldenQuestion]


def load_golden(path: str | Path) -> GoldenSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GoldenSet.model_validate(data)
