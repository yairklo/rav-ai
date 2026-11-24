"""Output schema for a system answer: typed claims, each backed by citations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ClaimType(StrEnum):
    QUOTE = "quote"  # verbatim quotation from a source
    INTERPRETATION = "interpretation"  # accepted commentary / explanation from sources
    SYSTEM_CONCLUSION = "system_conclusion"  # the system's own inference
    CHIDUSH = "chidush"  # a novel idea originating from the system


class Citation(BaseModel):
    ref: str = Field(description="Canonical reference of the source, e.g. as stored in the DB.")
    quoted_text: str = Field(description="Text quoted from that source.")
    attribution: str | None = Field(
        default=None,
        description='Who said it, e.g. "המשנה ברורה בשם המגן אברהם". None = the source itself.',
    )


class Claim(BaseModel):
    type: ClaimType
    text: str
    citations: list[Citation] = Field(default_factory=list)
    flagged_as_system: bool = Field(
        default=False,
        description="Explicit marking that this claim is the system's own (required for "
        "chidush and system_conclusion).",
    )


class Answer(BaseModel):
    claims: list[Claim]
    disclaimer: str = Field(
        default="",
        description="Statement that this is the system's opinion, not a ruling for practice.",
    )
    is_psak: bool = Field(default=False, description="Must always be False.")

    def full_text(self) -> str:
        """All answer text (claims and quoted passages) for deterministic text checks."""
        parts: list[str] = []
        for claim in self.claims:
            parts.append(claim.text)
            parts.extend(c.quoted_text for c in claim.citations)
        return "\n".join(parts)

    def cited_refs(self) -> set[str]:
        return {c.ref for claim in self.claims for c in claim.citations}
