"""Proposed ``runs`` table: one row per system run, kept for evals and future training data.

Uses its own declarative Base so it does not interfere with the corpus models. NOT wired
into Alembic yet: integrating it is left to whoever owns the DB migrations.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
JsonType = JSON().with_variant(JSONB(), "postgresql")


class EvalsBase(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


class Run(EvalsBase):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    # Input
    question_text: Mapped[str] = mapped_column(Text)
    question_id: Mapped[str | None] = mapped_column(String(128), index=True)  # golden id, if any
    golden_version: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128), index=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64))

    # What happened
    retrieved_sources: Mapped[list[Any]] = mapped_column(JsonType, default=list)  # refs (+scores)
    tool_calls: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    answer: Mapped[dict[str, Any] | None] = mapped_column(JsonType)  # Answer.model_dump()
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    # Verification
    verification: Mapped[dict[str, Any] | None] = mapped_column(JsonType)  # VerificationReport
    verification_passed: Mapped[bool | None] = mapped_column(Boolean, index=True)

    # Human/LLM review, filled later; gate for using the row as training data
    review_label: Mapped[str | None] = mapped_column(String(32))
    review_notes: Mapped[str | None] = mapped_column(Text)


def run_from_result(result: Any, question_text: str, golden_version: str | None = None) -> Run:
    """Build a ``Run`` row from a runner ``QuestionResult``."""
    out = result.output
    return Run(
        question_text=question_text,
        question_id=result.question_id,
        golden_version=golden_version,
        model=result.model or "unknown",
        retrieved_sources=list(out.retrieved_refs) if out else [],
        tool_calls=list(out.tool_calls) if out else [],
        answer=out.answer.model_dump(mode="json") if out else None,
        error=result.error,
        latency_ms=result.latency_ms,
        verification=result.verification.model_dump(mode="json") if result.verification else None,
        verification_passed=result.verification.passed if result.verification else None,
    )
