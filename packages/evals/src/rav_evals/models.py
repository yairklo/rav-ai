"""Run logging. The ``runs`` table lives on the shared corpus ``Base`` (``rav_ai.db.models``)
so Alembic migrates it with everything else; this module only builds rows from results."""

from __future__ import annotations

import hashlib
from typing import Any

from rav_ai.db.models import Run
from sqlalchemy.orm import Session

from .golden import GoldenSet
from .runner import EvalReport

__all__ = ["Run", "prompt_hash", "record_report", "run_from_result"]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def run_from_result(result: Any, question_text: str, golden_version: str | None = None) -> Run:
    """Build a ``Run`` row from a runner ``QuestionResult``: model, prompt hash, answer and
    verifier results."""
    out = result.output
    return Run(
        question_text=question_text,
        question_id=result.question_id,
        golden_version=golden_version,
        model=result.model or "unknown",
        prompt_version=out.prompt_version if out else None,
        prompt_hash=out.prompt_hash if out else None,
        retrieved_sources=list(out.retrieved_refs) if out else [],
        tool_calls=list(out.tool_calls) if out else [],
        answer=out.answer.model_dump(mode="json") if out else None,
        error=result.error,
        latency_ms=result.latency_ms,
        verification=result.verification.model_dump(mode="json") if result.verification else None,
        verification_passed=result.verification.passed if result.verification else None,
    )


def record_report(session: Session, golden: GoldenSet, report: EvalReport) -> list[Run]:
    """Persist every question result of ``report`` as a ``runs`` row (one commit)."""
    questions = {q.id: q.question for q in golden.questions}
    rows = [
        run_from_result(r, questions.get(r.question_id, ""), report.golden_version)
        for r in report.results
    ]
    session.add_all(rows)
    session.commit()
    return rows
