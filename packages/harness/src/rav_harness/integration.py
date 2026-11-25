"""Glue to rav-evals (``answer_fn``; store and normalizer come from rav-evals) and ``runs``."""

from __future__ import annotations

from collections.abc import Callable

from rav_ai.db.models import Run
from rav_evals.golden import GoldenQuestion, GoldenSet
from rav_evals.normalization import normalize
from rav_evals.pg_store import PostgresSourceStore
from rav_evals.runner import AnswerFn, EvalReport, RunOutput, run_eval
from sqlalchemy.orm import Session

from .config import HarnessConfig
from .llm import LLM
from .loop import Harness, HarnessResult
from .prompt import PROMPT_VERSION, prompt_hash
from .tools import CorpusTools


def db_recorder(session: Session) -> Callable[[HarnessResult, str | None, str | None], str]:
    def record(res: HarnessResult, question_id: str | None, golden_version: str | None) -> str:
        run = Run(
            question_text=res.question,
            question_id=question_id,
            golden_version=golden_version,
            model=res.model,
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash(),
            messages=res.messages,
            retrieved_sources=res.retrieved_refs,
            tool_calls=res.tool_calls,
            answer=res.answer.model_dump(mode="json") if res.answer else None,
            error=res.error,
            latency_ms=res.latency_ms,
            steps=res.steps,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            cost_usd=res.cost_usd,
        )
        session.add(run)
        session.commit()
        return run.id

    return record


def build_harness(
    session: Session, llm: LLM, config: HarnessConfig | None = None, *, record: bool = True
) -> Harness:
    """A harness over ``session``. Tools are per-run state, so call this once per question or
    use :func:`make_answer_fn`, which rebuilds the tools for every question."""
    return Harness(
        llm, CorpusTools(session), config, recorder=db_recorder(session) if record else None
    )


def make_answer_fn(
    session: Session,
    llm: LLM,
    config: HarnessConfig | None = None,
    *,
    record: bool = True,
    run_ids: dict[str, str] | None = None,
) -> AnswerFn:
    """``answer_fn`` for ``rav_evals.runner.run_eval``. Raises when no valid answer was
    produced (run_eval then records the question as an error)."""

    def answer_fn(q: GoldenQuestion) -> RunOutput:
        res = build_harness(session, llm, config, record=record).run(
            q.question, question_id=q.id, golden_version=None
        )
        if res.run_id and run_ids is not None:
            run_ids[q.id] = res.run_id
        if res.answer is None:
            raise RuntimeError(res.error or "harness produced no answer")
        return RunOutput(
            answer=res.answer,
            retrieved_refs=res.retrieved_refs,
            model=res.model,
            tool_calls=res.tool_calls,
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash(),
        )

    return answer_fn


def run_golden(
    session: Session,
    golden: GoldenSet,
    llm: LLM,
    config: HarnessConfig | None = None,
    *,
    record: bool = True,
) -> EvalReport:
    """Run the harness over a golden set, verify with rav-evals, and write the verifier
    outcome (and golden version) back onto each run row."""
    run_ids: dict[str, str] = {}
    report = run_eval(
        golden,
        make_answer_fn(session, llm, config, record=record, run_ids=run_ids),
        PostgresSourceStore(session.get_bind()),
        normalize=normalize,
    )
    if record:
        for r in report.results:
            run = session.get(Run, run_ids.get(r.question_id, ""))
            if run is not None:
                run.golden_version = golden.version
                if r.verification is not None:
                    run.verification = r.verification.model_dump(mode="json")
                    run.verification_passed = r.verification.passed
        session.commit()
    return report
