"""Glue to rav-evals (``answer_fn``; store and normalizer come from rav-evals) and ``runs``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rav_ai.db.models import Run
from rav_evals.golden import GoldenQuestion, GoldenSet
from rav_evals.normalization import normalize
from rav_evals.pg_store import PostgresSourceStore
from rav_evals.runner import AnswerFn, EvalReport, RunOutput, run_eval
from sqlalchemy.orm import Session

from .config import HarnessConfig, load_config
from .llm import LLM
from .loop import Harness, HarnessResult
from .tools import CorpusTools


def db_recorder(session: Session) -> Callable[[HarnessResult, str | None, str | None], str]:
    def record(res: HarnessResult, question_id: str | None, golden_version: str | None) -> str:
        run = Run(
            question_text=res.question,
            question_id=question_id,
            golden_version=golden_version,
            model=res.model,
            prompt_version=res.prompt_version,
            prompt_hash=res.prompt_hash,
            messages=res.messages,
            retrieved_sources=res.retrieved_refs,
            tool_calls=res.tool_calls,
            answer=res.answer.model_dump(mode="json") if res.answer else None,
            error=res.error,
            knowledge_items=res.knowledge_items,
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
    session: Session,
    llm: LLM,
    config: HarnessConfig | None = None,
    *,
    record: bool = True,
    mode: str | None = None,
) -> Harness:
    """A harness over ``session``. Tools are per-run state, so call this once per question or
    use :func:`make_answer_fn`, which rebuilds the tools for every question."""
    config = config or load_config()
    return Harness(
        llm,
        CorpusTools(session, mode, config),
        config,
        recorder=db_recorder(session) if record else None,
    )


def run_output_scope(res: HarnessResult) -> dict[str, Any]:
    """The RunOutput fields that describe the knowledge mode and the scope gate. scope_keys are
    only the keys the gate confirmed in scope (empty when out_of_scope), never the keys of every
    item the run touched (those stay in runs.knowledge_items). ``knowledge`` and ``scope_keys``
    are RunOutput fields from KL-3 (#14); until it is merged pydantic ignores them."""
    gate = res.scope_gate or {}
    return {
        "knowledge": res.knowledge,
        "scope_keys": list(gate.get("scope_keys", []))
        if gate.get("status") != "out_of_scope"
        else [],
        "scope_status": gate.get("status"),
    }


def make_answer_fn(
    session: Session,
    llm: LLM,
    config: HarnessConfig | None = None,
    *,
    record: bool = True,
    run_ids: dict[str, str] | None = None,
    mode: str | None = None,
) -> AnswerFn:
    """``answer_fn`` for ``rav_evals.runner.run_eval``. Raises when no valid answer was
    produced (run_eval then records the question as an error)."""

    def answer_fn(q: GoldenQuestion) -> RunOutput:
        res = build_harness(session, llm, config, record=record, mode=mode).run(
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
            prompt_version=res.prompt_version,
            prompt_hash=res.prompt_hash,
            knowledge_items=res.knowledge_items,
            **run_output_scope(res),
        )

    return answer_fn


def run_golden(
    session: Session,
    golden: GoldenSet,
    llm: LLM,
    config: HarnessConfig | None = None,
    *,
    record: bool = True,
    mode: str | None = None,
) -> EvalReport:
    """Run the harness over a golden set, verify with rav-evals, and write the verifier
    outcome (and golden version) back onto each run row."""
    run_ids: dict[str, str] = {}
    report = run_eval(
        golden,
        make_answer_fn(session, llm, config, record=record, run_ids=run_ids, mode=mode),
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
