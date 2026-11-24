"""Glue to rav-evals (answer_fn + DB-backed SourceStore) and to the ``runs`` table."""

from __future__ import annotations

from collections.abc import Callable

from rav_ai.lookup import get_texts, may_display
from rav_evals.golden import GoldenQuestion, GoldenSet
from rav_evals.runner import AnswerFn, EvalReport, RunOutput, run_eval
from sqlalchemy.orm import Session
from textnorm import normalize

from .config import HarnessConfig
from .llm import LLM
from .loop import Harness, HarnessResult
from .prompt import PROMPT_VERSION, prompt_hash
from .runs import Run, save_run
from .tools import CorpusTools


def verifier_normalizer(s: str) -> str:
    """Normalizer for the verifier: strip Sefaria markup, level B (punctuation-insensitive)."""
    return normalize(s, "B", markup="sefaria").text


class DbSourceStore:
    """rav-evals ``SourceStore``: displayable text of all versions of a ref, joined."""

    def __init__(self, session: Session, mode: str | None = None):
        self.session, self.mode = session, mode

    def get_text(self, ref: str) -> str | None:
        rows = get_texts(self.session, ref)
        if not rows:
            return None
        return "\n".join(
            r.text_raw for r in rows if may_display(r.can_index, r.can_display_public, self.mode)
        )


def db_recorder(session: Session) -> Callable[[HarnessResult, str | None, str | None], str]:
    def record(res: HarnessResult, question_id: str | None, golden_version: str | None) -> str:
        run = Run(
            question_text=res.question, question_id=question_id, golden_version=golden_version,
            model=res.model, prompt_version=PROMPT_VERSION, prompt_hash=prompt_hash(),
            messages=res.messages, retrieved_sources=res.retrieved_refs,
            tool_calls=res.tool_calls,
            answer=res.answer.model_dump(mode="json") if res.answer else None,
            error=res.error, latency_ms=res.latency_ms, steps=res.steps,
            input_tokens=res.input_tokens, output_tokens=res.output_tokens, cost_usd=res.cost_usd,
        )
        save_run(session, run)
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
            answer=res.answer, retrieved_refs=res.retrieved_refs, model=res.model,
            tool_calls=res.tool_calls,
        )

    return answer_fn


def run_golden(
    session: Session, golden: GoldenSet, llm: LLM, config: HarnessConfig | None = None,
    *, record: bool = True,
) -> EvalReport:
    """Run the harness over a golden set, verify with rav-evals, and write the verifier
    outcome (and golden version) back onto each run row."""
    run_ids: dict[str, str] = {}
    report = run_eval(
        golden,
        make_answer_fn(session, llm, config, record=record, run_ids=run_ids),
        DbSourceStore(session),
        normalize=verifier_normalizer,
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
