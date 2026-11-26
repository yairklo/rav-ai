"""V8: the golden question's scope decides, not the run's (PR #16)."""

from __future__ import annotations

import json

from rav_evals.config import FIXTURES_DIR, load_config
from rav_evals.golden import GoldenQuestion, GoldenSet
from rav_evals.kl_lookup import InMemoryKnowledgeLookup
from rav_evals.runner import RunOutput, question_in_scope, run_eval
from rav_evals.schema import Answer
from rav_evals.verifier import InMemorySourceStore

FULL = "Full 1a-1b"  # a scope where every core book is covered


def _seed():
    d = json.loads((FIXTURES_DIR / "seed-direct.json").read_text(encoding="utf-8"))
    kn = dict(d["knowledge"])
    kn["coverage"] = {
        **kn["coverage"],
        FULL: {w: "covered" for w in ("Seed Gemara", "Seed Rishon", "Seed Codes")},
    }
    return d, kn


def test_question_scope_wins_over_a_fully_covered_run_scope():
    d, kn = _seed()
    q = GoldenQuestion.model_validate(d["question"])
    assert q.scope_keys == ["Seed 1a-2b"]  # Seed Codes is not_ingested there
    answer = Answer.model_validate({**d["output"]["answer"], "unchecked_core_books": []})
    out = RunOutput.model_validate(
        {**d["output"], "answer": answer.model_dump(mode="json"), "scope_keys": [FULL]}
    )
    report = run_eval(
        GoldenSet(version="t", questions=[q]),
        lambda _q: out,
        InMemorySourceStore(d["sources"]),
        knowledge=InMemoryKnowledgeLookup(kn),
        markers=load_config().markers,
    )
    codes = report.results[0].verification.failure_codes()
    assert "V8:coverage_notice_missing" in codes


def test_out_of_scope_gate_makes_the_question_out_of_scope():
    d, _ = _seed()
    q = GoldenQuestion.model_validate(d["question"])
    out = RunOutput.model_validate(d["output"])
    assert question_in_scope(q, out) is True
    gated = out.model_copy(update={"scope_status": "out_of_scope"})
    assert question_in_scope(q, gated) is False
