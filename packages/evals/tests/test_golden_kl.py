"""Golden: TODO values are pending (never passed); heldout never enters dev."""

from __future__ import annotations

from pathlib import Path

from rav_evals.cli import golden_status
from rav_evals.golden import GoldenSet, load_golden, load_goldens
from rav_evals.runner import RunOutput, run_eval
from rav_evals.schema import Answer, Citation, Claim, ClaimType
from rav_evals.verifier import InMemorySourceStore

KL = Path(__file__).parent.parent / "golden" / "kl"
FILES = [KL / "sycophancy.draft.yaml", KL / "categories.draft.yaml"]


def test_drafts_load_and_every_todo_is_pending():
    for f in FILES:
        g = load_golden(f, split=None)
        assert g.questions and all(q.pending for q in g.questions)
        assert all(q.todo_fields() for q in g.questions)


def test_heldout_not_in_dev():
    dev = load_goldens(FILES, "dev")
    held = load_goldens(FILES, "heldout")
    assert dev.questions and held.questions
    assert {q.split for q in dev.questions} == {"dev"}
    assert not {q.id for q in dev.questions} & {q.id for q in held.questions}


def test_category_counts():
    st = golden_status(FILES)
    cats = st["by_category"]
    assert 8 <= sum(cats["sycophancy"].values()) <= 12
    for c in (
        "novel",
        "sugya_flow",
        "dispute_chain",
        "tzdadim",
        "girsa_dependent",
        "psak_not_accepted",
        "out_of_scope",
        "snif",
    ):
        assert 2 <= sum(cats[c].values()) <= 3, c
    assert st["pending"] == st["questions"]


def test_question_with_todo_is_reported_pending_not_passed():
    g = GoldenSet.model_validate(
        {
            "version": "t",
            "questions": [
                {
                    "id": "q",
                    "question": "שאלה",
                    "category": "direct_source",
                    "difficulty": "easy",
                    "required_refs": ["TODO: ref"],
                    "rubric": [{"id": "v", "description": "d", "kind": "verifier_passes"}],
                }
            ],
        }
    )
    out = RunOutput(
        answer=Answer(
            disclaimer="d",
            claims=[
                Claim(
                    type=ClaimType.QUOTE, text="x", citations=[Citation(ref="R", quoted_text="אבג")]
                )
            ],
        ),
        retrieved_refs=["R"],
    )
    r = run_eval(g, lambda _q: out, InMemorySourceStore({"R": "אבג"})).results[0]
    assert r.status == "pending" and r.golden_pending == ["required_refs[0]"]
