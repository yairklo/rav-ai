import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rav_evals.cli import main as cli_main
from rav_evals.golden import GoldenSet, is_placeholder, load_golden
from rav_evals.models import EvalsBase, Run, run_from_result
from rav_evals.runner import JudgeVerdict, RunOutput, run_eval
from rav_evals.schema import Answer, Citation, Claim, ClaimType
from rav_evals.verifier import InMemorySourceStore

DRAFT = Path(__file__).parent.parent / "golden" / "hanukkah.draft.yaml"
SRC = {"R 1": "בית שמאי אומרים מפחית והולך ובית הלל אומרים מוסיף והולך"}
STORE = InMemorySourceStore(SRC)

GOLDEN_YAML = """
version: t1
status: reviewed
questions:
  - id: q1
    question: שאלה
    category: dispute
    difficulty: easy
    status: reviewed
    required_refs: ["R 1"]
    required_sides:
      - {label: ש, terms: ["בית שמאי"]}
      - {label: ה, terms: ["בית הלל"]}
    rubric:
      - {id: both, description: d, kind: mentions_all, terms: ["בית שמאי", "בית הלל"]}
      - {id: none, description: d, kind: mentions_none, terms: ["אסור"]}
      - {id: cite, description: d, kind: cites_ref, ref: "R 1"}
      - {id: ctype, description: d, kind: has_claim_type, claim_type: system_conclusion}
      - {id: ver, description: d, kind: verifier_passes}
      - {id: j, description: d, kind: llm}
"""


def good_output(_q):
    return RunOutput(
        model="fake",
        retrieved_refs=["R 1"],
        answer=Answer(
            disclaimer="דעת המערכת",
            claims=[
                Claim(
                    type=ClaimType.QUOTE,
                    text="המחלוקת",
                    citations=[Citation(ref="R 1", quoted_text="בית שמאי אומרים מפחית והולך")],
                ),
                Claim(
                    type=ClaimType.INTERPRETATION,
                    text="בית הלל מוסיפים",
                ),
                Claim(type=ClaimType.SYSTEM_CONCLUSION, text="מסקנה", flagged_as_system=True),
            ],
        ),
    )


@pytest.fixture
def golden(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(GOLDEN_YAML, encoding="utf-8")
    return load_golden(p)


def status_map(result):
    return {c.id: c.status for c in result.checks}


def test_draft_file_loads_and_is_marked_draft():
    g = load_golden(DRAFT)
    assert g.status == "draft" and len(g.questions) == 3
    assert all(q.status == "draft" for q in g.questions)
    # no invented refs: every required ref is a placeholder
    assert all(is_placeholder(r) for q in g.questions for r in q.required_refs)
    assert {q.category.value for q in g.questions} <= {
        "direct_source",
        "dispute",
        "modern",
        "false_premise",
        "out_of_corpus",
    }


def test_invalid_category_rejected():
    with pytest.raises(ValueError):
        GoldenSet.model_validate(
            {
                "version": "x",
                "questions": [
                    {"id": "a", "question": "q", "category": "bogus", "difficulty": "easy"}
                ],
            }
        )


def test_run_good_answer(golden):
    report = run_eval(golden, good_output, STORE)
    r = report.results[0]
    st = status_map(r)
    assert r.passed
    assert st["j"] == "pending_judge"
    assert all(v == "passed" for k, v in st.items() if k != "j")


def test_judge_hook_is_called(golden):
    class J:
        def judge(self, q, out, item):
            return JudgeVerdict(passed=False, reason="meh")

    r = run_eval(golden, good_output, STORE, judge=J()).results[0]
    assert status_map(r)["j"] == "failed" and not r.passed


def test_missing_requirements_fail(golden):
    def bad(_q):
        return RunOutput(
            retrieved_refs=[],
            answer=Answer(
                disclaimer="d", claims=[Claim(type=ClaimType.INTERPRETATION, text="אסור")]
            ),
        )

    r = run_eval(golden, bad, STORE).results[0]
    st = status_map(r)
    assert not r.passed
    for k in ("required_ref:R 1", "side:ש", "side:ה", "both", "none", "cite", "ctype"):
        assert st[k] == "failed", k


def test_answer_fn_exception_is_captured(golden):
    def boom(_q):
        raise RuntimeError("kaput")

    report = run_eval(golden, boom, STORE)
    assert report.results[0].error and "kaput" in report.results[0].error
    assert report.summary()["errors"] == 1


def test_placeholders_are_skipped_not_failed():
    g = load_golden(DRAFT)
    report = run_eval(g, lambda q: good_output(q), STORE)
    for r in report.results:
        assert "failed" not in {
            c.status for c in r.checks if c.id.startswith(("required_", "side:"))
        }
    assert report.summary()["draft_questions"] == 3


def test_report_json_and_markdown(golden, tmp_path):
    def bad(_q):
        return RunOutput(
            retrieved_refs=["R 1"],
            answer=Answer(
                disclaimer="d",
                claims=[
                    Claim(
                        type=ClaimType.QUOTE,
                        text="x",
                        citations=[Citation(ref="R 1", quoted_text="לא קיים")],
                    )
                ],
            ),
        )

    report = run_eval(golden, bad, STORE)
    jp, mp = report.write(tmp_path / "out")
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["summary"]["verifier_failure_codes"] == {"V2:quote_not_in_source": 1}
    assert data["summary"]["by_category"]["dispute"] == {"questions": 1, "passed": 0}
    md = mp.read_text(encoding="utf-8")
    assert "V2:quote_not_in_source" in md and "FAIL" in md


def test_runs_table_roundtrip(golden):
    report = run_eval(golden, good_output, STORE)
    engine = create_engine("sqlite://")
    EvalsBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(run_from_result(report.results[0], "שאלה", golden.version))
        s.commit()
        row = s.scalars(select(Run)).one()
        assert row.verification_passed is True
        assert row.retrieved_sources == ["R 1"]
        assert row.answer["claims"][0]["type"] == "quote"
        assert row.model == "fake" and row.id


def test_cli(tmp_path, monkeypatch, golden):
    (tmp_path / "g.yaml").write_text(GOLDEN_YAML, encoding="utf-8")
    (tmp_path / "src.json").write_text(json.dumps(SRC, ensure_ascii=False), encoding="utf-8")
    # self-contained answer module
    (tmp_path / "fakeans.py").write_text(
        "from rav_evals.runner import RunOutput\n"
        "from rav_evals.schema import Answer, Claim, ClaimType\n"
        "def fn(q):\n"
        "    c = Claim(type=ClaimType.INTERPRETATION, text='x')\n"
        "    return RunOutput(answer=Answer(disclaimer='d', claims=[c]))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = cli_main(
        ["g.yaml", "--answer-fn", "fakeans:fn", "--sources", "src.json", "--out-dir", "o"]
    )
    assert code == 1  # required ref etc. not satisfied
    assert (tmp_path / "o" / "report.json").exists() and (tmp_path / "o" / "report.md").exists()
