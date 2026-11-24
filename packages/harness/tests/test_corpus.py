"""Tools and run logging against a real (throw-away) Postgres DB."""

from __future__ import annotations

from rav_evals.golden import GoldenQuestion, GoldenSet

from conftest import BER_TEXT, FakeLLM, answer_args, call, reply
from rav_harness import HarnessConfig
from rav_harness.integration import DbSourceStore, build_harness, run_golden
from rav_harness.runs import Run
from rav_harness.tools import CorpusTools


def test_get_segment_returns_raw_text_with_license_and_tracks_retrieval(session):
    t = CorpusTools(session, mode="personal")
    r = t.get_segment("Berakhot 2a:1")
    v = r["versions"][0]
    assert v["text"] == BER_TEXT and v["license"] == "Public Domain" and not v["withheld"]
    assert t.retrieved_refs == ["Berakhot 2a:1"]


def test_license_gates_apply_per_deployment_mode(session):
    personal, public = CorpusTools(session, mode="personal"), CorpusTools(session, mode="public")
    assert personal.get_segment("NC Book 1:1")["versions"][0]["text"] is not None
    assert public.get_segment("NC Book 1:1")["versions"][0]["withheld"] is True
    assert public.retrieved_refs == []  # withheld text is not "retrieved"
    secret = personal.get_segment("Secret Book 1:1")["versions"][0]
    assert secret["text"] is None and secret["withheld"]
    assert [r["ref"] for r in public.search_text("שמע")["results"]] == ["Berakhot 2a:1"]
    assert personal.search_text("חסוי")["results"] == []  # can_index=False: never searchable


def test_get_segment_unknown_ref_lists_children(session):
    r = CorpusTools(session).get_segment("Berakhot 2a")
    assert r["found"] is False and r["child_refs"] == ["Berakhot 2a:1", "Berakhot 2a:2"]


def test_neighbors_layers_and_edge_filter(session):
    t = CorpusTools(session, mode="personal")
    r = t.neighbors("Berakhot 2a:1")
    assert {n["ref"] for n in r["above"]} == {"Rashi on Berakhot 2a:1:1", "NC Book 1:1"}
    only = t.neighbors("Berakhot 2a:1", edge_types=["commentary"])
    assert [n["ref"] for n in only["above"]] == ["Rashi on Berakhot 2a:1:1"]
    assert "Rashi on Berakhot 2a:1:1" in t.retrieved_refs  # short snippet, fully seen
    assert t.neighbors("Nope 1:1")["found"] is False


def test_search_text_ranks_and_dedupes(session):
    t = CorpusTools(session, mode="personal")
    r = t.search_text('מאימתי קורין את שמע בערבין?')
    assert r["results"][0]["ref"] == "Berakhot 2a:1"
    assert t.search_text("???")["results"] == []


def test_resolve_citation_uses_textnorm_and_checks_existence(session):
    r = CorpusTools(session).resolve_citation("עיין בברכות ב ע\"א")
    assert r["citations"], r
    c = r["citations"][0]
    assert c["ref"] == "Berakhot 2a" and c["exists"] is True
    assert "Berakhot 2a:1" in c["segment_refs"]


def test_every_run_is_recorded_with_messages_tools_tokens_cost(session):
    llm = FakeLLM([reply(call("get_segment", ref="Berakhot 2a:1")),
                   reply(call("submit_answer", "c2", **answer_args()), cost=0.02)])
    res = build_harness(session, llm, HarnessConfig()).run("מתי?", question_id="q7")
    run = session.get(Run, res.run_id)
    assert run.question_id == "q7" and run.model == "fake/model" and run.error is None
    assert (run.input_tokens, run.output_tokens, run.steps) == (200, 40, 2)
    assert abs(run.cost_usd - 0.03) < 1e-9
    assert run.retrieved_sources == ["Berakhot 2a:1"]
    assert [c["name"] for c in run.tool_calls] == ["get_segment", "submit_answer"]
    assert run.messages[0]["role"] == "system" and run.prompt_hash and run.answer["claims"]


def test_failed_runs_are_recorded_too(session):
    llm = FakeLLM([reply(call("get_segment"), content="x")] * 2)
    res = build_harness(session, llm, HarnessConfig(max_steps=2)).run("q")
    run = session.get(Run, res.run_id)
    assert run.answer is None and "max_steps" in run.error


def _golden(*ids):
    return GoldenSet(version="g-test", questions=[
        GoldenQuestion(id=i, question="מתי קוראים קריאת שמע?", category="direct_source",
                       difficulty="easy") for i in ids])


def test_answer_fn_plugs_into_evals_and_verifier_passes_a_real_quote(session):
    llm = FakeLLM([reply(call("get_segment", ref="Berakhot 2a:1")),
                   reply(call("submit_answer", **answer_args()))])
    report = run_golden(session, _golden("q1"), llm, HarnessConfig())
    r = report.results[0]
    assert r.error is None and r.verification.passed, r.verification
    assert r.output.retrieved_refs == ["Berakhot 2a:1"]
    run = session.query(Run).filter_by(question_id="q1").one()
    assert run.golden_version == "g-test" and run.verification_passed is True


def test_verifier_catches_a_fabricated_quote(session):
    bad = answer_args(claims=[{
        "type": "quote", "text": "x",
        "citations": [{"ref": "Berakhot 2a:1", "quoted_text": "דברים שלא נאמרו מעולם"}]}])
    llm = FakeLLM([reply(call("get_segment", ref="Berakhot 2a:1")),
                   reply(call("submit_answer", **bad))])
    report = run_golden(session, _golden("q2"), llm, HarnessConfig())
    assert report.results[0].verification.failure_codes() == ["V2:quote_not_in_source"]
    assert session.query(Run).filter_by(question_id="q2").one().verification_passed is False


def test_answer_fn_raises_when_no_answer_so_run_eval_marks_error(session):
    report = run_golden(session, _golden("q3"), FakeLLM([reply(content="x")] * 2),
                        HarnessConfig(max_steps=2))
    assert report.results[0].error is not None


def test_db_source_store_respects_license(session):
    store = DbSourceStore(session, mode="public")
    assert store.get_text("Berakhot 2a:1") == BER_TEXT
    assert store.get_text("NC Book 1:1") == ""  # exists, nothing displayable
    assert store.get_text("Nope") is None
