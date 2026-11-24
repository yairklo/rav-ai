"""Agent loop with a fake model and fake tools: no network, no database."""

from __future__ import annotations

from conftest import FakeLLM, answer_args, call, reply
from rav_harness import Harness, HarnessConfig
from rav_harness.llm import ToolCall, parse_tool_call
from rav_harness.prompt import DEFAULT_DISCLAIMER


class FakeTools:
    def __init__(self, results=None, retrieved=("Berakhot 2a:1",)):
        self.retrieved_refs: list[str] = []
        self._pending = list(retrieved)
        self.results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def specs(self):
        return [{"type": "function", "function": {"name": "get_segment", "parameters": {}}}]

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "boom":
            raise RuntimeError("db down")
        self.retrieved_refs.extend(self._pending)
        self._pending = []
        return self.results.get(name, {"found": True})


def harness(llm, tools=None, **cfg):
    return Harness(llm, tools or FakeTools(), HarnessConfig(**cfg))


def test_tool_then_submit_records_everything():
    llm = FakeLLM([
        reply(call("get_segment", ref="Berakhot 2a:1")),
        reply(call("submit_answer", "c2", **answer_args()), cost=0.02, tin=200, tout=50),
    ])
    res = harness(llm).run("מתי קוראים קריאת שמע של ערבית?")
    assert res.error is None and res.answer is not None
    assert res.steps == 2
    assert (res.input_tokens, res.output_tokens) == (300, 70)
    assert abs(res.cost_usd - 0.03) < 1e-9
    assert res.retrieved_refs == ["Berakhot 2a:1"]
    assert [c["name"] for c in res.tool_calls] == ["get_segment", "submit_answer"]
    roles = [m["role"] for m in res.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert res.answer.is_psak is False
    # the model was offered the corpus tools plus submit_answer, with the configured model
    assert llm.calls[0]["tools"] == ["get_segment", "submit_answer"]
    assert llm.calls[0]["model"] == HarnessConfig().model


def test_empty_disclaimer_is_filled_and_psak_cannot_be_set():
    args = answer_args(disclaimer="", is_psak=True)
    llm = FakeLLM([reply(call("get_segment")), reply(call("submit_answer", **args))])
    ans = harness(llm).run("q").answer
    assert ans.disclaimer == DEFAULT_DISCLAIMER
    assert ans.is_psak is False


def test_unflagged_system_claim_is_bounced_then_accepted():
    bad = answer_args(claims=[{"type": "chidush", "text": "חידוש", "flagged_as_system": False}])
    good = answer_args(claims=[{"type": "chidush", "text": "חידוש של המערכת, בהסתייגות",
                                "flagged_as_system": True}])
    llm = FakeLLM([reply(call("submit_answer", **bad)), reply(call("submit_answer", **good))])
    res = harness(llm).run("q")
    assert res.answer is not None
    assert len(res.rejected_answers) == 1 and "flagged_as_system" in res.rejected_answers[0]
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert '"accepted": false' in tool_msg["content"]


def test_citing_an_unretrieved_ref_is_bounced():
    args = answer_args()  # cites Berakhot 2a:1, but no tool ever returned it
    llm = FakeLLM([reply(call("submit_answer", **args))] * 3)
    res = harness(llm, FakeTools(retrieved=()), max_answer_retries=2).run("q")
    assert res.answer is None
    assert "answer rejected 3 times" in res.error and "not retrieved" in res.error


def test_max_steps_without_answer_is_an_error_not_an_exception():
    llm = FakeLLM([reply(call("get_segment", cid=f"c{i}")) for i in range(5)])
    res = harness(llm, max_steps=3).run("q")
    assert res.answer is None and "max_steps" in res.error and res.steps == 3


def test_text_only_reply_is_nudged_to_submit():
    llm = FakeLLM([reply(content="I think it's evening"), reply(call("get_segment")),
                   reply(call("submit_answer", **answer_args()))])
    res = harness(llm).run("q")
    assert res.answer is not None
    assert "Call submit_answer" in llm.calls[1]["messages"][-1]["content"]


def test_tool_exception_and_bad_json_arguments_are_reported_to_the_model():
    broken = parse_tool_call("c9", "get_segment", "{not json")
    assert broken.arguments_error
    llm = FakeLLM([
        reply(ToolCall("c1", "boom", {}, "{}"), broken),
        reply(call("get_segment"), call("submit_answer", "c3", **answer_args())),
    ])
    res = harness(llm).run("q")
    assert res.answer is not None
    errors = [c["error"] for c in res.tool_calls if c["error"]]
    assert any("RuntimeError" in e for e in errors)
    assert any("invalid arguments" in e for e in errors)


def test_llm_failure_is_recorded_and_unknown_cost_stays_none():
    class Down:
        def complete(self, *a, **k):
            raise ConnectionError("no network")

    res = harness(Down()).run("q")
    assert res.answer is None and "ConnectionError" in res.error and res.cost_usd is None

    llm = FakeLLM([reply(call("get_segment"), cost=None),
                   reply(call("submit_answer", **answer_args()), cost=None)])
    assert harness(llm).run("q").cost_usd is None


def test_recorder_receives_result_and_ids():
    seen = []
    llm = FakeLLM([reply(call("get_segment")), reply(call("submit_answer", **answer_args()))])
    h = Harness(llm, FakeTools(), HarnessConfig(),
                recorder=lambda r, qid, gv: seen.append((r.question, qid, gv)) or "run-1")
    res = h.run("שאלה", question_id="q1", golden_version="g1")
    assert seen == [("שאלה", "q1", "g1")] and res.run_id == "run-1"
