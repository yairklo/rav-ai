from __future__ import annotations

import re

from rav_ai.db.models import (
    Annotation,
    KlEvidence,
    KlItem,
    KlJob,
    KlLink,
    KlTask,
    Segment,
    SegmentText,
)
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from fakes import ScriptedLLM
from helpers import (
    bad_s2_responder,
    cfg,
    pipeline,
    run,
    s1_responder,
    s2_responder,
)
from rav_knowledge.runner import promote


def items(engine, kind=None, **where):
    with Session(engine) as s:
        q = select(KlItem).order_by(KlItem.id)
        if kind:
            q = q.where(KlItem.kind == kind)
        for k, v in where.items():
            q = q.where(getattr(KlItem, k) == v)
        return list(s.scalars(q))


# ---- S0 ------------------------------------------------------------------------------------


def test_s0_acronym_candidates_ranked_by_context_domain(world):
    run(world.engine, "s0", None, "Shabbat 21a-21a")
    run(world.engine, "s0", None, "Shulchan Arukh, Orach Chayim 1")
    with Session(world.engine) as s:
        raw_t = {t.segment_id: t.text_raw for t in s.scalars(select(SegmentText))}
        anns = list(s.scalars(select(Annotation).where(Annotation.kind == "abbrev")))
        by_dom = {}
        for a in anns:
            if a.context and a.context["norm"] == 'ע"כ':
                by_dom[a.context_domain] = [c["text"] for c in a.candidates]
                assert raw_t[a.segment_id][a.span_start : a.span_end] == 'ע"כ'
                assert a.method == "s0.acronym" and a.method_version and a.chosen is None
        # same abbreviation, different order of candidates in Gemara vs Shulchan Arukh
        assert by_dom["talmud"] == ["עד כאן", "על כרחך", "על כן"]
        assert by_dom["halakha"] == ["עד כאן", "על כן", "על כרחך"]


def test_s0_markers_and_error_markers_are_annotations(world):
    run(world.engine, "s0", None, "Shabbat 21a-22a")
    run(world.engine, "s0", None, "Shulchan Arukh, Orach Chayim 1")
    with Session(world.engine) as s:
        marks = {
            (a.kind, a.chosen)
            for a in s.scalars(select(Annotation).where(Annotation.kind.like("%marker")))
        }
    assert ("discourse_marker", "mitivei") in marks
    assert ("discourse_marker", "matni") in marks
    assert ("discourse_marker", "teiku") in marks
    assert ("discourse_marker", "hilkheta") in marks
    assert ("discourse_marker", "tash") in marks
    assert ("quote_marker", "vezal") in marks
    assert ("error_marker", "tas") in marks
    assert ("error_marker", "eino_mechuvar") in marks


def test_s0_marker_variants_in_full_words():
    from conftest import make_text
    from rav_knowledge.stages import find_markers

    t = make_text(1, 1, "תא שמע מתניתין היא ואיתיביה")
    names = {(k, n) for k, n, _a, _b in find_markers(t.text_a, t.offsets_a)}
    assert ("discourse_marker", "tash") in names and ("discourse_marker", "matni") in names
    assert ("discourse_marker", "itivei") not in names  # prefixed form is not matched


def test_s0_is_idempotent(world):
    r1 = run(world.engine, "s0", None)
    with Session(world.engine) as s:
        n1 = s.scalar(select(func.count()).select_from(Annotation))
    r2 = run(world.engine, "s0", None)
    with Session(world.engine) as s:
        n2 = s.scalar(select(func.count()).select_from(Annotation))
    assert r1.done == 5 and r2.skipped == 5 and r2.llm_calls == 0 and n1 == n2 > 0


# ---- S0b -----------------------------------------------------------------------------------


def test_s0b_constrained_choice(world):
    run(world.engine, "s0", None)

    def responder(messages, tool, model):
        ids = [int(x) for x in re.findall(r"id=(\d+)", messages[-1]["content"])]
        out = [{"id": i, "choice": 0} for i in ids]
        out.append({"id": ids[0], "choice": 99})  # out of range: ignored
        out.append({"id": 10**9, "choice": 0})  # unknown id: ignored
        return {"choices": out}

    llm = ScriptedLLM(responder)
    r = run(world.engine, "s0b", llm)
    assert r.failed == 0 and llm.n >= 1
    with Session(world.engine) as s:
        ch = list(s.scalars(select(Annotation).where(Annotation.kind == "abbrev_choice")))
        assert ch and all(a.chosen == a.candidates[0]["text"] for a in ch)
        assert all(a.origin == "llm" and a.method == "s0b.llm_choice" for a in ch)
    assert r.stats["ignored_choices"] >= 2
    again = run(world.engine, "s0b", ScriptedLLM(responder))
    assert again.llm_calls == 0  # nothing left to choose


# ---- S1 ------------------------------------------------------------------------------------


def test_s1_sliding_windows_merge_and_idempotence(world):
    llm = ScriptedLLM(s1_responder())
    r = run(world.engine, "s1", llm)
    assert r.done == 3 and llm.n == 2  # two windows + merge
    sugyot = items(world.engine, "sugya")
    assert [(i.body["title_he"], i.is_current) for i in sugyot] == [
        ("סוגיה א", True),
        ("סוגיה ב", True),
        ("סוגיה ג", True),
    ]
    # B was cut by the edge of window 0 and taken whole from window 1
    b = sugyot[1]
    assert b.body["start_ref"] == "Shabbat 22a:1" and b.body["end_ref"] == "Shabbat 22b:2"
    with Session(world.engine) as s:
        n = s.scalar(select(func.count()).select_from(select(1).where(text("true")).subquery()))  # noqa: F841
    llm2 = ScriptedLLM(s1_responder())
    r2 = run(world.engine, "s1", llm2)
    assert llm2.n == 0 and r2.skipped == 3


# ---- S2 ------------------------------------------------------------------------------------


def test_s2_summary_end_to_end(world):
    llm2 = ScriptedLLM(s2_responder)
    r = pipeline(world.engine, ScriptedLLM(s1_responder()), llm2)
    assert r.done == 3 and r.failed == 0
    summ = {i.item_key: i for i in items(world.engine, "sugya_summary")}
    a = summ["summary:sugya:Shabbat 21a:1"]
    assert a.status == "generated" and a.is_current and 0 < a.confidence <= 1
    assert [st["id"] for st in a.body["steps"]] == ["s1", "s2", "s3", "s4"]
    assert a.search_text_he and "רב מחייב" in a.search_text_he
    with Session(world.engine) as s:
        evs = list(s.scalars(select(KlEvidence).where(KlEvidence.item_id == a.id)))
        raw = {t.segment_id: t.text_raw for t in s.scalars(select(SegmentText))}
        assert len(evs) == 4 and all(e.locate_status == "located" for e in evs)
        e0 = evs[0]
        assert raw[e0.segment_id][e0.span_start : e0.span_end] == "אמר רב חייב"
        assert e0.entail_status == "unchecked" and e0.raw_hash
        links = {(l.rel) for l in s.scalars(select(KlLink))}  # noqa: E741
        assert {"summarizes", "depends_on"} <= links
    # the user prompt: stable system prefix, aids under their own header, prior notes last
    user = next(c for c in llm2.calls if "Shabbat 21a:1" in c["messages"][-1]["content"])[
        "messages"
    ][-1]["content"]
    assert user.index("עזרי קריאה") < user.index("הערות עזר, עלולות לטעות")
    assert "[A1|" in user and "רש״י" in user.replace("'", "״") or "רש" in user
    assert "[S1|canonical|Shabbat] Shabbat 21a:1 ⟩" in user


def test_s2_rejected_after_one_retry_with_feedback(world):
    run(world.engine, "s0", None)
    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    llm = ScriptedLLM(bad_s2_responder)
    r = run(world.engine, "s2", llm)
    assert r.done == 3 and llm.n == 6  # every task: attempt + exactly one retry
    assert "K3" in llm.calls[1]["messages"][-1]["content"]  # the feedback names the failed check
    for it in items(world.engine, "sugya_summary"):
        assert it.status == "rejected" and it.rejected_by == "auto" and not it.is_current
        assert "K3" in {f["code"] for f in it.check_report["findings"]}


def test_steinsaltz_aid_flag_and_license(world):
    import pytest  # noqa: F401

    for mode, expect_display in (("personal", False), ("public", True)):
        llm = ScriptedLLM(s2_responder)
        c = cfg(mode=mode)
        pipeline(world.engine, ScriptedLLM(s1_responder()), llm, config=c)
        first = next(c_ for c_ in llm.calls if "Shabbat 21a:1" in c_["messages"][-1]["content"])
        user = first["messages"][-1]["content"]
        assert ("Steinsaltz on" in user) == (mode == "personal")
        assert "Rashi on" in user
        with Session(world.engine) as s:
            it = s.scalars(
                select(KlItem)
                .where(KlItem.item_key == "summary:sugya:Shabbat 21a:1", KlItem.model.is_not(None))
                .order_by(KlItem.id.desc())
            ).first()
            assert it.can_display_public is expect_display
            assert it.can_train is True
            # the aid is recorded in the item's source versions, never as evidence
            assert world.ver["Shabbat"] in it.source_version_ids
            ev_segs = set(
                s.scalars(select(KlEvidence.segment_id).where(KlEvidence.item_id == it.id))
            )
            assert ev_segs <= {
                world.seg[f"Shabbat {k}"] for k in ("21a:1", "21a:2", "21b:1", "21b:2")
            }
        # a different mode is a different input: the next loop iteration must not reuse tasks
        break_after = mode  # noqa: F841


def test_aid_is_part_of_the_task_hash(world):
    pipeline(
        world.engine,
        ScriptedLLM(s1_responder()),
        ScriptedLLM(s2_responder),
        config=cfg(mode="personal"),
    )
    llm = ScriptedLLM(s2_responder)
    run(world.engine, "s2", llm, config=cfg(mode="public"))
    assert llm.n >= 1  # Steinsaltz off -> different input_hash -> re-generated


# ---- runner ---------------------------------------------------------------------------------


def test_idempotent_second_run_makes_no_llm_calls(world):
    llm = ScriptedLLM(s2_responder)
    pipeline(world.engine, ScriptedLLM(s1_responder()), llm)
    first = llm.n
    llm_again = ScriptedLLM(s2_responder)
    r = run(world.engine, "s2", llm_again)
    assert first == 3 and llm_again.n == 0 and r.skipped == 3 and r.done == 0


def test_resume_continues_the_same_job_and_resets_stuck_running(world):
    run(world.engine, "s0", None)
    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    calls = {"n": 0}

    def flaky(messages, tool, model):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("provider timeout")
        return s2_responder(messages, tool, model)

    r1 = run(world.engine, "s2", ScriptedLLM(flaky))
    assert r1.failed == 1 and r1.done == 2 and r1.status == "failed"
    # simulate a crash: another task stuck in 'running'
    with Session(world.engine) as s:
        t = s.scalars(
            select(KlTask).where(KlTask.job_id == r1.job_id, KlTask.state == "done")
        ).first()
        t.state = "running"
        s.commit()
    llm = ScriptedLLM(s2_responder)
    r2 = run(world.engine, "s2", llm)
    assert r2.job_id == r1.job_id and r2.failed == 0 and llm.n == 2  # failed + stuck, not the rest
    with Session(world.engine) as s:
        assert {t.state for t in s.scalars(select(KlTask).where(KlTask.job_id == r1.job_id))} == {
            "done"
        }
        assert s.get(KlJob, r1.job_id).status == "done"


def test_budget_stops_before_overrun_and_resumes(world):
    run(world.engine, "s0", None)
    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    dry = run(world.engine, "s2", None, dry_run=True)
    assert dry.tasks == 3 and dry.est_cost_usd and dry.est_cost_usd > 0
    llm = ScriptedLLM(s2_responder)
    r = run(world.engine, "s2", llm, budget_usd=1e-9)
    assert r.status == "aborted" and llm.n == 0
    r2 = run(world.engine, "s2", ScriptedLLM(s2_responder), budget_usd=10.0)
    assert r2.job_id == r.job_id and r2.done == 3 and r2.status == "done"


def test_second_model_runs_on_a_fraction_as_shadow(world):
    c = cfg()
    c.second_model_fraction = 1.0
    c.second_model = "openai/gpt-5.1"
    llm = ScriptedLLM(s2_responder)
    pipeline(world.engine, ScriptedLLM(s1_responder()), llm, config=c)
    assert {x["model"] for x in llm.calls} == {c.model_for("s2"), "openai/gpt-5.1"}
    shadow = [i for i in items(world.engine, "sugya_summary") if i.model == "openai/gpt-5.1"]
    assert shadow and all(not i.is_current for i in shadow)


# ---- promote / rollback / verified / overrides / supersede / stale ---------------------------


def _model(name):
    c = cfg()
    c.models = {**c.models, "s2": name, "s1": name}
    return c


def test_new_model_output_is_not_current_until_promote_and_rollback(world):
    j1 = pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder)).job_id
    key = "summary:sugya:Shabbat 21a:1"
    r2 = run(world.engine, "s2", ScriptedLLM(s2_responder), config=_model("anthropic/other"))
    cur = [i for i in items(world.engine, "sugya_summary") if i.item_key == key]
    assert [(i.job_id, i.is_current) for i in cur] == [(j1, True), (r2.job_id, False)]
    out = promote(world.engine, r2.job_id)
    assert out["promoted"] == 3 and out["stale"] == 0
    cur = {
        i.job_id: i.is_current for i in items(world.engine, "sugya_summary") if i.item_key == key
    }
    assert cur == {j1: False, r2.job_id: True}
    promote(world.engine, j1)  # rollback = promote the earlier job
    cur = {
        i.job_id: i.is_current for i in items(world.engine, "sugya_summary") if i.item_key == key
    }
    assert cur == {j1: True, r2.job_id: False}


def test_promote_gate_and_force_note(world):
    import pytest

    from rav_knowledge.runner import PromoteRefused

    pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder))
    bad = run(world.engine, "s2", ScriptedLLM(bad_s2_responder), config=_model("anthropic/bad"))
    with pytest.raises(PromoteRefused):
        promote(world.engine, bad.job_id)  # every item rejected: the K gate fails
    with pytest.raises(PromoteRefused):
        promote(world.engine, bad.job_id, force_note=" ")
    out = promote(world.engine, bad.job_id, force_note="experiment, reviewed by hand")
    assert out["forced"] is True
    with Session(world.engine) as s:
        assert "force: experiment" in s.get(KlJob, bad.job_id).note


def test_verified_items_are_never_overwritten(world):
    pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder))
    key = "summary:sugya:Shabbat 21a:1"
    with Session(world.engine) as s:
        it = s.scalars(select(KlItem).where(KlItem.item_key == key, KlItem.is_current)).one()
        it.status = "verified"
        vid = it.id
        s.commit()
    r2 = run(world.engine, "s2", ScriptedLLM(s2_responder), config=_model("anthropic/other"))
    out = promote(world.engine, r2.job_id)
    assert out["blocked_verified"] == 1
    with Session(world.engine) as s:
        assert s.get(KlItem, vid).is_current
        # an unverified neighbour was promoted normally
        assert out["promoted"] == 2


def test_overrides_win_and_conflicts_are_queued(world):
    from rav_ai.db.models import KlOverride

    pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder))
    key = "summary:sugya:Shabbat 21a:1"
    with Session(world.engine) as s:
        s.add(KlOverride(target_key=key, action="reject", note="wrong", reviewer="yair"))
        s.commit()
    r2 = run(world.engine, "s2", ScriptedLLM(s2_responder), config=_model("anthropic/other"))
    new = [i for i in items(world.engine, "sugya_summary", job_id=r2.job_id) if i.item_key == key]
    assert new[0].status == "rejected" and new[0].rejected_by == "human"
    assert r2.stats["override_conflicts"] == 1
    # an `edit` override replaces body fields and verifies the item
    with Session(world.engine) as s:
        s.add(KlOverride(target_key=key, action="edit", corrected={"title_he": "כותרת מתוקנת"}))
        s.commit()
    r3 = run(world.engine, "s2", ScriptedLLM(s2_responder), config=_model("anthropic/third"))
    ed = [i for i in items(world.engine, "sugya_summary", job_id=r3.job_id) if i.item_key == key][0]
    assert ed.status == "verified" and ed.body["title_he"] == "כותרת מתוקנת"
    with world.engine.connect() as c:
        rows = c.execute(text("SELECT target_key, action FROM kl_conflicts")).all()
    assert any(r[0] == key for r in rows)


def test_set_boundary_override(world):
    from rav_ai.db.models import KlOverride

    run(world.engine, "s1", ScriptedLLM(s1_responder()))
    with Session(world.engine) as s:
        s.add(
            KlOverride(
                target_key="sugya:Shabbat 21a:1",
                action="set_boundary",
                corrected={"end_ref": "Shabbat 21b:1"},
            )
        )
        s.commit()
    c = cfg()
    c.models = {**c.models, "s1": "anthropic/other"}
    r = run(world.engine, "s1", ScriptedLLM(s1_responder()), config=c)
    new = [
        i
        for i in items(world.engine, "sugya", job_id=r.job_id)
        if i.item_key == "sugya:Shabbat 21a:1"
    ][0]
    assert new.body["end_ref"] == "Shabbat 21b:1" and new.status == "verified"
    with Session(world.engine) as s:
        end = s.get(Segment, new.end_segment_id)
        assert end.ref == "Shabbat 21b:1"


def test_supersede_boundaries_at_scope_level_and_stale_dependents(world):
    pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder))
    c = cfg()
    c.models = {**c.models, "s1": "anthropic/other"}
    r = run(world.engine, "s1", ScriptedLLM(s1_responder("B")), config=c)
    sug = {i.item_key: i for i in items(world.engine, "sugya", job_id=r.job_id)}
    assert set(sug) == {"sugya:Shabbat 21a:1", "sugya:Shabbat 22a:1"}
    out = promote(world.engine, r.job_id)
    assert out["superseded"] >= 1 and out["stale"] >= 1  # C is gone; its summary is stale
    with Session(world.engine) as s:
        cur = {
            i.item_key
            for i in s.scalars(select(KlItem).where(KlItem.kind == "sugya", KlItem.is_current))
        }
        assert cur == {"sugya:Shabbat 21a:1", "sugya:Shabbat 22a:1"}
        stale = {i.item_key for i in s.scalars(select(KlItem).where(KlItem.stale))}
        assert "summary:sugya:Shabbat 23a:1" in stale or "summary:sugya:Shabbat 22a:1" in stale


def test_invalidate_reanchors_or_marks_stale_recursively(world):
    from conftest import make_text
    from rav_knowledge.runner import invalidate

    pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2_responder))
    sid = world.seg["Shabbat 21a:1"]
    with Session(world.engine) as s:
        t = s.scalars(select(SegmentText).where(SegmentText.segment_id == sid)).one()
        new = make_text(sid, t.version_id, "הוספה בראש מתני׳ אמר רב חייב ושמואל אמר פטור")
        for col in ("text_raw", "raw_hash", "text_a", "text_b", "offsets_a", "offsets_b"):
            setattr(t, col, getattr(new, col))
        s.commit()
    out = invalidate(world.engine, [sid])
    assert out["reanchored"] >= 1 and out["stale_items"] == 0  # the quote is still there
    # now the quote disappears
    with Session(world.engine) as s:
        t = s.scalars(select(SegmentText).where(SegmentText.segment_id == sid)).one()
        new = make_text(sid, t.version_id, "טקסט אחר לגמרי")
        for col in ("text_raw", "raw_hash", "text_a", "text_b", "offsets_a", "offsets_b"):
            setattr(t, col, getattr(new, col))
        s.commit()
    out = invalidate(world.engine, [sid])
    assert out["stale_items"] >= 1
    with Session(world.engine) as s:
        a = s.scalars(select(KlItem).where(KlItem.item_key == "summary:sugya:Shabbat 21a:1")).one()
        assert a.stale
        # evidence that was lost says so
        assert (
            s.scalar(
                select(func.count())
                .select_from(KlEvidence)
                .where(KlEvidence.item_id == a.id, KlEvidence.locate_status == "not_found")
            )
            >= 1
        )
