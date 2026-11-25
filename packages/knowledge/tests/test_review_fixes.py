"""KL-1 review fixes (fix-1) and KL-1-fix-2 additions. Each test names the item it covers."""

from __future__ import annotations

import pytest
from alembic import command
from pydantic import ValidationError
from rav_ai.db.models import (
    KlItem,
    KlLink,
    KlReception,
    KlRelation,
    KlSource,
)
from rav_evals.normalization import normalize as nb
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fakes import ScriptedLLM
from helpers import cfg, input_lines, pipeline, run, s1_responder, s2_responder
from rav_knowledge import queries
from rav_knowledge.checks import SegInfo, coverage_gaps
from rav_knowledge.config import load_config
from rav_knowledge.evidence import find_span, locate
from rav_knowledge.machloket import build_machloket_candidate
from rav_knowledge.runner import PromoteRefused, apply_overrides, promote
from rav_knowledge.schemas import (
    Ev,
    Holder,
    Position,
    SourceClaim,
    SourceClaimTarget,
    Step,
    schema_version_for,
)
from rav_knowledge.sources import register_work, validate_authority_areas
from rav_knowledge.stages import merge_windows
from rav_knowledge.store import write_item
from test_checks import codes, pos, rec
from test_checks import run as run_checks_
from test_schema_migration import _cfg

# ---- 1: S1 long sugyot -------------------------------------------------------------------


def _long_responder(flags: bool):
    def r(messages, tool, model):
        if tool != "sugya":
            return None
        labels = list(input_lines(messages))
        first, last = labels[0], labels[-1]
        first_window = "Shabbat 21a:1" in messages[-1]["content"]
        base = {"start_seg": first, "end_seg": last}
        if first_window:
            return {
                "sugyot": [{"title_he": "ארוכה", **base, **({"open_end": True} if flags else {})}]
            }
        return {"sugyot": [{"title_he": "המשך", **base, **({"open_start": True} if flags else {})}]}

    return r


@pytest.mark.parametrize("flags", [True, False])  # LLM flags, or derived from the window edge
def test_1_five_page_sugya_with_window3_overlap1_comes_out_whole(world, flags):
    c = cfg()
    assert (c.window_amudim, c.window_overlap) == (3, 1)
    run(world.engine, "s0", None)
    r = run(world.engine, "s1", ScriptedLLM(_long_responder(flags)))
    assert r.failed == 0 and "coverage_holes" not in r.stats and "coverage_overlaps" not in r.stats
    with Session(world.engine) as s:
        sg = list(s.scalars(select(KlItem).where(KlItem.kind == "sugya")))
        assert len(sg) == 1 and sg[0].item_key == "sugya:Shabbat 21a:1"
        assert sg[0].body["end_ref"] == "Shabbat 23a:2" and sg[0].title_he == "ארוכה"
        assert sg[0].check_report is None


def test_1_hole_is_a_report_error_and_queues_the_neighbours(world):
    def r(messages, tool, model):
        if tool != "sugya":
            return None
        labels = list(input_lines(messages))
        if "Shabbat 21a:1" in messages[-1]["content"]:
            return {"sugyot": [{"title_he": "א", "start_seg": "S1", "end_seg": "S3"}]}
        return {"sugyot": [{"title_he": "ב", "start_seg": "S2", "end_seg": labels[-1]}]}

    run(world.engine, "s0", None)
    rep = run(world.engine, "s1", ScriptedLLM(r))
    assert rep.stats["coverage_holes"] == 2  # segments 4 and 5 (index 3, 4) belong to no sugya
    with Session(world.engine) as s:
        sg = list(s.scalars(select(KlItem).where(KlItem.kind == "sugya").order_by(KlItem.id)))
        assert len(sg) == 2
        for it in sg:
            assert it.check_report["ok"] is False and it.check_report["queue_priority"] is True
            assert it.check_report["findings"][0]["code"] == "K8"
            assert "hole" in it.check_report["findings"][0]["message"]


def test_1_overlap_is_reported_too(world):
    def r(messages, tool, model):
        if tool != "sugya":
            return None
        if "Shabbat 21a:1" in messages[-1]["content"]:
            return {"sugyot": [{"title_he": "א", "start_seg": "S1", "end_seg": "S5"}]}
        return {"sugyot": [{"title_he": "ב", "start_seg": "S1", "end_seg": "S6"}]}

    run(world.engine, "s0", None)
    rep = run(world.engine, "s1", ScriptedLLM(r))
    assert rep.stats["coverage_overlaps"] >= 1


def test_1_coverage_gaps_and_merge_are_pure():
    assert coverage_gaps([(0, 2), (3, 5)], list(range(6))) == ([], [])
    assert coverage_gaps([(0, 2), (4, 5)], list(range(6))) == ([3], [])
    assert coverage_gaps([(0, 3), (3, 5)], list(range(6))) == ([], [3])

    def sg(a, b, os=False, oe=False, t="x"):
        return dict(title_he=t, start_idx=a, end_idx=b, start_id=a, end_id=b,
                    start_ref=f"r{a}", end_ref=f"r{b}", open_start=os, open_end=oe)  # fmt: skip

    w = [
        {"idx": [0, 1, 2], "sugyot": [sg(0, 2, oe=True, t="first")]},
        {"idx": [2, 3, 4], "sugyot": [sg(2, 3, os=True, oe=True)]},
        {
            "idx": [3, 4, 5],
            "sugyot": [
                sg(3, 5, os=True),
            ],
        },
    ]
    spans, universe = merge_windows(w)
    assert [(x["start_idx"], x["end_idx"], x["title_he"]) for x in spans] == [(0, 5, "first")]
    assert universe == [0, 1, 2, 3, 4, 5]


# ---- 2: evidence location ----------------------------------------------------------------


def test_2_word_boundaries_short_quotes_and_multifragment_uniqueness():
    # a quote must start and end on a word boundary: "רב" is not inside "רבא"
    assert find_span("רב", nb("אמר רבא")) is None
    assert find_span("אמר ר", nb("אמר רב חייב")) is None
    assert find_span("אמר רב", nb("אמר רב חייב"))[:2] == (0, 6)
    # shorter than the threshold (< 2 words or < 6 chars) is ambiguous even if it occurs once
    assert find_span("רבא", nb("אמר רבא חייב"))[2] is False
    assert find_span("אמר רב חייב", nb("אמר רב חייב"))[2] is True
    assert find_span("רב חי", nb("אמר רב חי שם"))[2] is False  # 5 chars
    assert find_span("אמר רב", nb("אמר רב חייב"), min_words=3)[2] is False  # config threshold
    # a multi-fragment quote is unique only with exactly one ordered occurrence
    assert find_span("אמר רב ... פטור", nb("אמר רב חייב ושמואל פטור"))[2] is True
    twice = "אמר רב חייב ושמואל פטור אמר רב חייב ושמואל פטור"
    assert find_span("אמר רב ... פטור", nb(twice))[2] is False
    assert find_span("פטור ... אמר רב", nb("אמר רב חייב ושמואל פטור")) is None  # wrong order


def test_2_locate_marks_short_quote_ambiguous():
    from test_evidence import seg

    assert locate("ושמואל", seg()).status == "ambiguous"
    assert locate("ושמואל אמר פטור", seg()).status == "located"


# ---- 3: K3 and default roles -------------------------------------------------------------


def test_3_default_roles_are_claim_bearing():
    st = Step(id="s1", move="statement", text_he="x", evidence=[Ev(seg="S1", quote="q")])
    assert st.evidence[0].role == "states"
    st2 = Step(id="s1", move="statement", text_he="x",
               evidence=[Ev(seg="S1", quote="q", role="supports")])  # fmt: skip
    assert st2.evidence[0].role == "supports"  # an explicit role is kept
    p = pos(evidence=[Ev(seg="S1", quote="q")])
    assert p.evidence[0].role == "states"
    sc = SourceClaim(claim_kind="girsa_error", target=SourceClaimTarget(), holder=Holder(label="x"),
                     holder_evidence=[Ev(seg="S1", quote="q")])  # fmt: skip
    assert sc.holder_evidence[0].role == "attributes"


def test_3_s2_with_one_invented_quote_out_of_eleven_is_rejected(world):
    def s2(messages, tool, model):
        if tool != "sugya_summary":
            return None
        if "והלכתא" not in " ".join(input_lines(messages).values()):
            return s2_responder(messages, tool, model)
        steps = [
            {
                "id": f"s{i}",
                "move": "statement",
                **({"responds_to": f"s{i - 1}"} if i > 1 else {}),
                "text_he": "טענה",
                "evidence": [{"seg": "S1", "quote": "אמר רב חייב"}],  # no role given
            }
            for i in range(1, 12)
        ]
        steps[5]["evidence"] = [{"seg": "S1", "quote": "ציטוט שהומצא לגמרי"}]  # 1 of 11 = 9%
        return {"title_he": "סוגיה א", "steps": steps}

    pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(s2))
    with Session(world.engine) as s:
        it = s.scalars(select(KlItem).where(KlItem.item_key == "summary:sugya:Shabbat 21a:1")).one()
        assert it.status == "rejected" and it.rejected_by == "auto"
        assert any(f["code"] == "K3" for f in it.check_report["findings"])


def test_3_s2_prompt_asks_for_role():
    from rav_knowledge.prompts import load_prompt

    assert "role" in load_prompt("s2").text


# ---- 4: K5 ---------------------------------------------------------------------------------


def lens(work="Minchat Asher", slug="rav-weiss"):
    return SegInfo("S1", 1, "x", 1, work, slug, "lamdanut", "lens", "modern", 0)


def test_4a_holder_who_is_the_lens_author_is_not_downgraded():
    m = pos(holder=Holder(label="הרב וייס", entity_slug="rav-weiss"))
    r = run_checks_(m, [rec("evidence[0]", info=lens())])
    assert r.ok and not m.reading_only and "K5" not in codes(r)
    m2 = pos(holder=Holder(label="מנחת אשר", work="Minchat Asher"))
    assert not run_checks_(m2, [rec("evidence[0]", info=lens())]).warnings and not m2.reading_only
    # someone else's view found only in the lens is still a reading
    m3 = pos(holder=Holder(label="רב", entity_slug="rav"))
    run_checks_(m3, [rec("evidence[0]", info=lens())])
    assert m3.reading_only


def test_4b_reported_needs_attributes_evidence_from_the_reporter():
    h = Holder(label="רב", attribution="reported", reported_by="Rashi on Shabbat")
    rashi = SegInfo("S1", 1, "x", 1, "Rashi on Shabbat", "rashi", "rishonim", "primary", None, 0)
    ok = run_checks_(pos(holder=h), [rec("evidence[0]", "attributes", info=rashi)])
    assert ok.ok
    # evidence in another text, or in the right text but not as `attributes`: error
    assert "K5" in codes(run_checks_(pos(holder=h), [rec("evidence[0]", "attributes")]), "error")
    assert "K5" in codes(
        run_checks_(pos(holder=h), [rec("evidence[0]", "states", info=rashi)]), "error"
    )
    # matching by author slug works too
    h2 = Holder(label="רב", attribution="reported", reported_by="rashi")
    assert run_checks_(pos(holder=h2), [rec("evidence[0]", "attributes", info=rashi)]).ok


# ---- 5: reading_only ---------------------------------------------------------------------------


def test_5_reading_only_records_read_by_and_grade_and_is_left_out_of_machloket(world):
    m = pos()
    run_checks_(m, [rec("evidence[0]", info=lens())])
    assert m.reading_only and m.attribution_grade == "inferred_by_source"
    assert m.read_by.entity_slug == "rav-weiss" and m.read_by.work == "Minchat Asher"
    assert m.model_dump(mode="json")["read_by"]["label"] == "rav-weiss"

    eng = world.engine
    with Session(eng) as s:
        sg = _item(s, "sugya", "sugya:S1", {})
        iss = _item(s, "issue", "issue:1", {})
        s.add(KlLink(from_item_id=iss.id, to_item_id=sg.id, rel="raised_in"))
        for i, (side, ro) in enumerate([("א", False), ("ב", False), ("ג", True)]):
            p = _item(s, "position", f"position:{i}",
                      {"issue": "issue:1", "side_label": side, "reading_only": ro,
                       "holder": {"tier": "canonical"}})  # fmt: skip
            s.add(KlLink(from_item_id=p.id, to_item_id=iss.id, rel="position_on"))
        s.commit()
    with Session(eng) as s:
        assert [p.item_key for p in queries.positions_by_issue(s, "issue:1")] == [
            "position:0",
            "position:1",
        ]
        iss = s.scalars(select(KlItem).where(KlItem.item_key == "issue:1")).one()
        mk = build_machloket_candidate(s, iss)
        assert len(mk.body["sides"]) == 2 and "position:2" not in str(mk.body["sides"])


# ---- 6: overrides (edit / set_boundary create versions) are in test_pipeline.py;
# ---- 10: the annotation / test_k3 cleanups ---------------------------------------------------


def test_10_apply_overrides_missing_annotation_id(world):
    with Session(world.engine) as s:
        assert apply_overrides(s, [], [987654321]) == 0


# ---- 7: fail-closed ---------------------------------------------------------------------------


def test_7_write_item_needs_explicit_license_flags(world):
    with Session(world.engine) as s, pytest.raises(TypeError):
        write_item(s, kind="issue", item_key="k", scope_key="s", body={}, input_hash="h",
                   job_id=None, model=None, prompt_version="t", prompt_hash=None,
                   schema_version=1)  # fmt: skip


def test_7_register_work_cli_requires_license_tier(world):
    from rav_knowledge.cli import main

    with pytest.raises(SystemExit) as e:
        main(["register-work", "Shabbat", "--tier", "canonical", "--role", "primary"])
    assert e.value.code == 2
    with Session(world.engine) as s, pytest.raises(TypeError):
        register_work(s, "Shabbat", tier="canonical", role="primary")  # type: ignore[call-arg]


# ---- 8: links survive a new version of the target -----------------------------------------------


def _item(s, kind, key, body, **kw):
    return write_item(s, kind=kind, item_key=key, scope_key="s", body=body, input_hash=key,
                      job_id=None, model=None, prompt_version="t", prompt_hash=None,
                      schema_version=1, can_display_public=True, can_train=True,
                      title_he=key, **kw)  # fmt: skip


def _promote_new_version(s, kind, key):
    return write_item(s, kind=kind, item_key=key, scope_key="s", body={"v": 2},
                      input_hash=key + "#2", job_id=None, model=None, prompt_version="t",
                      prompt_hash=None, schema_version=1, can_display_public=True,
                      can_train=True, title_he=key, make_current=True)  # fmt: skip


def test_8_axis_queries_survive_new_versions_of_the_target(world):
    with Session(world.engine) as s:
        sg = _item(s, "sugya", "sugya:S1", {})
        iss = _item(s, "issue", "issue:1", {})
        s.add(KlLink(from_item_id=iss.id, to_item_id=sg.id, rel="raised_in"))
        for i, side in enumerate(["א", "ב"]):
            p = _item(s, "position", f"position:{i}", {"side_label": side, "holder": {}})
            s.add(KlLink(from_item_id=p.id, to_item_id=iss.id, rel="position_on"))
        s.commit()
        build_machloket_candidate(s, iss)
        s.commit()
    with Session(world.engine) as s:
        _promote_new_version(s, "sugya", "sugya:S1")  # a new, current version of the sugya
        s.commit()
        got = queries.issues_by_sugya(s, "sugya:S1")
        assert [i.item_key for i in got] == ["issue:1"] and all(i.is_current for i in got)
        _promote_new_version(s, "issue", "issue:1")
        _promote_new_version(s, "machloket", "machloket:issue:1")
        s.commit()
        assert [i.item_key for i in queries.issues_by_sugya(s, "sugya:S1")] == ["issue:1"]
        iss = queries.issues_by_sugya(s, "sugya:S1")[0]
        assert iss.version == 2 and iss.is_current  # the NEW version, not the linked old one
        assert [m.item_key for m in queries.machlokot_by_issue(s, "issue:1")] == [
            "machloket:issue:1"
        ]
        out = queries.sugyot_by_machloket(s, "machloket:issue:1")
        assert [x.item_key for x in out] == ["sugya:S1"] and out[0].version == 2
        assert [p.item_key for p in queries.positions_by_issue(s, "issue:1")] == [
            "position:0",
            "position:1",
        ]
        assert queries.issues_by_sugya(s, "sugya:none") == []


# ---- 9: gate threshold from config -----------------------------------------------------------


def test_9_gate_rejected_fraction_comes_from_config(world):
    assert load_config().gate_max_rejected_fraction == 0.1

    def half_bad(messages, tool, model):
        if tool == "sugya_summary" and "והלכתא" in " ".join(input_lines(messages).values()):
            return {"title_he": "x", "steps": [{"id": "s1", "move": "statement", "text_he": "x",
                    "evidence": [{"seg": "S1", "quote": "ציטוט שאינו קיים בטקסט"}]}]}  # fmt: skip
        return s2_responder(messages, tool, model)

    r = pipeline(world.engine, ScriptedLLM(s1_responder()), ScriptedLLM(half_bad))
    with pytest.raises(PromoteRefused):  # 1 of 2 rejected (50%) > 10%
        promote(world.engine, r.job_id)
    loose = cfg(gate_max_rejected_fraction=0.6)
    assert promote(world.engine, r.job_id, cfg=loose)["gate"]["source"] == "k-checks"


# ---- 11: KM, Asif tier, S0b model, COMMENT ON ---------------------------------------------------


def test_11_machloket_has_its_own_code():
    from rav_knowledge.schemas import Machloket

    m = Machloket(issue_id="issue:1", sides=[["p1"], ["p2"]])
    r = run_checks_(m, position_issue={"p1": "issue:1", "p2": "issue:2"})
    assert codes(r, "error") == ["KM"]


def test_11_asif_tier_and_fast_s0b_model(world):
    from rav_knowledge.sources import LENS_RECORDS

    assert {t: tier for t, _, tier, *_ in LENS_RECORDS}["Asif"] == "sugya_compendium"
    c = load_config()
    assert "haiku" in c.model_for("s0b") and "opus" in c.model_for("s2")
    assert c.price(c.model_for("s0b")) is not None


def test_11_deprecation_comments_added_and_removed(pg_engine):
    q = text("SELECT obj_description(to_regclass(:t), 'pg_class')")

    def comment(t):
        with pg_engine.connect() as c:
            return c.execute(q, {"t": t}).scalar()

    for t in ("cards", "card_dependencies"):
        assert comment(t) == "DEPRECATED: superseded by kl_items"
    cf = _cfg(pg_engine)
    command.downgrade(cf, "0004")
    pg_engine.dispose()
    for t in ("cards", "card_dependencies"):
        assert comment(t) is None
    command.upgrade(cf, "head")


# ---- fix-2 -------------------------------------------------------------------------------------


def test_fix2_relation_takes_precedence_over(pg_engine):
    with Session(pg_engine) as s:
        a, b = _item(s, "issue", "a", {}), _item(s, "issue", "b", {})
        s.add(KlRelation(from_item_id=a.id, to_item_id=b.id, rel="takes_precedence_over"))
        s.commit()
        s.add(KlRelation(from_item_id=a.id, to_item_id=b.id, rel="precedes_maybe"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_fix2_reception_contexts(pg_engine):
    with Session(pg_engine) as s:
        it = _item(s, "issue", "r", {})
        for ctx in ("public_instruction", "bediavad", "minhag:ashkenaz", "snif"):
            s.add(KlReception(target_item_id=it.id, target_claim="c", context=ctx,
                              weight_label="primary", policy_version="p"))  # fmt: skip
        s.commit()
        it_id = it.id
    for ctx, label in (("bediavad", "rejected"), ("made_up", "primary")):
        with Session(pg_engine) as s:
            s.add(KlReception(target_item_id=it_id, target_claim="d", context=ctx,
                              weight_label=label, policy_version="p"))  # fmt: skip
            with pytest.raises(IntegrityError):
                s.commit()
    with Session(pg_engine) as s:  # `rejected` is still fine in lemaase
        s.add(KlReception(target_item_id=it_id, target_claim="d", context="lemaase",
                          weight_label="rejected", policy_version="p"))  # fmt: skip
        s.commit()


def test_fix2_authority_areas(world):
    assert validate_authority_areas(None) is None
    assert validate_authority_areas(["responsa", "lamdanut"]) == ["responsa", "lamdanut"]
    with pytest.raises(ValidationError):
        validate_authority_areas(["responsa", "astrology"])
    with Session(world.engine) as s:
        src = register_work(s, "Rashi on Shabbat", tier="rishonim", role="lens",
                            license_tier="PUBLIC_COMMERCIAL",
                            authority_areas=["talmud_commentary"])  # fmt: skip
        s.commit()
        assert s.get(KlSource, src.work_id).authority_areas == ["talmud_commentary"]
        with pytest.raises(ValidationError):
            register_work(s, "Rashi on Shabbat", tier="rishonim", role="lens",
                          license_tier="PUBLIC_COMMERCIAL", authority_areas=["nope"])  # fmt: skip


def test_fix2_holder_quoted_only_needs_quoted_in():
    with pytest.raises(ValidationError):
        Holder(label="x", known_via="quoted_only")
    with pytest.raises(ValidationError):
        Holder(label="x", known_via="quoted_only", quoted_in=[])
    h = Holder(
        label="x", known_via="quoted_only", quoted_in=[{"work": "Rashi", "evidence_ids": [1]}]
    )
    assert h.quoted_in[0].work == "Rashi"
    assert Holder(label="x").known_via == "direct" and Holder(label="x").quoted_in is None


def test_fix2_position_new_fields_need_evidence_and_versions_are_bumped():
    ev = [Ev(seg="S1", quote="q")]
    for field, value in (("meakev", "yes"), ("obligation_type", "minhag_taut"),
                         ("instruction_scope", "public")):  # fmt: skip
        with pytest.raises(ValidationError):
            pos(**{field: value})
        assert getattr(pos(evidence=ev, **{field: value}), field) == value
    assert pos().meakev is None and pos().leniency_reason is None
    with pytest.raises(ValidationError):
        pos(evidence=ev, meakev="maybe")
    with pytest.raises(ValidationError):
        pos(evidence=ev, leniency_reason={"text_he": "x"})  # a leniency reason carries evidence
    ok = pos(leniency_reason={"text_he": "x", "evidence": [{"seg": "S1", "quote": "q"}]})
    assert ok.leniency_reason.evidence[0].role == "states"
    assert schema_version_for("position") == 2 and schema_version_for("issue") == 1
    assert Position.model_validate(pos(evidence=ev).model_dump())  # round trip


# ---- fix-3 -------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [True, False])
def test_fix3_1_other_sugya_in_the_overlap_page_does_not_merge_with_the_carry(world, flags):
    """A=0-3, X=4-4, B=5-9 where the overlap page (idx 4,5) holds X and the start of B."""

    def r(messages, tool, model):
        if tool != "sugya":
            return None
        if "Shabbat 21a:1" in messages[-1]["content"]:
            return {"sugyot": [
                {"title_he": "A", "start_seg": "S1", "end_seg": "S4"},
                {"title_he": "X", "start_seg": "S5", "end_seg": "S5"},
                {"title_he": "B", "start_seg": "S6", "end_seg": "S6",
                 **({"open_end": True} if flags else {})},
            ]}  # fmt: skip
        return {"sugyot": [
            {"title_he": "X", "start_seg": "S1", "end_seg": "S1",
             **({"open_start": True} if flags else {})},
            {"title_he": "B2", "start_seg": "S2", "end_seg": "S6"},
        ]}  # fmt: skip

    run(world.engine, "s0", None)
    rep = run(world.engine, "s1", ScriptedLLM(r))
    assert "coverage_holes" not in rep.stats and "coverage_overlaps" not in rep.stats
    with Session(world.engine) as s:
        got = list(s.scalars(select(KlItem).where(KlItem.kind == "sugya").order_by(KlItem.id)))
        assert [(i.item_key, i.body["end_ref"], i.is_current) for i in got] == [
            ("sugya:Shabbat 21a:1", "Shabbat 21b:2", True),
            ("sugya:Shabbat 22a:1", "Shabbat 22a:1", True),
            ("sugya:Shabbat 22a:2", "Shabbat 23a:2", True),
        ]
        assert all(i.check_report is None for i in got)


def test_fix3_2_table_comments_do_not_drift(pg_engine):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from rav_ai.db.models import Base

    with pg_engine.connect() as c:
        diffs = compare_metadata(MigrationContext.configure(c), Base.metadata)
    flat = [d for x in diffs for d in (x if isinstance(x, list) else [x])]
    assert not [d for d in flat if "table_comment" in d[0]]


def test_fix3_3_license_tier_has_no_model_default():
    assert KlSource.__table__.c.license_tier.default is None


def test_fix3_4_copy_evidence_uses_the_config_thresholds(world):
    from rav_ai.db.models import KlEvidence, Segment, SegmentText

    from rav_knowledge.store import copy_evidence

    with Session(world.engine) as s:
        sid = world.seg["Shabbat 21a:1"]
        st = s.scalars(select(SegmentText).where(SegmentText.segment_id == sid)).first()
        it, new = _item(s, "issue", "ev1", {}), _item(s, "issue", "ev2", {})
        quote = "ושמואל אמר פטור"
        s.add(KlEvidence(item_id=it.id, segment_id=sid, version_id=st.version_id, quote=quote,
                         role="states", locate_status="located"))  # fmt: skip
        s.flush()
        copy_evidence(s, it.id, new.id)
        assert (
            s.scalars(select(KlEvidence).where(KlEvidence.item_id == new.id)).one().locate_status
            == "located"
        )
        copy_evidence(s, it.id, new.id, min_words=4)  # stricter config: 3 words is now too short
        rows = list(s.scalars(select(KlEvidence).where(KlEvidence.item_id == new.id)))
        assert [r.locate_status for r in rows] == ["located", "ambiguous"]
        _ = Segment
