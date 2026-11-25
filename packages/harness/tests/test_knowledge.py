"""Knowledge-layer tools, scope gate and answer validators (KL-2). Seeded kl_* + FakeLLM."""

from __future__ import annotations

import tomllib
from dataclasses import replace

import pytest
from rav_ai.db.models import KlCoverage, KlItem, Run
from sqlalchemy import text

from conftest import FakeLLM, call, reply
from kl_seed import SCOPE
from rav_harness import HarnessConfig, load_config
from rav_harness.config import parse_phrases
from rav_harness.integration import build_harness
from rav_harness.loop import validate_answer
from rav_harness.prompt import PROMPT_VERSION, PROMPT_VERSION_KL, knowledge_prompt
from rav_harness.scope import extract_refs, hebrew_number, ref_in_scope, scope_key_in_scope
from rav_harness.tools import CorpusTools
from rav_harness.validators import ALL_CODES

Q_IN = "מה אומרת הגמרא בשבת כב על נר חנוכה"
Q_OUT = "מה הדין במצה בפסח קיב"
Q_PRACTICAL = "האם מותר להדליק נר חנוכה בשבת כב מבחוץ"


def cfg(knowledge="base+lens", allow=True, **kw) -> HarnessConfig:
    return replace(load_config(), knowledge=knowledge, allow_personal_only=allow, **kw)


def mk(
    session, knowledge="base+lens", mode="personal", allow=True, question=Q_IN, config=None
) -> CorpusTools:
    t = CorpusTools(session, mode, config or cfg(knowledge, allow))
    t.kl.run_scope_gate(question)
    return t


def keys(res, field="positions") -> set[str]:
    return {p["key"] for i in res["issues"] for p in i[field]}


def cit(ref, quoted="x", **kw):
    return {"ref": ref, "quoted_text": quoted, **kw}


def claim(text, citations=(), type="interpretation", **kw):
    return {"type": type, "text": text, "citations": list(citations), **kw}


def answer(*claims, scope="in_scope", **kw):
    return {"claims": list(claims), "disclaimer": "דעת המערכת בלבד", "scope": scope, **kw}


def check(t: CorpusTools, args: dict, read: tuple[str, ...] = ()):
    """validate_answer against what the tools returned; ``read`` refs go through get_segment."""
    for r in read:
        assert t.call("get_segment", {"ref": r})["found"]
    return validate_answer(args, set(t.retrieved_refs), t.kl.ctx)


def rejected_with(t, args, code, read=()):
    ans, err = check(t, args, read)
    assert ans is None and f"[{code}]" in err, err
    return err


# ---- scope helpers -------------------------------------------------------------------------


def test_scope_rules_and_citation_extraction():
    rules = load_config().scope
    assert ref_in_scope("Shabbat 22a:3", rules) and ref_in_scope("Shabbat 24b", rules)
    assert not ref_in_scope("Shabbat 25a", rules) and not ref_in_scope("Shabbat 20b", rules)
    assert ref_in_scope("Shabbat 22", rules) and not ref_in_scope("Pesachim 112a", rules)
    assert ref_in_scope("Tur, Orach Chayim 670:1", rules)
    assert not ref_in_scope("Tur, Orach Chayim 669", rules)
    assert ref_in_scope("Mishneh Torah, Scroll of Esther and Hanukkah 4:2", rules)
    assert not ref_in_scope("Mishneh Torah, Scroll of Esther and Hanukkah 5", rules)
    assert scope_key_in_scope("Shabbat 21a-24a", rules)
    assert not scope_key_in_scope("Shabbat 21a-26a", rules)
    assert hebrew_number('תרע"א') == 671 and hebrew_number("תרעא") == 671
    assert extract_refs('מה כתב שו"ע או"ח תרעא וגם טור או"ח תרע') == [
        "Shulchan Arukh, Orach Chayim 671",
        "Tur, Orach Chayim 670",
    ]


# ---- tools: filter, envelope, never retrieved ----------------------------------------------


def test_every_tool_wraps_in_the_envelope_and_adds_no_retrieved_refs(session, kl):
    t = mk(session)
    calls = [
        ("coverage", {"ref_or_query": Q_IN}),
        ("find_issues", {"query": "נר חנוכה"}),
        ("get_sugya", {"ref": "Shabbat 22a"}),
        ("get_shitot", {"issue_key": "issue:candle-place"}),
        ("get_psak_chain", {"siman_or_ref": "670"}),
        ("get_chakirot", {"ref": "Shabbat 22a:1"}),
        ("explain_abbrev", {"ref": "Shabbat 22a:1", "token": "מצוה"}),
        ("get_links", {"ref": "Shabbat 22a:1"}),
        ("get_method_principles", {}),
    ]
    for name, args in calls:
        res = t.call(name, args)
        assert res["layer"] == "knowledge" and res["tool"] == name, res
        assert res["notice"].startswith("עזר בלבד") and "error" not in res
    assert t.retrieved_refs == []  # the layer is a map: nothing counts as read
    assert t.kl.ctx.layer_refs  # ... but evidence refs were seen


def test_filter_hides_rejected_stale_low_confidence_old_and_non_current(session, kl):
    res = mk(session).call("get_shitot", {"issue_key": "issue:candle-place"})
    assert keys(res) == {
        "pos:rashi",
        "pos:tur",
        "pos:sa",
        "pos:by",
        "pos:mag",
        "pos:lens",
        "pos:gemara",
        "pos:nopub",
    }  # not: rejected, stale, low confidence, non-current
    allkeys = set()
    for r in res["relations"]:
        allkeys |= {r["from_item"], r["to_item"]}
    hidden = {kl.p_rej.id, kl.p_stale.id, kl.p_low.id, kl.p_old.id}
    assert not hidden & allkeys


def test_public_mode_hides_non_displayable_item(session, kl):
    it = session.get(KlItem, kl.p_by.id)
    it.can_display_public = False
    session.commit()
    pers = keys(mk(session).call("get_shitot", {"issue_key": "issue:candle-place"}))
    pub = keys(mk(session, mode="public").call("get_shitot", {"issue_key": "issue:candle-place"}))
    assert "pos:by" in pers and "pos:by" not in pub
    assert "pos:nopub" in pers and "pos:nopub" not in pub


def test_position_fields_reception_and_telegraphic_limits(session, kl):
    t = mk(session)
    res = t.call("get_shitot", {"ref": "Shabbat 22a:1"})
    iss = res["issues"][0]
    assert iss["key"] == "issue:candle-place" and iss["promoted_job"]
    pos = {p["key"]: p for p in iss["positions"]}
    r = pos["pos:rashi"]
    assert r["tier"] == "rishonim" and r["status"] == "verified" and r["confidence"] == 0.9
    assert (
        r["evidence_refs"]
        and r["promoted_job"] is not None
        and pos["pos:tur"]["promoted_job"] is None
    )
    assert len(r["view"].split()) <= 20
    ctxs = {x["context"]: x["weight_label"] for x in r["reception"]}
    assert ctxs == {"iyun": "primary", "lemaase": "rejected"}  # rejected shown, not filtered
    assert pos["pos:tur"]["unusual_interpretation"] and not r["unusual_interpretation"]
    rel = [x for x in r["relations"] if x["rel"] == "derived_from"][0]
    assert rel["evidence_origin"] == "llm" and rel["candidate"] is False and rel["evidence_refs"]
    assert iss["machloket"][0]["sides"] == [["pos:rashi"], ["pos:by"]]
    assert iss["open_questions"][0]["q_kind"] == "teiku"
    sc = res["source_claims"][0]
    assert sc["target"]["ref"] == "Rashi on Shabbat 22a:1:1" and sc["target"]["quote"] == "בטפח"


def test_depends_on_is_shown_as_a_typed_dependency_not_the_same_nidon(session, kl):
    res = mk(session).call("get_shitot", {"issue_key": "issue:candle-place"})
    rel = res["issues"][0]["related_issues"]
    assert len(rel) == 1 and rel[0]["rel"] == "depends_on" and rel[0]["same_nidon"] is False
    assert rel[0]["table"] == "kl_links" and rel[0]["evidence_origin"] and rel[0]["evidence_refs"]
    assert "issue:gamal-pishtan" not in {i["key"] for i in res["issues"]}  # not folded in


def test_sugya_steps_are_clipped_and_readings_need_the_lens(session, kl):
    base = mk(session, "base").call("get_sugya", {"ref": "Shabbat 22a", "with_readings": True})
    step = base["items"][0]["summaries"][0]["steps"][0]
    assert len(step["text"].split()) <= 26 and "readings" not in step
    assert step["readings_hidden"] == 2 and "readings_note" in base
    t = mk(session)
    full = t.call("get_sugya", {"ref": "Shabbat 22a", "with_readings": True})
    rs = full["items"][0]["summaries"][0]["steps"][0]["readings"]
    assert [r["holder"] for r in rs] == ['רשב"א', 'ריטב"א']
    plain = t.call("get_sugya", {"ref": "Shabbat 22a"})
    assert "readings" not in plain["items"][0]["summaries"][0]["steps"][0]


def test_psak_chain_has_four_levels_and_base_hides_the_lens(session, kl):
    res = mk(session, "base").call("get_psak_chain", {"siman_or_ref": "670"})
    assert [(lv["level"], lv["tier"]) for lv in res["levels"]] == [
        (1, "canonical"),
        (2, "rishonim"),
        (3, "codes"),
        (4, "acharonim"),
    ]
    assert {r["rel"] for r in res["relations"]} == {"derived_from", "rules_like"}
    by_depth = {i["key"]: i["depth"] for lv in res["levels"] for i in lv["items"]}
    assert by_depth["pos:tur"] == 0 and by_depth["pos:rashi"] == 1 and by_depth["pos:mag"] == 1
    ref = {"siman_or_ref": "Beit Yosef, Orach Chayim 670"}
    lens = mk(session).call("get_psak_chain", ref)
    assert "pos:lens" in {i["key"] for lv in lens["levels"] for i in lv["items"]}
    base = mk(session, "base").call("get_psak_chain", ref)
    assert "pos:lens" not in {i["key"] for lv in base["levels"] for i in lv["items"]}
    assert base["relations"] == []  # a relation to a hidden item does not leak


def test_base_has_no_lens_tools_secondary_items_or_readings(session, kl):
    t = mk(session, "base")
    names = [s["function"]["name"] for s in t.specs()]
    assert "get_chakirot" not in names and "get_method_principles" not in names
    assert "get_shitot" in names
    assert "error" in t.call("get_chakirot", {"ref": "Shabbat 22a:1"})
    assert "pos:lens" not in keys(t.call("get_shitot", {"issue_key": "issue:candle-place"}))
    full = [s["function"]["name"] for s in mk(session).specs()]
    assert "get_chakirot" in full and "get_method_principles" in full


def test_chakira_matrix_only_for_verified_with_empty_cells(session, kl):
    res = mk(session).call("get_chakirot", {"ref": "Shabbat 22a:1"})
    assert [m["key"] for m in res["items"]] == ["chak:ok"]
    cells = {m["method"]: m["cells"] for m in res["items"][0]["matrix"]}
    assert cells['רש"י'][1] == "אין מקור" and cells["בית יוסף"][0] == "אין מקור"
    assert cells["טור"][0]["role"] == "aligned" and cells["טור"][1] == "אין מקור"
    assert [m["key"] for m in res["retrieval_only"]] == ["chak:no"]
    assert res["retrieval_only"][0]["structure"] is None


def test_method_principles_empty_table_returns_empty_list(session, kl):
    session.execute(
        text(
            "DELETE FROM kl_item_segments WHERE item_id IN "
            "(SELECT id FROM kl_items WHERE kind = 'method_principle')"
        )
    )
    session.execute(text("DELETE FROM kl_items WHERE kind = 'method_principle'"))
    session.commit()
    res = mk(session).call("get_method_principles", {"query": "כוונה"})
    assert res["items"] == [] and "error" not in res


def test_method_principles_are_personal_only_and_fail_closed(session, kl):
    t = mk(session)
    items = t.call("get_method_principles", {})["items"]
    assert [i["key"] for i in items] == ["mp:1"] and "לא ראיה" in items[0]["role"]
    assert t.call("get_method_principles", {"query": "כוונה"})["items"]
    assert mk(session, allow=False).call("get_method_principles", {})["items"] == []
    pub = mk(session, mode="public").call("get_method_principles", {})
    assert pub["items"] == [] and "error" not in pub
    assert "error" in mk(session, "base").call("get_method_principles", {})


def test_get_links_are_candidates_with_the_sefaria_label(session, kl):
    res = mk(session).call("get_links", {"ref": "Shabbat 22a:1"})
    e = res["items"][0]
    assert e["candidate"] is True and e["evidence_origin"] == "sefaria"
    assert e["label"] == "קישור ספריא: מועמד, לא ראיה" and e["edge_type"].startswith("ein mishpat")
    assert {e["ref_a"], e["ref_b"]} == {"Shabbat 22a:1", "Tur, Orach Chayim 670:1"}
    assert e["citation_a"] and e["citation_b"]
    assert (
        mk(session).call("get_links", {"ref": "Shabbat 22a:1", "kinds": ["commentary"]})["items"]
        == []
    )


def test_explain_abbrev_and_find_issues(session, kl):
    t = mk(session)
    ab = t.call("explain_abbrev", {"ref": "Shabbat 22a:1", "token": "מצוה"})["items"]
    assert ab and ab[0]["chosen"] == "מצוה להניח"
    assert t.call("explain_abbrev", {"ref": "Shabbat 22a:1", "token": "אחר"})["items"] == []
    found = t.call("find_issues", {"query": "היכן מניחים נר", "limit": 5})["items"]
    assert [i["key"] for i in found][0] == "issue:candle-place"
    assert t.call("find_issues", {"query": "!!"})["items"] == []


# ---- scope gate ----------------------------------------------------------------------------


def test_scope_gate_in_out_and_partial(session, kl):
    t = CorpusTools(session, "personal", cfg())
    assert t.kl.run_scope_gate(Q_IN).status == "in_scope"
    out = t.kl.run_scope_gate(Q_OUT)
    assert out.status == "out_of_scope" and out.reasons
    mixed = t.kl.run_scope_gate("שבת כב ופסחים קיב")
    assert mixed.status in ("out_of_scope", "partial")
    session.add(KlCoverage(scope_key=SCOPE, work_id=kl.works.tos.id, status="not_ingested"))
    session.commit()
    gate = t.kl.run_scope_gate(Q_IN)
    assert gate.status == "partial" and gate.unchecked_core_books == ["Tosafot on Shabbat"]
    cov = t.call("coverage", {"ref_or_query": "Shabbat 22a:1"})
    assert cov["status"] == "partial" and cov["unchecked_core_books"] == ["Tosafot on Shabbat"]


# ---- validators 4a-4j: negative and positive -----------------------------------------------


def test_codes_are_unique():
    assert len(set(ALL_CODES)) == len(ALL_CODES)


def test_4a_evidence_ref_never_read_is_rejected(session, kl):
    t = mk(session)
    t.call("get_shitot", {"issue_key": "issue:candle-place"})
    args = answer(claim("הגמרא", [cit("Shabbat 22a:1")]))
    err = rejected_with(t, args, "KL_UNREAD_EVIDENCE_REF")
    assert "get_segment" in err
    ans, _ = check(t, args, read=("Shabbat 22a:1",))
    assert ans is not None


def test_4b_lens_role_attribution_and_rishon_attribution(session, kl):
    t = mk(session)
    rashi = "Lens on Rashi 1:1"
    bad_role = answer(claim("עדשה", [cit(rashi)]))
    rejected_with(t, bad_role, "KL_LENS_ROLE", read=(rashi,))
    no_attr = answer(claim("עדשה", [cit(rashi, source_role="secondary_lens")]))
    rejected_with(t, no_attr, "KL_LENS_ROLE")
    lens = cit(rashi, source_role="secondary_lens", attribution='הלנס ברש"י')
    err = rejected_with(t, answer(claim("הראשון סובר צד א", [lens])), "KL_LENS_ONLY_RISHON")
    assert "get_segment" in err
    good = answer(claim('לפי הבנת הלנס ברש"י, הלשון מתיישבת עם צד א', [lens]))
    assert check(t, good)[0] is not None
    # the same words backed by the Rishon himself are fine
    first = answer(claim("הראשון סובר צד א", [cit("Rashi on Shabbat 22a:1:1")]))
    assert check(t, first, read=("Rashi on Shabbat 22a:1:1",))[0] is not None


def test_4c_relation_fields_and_candidate_only(session, kl):
    t = mk(session)
    link = t.call("get_links", {"ref": "Shabbat 22a:1"})["items"][0]
    rel = {
        "table": "edges",
        "id": link["id"],
        "rel": link["rel"],
        "evidence_origin": "sefaria",
        "candidate": True,
        "evidence_refs": [],
    }
    missing = {**rel, "evidence_origin": ""}
    rejected_with(
        t,
        answer(claim("יש קישור", [cit("Shabbat 22a:1")], relations=[missing])),
        "KL_RELATION_FIELDS",
    )
    wrong = {**rel, "candidate": False}
    rejected_with(
        t,
        answer(claim("יש קישור", [cit("Shabbat 22a:1")], relations=[wrong])),
        "KL_RELATION_MISMATCH",
    )
    ghost = {**rel, "id": 99999}
    rejected_with(t, answer(claim("יש קישור", relations=[ghost])), "KL_RELATION_UNKNOWN")
    # the Ein Mishpat link as the only support, one end read: rejected
    one_end = answer(claim("הטור מקביל", [cit("Shabbat 22a:1")], relations=[rel]))
    err = rejected_with(t, one_end, "KL_CANDIDATE_ONLY", read=("Shabbat 22a:1",))
    assert "מועמד, לא ראיה" in err and "Tur, Orach Chayim 670:1" in err
    both = answer(
        claim("הטור מקביל", [cit("Shabbat 22a:1"), cit("Tur, Orach Chayim 670:1")], relations=[rel])
    )
    assert check(t, both, read=("Tur, Orach Chayim 670:1",))[0] is not None


def test_4c_a_typed_dependency_with_origin_and_evidence_is_accepted(session, kl):
    t = mk(session)
    res = t.call("get_shitot", {"issue_key": "issue:candle-place"})
    dep = res["issues"][0]["related_issues"][0]
    rel = {
        k: dep[k] for k in ("table", "id", "rel", "evidence_origin", "candidate", "evidence_refs")
    }
    text = 'נידון חו"מ תי"ח (גמל טעון פשתן) תלוי בנידון זה, ואינו אותו נידון'
    args = answer(claim(text, [cit("Shulchan Arukh, Choshen Mishpat 418:1")], relations=[rel]))
    assert check(t, args, read=("Shulchan Arukh, Choshen Mishpat 418:1",))[0] is not None
    rejected_with(t, answer(claim(text, relations=[{**rel, "rel": ""}])), "KL_RELATION_FIELDS")


def test_4d_girsa_claims_need_a_witness_or_an_attributed_interpretation(session, kl):
    t = mk(session)
    system = answer(
        claim(
            "הדפוס שגוי כאן",
            [cit("Shabbat 22a:1")],
            type="system_conclusion",
            flagged_as_system=True,
        )
    )
    rejected_with(t, system, "KL_GIRSA_WITNESS", read=("Shabbat 22a:1",))
    plain = answer(claim("הגרסה שגויה", [cit("Shabbat 22a:1")]))
    rejected_with(t, plain, "KL_GIRSA_WITNESS")
    unknown_w = answer(
        claim('בכ"י Y ליתא', [cit("Shabbat 22a:1", witness_id=kl.witness_lonely.id)])
    )
    rejected_with(t, unknown_w, "KL_GIRSA_WITNESS")
    missing_w = answer(claim('בכ"י ליתא', [cit("Shabbat 22a:1", witness_id=9999)]))
    rejected_with(t, missing_w, "KL_GIRSA_WITNESS")
    ok_w = answer(
        claim('בכ"י X ליתא, לפי הרשב"א', [cit("Shabbat 22a:1", witness_id=kl.witness.id)])
    )
    assert check(t, ok_w)[0] is not None
    named = answer(
        claim('הב"ח מגיה: בטפחיים', [cit("Rashi on Shabbat 22a:1:1", attribution='הב"ח')])
    )
    assert check(t, named, read=("Rashi on Shabbat 22a:1:1",))[0] is not None
    forbidden = answer(
        claim('הב"ח אומר שהדפוס שגוי', [cit("Rashi on Shabbat 22a:1:1", attribution='הב"ח')])
    )
    rejected_with(t, forbidden, "KL_GIRSA_WITNESS")


def test_4e_unchecked_core_books_must_be_listed_and_stated(session, kl):
    session.add(KlCoverage(scope_key=SCOPE, work_id=kl.works.tos.id, status="unknown"))
    session.commit()
    t = mk(session)
    base = claim("הגמרא", [cit("Shabbat 22a:1")])
    rejected_with(t, answer(base, scope="partial"), "KL_COVERAGE_BOOKS", read=("Shabbat 22a:1",))
    notice = claim("כיסוי חלקי. ספרים מרכזיים שלא נבדקו כאן: Tosafot on Shabbat")
    no_list = answer(base, notice, scope="partial")
    assert "unchecked_core_books must list" in rejected_with(t, no_list, "KL_COVERAGE_BOOKS")
    no_text = answer(
        base, claim("כיסוי חלקי"), scope="partial", unchecked_core_books=["Tosafot on Shabbat"]
    )
    rejected_with(t, no_text, "KL_COVERAGE_BOOKS")
    silent = claim("אין מחלוקת בדבר")
    rejected_with(
        t,
        answer(base, notice, silent, scope="partial", unchecked_core_books=["Tosafot on Shabbat"]),
        "KL_SILENCE_IS_NOT_AGREEMENT",
    )
    hedged = claim("לא נמצא חולק בקורפוס שנבדק; אין מחלוקת שנמצאה")
    ok = answer(base, notice, hedged, scope="partial", unchecked_core_books=["Tosafot on Shabbat"])
    assert check(t, ok)[0] is not None


def test_4f_scope_must_match_the_gate_and_out_of_scope_has_no_conclusion(session, kl):
    t = mk(session)
    rejected_with(t, answer(claim("x"), scope=None), "KL_SCOPE_MISMATCH")
    rejected_with(t, answer(claim("x"), scope="partial"), "KL_SCOPE_MISMATCH")
    o = mk(session, question=Q_OUT)
    msg = "השאלה מחוץ למה שנקלט ונבדק: חסר פסחים קיב"
    for kind in ("chidush", "system_conclusion"):
        bad = answer(
            claim(msg), claim("חידוש", type=kind, flagged_as_system=True), scope="out_of_scope"
        )
        rejected_with(o, bad, "KL_OUT_OF_SCOPE_CONCLUSION")
    rejected_with(o, answer(claim("אין מידע"), scope="out_of_scope"), "KL_SCOPE_NOTICE")
    good = answer(claim(msg, [cit("Pesachim 112a:1")]), scope="out_of_scope")
    assert check(o, good, read=("Pesachim 112a:1",))[0] is not None  # may point at what was read


def test_4g_unusual_interpretation_must_be_flagged_and_marked(session, kl):
    t = mk(session)
    t.call("get_shitot", {"issue_key": "issue:candle-place"})
    tur = [kl.p_tur.id]
    plain = answer(claim("הטור סובר כך", kl_item_ids=tur))
    rejected_with(t, plain, "KL_UNUSUAL_FLAG")
    only_flag = answer(claim("הטור סובר כך", kl_item_ids=tur, flags=["unusual_interpretation"]))
    rejected_with(t, only_flag, "KL_UNUSUAL_FLAG")
    ok = answer(
        claim("פירוש חריג: הטור סובר כך", kl_item_ids=tur, flags=["unusual_interpretation"])
    )
    assert check(t, ok)[0] is not None
    other = answer(claim('רש"י סובר כך', kl_item_ids=[kl.p_rashi.id]))
    assert check(t, other)[0] is not None
    rejected_with(t, answer(claim("x", kl_item_ids=[424242])), "KL_UNKNOWN_ITEM")


def test_4h_maran_instructions_need_documented_practice(session, kl):
    t = mk(session)
    t.call("get_shitot", {"issue_key": "issue:candle-place"})
    text = "קיבלנו הוראות מרן להניח בחוץ"
    rejected_with(
        t,
        answer(claim(text, [cit("Shulchan Arukh, Orach Chayim 670:1")])),
        "KL_MARAN_INSTRUCTIONS",
        read=("Shulchan Arukh, Orach Chayim 670:1",),
    )
    wrong_item = answer(
        claim(text, [cit("Shulchan Arukh, Orach Chayim 670:1")], kl_item_ids=[kl.p_rashi.id])
    )
    rejected_with(t, wrong_item, "KL_MARAN_INSTRUCTIONS")
    ok = answer(claim(text, [cit("Shulchan Arukh, Orach Chayim 670:1")], kl_item_ids=[kl.p_sa.id]))
    assert check(t, ok)[0] is not None


def test_4i_shulchan_arukh_is_not_a_verdict_by_fiat(session, kl):
    t = mk(session)
    for txt in ('השו"ע פסק ולכן זו ההלכה', 'דעת הרמב"ם נדחה מפני השו"ע'):
        rejected_with(t, answer(claim(txt)), "KL_SA_PSAK")
    assert check(t, answer(claim("המחבר פסק כך, והטור כותב אחרת")))[0] is not None
    p = mk(session, question=Q_PRACTICAL)
    assert p.kl.ctx.gate.practical and p.kl.ctx.gate.status == "in_scope"
    rejected_with(p, answer(claim("המחבר מתיר")), "KL_SA_PRACTICAL")
    lacks = answer(claim('המחבר והרמ"א מתירים'))
    rejected_with(p, lacks, "KL_SA_PRACTICAL")
    full = answer(claim('המחבר והרמ"א מתירים; הטור חולק מטעמו'))
    assert check(p, full)[0] is not None
    none_found = answer(claim('המחבר והרמ"א מתירים; לא נמצא חולק בקורפוס שנבדק'))
    assert check(p, none_found)[0] is not None


def test_4j_several_readings_are_all_shown_or_flagged(session, kl):
    t = mk(session)
    t.call("get_sugya", {"ref": "Shabbat 22a", "with_readings": True})
    item = [kl.summary.id]
    rejected_with(t, answer(claim('הסבר אחד: רשב"א', kl_item_ids=item)), "KL_READINGS")
    assert check(t, answer(claim('רשב"א; ריטב"א', kl_item_ids=item)))[0] is not None
    assert check(t, answer(claim('רשב"א. יש הסברים נוספים', kl_item_ids=item)))[0] is not None


# ---- end to end with FakeLLM ---------------------------------------------------------------

GOOD_CITE = answer(
    claim("הגמרא", [cit("Shabbat 22a:1", "נר חנוכה")], type="quote"),
    kl_item_ids=None,
)


def run(session, llm, question=Q_IN, knowledge="base+lens", mode="personal", **kw):
    c = cfg(knowledge, **kw)
    h = build_harness(session, llm, c, mode=mode)
    return h.run(question, question_id="q1"), c


def good_args():
    a = answer(claim("מצוה להניח נר", [cit("Shabbat 22a:1", "מצוה להניח נר חנוכה")], type="quote"))
    a["claims"][0]["kl_item_ids"] = []
    return a


def test_fake_llm_reads_the_text_and_cites_it(session, kl):
    llm = FakeLLM(
        [
            reply(call("coverage", ref_or_query=Q_IN), call("get_sugya", "c2", ref="Shabbat 22a")),
            reply(call("get_segment", "c3", ref="Shabbat 22a:1")),
            reply(call("submit_answer", "c4", **good_args())),
        ]
    )
    res, _ = run(session, llm)
    assert res.error is None and res.answer.scope == "in_scope"
    assert res.retrieved_refs == ["Shabbat 22a:1"]
    assert res.prompt_version == PROMPT_VERSION_KL and res.knowledge == "base+lens"
    sys_msg = llm.calls[0]["messages"][0]["content"]
    assert "השכבה עלולה לטעות; הטקסט קובע" in sys_msg
    first_user = llm.calls[0]["messages"][1]["content"]
    assert "שער תחום" in first_user and '"status": "in_scope"' in first_user
    assert "get_sugya" in llm.calls[0]["tools"] and "get_segment" in llm.calls[0]["tools"]
    assert res.tool_calls[0]["name"] == "scope_gate"


def test_fake_llm_citing_straight_from_the_layer_is_rejected(session, kl):
    llm = FakeLLM(
        [reply(call("get_sugya", ref="Shabbat 22a"))]
        + [reply(call("submit_answer", f"s{i}", **good_args())) for i in range(3)]
    )
    res, _ = run(session, llm)
    assert res.answer is None and "KL_UNREAD_EVIDENCE_REF" in res.error
    assert "Shabbat 22a:1" not in res.retrieved_refs


def test_out_of_scope_question_rejects_chidush_then_accepts_honest_answer(session, kl):
    bad = answer(
        claim("השאלה מחוץ למה שנקלט ונבדק"),
        claim("חידוש", type="chidush", flagged_as_system=True),
        scope="out_of_scope",
    )
    good = answer(claim("השאלה מחוץ למה שנקלט ונבדק: חסרים פסחים"), scope="out_of_scope")
    llm = FakeLLM(
        [
            reply(call("coverage", ref_or_query=Q_OUT)),
            reply(call("submit_answer", "a", **bad)),
            reply(call("submit_answer", "b", **good)),
        ]
    )
    res, _ = run(session, llm, Q_OUT)
    assert res.answer is not None and res.answer.scope == "out_of_scope"
    assert (
        len(res.rejected_answers) == 1 and "KL_OUT_OF_SCOPE_CONCLUSION" in res.rejected_answers[0]
    )
    assert res.scope_gate["status"] == "out_of_scope"


def test_knowledge_items_and_gate_are_recorded_on_the_run(session, kl):
    llm = FakeLLM(
        [
            reply(call("get_shitot", ref="Shabbat 22a:1")),
            reply(call("get_segment", "c2", ref="Shabbat 22a:1")),
            reply(call("submit_answer", "c3", **good_args())),
        ]
    )
    res, _ = run(session, llm)
    row = session.get(Run, res.run_id)
    assert row.prompt_version == "v1-kl" and row.prompt_hash == res.prompt_hash
    meta, *items = row.knowledge_items
    assert (
        meta["meta"]["knowledge"] == "base+lens"
        and meta["meta"]["scope_gate"]["status"] == "in_scope"
    )
    assert items and all(set(i) == {"item_id", "version", "tool", "tier"} for i in items)
    assert "get_shitot" in {i["tool"] for i in items}
    assert any(i["tier"] == "rishonim" for i in items)


def test_knowledge_off_is_v0_with_no_new_tools(session, kl):
    llm = FakeLLM(
        [
            reply(call("get_segment", ref="Shabbat 22a:1")),
            reply(call("submit_answer", "c2", **good_args())),
        ]
    )
    res, _ = run(session, llm, knowledge="off")
    assert res.error is None and res.prompt_version == PROMPT_VERSION == "v0"
    assert llm.calls[0]["tools"] == [
        "get_segment",
        "neighbors",
        "search_text",
        "resolve_citation",
        "submit_answer",
    ]
    assert res.scope_gate is None and res.knowledge_items == []
    assert "שער תחום" not in llm.calls[0]["messages"][1]["content"]
    assert session.get(Run, res.run_id).knowledge_items == []


def test_make_answer_fn_passes_knowledge_items_to_run_output(session, kl):
    from rav_evals.golden import GoldenQuestion

    from rav_harness.integration import make_answer_fn

    llm = FakeLLM(
        [
            reply(call("get_shitot", ref="Shabbat 22a:1")),
            reply(call("get_segment", "c2", ref="Shabbat 22a:1")),
            reply(call("submit_answer", "c3", **good_args())),
        ]
    )
    fn = make_answer_fn(session, llm, cfg(), record=False)
    out = fn(GoldenQuestion(id="g1", question=Q_IN, category="direct_source", difficulty="easy"))
    assert out.prompt_version == "v1-kl" and out.knowledge_items[0]["tool"] == "_meta"
    assert len(out.knowledge_items) > 1


def test_sample_answer_with_coverage_notice_and_candidate_relation(session, kl):
    """The documented example: partial coverage, a Sefaria link shown as a candidate only."""
    session.add(KlCoverage(scope_key=SCOPE, work_id=kl.works.tos.id, status="not_ingested"))
    session.commit()
    link_id = session.execute(text("SELECT id FROM edges WHERE edge_type LIKE 'ein%'")).scalar_one()
    rel = {
        "table": "edges",
        "id": link_id,
        "rel": "ein mishpat / ner mitsvah",
        "evidence_origin": "sefaria",
        "candidate": True,
        "evidence_refs": [],
    }
    sample = answer(
        claim(
            "מצוה להניח נר חנוכה על הפתח",
            [cit("Shabbat 22a:1", "מצוה להניח נר חנוכה")],
            type="quote",
        ),
        claim(
            "יש קישור ספריא (עין משפט) בין הגמרא לטור; זה מועמד, לא ראיה, ולכן קראנו את שני הצדדים",
            [cit("Shabbat 22a:1", "נר חנוכה"), cit("Tur, Orach Chayim 670:1", "להניח על פתח ביתו")],
            relations=[rel],
        ),
        claim(
            "כיסוי חלקי. ספרים מרכזיים שלא נבדקו כאן: Tosafot on Shabbat. "
            "לא נמצא חולק בקורפוס שנבדק."
        ),
        scope="partial",
        unchecked_core_books=["Tosafot on Shabbat"],
    )
    llm = FakeLLM(
        [
            reply(
                call("coverage", ref_or_query=Q_IN), call("get_links", "c2", ref="Shabbat 22a:1")
            ),
            reply(
                call("get_segment", "c3", ref="Shabbat 22a:1"),
                call("get_segment", "c4", ref="Tur, Orach Chayim 670:1"),
            ),
            reply(call("submit_answer", "c5", **sample)),
        ]
    )
    res, _ = run(session, llm)
    assert res.error is None, res.error
    assert res.answer.scope == "partial" and res.answer.claims[1].relations[0].candidate
    import json

    print(json.dumps(res.answer.model_dump(mode="json"), ensure_ascii=False, indent=2))


@pytest.mark.parametrize("bad", ["x", "knowledge"])
def test_config_rejects_unknown_knowledge_mode(bad):
    with pytest.raises(ValueError):
        HarnessConfig().with_knowledge(bad)


# ---- KL-2 fix-1 -----------------------------------------------------------------------------


def _edge_rel(t, origin_label):
    link = t.call("get_links", {"ref": "Shabbat 22a:1"})["items"]
    link = [x for x in link if x["edge_type"].startswith("ein")][0]
    assert link["candidate"] is True and link["evidence_origin"] == origin_label
    return link, {
        "table": "edges",
        "id": link["id"],
        "rel": link["rel"],
        "evidence_origin": link["evidence_origin"],
        "candidate": True,
        "evidence_refs": [],
    }


@pytest.mark.parametrize("origin", ["sefaria", "heuristic", "llm", "weird"])
def test_fix1_every_non_human_edge_is_a_candidate_whatever_its_origin(session, kl, origin):
    session.execute(text("UPDATE edges SET origin = :o WHERE edge_type LIKE 'ein%'"), {"o": origin})
    session.commit()
    t = mk(session)
    link, rel = _edge_rel(t, origin)
    cfgp = t.kl.phrases
    assert link["label"] == (cfgp.sefaria_label if origin == "sefaria" else cfgp.auto_link_label)
    args = answer(claim("הטור מקביל", [cit("Shabbat 22a:1")], relations=[rel]))
    err = rejected_with(t, args, "KL_CANDIDATE_ONLY", read=("Shabbat 22a:1",))
    assert "Tur, Orach Chayim 670:1" in err
    both = answer(
        claim("הטור מקביל", [cit("Shabbat 22a:1"), cit("Tur, Orach Chayim 670:1")], relations=[rel])
    )
    assert check(t, both, read=("Tur, Orach Chayim 670:1",))[0] is not None


def test_fix1_a_human_edge_is_not_a_candidate(session, kl):
    session.execute(text("UPDATE edges SET origin = 'human' WHERE edge_type LIKE 'ein%'"))
    session.commit()
    link = mk(session).call("get_links", {"ref": "Shabbat 22a:1"})["items"]
    link = [x for x in link if x["edge_type"].startswith("ein")][0]
    assert link["candidate"] is False and link["label"] is None


def test_fix1_candidate_check_cannot_be_bypassed_by_adding_another_relation(session, kl):
    t = mk(session)
    _, edge = _edge_rel(t, "sefaria")
    dep = t.call("get_shitot", {"issue_key": "issue:candle-place"})["issues"][0]["related_issues"][
        0
    ]
    assert dep["candidate"] is False
    dep_rel = {k: dep[k] for k in ("table", "id", "rel", "evidence_origin", "candidate")}
    dep_rel["evidence_refs"] = dep["evidence_refs"]
    cm = "Shulchan Arukh, Choshen Mishpat 418:1"
    only_one_end = answer(
        claim("הטור מקביל ונידון תלוי", [cit("Shabbat 22a:1"), cit(cm)], relations=[edge, dep_rel])
    )
    rejected_with(t, only_one_end, "KL_CANDIDATE_ONLY", read=("Shabbat 22a:1", cm))
    dep_first = answer(
        claim("נידון תלוי והטור מקביל", [cit("Shabbat 22a:1"), cit(cm)], relations=[dep_rel, edge])
    )
    rejected_with(t, dep_first, "KL_CANDIDATE_ONLY")
    ok = answer(
        claim(
            "הטור מקביל ונידון תלוי",
            [cit("Shabbat 22a:1"), cit("Tur, Orach Chayim 670:1"), cit(cm)],
            relations=[edge, dep_rel],
        )
    )
    assert check(t, ok, read=("Tur, Orach Chayim 670:1",))[0] is not None


def test_fix1_phrases_come_from_the_config_for_prompt_and_validator(session, kl):
    base = load_config()
    changed = replace(
        base.phrases, out_of_scope="OUT-OF-SCOPE-X", unchecked="UNCHECKED-X:", unusual="UNUSUAL-X"
    )
    c = replace(base, knowledge="base+lens", phrases=changed)
    prompt = knowledge_prompt(changed)
    assert "OUT-OF-SCOPE-X" in prompt and "UNCHECKED-X:" in prompt and "UNUSUAL-X" in prompt
    assert "מחוץ למה שנקלט ונבדק" not in prompt and "<<" not in prompt
    assert knowledge_prompt(base.phrases) != prompt
    # the validator follows: the old wording is now rejected, the new one accepted
    o = mk(session, question=Q_OUT, config=c)
    old = answer(claim("השאלה מחוץ למה שנקלט ונבדק"), scope="out_of_scope")
    rejected_with(o, old, "KL_SCOPE_NOTICE")
    assert check(o, answer(claim("OUT-OF-SCOPE-X: חסר"), scope="out_of_scope"))[0] is not None
    # and so does the prompt the model is actually sent
    llm = FakeLLM([reply(call("coverage", ref_or_query=Q_OUT))] * 1 + [reply()] * 0)
    llm.responses.append(
        reply(call("submit_answer", "z", **answer(claim("OUT-OF-SCOPE-X"), scope="out_of_scope")))
    )
    res = build_harness(session, llm, c, mode="personal").run(Q_OUT)
    assert res.error is None and "OUT-OF-SCOPE-X" in llm.calls[0]["messages"][0]["content"]
    assert res.prompt_hash != knowledge_prompt(base.phrases)  # hash is of the text actually used


def _raw_phrases():
    import pathlib

    from rav_harness.config import CONFIG_PATH

    return tomllib.loads(pathlib.Path(CONFIG_PATH).read_text(encoding="utf-8"))["knowledge"][
        "phrases"
    ]


def test_fix1_phrases_config_is_validated():
    assert parse_phrases(_raw_phrases()).practical.search("האם מותר")
    for key, bad in [
        ("unchecked", "  "),
        ("notice", ""),
        ("practical_regex", "("),
        ("practical_regex", ""),
        ("girsa_patterns", []),
        ("maran_patterns", ["ok", "["]),
        ("sa_psak_patterns", ["  "]),
    ]:
        raw = {**_raw_phrases(), key: bad}
        with pytest.raises(ValueError, match="knowledge.phrases"):
            parse_phrases(raw)
    raw = _raw_phrases()
    del raw["partial"]
    with pytest.raises(ValueError, match="partial"):
        parse_phrases(raw)
    with pytest.raises(ValueError):  # knowledge on without phrases
        HarnessConfig().with_knowledge("base")


@pytest.mark.parametrize(
    "txt",
    ['ט"ס בדפוס כאן', 'ל"ג בכ"י', "לא גרסינן כך בגמרא", "הנוסח משובש כאן"],
)
def test_fix1_4d_more_girsa_wordings_need_a_witness(session, kl, txt):
    rejected_with(mk(session), answer(claim(txt)), "KL_GIRSA_WITNESS")


@pytest.mark.parametrize("txt", ["נוהגים כמרן בזה", "נהגו כמרן", "קיבלו פסקי מרן"])
def test_fix1_4h_more_maran_wordings_need_documented_practice(session, kl, txt):
    t = mk(session)
    rejected_with(t, answer(claim(txt)), "KL_MARAN_INSTRUCTIONS")
    t.call("get_shitot", {"issue_key": "issue:candle-place"})
    sa = "Shulchan Arukh, Orach Chayim 670:1"
    ok = answer(claim(txt, [cit(sa)], kl_item_ids=[kl.p_sa.id]))
    assert check(t, ok, read=(sa,))[0] is not None


@pytest.mark.parametrize("txt", ['הלכה כשו"ע', "מרן הכריע וכן הלכה", "וכן נקטינן כמרן"])
def test_fix1_4i_more_shulchan_arukh_wordings_are_rejected(session, kl, txt):
    rejected_with(mk(session), answer(claim(txt)), "KL_SA_PSAK")


def test_fix1_relation_origin_is_the_weakest_of_both_ends(session, kl):
    def origin_of_rules_like():
        res = mk(session).call("get_shitot", {"issue_key": "issue:candle-place"})
        return [
            r
            for r in res["relations"]
            if r["rel"] == "rules_like" and r["from_item"] == kl.p_tur.id
        ][0]["evidence_origin"]

    session.get(KlItem, kl.p_tur.id).prompt_version = "human"
    session.commit()
    assert origin_of_rules_like() == "llm"  # from=human, to=llm -> weakest
    session.get(KlItem, kl.p_rashi.id).prompt_version = "human"
    session.commit()
    assert origin_of_rules_like() == "human"
    session.get(KlItem, kl.p_tur.id).prompt_version = "v1"
    session.get(KlItem, kl.p_tur.id).check_report = {"judge": {"ok": True}}
    session.commit()
    assert origin_of_rules_like() == "judge"  # judge < human


def test_fix1_unchecked_books_are_shown_by_hebrew_title_when_there_is_one(session, kl):
    kl.works.tos.he_title = "תוספות על שבת"
    session.add(KlCoverage(scope_key=SCOPE, work_id=kl.works.tos.id, status="not_ingested"))
    session.commit()
    t = mk(session)
    gate = t.kl.ctx.gate
    assert gate.unchecked_core_books == ["Tosafot on Shabbat"]
    assert gate.unchecked_display == {"Tosafot on Shabbat": "תוספות על שבת"}
    cov = t.call("coverage", {"ref_or_query": Q_IN})
    assert cov["unchecked_display"] == {"Tosafot on Shabbat": "תוספות על שבת"}
    base = claim("הגמרא", [cit("Shabbat 22a:1")])
    hebrew = claim("כיסוי חלקי. ספרים מרכזיים שלא נבדקו כאן: תוספות על שבת")
    english = claim("כיסוי חלקי. ספרים מרכזיים שלא נבדקו כאן: Tosafot on Shabbat")
    kw = {"scope": "partial", "unchecked_core_books": ["Tosafot on Shabbat"]}
    rejected_with(t, answer(base, english, **kw), "KL_COVERAGE_BOOKS", read=("Shabbat 22a:1",))
    assert check(t, answer(base, hebrew, **kw))[0] is not None
