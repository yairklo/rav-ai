from rav_evals.schema import Answer, Citation, Claim, ClaimType
from rav_evals.textnorm_stub import normalize
from rav_evals.verifier import (
    AttributionVerdict,
    InMemorySourceStore,
    verify_answer,
)

SRC = "מצות חנוכה <b>נר</b> איש וביתו, והמהדרין נר לכל אחד ואחד."
STORE = InMemorySourceStore({"A 1": SRC, "NIKUD": "מִצְוַת חֲנֻכָּה נֵר אִישׁ וּבֵיתוֹ"})


def ans(*claims, disclaimer="דעת המערכת, לא פסק"):
    return Answer(claims=list(claims), disclaimer=disclaimer)


def quote(ref="A 1", text="נר איש וביתו", **kw):
    return Claim(
        type=ClaimType.QUOTE, text="x", citations=[Citation(ref=ref, quoted_text=text, **kw)]
    )


def codes(report):
    return report.failure_codes()


def test_normalize_strips_nikud_html_punct_finals():
    assert normalize("מִצְוַת <b>חנוכה</b>, ") == normalize("מצות חנוכה")
    assert normalize("ךםןףץ") == normalize("כמנפצ")
    assert normalize('מג"א א׳־ב') == "מג א א ב"


def test_clean_answer_passes():
    r = verify_answer(ans(quote()), ["A 1"], STORE)
    assert r.passed and r.citations_checked == 1


def test_v1_ref_not_found():
    r = verify_answer(ans(quote(ref="Nope 9")), ["Nope 9"], STORE)
    assert codes(r) == ["V1:ref_not_found"]


def test_v1_ref_not_retrieved():
    r = verify_answer(ans(quote()), [], STORE)
    assert codes(r) == ["V1:ref_not_retrieved"]
    assert r.failures[0].ref == "A 1" and r.failures[0].claim_index == 0


def test_v2_quote_missing():
    r = verify_answer(ans(quote(text="נר לכל בית")), ["A 1"], STORE)
    assert codes(r) == ["V2:quote_not_in_source"]


def test_v2_nikud_and_html_tolerated():
    assert verify_answer(ans(quote(ref="NIKUD", text="מצות חנכה נר")), ["NIKUD"], STORE).passed
    assert verify_answer(ans(quote(text="מצות חנוכה נר איש")), ["A 1"], STORE).passed


def test_v2_ellipsis_ordered_fragments():
    assert verify_answer(ans(quote(text="מצות חנוכה ... נר לכל אחד")), ["A 1"], STORE).passed
    r = verify_answer(ans(quote(text="נר לכל אחד ... מצות חנוכה")), ["A 1"], STORE)
    assert codes(r) == ["V2:quote_not_in_source"]


def test_v2_empty_quote_fails():
    r = verify_answer(ans(quote(text="  ...  ")), ["A 1"], STORE)
    assert codes(r) == ["V2:quote_not_in_source"]


def test_v3_quote_without_citation():
    r = verify_answer(ans(Claim(type=ClaimType.QUOTE, text="x")), [], STORE)
    assert codes(r) == ["V3:quote_without_citation"]


def test_v3_system_claims_need_flag():
    for t in (ClaimType.CHIDUSH, ClaimType.SYSTEM_CONCLUSION):
        r = verify_answer(ans(Claim(type=t, text="x")), [], STORE)
        assert codes(r) == ["V3:unflagged_system_claim"]
        ok = verify_answer(ans(Claim(type=t, text="x", flagged_as_system=True)), [], STORE)
        assert ok.passed


def test_v3_disclaimer_and_psak():
    assert codes(verify_answer(ans(disclaimer=""), [], STORE)) == ["V3:missing_disclaimer"]
    a = Answer(claims=[], disclaimer="d", is_psak=True)
    assert codes(verify_answer(a, [], STORE)) == ["V3:marked_as_psak"]


def test_v4_stub_skips_and_custom_checker_can_fail():
    a = ans(quote(attribution='המ"ב בשם המג"א'))
    r = verify_answer(a, ["A 1"], STORE)
    assert r.passed and len(r.skipped) == 1 and r.skipped[0].startswith("V4")

    class Bad:
        def check(self, citation, source_text):
            return AttributionVerdict("fail", "wrong author")

    r = verify_answer(a, ["A 1"], STORE, attribution_checker=Bad())
    assert codes(r) == ["V4:attribution_mismatch"]
    assert r.failures[0].reason == "wrong author"


def test_custom_normalizer_is_used():
    r = verify_answer(ans(quote(text="NR")), ["A 1"], STORE, normalize=lambda s: "nr")
    assert r.passed
