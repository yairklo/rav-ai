import pytest

from textnorm import find_acronyms, load_dictionary, rank_candidates
from textnorm.acronyms import DOMAINS


def texts(found):
    return [c.norm for c in found]


def test_finds_basic_candidates():
    t = 'אמר ר"ל ע"כ וכתב מ"מ'
    found = find_acronyms(t)
    assert texts(found) == ['ר"ל', 'ע"כ', 'מ"מ']
    assert [t[c.span[0] : c.span[1]] for c in found] == ['ר"ל', 'ע"כ', 'מ"מ']
    rl = found[0]
    assert "ריש לקיש" in [e.text for e in rl.expansions]
    assert all(e.source == "curated-v1" for e in rl.expansions)


def test_variant_quotes_and_niqqud():
    t = "כתב רַשִׁ״י וְרֵ”ל ועוד"
    assert texts(find_acronyms(t)) == ['רש"י', 'ור"ל']
    assert find_acronyms(t)[0].span == (t.index("רַ"), t.index("י", t.index("״")) + 1)


def test_prefix_stripping():
    (c,) = find_acronyms('והוא אמר וע"כ')
    assert c.prefix == "ו" and c.known


def test_unknown_is_candidate_without_expansions():
    (c,) = find_acronyms('אמר זק"ף')
    assert not c.known and c.expansions == ()


def test_numerals_flagged():
    t = 'בכ"ה בכסליו ותרע"א'
    c1, c2 = find_acronyms(t)
    assert c1.could_be_numeral and c2.could_be_numeral


def test_geresh_abbreviations():
    t = "ראה תוס' וכו' בסי' ג' ואמר 'שלום' לכולם"
    found = find_acronyms(t)
    assert [c.norm for c in found] == ["תוס'", "וכו'", "בסי'", "ג'"]  # 'שלום' quote not reported
    assert all(c.kind == "abbreviation" for c in found)


def test_does_not_expand_text():
    t = 'ר"ל'
    assert find_acronyms(t)[0].text == t  # reports the original slice; text is never rewritten


def test_dictionary_integrity():
    d = load_dictionary()
    assert len(d) > 140
    assert d["רל"]  # ר"ל, key without quotes
    import json
    from importlib import resources

    data = json.loads(resources.files("textnorm").joinpath("data/acronyms.json").read_text("utf8"))
    for e in data["entries"]:
        assert e["expansions"] and e["source"] in data["meta"]["sources"]
        assert all(x["domain"] in data["meta"]["domains"] for x in e["expansions"])
        assert e["abbr"].count('"') == 1 or e["abbr"].endswith("'")


def test_added_expansions_exist_with_domain_and_prior():
    d = load_dictionary()
    want = {'ע"כ': "על כן", 'ע"ז': "על זה", 'ח"ו': "חס וחלילה", 'ת"י': "תלמוד ירושלמי"}
    for abbr, text in want.items():
        (ex,) = [e for e in d[abbr.replace('"', "")] if e.text == text]
        assert ex.domain in DOMAINS and set(ex.prior) == set(DOMAINS)
        assert all(0 <= v <= 1 for v in ex.prior.values())


def test_every_expansion_has_domain_and_full_prior():
    for exps in load_dictionary().values():
        for e in exps:
            assert e.domain in DOMAINS and set(e.prior) == set(DOMAINS)
            assert e.domain == max(DOMAINS, key=lambda k: e.prior[k]) or e.prior[e.domain] == max(
                e.prior.values()
            )


def test_rank_candidates_orders_by_context_without_choosing():
    talmud = [e.text for e in rank_candidates('ע"ז', "talmud")]
    general = [e.text for e in rank_candidates('ע"ז', "general")]
    assert talmud[0] == "עבודה זרה" and general[0] == "על זה"
    assert sorted(talmud) == sorted(general) == sorted(["עבודה זרה", "על זה"])  # same set
    assert rank_candidates('ת"י', "talmud")[0].text == "תלמוד ירושלמי"
    assert rank_candidates('ת"י', "general")[0].text == "תרגום יונתן"
    assert rank_candidates('ח"ו', "responsa")[0].text in {"חס ושלום", "חס וחלילה"}
    ranks = [e.prior["halakha"] for e in rank_candidates('ר"י', "halakha")]
    assert ranks == sorted(ranks, reverse=True)


def test_rank_candidates_edges():
    assert rank_candidates('זק"ף', "talmud") == []
    assert [e.text for e in rank_candidates('וע"כ', "general")]  # clitic prefix handled
    with pytest.raises(ValueError):
        rank_candidates('ר"ל', "poskim")
