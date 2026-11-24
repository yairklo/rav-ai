from textnorm import find_acronyms, load_dictionary


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
        assert e["abbr"].count('"') == 1 or e["abbr"].endswith("'")
