import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from textnorm import normalize

HEB = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
MARKS = "ְֱִַָׇֹּֽׁׂ֑֣"
ALPHABET = HEB + MARKS + " \t\n ‏​\"'׳״־-.,:;()abcZ’“שׁוֹ1"

texts = st.text(alphabet=ALPHABET, max_size=60)


def test_level_a_basic():
    r = normalize("מֵאֵימָתַי קוֹרִין אֶת שְׁמַע בָּעַרְבִית")  # Mishnah Berakhot 1:1
    assert r.text == "מאימתי קורינ את שמע בערבית"
    assert normalize("בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים").text == "בראשית ברא אלהימ"  # Gen 1:1 w/ taamim


def test_level_a_unifies_forms():
    r = normalize("ר״ל  ‏ ר″ל ע׳ ע’ א־ב א–ב ג")
    assert r.text == "ר\"ל ר\"ל ע' ע' א-ב א-ב ג"


def test_level_b():
    assert normalize('ר"ל, אמר: "שלום"!', "B").text == "רל אמר שלומ"
    assert normalize("אל־תיקח", "B").text == "אל תיקח"
    assert normalize("מצווה ייתכן", "B").text == "מצוה יתכנ"
    assert normalize("מצווה", "B", plene="none").text == "מצווה"
    assert normalize("כתוב מצוה טוב", "B", plene="vav").text == "כתב מצה טוב"


def test_presentation_forms():
    r = normalize("שׁוֹ")  # שׁ , וֹ presentation forms
    assert r.text == "שו"
    assert r.spans == ((0, 1), (1, 2))


def test_map_span_examples():
    t = "אָמַר רַבִּי"
    r = normalize(t)
    assert r.text == "אמר רבי"
    assert r.original_slice(0, 3) == "אָמַר"  # trailing niqqud included
    assert r.original_slice(4, 7) == "רַבִּי"
    assert r.map_span(3, 3) == (r.spans[3][0],) * 2
    with pytest.raises(ValueError):
        r.map_span(2, 99)
    with pytest.raises(ValueError):
        normalize("x", "C")


def test_quote_validation_use_case():
    original = "תָּנוּ רַבָּנַן: מִצְוַת חֲנוּכָּה נֵר אִישׁ וּבֵיתוֹ"
    r = normalize(original)
    quote = "מצות חנוכה נר איש"
    i = r.text.index(quote)
    s, e = r.map_span(i, i + len(quote))
    assert original[s:e] == "מִצְוַת חֲנוּכָּה נֵר אִישׁ"


def test_empty():
    r = normalize("")
    assert r.text == "" and r.map_span(0, 0) == (0, 0)
    assert normalize("  ​ ").text == ""


# ---------------------------------------------------------------- property tests
def _ranges(n):
    return st.tuples(st.integers(0, n), st.integers(0, n)).map(lambda t: tuple(sorted(t)))


@settings(max_examples=300, deadline=None)
@given(texts, st.sampled_from(["A", "B"]), st.sampled_from(["none", "double"]), st.data())
def test_map_span_returns_correct_chars(text, level, plene, data):
    r = normalize(text, level, plene=plene)
    assert len(r.text) == len(r.spans)
    # spans are in bounds and ordered
    prev = (0, 0)
    for s, e in r.spans:
        assert 0 <= s < e <= len(text)
        assert s >= prev[0] and e >= prev[1]
        prev = (s, e)
    a, b = data.draw(_ranges(len(r.text)))
    s, e = r.map_span(a, b)
    assert 0 <= s <= e <= len(text)
    sub = r.text[a:b]
    if sub and sub == sub.strip():
        # re-normalizing the mapped original slice reproduces exactly the requested range
        assert normalize(text[s:e], level, plene=plene).text == sub


@settings(max_examples=200, deadline=None)
@given(texts, st.sampled_from(["A", "B"]))
def test_idempotent(text, level):
    r = normalize(text, level)
    assert normalize(r.text, level).text == r.text


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=HEB + " ", max_size=30), st.data())
def test_diacritics_are_invisible(base, data):
    marks = st.text(alphabet=MARKS, max_size=3)
    noisy = "".join(c + data.draw(marks) for c in base)
    r = normalize(noisy)
    assert r.text == normalize(base).text
    # each non-space output char maps back onto the right base letter (+ its marks)
    for k, (s, e) in enumerate(r.spans):
        if r.text[k] != " ":
            assert normalize(noisy[s:e]).text == r.text[k]


@settings(max_examples=200, deadline=None)
@given(texts)
def test_output_has_no_marks_or_runs_of_space(text):
    for lvl in ("A", "B"):
        out = normalize(text, lvl).text
        assert "  " not in out and out == out.strip()
        assert not any("֑" <= c <= "ׇ" for c in out)
        assert not any(c in "ךםןףץ" for c in out)
