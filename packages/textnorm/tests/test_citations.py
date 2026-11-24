import pytest

from textnorm import find_citations, parse_numeral, to_gematria

CASES = [
    ('שבת כא ע"ב', "Shabbat 21b", "bavli"),
    ('גמרא שבת כ"א ע"א', "Shabbat 21a", "bavli"),
    ("שבת דף כא:", "Shabbat 21b", "bavli"),
    ("במסכת ברכות דף ב' ע\"ב", None, None),  # geresh numeral without final: not required
    ('בבא קמא ס"ב ע"ב', "Bava Kamma 62b", "bavli"),
    ('ב"מ קי"ד ע"א', "Bava Metzia 114a", "bavli"),
    ('רש"י בשבת לא ע"א ד"ה מאי', "Shabbat 31a", "bavli"),
    ('או"ח סי\' תרע"א ס"ב', "Shulchan Arukh, Orach Chayim 671:2", "shulchan_arukh"),
    ('שו"ע יו"ד סימן קפ"ז סעיף ג', "Shulchan Arukh, Yoreh De'ah 187:3", "shulchan_arukh"),
    ('משנ"ב סי\' תרע"א ס"ק ג', "Mishnah Berurah 671:3", "mishnah_berurah"),
    ('רמב"ם הל\' חנוכה פ"ג ה"א', "Mishneh Torah, Scroll of Esther and Hanukkah 3:1", "rambam"),
    ('הרמב"ם הלכות שבת פרק ה הלכה ג', "Mishneh Torah, Sabbath 5:3", "rambam"),
    ('משנה ברכות פ"א מ"א', "Mishnah Berakhot 1:1", "mishnah"),
    ("מסכת אבות פרק ב משנה ד", "Mishnah Pirkei Avot 2:4", "mishnah"),
]


@pytest.mark.parametrize("text,ref,kind", CASES)
def test_cases(text, ref, kind):
    found = find_citations(text)
    if ref is None:
        return
    assert found, text
    c = found[0]
    assert (c.id, c.kind) == (ref, kind)
    assert 0.5 <= c.confidence <= 1
    assert text[c.span[0] : c.span[1]] == c.text


def test_span_inside_sentence_with_niqqud():
    text = "כמו שאמרו בְּשַׁבָּת כא ע״ב, ועוד"
    (c,) = find_citations(text)
    assert c.id == "Shabbat 21b"
    assert c.text.endswith("שַׁבָּת כא ע״ב")  # span covers the (prefixed) tractate token


def test_bare_daf_needs_context():
    (a,) = find_citations('ראה דף כ"א ע"ב')
    assert a.id is None and a.kind == "daf" and a.parts["daf"] == 21
    (b,) = find_citations('ראה דף כ"א ע"ב', context="Shabbat")
    assert b.id == "Shabbat 21b"


def test_context_siman():
    (c,) = find_citations('סי\' תרע"א ס"ב', context="Shulchan Arukh, Orach Chayim")
    assert c.id == "Shulchan Arukh, Orach Chayim 671:2"
    assert find_citations('סי\' תרע"א ס"ב') == []


def test_plain_words_not_citations():
    assert find_citations("שבת שלום ומועדים לשמחה") == []
    assert find_citations("אמר רב ברכות רבות") == []


def test_confidence_ordering():
    full = find_citations('שבת דף כא ע"ב')[0].confidence
    bare = find_citations("שבת כא")[0].confidence
    assert full > bare


def test_multiple():
    t = 'עיין שבת כא ע"ב וברמב"ם הל\' חנוכה פ"ג ה"א ובמשנ"ב סי\' תרע"א ס"ק א.'
    ids = [c.id for c in find_citations(t)]
    assert ids == [
        "Shabbat 21b",
        "Mishneh Torah, Scroll of Esther and Hanukkah 3:1",
        "Mishnah Berurah 671:1",
    ]


@pytest.mark.parametrize(
    "tok,val",
    [('כ"א', 21), ('תרע"א', 671), ('קפ"ז', 187), ('ט"ו', 15), ('ט"ז', 16), ("ה'תשפ\"ד", 5784),
     ('תתק', 900), ('ס"ב', 62), ("12", 12)],
)  # fmt: skip
def test_gematria(tok, val):
    assert parse_numeral(tok) == val


@pytest.mark.parametrize("tok", ["אב", "שלום", "תתת", "", "ככ", "בא"])
def test_gematria_invalid(tok):
    assert parse_numeral(tok) is None


def test_gematria_roundtrip():
    for n in range(1, 1000):
        assert parse_numeral(to_gematria(n)) == n
