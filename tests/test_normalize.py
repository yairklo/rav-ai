import random

import pytest

from rav_ai.normalize import OffsetMap, normalize_a, normalize_b

RAW = '<b>דין</b>  השכמת&nbsp;הבוקר<br>יִתְגַּבֵּר כַּאֲרִי "רמב״ם" (טור):  <i data-commentator="x"></i>end'


def slice_of(norm, word, raw):
    s = norm.text.index(word)
    rs, re_ = norm.offsets.to_raw(s, s + len(word))
    return raw[rs:re_]


def test_form_a_strips_markup_and_niqqud_unifies_glyphs():
    a = normalize_a(RAW)
    assert a.text == 'דין השכמת הבוקר יתגבר כארי "רמב"ם" (טור): end'
    assert "ִ" not in a.text and "<" not in a.text and "  " not in a.text


def test_form_b_drops_punctuation():
    b = normalize_b(RAW)
    assert b.text == "דין השכמת הבוקר יתגבר כארי רמבם טור end"


@pytest.mark.parametrize("norm", [normalize_a, normalize_b])
def test_offsets_return_exact_raw_characters(norm):
    n = norm(RAW)
    assert slice_of(n, "יתגבר", RAW) == "יִתְגַּבֵּר"
    assert slice_of(n, "כארי", RAW) == "כַּאֲרִי"
    assert slice_of(n, "דין", RAW) == "דין"  # tags around it are outside the range
    assert slice_of(n, "הבוקר", RAW) == "הבוקר"
    assert slice_of(n, "end", RAW) == "end"
    # whole text covers from first to last kept raw char
    rs, re_ = n.offsets.to_raw(0, len(n.text))
    assert RAW[rs:re_].startswith("<b>דין") or RAW[rs:re_].startswith("דין")
    assert RAW[rs:re_].endswith("end")


def test_offsets_for_replaced_and_collapsed_sequences():
    raw = "א&nbsp;ב   ג"
    a = normalize_a(raw)
    assert a.text == "א ב ג"
    s = a.text.index(" ")
    assert raw[slice(*a.offsets.to_raw(s, s + 1))] == "&nbsp;"  # entity -> one space
    b2 = a.text.index("ב")
    assert raw[slice(*a.offsets.to_raw(b2, b2 + 1))] == "ב"
    assert raw[slice(*a.offsets.to_raw(a.text.index("ג"), len(a.text)))] == "ג"


def test_gershayim_variants_unified_and_abbreviations_not_expanded():
    for raw in ['רמב"ם', "רמב״ם", "רמב”ם"]:
        assert normalize_a(raw).text == 'רמב"ם'
        assert normalize_b(raw).text == "רמבם"  # abbreviation kept as written
    assert normalize_a("ר' יוחנן").text == "ר' יוחנן"


def test_empty_and_markup_only():
    assert normalize_a("").text == "" and normalize_b("<br><i></i>").text == ""


def test_roundtrip_serialization():
    n = normalize_b(RAW)
    assert OffsetMap.from_bytes(n.offsets.to_bytes()).chunks == n.offsets.chunks


@pytest.mark.parametrize("norm", [normalize_a, normalize_b])
def test_offsets_property_random(norm):
    rnd = random.Random(7)
    alphabet = list("אבגדהוזחטיכלמנסעפצקרשת") + [
        "ִ",
        "ּ",
        '"',
        "'",
        " ",
        "  ",
        "<b>",
        "</b>",
        "<br>",
        "&nbsp;",
        ":",
        "-",
        "־",
        "\n",
        "״",
        "x",
    ]
    for _ in range(1500):
        raw = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 40)))
        n = norm(raw)
        assert n.text == n.text.strip() and "  " not in n.text
        assert sum(c[2] for c in n.offsets.chunks) == len(n.text)
        for i, ch in enumerate(n.text):
            rs, re_ = n.offsets.to_raw(i, i + 1)
            assert 0 <= rs < re_ <= len(raw)
            piece = raw[rs:re_]
            if ch.isalpha() and len(piece) == 1:  # a plain letter maps to itself
                assert piece == ch
            if ch.isalpha():  # and the raw span always contains that letter
                assert ch in piece
