"""Wrapper tests: forms A/B as stored in the DB (textnorm does the work; see packages/textnorm)."""

import random

import pytest

from rav_ai.normalize import OffsetMap, normalize_a, normalize_b

RAW = '<b>דין</b>  השכמת&nbsp;הבוקר<br>יִתְגַּבֵּר כַּאֲרִי "רמב״ם" (טור):  <i data-commentator="x"></i>end'


def raw_of(norm, word):
    s = norm.text.index(word)
    rs, re_ = norm.offsets.to_raw(s, s + len(word))
    return RAW[rs:re_]


def test_form_a():
    a = normalize_a(RAW)
    assert a.text == 'דינ השכמת הבוקר יתגבר כארי "רמב"מ" (טור): end'  # finals folded, markup gone


def test_form_b_no_punctuation_plene_double():
    assert normalize_b(RAW).text == "דינ השכמת הבוקר יתגבר כארי רמבמ טור end"
    assert normalize_b("מצווה ייתכן").text == "מצוה יתכנ"  # plene="double", never "vav"
    assert normalize_b("כתוב").text == "כתוב"  # medial vav kept (no plene="vav")


@pytest.mark.parametrize("norm", [normalize_a, normalize_b])
def test_offsets_return_exact_raw_characters(norm):
    n = norm(RAW)
    assert raw_of(n, "יתגבר") == "יִתְגַּבֵּר"
    assert raw_of(n, "כארי") == "כַּאֲרִי"
    assert raw_of(n, "דינ") == "דין"
    assert raw_of(n, "end") == "end"


def test_entity_and_abbreviations():
    a = normalize_a("א&nbsp;ב")
    assert a.text == "א ב" and "א&nbsp;ב"[slice(*a.offsets.to_raw(1, 2))] == "&nbsp;"
    assert normalize_a('רמב"ם').text == normalize_a("רמב״ם").text  # not expanded


def test_roundtrip_serialization():
    n = normalize_b(RAW)
    assert OffsetMap.from_bytes(n.offsets.to_bytes()).spans == n.offsets.spans


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
        "\u05be",
        "\n",
        "\u05f4",
        "x",
    ]
    for _ in range(1500):
        raw = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 40)))
        n = norm(raw)
        assert n.text == n.text.strip() and "  " not in n.text
        for i in range(len(n.text)):
            rs, re_ = n.offsets.to_raw(i, i + 1)
            assert 0 <= rs < re_ <= len(raw)
