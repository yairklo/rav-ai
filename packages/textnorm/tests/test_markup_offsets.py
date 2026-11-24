from hypothesis import given
from hypothesis import strategies as st

from textnorm import decode_spans, encode_spans, normalize, strip_sefaria_markup

RAW = (
    '<b>דין</b>  השכמת&nbsp;הבוקר<br>יִתְגַּבֵּר "רמב״ם" (טור):'
    '<sup class="footnote-marker">1</sup><i class="footnote">note</i> end'
)


def test_strip_markup_visible_text_and_origin():
    vis, origin = strip_sefaria_markup('a<i data-x="1"></i>&amp;b<br>c&bogus;')
    assert vis == "a&b c&bogus;"
    assert origin[1] == (len('a<i data-x="1"></i>'), len('a<i data-x="1"></i>&amp;'))  # entity span
    assert origin[0] == (0, 1)


def test_markup_sefaria_offsets_point_into_original():
    for level in ("A", "B"):
        n = normalize(RAW, level, markup="sefaria")
        assert "<" not in n.text and "note" not in n.text and "&" not in n.text
        i = n.text.index("יתגבר")
        assert n.original_slice(i, i + 5) == "יִתְגַּבֵּר"
        i = n.text.index("end")
        assert n.original_slice(i, i + 3) == "end"
    a = normalize(RAW, "A", markup="sefaria")
    i = a.text.index("הבוקר") - 1
    assert a.original_slice(i, i + 1) == "&nbsp;"  # decoded entity maps to the whole entity


def test_markup_none_is_unchanged_behaviour():
    assert "<" in normalize("<b>x</b>", "A").text


def test_offsets_roundtrip_and_text_binding():
    n = normalize(RAW, "B", markup="sefaria")
    data = n.offsets_bytes()
    assert decode_spans(data) == n.spans
    from textnorm import NormResult

    m = NormResult.from_offsets(RAW, n.text, data, "B")
    assert m.original_slice(0, 3) == "דין"
    big = normalize(RAW * 40, "B", markup="sefaria")
    assert len(big.offsets_bytes()) < len(big.text) // 4  # compact on realistic lengths


@given(st.lists(st.tuples(st.integers(0, 10**6), st.integers(0, 50)), max_size=200))
def test_encode_decode_property(items):
    spans = tuple(sorted((s, s + ln) for s, ln in items))
    assert decode_spans(encode_spans(spans)) == spans


@given(st.text(alphabet="אבגדהוזחטיכלמנסעפצקרשת ִּ\"'<>&;nbsp\n-:", max_size=60))
def test_sefaria_normalize_offsets_are_in_range(raw):
    for level in ("A", "B"):
        n = normalize(raw, level, markup="sefaria")
        assert all(0 <= s < e <= len(raw) for s, e in n.spans)
