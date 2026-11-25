from __future__ import annotations

from conftest import make_text
from rav_knowledge.evidence import SegText, find_span, fragments, locate, reanchor

RAW = "<b>מתני׳</b> אָמַר רַב חַיָּיב, וּשְׁמוּאֵל אָמַר פָּטוּר. אמר רב חייב שוב"


def seg(raw: str = RAW) -> SegText:
    t = make_text(1, 1, raw)
    return SegText(1, 1, "X 1:1", t.raw_hash, t.text_a, t.text_b, t.offsets_b)


def test_locate_maps_to_raw_offsets_through_niqqud_and_markup():
    s = seg()
    loc = locate("ושמואל אמר פטור", s)
    assert loc.status == "located"
    # the raw span covers the vocalized words, markup and punctuation excluded
    assert RAW[loc.span_start : loc.span_end].startswith("וּשְׁמוּאֵל")
    assert RAW[loc.span_start : loc.span_end].endswith("פָּטוּר")
    assert loc.quote_prefix and loc.quote_suffix is not None
    assert len(loc.quote_prefix) <= 32


def test_orthography_and_final_letters_do_not_matter():
    s = seg()
    assert locate("אָמַר רב חַיב", s).status in ("located", "ambiguous")  # plene/niqqud


def test_ellipsis_fragments_in_order():
    s = seg()
    assert fragments("אמר רב ... פטור") == ["אמר רב", "פטור"]
    assert locate("מתני ... ושמואל ... פטור", s).status == "located"
    assert locate("פטור ... מתני", s).status == "not_found"  # wrong order


def test_not_found_and_out_of_scope():
    assert locate("דבר שלא כתוב", seg()).status == "not_found"
    assert locate("כל דבר", None).status == "out_of_scope"


def test_ambiguous_when_quote_repeats():
    loc = locate("אמר רב חייב", seg())
    assert loc.status == "ambiguous"
    # numbered anchors pick the occurrence
    s = seg()
    idx = s.text_b.split().index("אמר", 4)
    loc2 = locate("אמר רב חייב", s, w_start=idx)
    assert loc2.status == "located"
    assert loc2.span_start > loc.span_start


def test_find_span_none():
    assert find_span("", "abc") is None


def test_reanchor_after_text_change():
    old = seg()
    loc = locate("ושמואל אמר פטור", old)
    new_raw = "הוספה בהתחלה " + RAW
    new = seg(new_raw)
    assert new.raw_hash != old.raw_hash
    moved, stale = reanchor(
        "ושמואל אמר פטור", loc.quote_prefix, loc.quote_suffix, old.raw_hash, new
    )
    assert not stale and moved.span_start > loc.span_start
    assert new_raw[moved.span_start : moved.span_end].startswith("וּשְׁמוּאֵל")


def test_reanchor_unchanged_hash_is_noop_and_lost_quote_is_stale():
    s = seg()
    assert reanchor("אמר", None, None, s.raw_hash, s) == (None, False)
    changed = seg("טקסט אחר לגמרי")
    assert reanchor("ושמואל אמר פטור", None, None, "old", changed) == (None, True)


def test_reanchor_prefers_matching_context_among_repeats():
    old = seg("אמר רב חייב ראשון ואמר רב חייב שני")
    loc = locate("אמר רב חייב", old, w_start=4)
    new = seg("תוספת קדמית אמר רב חייב ראשון ואמר רב חייב שני")
    moved, stale = reanchor("אמר רב חייב", loc.quote_prefix, loc.quote_suffix, old.raw_hash, new)
    assert not stale
    assert moved.quote_prefix.endswith(loc.quote_prefix)  # the second occurrence, same context
