"""Range endpoints in a hole of the Hebrew text (real citations from the Sefaria links export)."""

from rav_ai.ingest import rnf_classify as rc
from rav_ai.sefaria.linkrefs import WorkIndex


def index(title, labels_list):
    rows = [
        (f"{title} {':'.join(lab)}", 0, tuple(range(len(lab))), [], lab, i)
        for i, lab in enumerate(labels_list, 1)
    ]
    return WorkIndex.build(title, rows)


# Tosafot on Bava Kamma: 55b:1:1 .. 62b:5:1 exist, but 62b:6 has no Hebrew text (a hole).
TOSAFOT = "Tosafot on Bava Kamma"
TOS = [("55b", "1", "1"), ("55b", "2", "1"), ("56a", "1", "1"), ("56a", "3", "1")] + [
    ("62b", str(i), "1") for i in (1, 2, 3, 5, 7)
]


def test_range_with_end_in_a_hole_resolves_to_existing_segments_inside_the_range():
    idx = index(TOSAFOT, TOS)
    assert idx._prefix.get(("", ("62b", "6"))) is None  # 62b:6 really is absent
    got = idx.resolve("Tosafot on Bava Kamma 55b:1-62b:6")
    assert got == [1, 2, 3, 4, 5, 6, 7, 8]  # up to 62b:5, not 62b:7 (outside the range)


def test_range_with_start_in_a_hole():
    idx = index(TOSAFOT, TOS)
    assert idx.resolve("Tosafot on Bava Kamma 56a:2-62b:3") == [4, 5, 6, 7]


def test_uncertain_ranges_stay_unresolved():
    idx = index(TOSAFOT, TOS)
    # ends after the last existing segment / starts before the first: never clamped
    assert idx.resolve("Tosafot on Bava Kamma 55b:1-70a:4") == []
    assert idx.resolve("Tosafot on Bava Kamma 2a:1-62b:3") == []
    # single refs in a hole are not ranges: still unresolved
    assert idx.resolve("Tosafot on Bava Kamma 62b:6") == []
    psalms = index("Psalms", [(str(c), str(v)) for c in (116, 117, 118) for v in (1, 2)])
    assert psalms.resolve("Psalms 117:7-8") == []


def chagigah(complete=False):
    """The real structure: chapters 1-3 with 8 / 7 / 8 mishnayot."""
    labels = [(str(c), str(m)) for c, n in ((1, 8), (2, 7), (3, 8)) for m in range(1, n + 1)]
    idx = index("Mishnah Chagigah", labels)
    idx.allow_hole_ranges = not complete
    return idx


def test_end_of_chapter_is_not_a_hole_the_range_is_not_truncated():
    idx = chagigah()
    # real link "Mishnah Chagigah 2:1-10": chapter 2 ends at 2:7, 2:10 is not a hole
    assert idx.resolve("Mishnah Chagigah 2:1-10") == []
    assert idx.resolve("Mishnah Chagigah 2:9-3:1") == []  # start after the chapter's last mishnah
    assert idx.resolve("Mishnah Chagigah 1:5-2:12") == []
    assert idx.resolve("Mishnah Chagigah 2:1-7") == list(range(9, 16))  # exact range still fine


def test_hole_range_off_for_complete_base_texts():
    labels = [("2", "1"), ("2", "3")]  # a real interior hole (2:2)...
    idx = index("Mishnah Berakhot", labels)
    assert idx.resolve("Mishnah Berakhot 2:1-2:2") == [1]  # ...is resolved for other works
    idx.allow_hole_ranges = False
    assert idx.resolve("Mishnah Berakhot 2:1-2:2") == []  # but never for a complete base text


def test_daf_address_in_integer_node_is_not_a_hole_range():
    psalms = index("Psalms", [(str(c), str(v)) for c in range(1, 151) for v in (1, 2, 3)])
    # '12a' used to become position 24 -> "Psalms 24:1 ... 25:3": a wrong edge
    assert psalms.resolve("Psalms 12a:1-12b:3") == []
    assert psalms.resolve("Psalms 12:1-13:3")  # a real range still resolves


def test_exact_and_plain_ranges_unchanged():
    idx = index(TOSAFOT, TOS)
    assert idx.resolve("Tosafot on Bava Kamma 62b:1:1") == [5]
    assert idx.resolve("Tosafot on Bava Kamma 62b:1-62b:3") == [5, 6, 7]


def profile(title, labels_list, types, complete=False):
    idx = index(title, labels_list)
    return rc.WorkProfile(idx, {"": types}, complete)


def test_classes():
    p = profile(TOSAFOT, TOS, ["Talmud", "Integer", "Integer"])
    assert rc.classify(p, "Tosafot on Bava Kamma 55b:1-62b:6")[0] == rc.BUG
    assert rc.classify(p, "Tosafot on Bava Kamma 62b:6:1") == (rc.PARTIAL, "hole_start")
    assert rc.classify(p, "Tosafot on Bava Kamma 70a:1:1") == (rc.PARTIAL, "beyond_start")
    assert rc.classify(p, "Tosafot on Bava Kamma 62b:1:1:1")[0] == rc.NUMBERING
    assert rc.classify(p, "Tosafot on Bava Kamma 62b:x") == (rc.OTHER, "unparseable")
    ps = profile("Psalms", [("117", "1"), ("117", "2")], ["Integer", "Integer"], complete=True)
    assert rc.classify(ps, "Psalms 117:7") == (rc.OTHER, "invalid_ref_in_complete_text")
    assert rc.classify(ps, "Psalms 12a:1")[0] == rc.NUMBERING
    assert rc.classify(ps, "Psalms 12a:1-12b:3")[0] == rc.NUMBERING


def test_classifier_end_of_chapter_is_not_a_resolver_bug():
    ch = [(str(c), str(m)) for c, n in ((1, 8), (2, 7), (3, 8)) for m in range(1, n + 1)]
    p = profile("Mishnah Chagigah", ch, ["Integer", "Integer"], complete=True)
    assert rc.classify(p, "Mishnah Chagigah 2:1-10") == (rc.OTHER, "invalid_ref_in_complete_text")
    other = profile("Commentary on X", ch, ["Integer", "Integer"])
    assert rc.classify(other, "Commentary on X 2:1-10") == (rc.PARTIAL, "beyond_end")


def test_stratified_sample_random_part_never_repeats_a_stratified_link():
    rows = [rc.Classified(i, f"c{i}", f"B{i % 40}", rc.PARTIAL, "x") for i in range(3000)]
    got = rc.stratified_sample(rows)
    assert len(got) == 600 + 200 and len({r.link_id for r in got}) == len(got)
