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
    # Mishnah Chagigah 2:1-10 (real link): chapter 2 has 7 mishnayot, nothing is invented
    mish = index("Mishnah Chagigah", [("2", str(i)) for i in range(1, 8)])
    assert mish.resolve("Mishnah Chagigah 2:1-10") == []
    psalms = index("Psalms", [(str(c), str(v)) for c in (116, 117, 118) for v in (1, 2)])
    assert psalms.resolve("Psalms 117:7-8") == []


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
