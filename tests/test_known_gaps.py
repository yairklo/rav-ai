import copy

import pytest

from rav_ai import known_gaps as kg


def test_shipped_file_loads_and_validates():
    gaps = kg.load()
    ritva = [g for g in gaps if g.work == "Ritva on Shabbat" and g.source == "he.wikisource"]
    assert ritva and ritva[0].status == "partial_external"
    assert ritva[0].revid > 0 and ritva[0].license == "CC BY-SA 4.0" and not ritva[0].ingested


GOOD = {
    "work": "W",
    "scope": "s",
    "status": "partial_external",
    "source": "he.wikisource",
    "revid": 5,
    "checked": "2026-11-25",
    "license": "CC BY-SA 4.0",
    "ingested": False,
    "note": "n",
}


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "done"},
        {"revid": 0},  # wikisource needs a pinned revid
        {"revid": "5"},
        {"checked": "yesterday"},
        {"ingested": "no"},
        {"license": ""},
        {"extra": 1},
    ],
)
def test_bad_schema_is_rejected(patch):
    with pytest.raises(kg.KnownGapsError):
        kg.parse({"gap": [{**copy.deepcopy(GOOD), **patch}]})
    kg.parse({"gap": [GOOD]})


def test_missing_key_and_empty_file_are_rejected():
    with pytest.raises(kg.KnownGapsError):
        kg.parse({"gap": [{k: v for k, v in GOOD.items() if k != "license"}]})
    with pytest.raises(kg.KnownGapsError):
        kg.parse({})
