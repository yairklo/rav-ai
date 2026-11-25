from rav_ai.sefaria import gcs
from rav_ai.sefaria.layout import parse_version_object, priority_tier


def _tier(path):
    return priority_tier(parse_version_object(path, 1))


def test_download_priority_tiers():
    b = "json/Talmud/Bavli/Seder Moed"
    assert _tier(f"{b}/Shabbat/Hebrew/Vilna.json") == 0
    assert (
        _tier(
            "json/Talmud/Bavli/Rishonim on Talmud/Rashi/Seder Moed/Rashi on Shabbat/Hebrew/x.json"
        )
        == 0
    )
    assert _tier("json/Mishnah/Seder Moed/Mishnah Shabbat/Hebrew/x.json") == 0
    assert _tier("json/Tanakh/Torah/Genesis/Hebrew/x.json") == 1
    assert _tier("json/Halakhah/Mishneh Torah/Mishneh Torah, Foo/Hebrew/x.json") == 1
    assert _tier("json/Halakhah/Tur/Tur, Orach Chayim/Hebrew/x.json") == 1
    assert _tier("json/Halakhah/Shulchan Arukh/Commentary/Mishnah Berurah/Hebrew/x.json") == 1
    assert _tier("json/Midrash/Foo/Foo/Hebrew/x.json") == 2


def test_backoff_is_exponential_and_capped(monkeypatch):
    slept = []
    monkeypatch.setattr(gcs.time, "sleep", slept.append)
    monkeypatch.setattr(gcs.random, "random", lambda: 1.0)
    for a in range(10):
        gcs.backoff_sleep(a)
    assert slept[:4] == [2.0, 4.0, 8.0, 16.0] and max(slept) == 120.0
