"""Layer order for ingestion: foundations first, then each later layer (all books are ingested)."""

from __future__ import annotations

import re

LAYER_NAMES = [
    "mishnah+bavli", "rashi", "tosafot", "rambam", "ramban", "rashba", "ritva", "ran", "rosh",
    "tur+beit_yosef", "shulchan_arukh+rema+nosei_keilim", "rest",
]  # fmt: skip

_PATTERNS: list[tuple[int, str]] = [
    (1, r"^Rashi on "),
    (2, r"^Tosafot on "),
    (3, r"^Mishneh Torah"),
    (4, r"^(Chiddushei HaRamban|Ramban) on "),
    (5, r"^(Chiddushei HaRashba|Chidushei HaRashba|Rashba) on "),
    (6, r"^(Chiddushei HaRitva|Chidushei HaRitva|Ritva) on "),
    (7, r"^(Chiddushei HaRan|Chidushei HaRan|Ran) on "),
    (8, r"^Rosh on "),
    (9, r"^(Tur|Beit Yosef|Darchei Moshe|Bayit Chadash|Bach)[, ]"),
    (10, r"^(Shulchan Arukh|Mishnah Berurah|Magen Avraham|Turei Zahav|Ba'er Hetev|"
         r"Be'er HaGolah|Rema|Sha'arei Teshuvah|Biur Halakha|Eshel Avraham|Ateret Zekenim|"
         r"Pri Megadim|Peri Megadim|Mishbetzot Zahav|Machatzit HaShekel|Birkei Yosef|"
         r"Rabbi Akiva Eiger on Shulchan Arukh|.* on Shulchan Arukh)"),
]  # fmt: skip


def layer_of(title: str, category_path: tuple[str, ...] | list[str]) -> int:
    cats = list(category_path)
    if "Mishnah" in cats[:1] and " on " not in title:
        return 0
    if "Bavli" in cats and " on " not in title and cats[:1] == ["Talmud"]:
        return 0
    for tier, rx in _PATTERNS:
        if re.match(rx, title):
            return tier
    return len(LAYER_NAMES) - 1
