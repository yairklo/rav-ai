"""Interpretation of object names in the Sefaria export bucket.

Observed layout (2026-11):
  json/<cat>[/<subcat>...]/<Book Title>/<Language>/<versionTitle>.json   (+ merged.json)
  schemas/<Book_Title_with_underscores>.json
  links/links<N>.csv
  table_of_contents.json, last_export.txt
"""

from __future__ import annotations

from dataclasses import dataclass

LANG_DIRS = {"Hebrew", "English"}


@dataclass(frozen=True)
class VersionObject:
    name: str  # full object name
    size: int
    category_path: tuple[str, ...]
    book: str
    language_dir: str
    version_file: str  # e.g. "Torat Emet 363.json"

    @property
    def is_merged(self) -> bool:
        return self.version_file == "merged.json"


def parse_version_object(name: str, size: int = 0) -> VersionObject | None:
    parts = name.split("/")
    if len(parts) < 5 or parts[0] != "json" or not parts[-1].endswith(".json"):
        return None
    lang_dir = parts[-2]
    if lang_dir not in LANG_DIRS:
        return None
    return VersionObject(
        name=name,
        size=size,
        category_path=tuple(parts[1:-3]),
        book=parts[-3],
        language_dir=lang_dir,
        version_file=parts[-1],
    )


def schema_object_name(book: str) -> str:
    return f"schemas/{book.replace(' ', '_')}.json"


# Download priority: 0 = Chanukah pilot core, 1 = other pilot sources, 2 = everything else.
_PILOT_TIER1_TITLES = {
    "Tur, Orach Chayim",
    "Beit Yosef, Orach Chayim",
    "Shulchan Arukh, Orach Chayim",
    "Mishnah Berurah",
}


def priority_tier(v: VersionObject) -> int:
    cats, book = v.category_path, v.book
    if book == "Shabbat" and "Bavli" in cats:
        return 0
    if book == "Mishnah Shabbat":
        return 0
    if book.endswith(" on Shabbat") and "Talmud" in cats:  # Rashi, Tosafot, Ramban, Rashba, ...
        return 0
    if "Tanakh" in cats[:1] or "Mishnah" in cats[:1]:
        return 1
    if book.startswith("Mishneh Torah") or book in _PILOT_TIER1_TITLES:
        return 1
    if book.endswith(" on Shulchan Arukh, Orach Chayim") or "Peri Megadim on Orach Chayim" in book:
        return 1
    return 2
