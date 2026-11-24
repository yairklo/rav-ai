"""Derive a coarse 'period' (תקופה) for a work from Sefaria categories + era code."""

from __future__ import annotations

ERA_PERIODS = {
    "T": "tannaim",
    "A": "amoraim",
    "GN": "geonim",
    "RI": "rishonim",
    "AH": "acharonim",
    "CO": "contemporary",
}

BASE_CATEGORY_PERIODS = {
    "Tanakh": "tanakh",
    "Mishnah": "mishnah",
    "Talmud": "talmud",
    "Tosefta": "tosefta",
    "Second Temple": "second_temple",
}


def derive_period(categories: list[str], era: str | None, dependence: str | None) -> str:
    """Base texts take their period from category; commentaries/everything else from era."""
    top = categories[0] if categories else ""
    is_commentary = dependence in {"Commentary", "Targum"} or "Commentary" in categories
    if not is_commentary and top in BASE_CATEGORY_PERIODS:
        return BASE_CATEGORY_PERIODS[top]
    if era in ERA_PERIODS:
        return ERA_PERIODS[era]
    if top in BASE_CATEGORY_PERIODS:
        return BASE_CATEGORY_PERIODS[top]
    return "unknown"
