"""Context domain of a work (for ranking acronym expansions) and period ranks."""

from __future__ import annotations


def domain_for(category: str | None, category_path: list[str] | None = None) -> str:
    path = " ".join([category or "", *(category_path or [])]).lower()
    if "responsa" in path:
        return "responsa"
    if "halakhah" in path or "shulchan" in path or "rambam" in path or "mishneh torah" in path:
        return "halakha"
    if "talmud" in path or "mishnah" in path or "bavli" in path:
        return "talmud"
    return "general"
