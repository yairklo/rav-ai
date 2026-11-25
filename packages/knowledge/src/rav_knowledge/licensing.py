"""Fail-closed licensing for the knowledge layer.

An item's permission is the AND of the permissions of every version it was built from.
"""

from __future__ import annotations

from dataclasses import dataclass

from rav_ai.db.models import KlSource, Version

from .config import KLConfig

PERSONAL = "PERSONAL_ONLY_RESTRICTED"
NONCOMMERCIAL = "NON_COMMERCIAL_ONLY"


def admit(source: KlSource | None, version: Version, cfg: KLConfig) -> tuple[bool, str]:
    """May this version be used as *input* for generation? (reason when not)"""
    if source is None:
        return False, "unregistered_source"
    if source.role == "reference_only":
        return False, "reference_only"
    if not version.can_index:
        return False, "cannot_index"
    if source.license_tier == PERSONAL and not (cfg.mode == "personal" and cfg.allow_personal_only):
        return False, "personal_only_restricted"
    return True, "ok"


def admit_aid(source: KlSource | None, version: Version, cfg: KLConfig) -> tuple[bool, str]:
    """Reading aids (Rashi, Steinsaltz): never evidence, so an unregistered work is fine, but
    the version must be indexable and a personal-only source follows the same gate."""
    if not version.can_index:
        return False, "cannot_index"
    if (
        source is not None
        and source.license_tier == PERSONAL
        and not (cfg.mode == "personal" and cfg.allow_personal_only)
    ):
        return False, "personal_only_restricted"
    return True, "ok"


@dataclass(frozen=True)
class Permission:
    can_display_public: bool
    can_train: bool


def item_permission(
    used: list[tuple[Version, KlSource | None]], *, steinsaltz_aid: bool = False
) -> Permission:
    """AND over all (version, source) pairs; fail closed on an empty list."""
    if not used:
        return Permission(False, False)
    display = all(v.can_display_public for v, _ in used)
    train = all(v.can_train for v, _ in used)
    for _, s in used:
        if s is None:
            continue
        if s.license_tier == PERSONAL:
            display = train = False
        elif s.license_tier == NONCOMMERCIAL:
            train = False
    if steinsaltz_aid:
        display = False
    return Permission(display, train)
