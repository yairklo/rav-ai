"""kl_sources registry + the personal-mode lens records. Ingestion of the lenses is KL-5; here
only the registry rows exist (placeholder ``works`` rows with no versions, so FKs hold)."""

from __future__ import annotations

from typing import Literal

from pydantic import TypeAdapter
from rav_ai.db.models import KlSource, Work
from sqlalchemy import select
from sqlalchemy.orm import Session

AuthorityArea = Literal[
    "talmud_commentary",
    "halakhic_ruling",
    "codification",
    "mitzvot_rationale",
    "mitzvot_enumeration",
    "responsa",
    "minhag_documentation",
    "girsa",
    "lamdanut",
    "derush_musar",
]
AUTHORITY_AREAS = AuthorityArea.__args__  # type: ignore[attr-defined]
_AREAS = TypeAdapter(list[AuthorityArea] | None)


def validate_authority_areas(areas: list[str] | None) -> list[str] | None:
    """The closed list lives here, not in the DB. None = not classified (no effect)."""
    return _AREAS.validate_python(areas)


PERSONAL = "PERSONAL_ONLY_RESTRICTED"
NOTE_PERSONAL = (
    "pending written permission: can_index only in personal mode; can_display_public=false and "
    "can_train=false; every item derived from it inherits the restriction"
)

# (title, he_title, tier, role, license_tier, note)
LENS_RECORDS = [
    ("Asif", "אסיף", "sugya_compendium", "lens", PERSONAL, NOTE_PERSONAL),
    ("Minchat Asher", "מנחת אשר", "lamdanut", "lens", PERSONAL, NOTE_PERSONAL),
    (
        "Rav Asher Weiss - Shiurim (written)",
        "הרב אשר וייס - שיעורים כתובים",
        "lamdanut",
        "lens",
        PERSONAL,
        NOTE_PERSONAL + "; also a methodological lens (method_principle)",
    ),
    (
        "Rav Asher Weiss - Shiurim (transcripts)",
        "הרב אשר וייס - תמלולי שיעורים",
        "lamdanut",
        "lens",
        PERSONAL,
        NOTE_PERSONAL + "; transcripts of oral shiurim",
    ),
    (
        "Virtual Beit Midrash (VBM)",
        "בית המדרש הוירטואלי",
        "modern_secondary",
        "reference_only",
        "NON_COMMERCIAL_ONLY",
        "reference only for now",
    ),
    (
        "HebrewBooks",
        "היברו בוקס",
        "modern_secondary",
        "reference_only",
        "NON_COMMERCIAL_ONLY",
        "reference only for now",
    ),
]


def _ensure_work(s: Session, title: str, he_title: str) -> Work:
    w = s.scalars(select(Work).where(Work.title == title)).first()
    if w is None:
        w = Work(
            title=title,
            he_title=he_title,
            category="Personal",
            category_path=["Personal"],
            section_names=[],
            schema={},
        )
        s.add(w)
        s.flush()
    return w


def seed_lens_records(s: Session, added_by: str = "kl-1") -> int:
    n = 0
    for title, he, tier, role, lic, note in LENS_RECORDS:
        w = _ensure_work(s, title, he)
        if s.get(KlSource, w.id) is None:
            s.add(
                KlSource(
                    work_id=w.id,
                    tier=tier,
                    role=role,
                    origin="personal" if lic == PERSONAL else "reference",
                    license_tier=lic,
                    license_basis=note,
                    weight_tier=None,
                    added_by=added_by,
                    note=note,
                )
            )
            n += 1
    s.flush()
    return n


def register_work(
    s: Session,
    title: str,
    *,
    tier: str,
    role: str,
    license_tier: str,  # no default: licensing is fail-closed
    weight_tier: str | None = None,
    authority_areas: list[str] | None = None,
    note: str | None = None,
    added_by: str = "cli",
) -> KlSource:
    areas = validate_authority_areas(authority_areas)
    w = s.scalars(select(Work).where(Work.title == title)).one()
    src = s.get(KlSource, w.id)
    if src is None:
        src = KlSource(work_id=w.id, tier=tier, role=role, added_by=added_by)
        s.add(src)
    src.tier, src.role, src.license_tier = tier, role, license_tier
    src.weight_tier, src.note, src.authority_areas = weight_tier, note, areas
    s.flush()
    return src
