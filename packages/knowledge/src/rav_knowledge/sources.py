"""kl_sources registry + the personal-mode lens records. Ingestion of the lenses is KL-5; here
only the registry rows exist (placeholder ``works`` rows with no versions, so FKs hold)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter
from rav_ai.config import POLICY_PATH, ROOT, LicensePolicy
from rav_ai.db.models import KL_TIERS, KlSource, Version, Work
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


# ---- reproducible registry seed (config/kl_sources_seed.toml) --------------------------------

SEED_PATH = ROOT / "config" / "kl_sources_seed.toml"
_LICENSE_RANK = {"PUBLIC_COMMERCIAL": 0, "NON_COMMERCIAL_ONLY": 1, "PERSONAL_ONLY_RESTRICTED": 2}
_WEIGHT_TIERS = ("core", "major", "extended", "long_tail")
_ROLES = ("primary", "lens", "witness", "reference_only")


def load_seed(path: Path | None = None) -> dict:
    """Parse and validate the seed file (no DB). Raises ValueError on a malformed entry."""
    raw = tomllib.loads((path or SEED_PATH).read_text(encoding="utf-8"))
    entries, skipped = raw.get("source", []), raw.get("not_registered", [])
    titles = [e.get("title") for e in entries] + [e.get("title") for e in skipped]
    if len(titles) != len(set(titles)):
        raise ValueError("seed: a title appears twice")
    for e in entries:
        t = e.get("title")
        if e.get("tier") not in KL_TIERS:
            raise ValueError(f"seed {t!r}: tier {e.get('tier')!r} not in {KL_TIERS}")
        if e.get("role") not in _ROLES:
            raise ValueError(f"seed {t!r}: role {e.get('role')!r}")
        if e.get("license_tier") not in _LICENSE_RANK:
            raise ValueError(f"seed {t!r}: license_tier is required ({list(_LICENSE_RANK)})")
        if e.get("weight_tier") not in (None, *_WEIGHT_TIERS):
            raise ValueError(f"seed {t!r}: weight_tier {e.get('weight_tier')!r}")
        validate_authority_areas(e.get("authority_areas"))
    for e in skipped:
        if not e.get("reason"):
            raise ValueError(f"seed not_registered {e.get('title')!r}: a reason is required")
    return {"source": entries, "not_registered": skipped}


def personal_only_licenses(policy_path: Path = POLICY_PATH) -> tuple[LicensePolicy, set[str]]:
    """The license-policy keys marked ``personal_only = true`` (an optional attribute read only by
    the knowledge layer; it does not change the three flags ingest stores on a version)."""
    policy = LicensePolicy.load(policy_path)
    raw = tomllib.loads(policy_path.read_text(encoding="utf-8")).get("licenses", {})
    keys = {
        policy.match(k)
        for name, v in raw.items()
        if v.get("personal_only")
        for k in (name, *v.get("aliases", []))
    }
    return policy, keys - {"unknown"}


def version_license_tier(v: Version, policy: LicensePolicy, personal: set[str]) -> str | None:
    """None: not indexable (unknown / missing license) -> the version is excluded anyway."""
    if not v.can_index:
        return None
    if policy.match(v.license) in personal:
        return PERSONAL
    return "PUBLIC_COMMERCIAL" if v.can_display_public else "NON_COMMERCIAL_ONLY"


def required_license_tier(
    s: Session, work: Work, policy_path: Path = POLICY_PATH
) -> tuple[str | None, list[str]]:
    """The loosest license_tier the work's indexable versions allow (= its most restrictive one),
    from Sefaria's per-version declaration via the license policy: a personal-only license ->
    PERSONAL_ONLY_RESTRICTED, else the stored flags. None: no indexable version (unknown /
    missing license everywhere) -> must not be registered."""
    policy, personal = personal_only_licenses(policy_path)
    need, basis = None, []
    for v in s.scalars(select(Version).where(Version.work_id == work.id).order_by(Version.id)):
        basis.append(f"{v.version_title}: {v.license or 'none'}")
        t = version_license_tier(v, policy, personal)
        if t is None:
            continue  # unknown / missing license: excluded per version anyway (fail-closed)
        if need is None or _LICENSE_RANK[t] > _LICENSE_RANK[need]:
            need = t
    return need, basis


def seed_sources(
    s: Session,
    path: Path | None = None,
    *,
    dry_run: bool = False,
    added_by: str = "seed",
    policy_path: Path = POLICY_PATH,
) -> dict:
    """Register every seed entry whose work exists and whose declared license_tier is not looser
    than its versions allow. Idempotent; refuses (and reports) everything else."""
    seed = load_seed(path)
    out: dict[str, list] = {"registered": [], "refused": [], "not_registered": []}
    for e in seed["source"]:
        t = e["title"]
        w = s.scalars(select(Work).where(Work.title == t)).first()
        if w is None:
            out["refused"].append({"title": t, "reason": "work not in the corpus"})
            continue
        need, basis = required_license_tier(s, w, policy_path)
        if need is None:
            out["refused"].append({"title": t, "reason": f"no indexable version: {basis}"})
            continue
        if _LICENSE_RANK[e["license_tier"]] < _LICENSE_RANK[need]:
            out["refused"].append(
                {"title": t, "reason": f"declared {e['license_tier']} looser than {need}: {basis}"}
            )
            continue
        row = {k: e.get(k) for k in ("title", "tier", "role", "license_tier", "weight_tier")}
        out["registered"].append(row)
        if dry_run:
            continue
        src = register_work(
            s,
            t,
            tier=e["tier"],
            role=e["role"],
            license_tier=e["license_tier"],
            weight_tier=e.get("weight_tier"),
            authority_areas=e.get("authority_areas"),
            note=f"core-books-draft {e.get('draft', '')}".strip(),
            added_by=added_by,
        )
        src.license_basis = "Sefaria: " + "; ".join(basis)
    for e in seed["not_registered"]:
        out["not_registered"].append({"title": e["title"], "reason": e["reason"]})
    s.flush()
    return out
