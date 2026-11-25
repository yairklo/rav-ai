"""Direct queries along the three axes: sugya (text + flow), issue (positions), machloket.

Links point at a specific VERSION of an item (kl_links.to_item_id), but a new version of the
target (a re-run, a human edit) must not lose them: every query resolves the key to ALL versions
of the item_key, follows the links, and returns the CURRENT version of what it finds.
Positions that are only a lens's reading (``reading_only``) are not positions of the issue and
are left out of every query here (KL-4 turns them into reads_as / aligns_with_side relations).
"""

from __future__ import annotations

from rav_ai.db.models import KlItem, KlLink
from sqlalchemy import select
from sqlalchemy.orm import Session


def _ids(s: Session, key: str, kind: str) -> list[int]:
    """Ids of every version of the item (a current one must exist, else the key is unknown)."""
    rows = s.execute(
        select(KlItem.id, KlItem.is_current).where(KlItem.item_key == key, KlItem.kind == kind)
    ).all()
    return [i for i, _ in rows] if any(cur for _, cur in rows) else []


def _current_of(s: Session, item_ids: list[int], kind: str, *, direct_only: bool = True):
    """Current versions of the item_keys that ``item_ids`` belong to (ordered by id)."""
    if not item_ids:
        return []
    keys = select(KlItem.item_key).where(KlItem.id.in_(item_ids), KlItem.kind == kind)
    q = select(KlItem).where(KlItem.item_key.in_(keys), KlItem.kind == kind, KlItem.is_current)
    if direct_only:
        q = q.where(KlItem.body["reading_only"].as_boolean().is_not(True))
    return list(s.scalars(q.order_by(KlItem.id)))


def _sources(s: Session, rel: str, target_key: str, target_kind: str) -> list[int]:
    """Item ids that link TO any version of the target with ``rel``."""
    ids = _ids(s, target_key, target_kind)
    if not ids:
        return []
    return list(
        s.scalars(select(KlLink.from_item_id).where(KlLink.rel == rel, KlLink.to_item_id.in_(ids)))
    )


def _targets(s: Session, rel: str, source_key: str, source_kind: str) -> list[int]:
    """Item ids that any version of the source links TO with ``rel``."""
    ids = _ids(s, source_key, source_kind)
    if not ids:
        return []
    return list(
        s.scalars(select(KlLink.to_item_id).where(KlLink.rel == rel, KlLink.from_item_id.in_(ids)))
    )


def issues_by_sugya(s: Session, sugya_key: str) -> list[KlItem]:
    return _current_of(s, _sources(s, "raised_in", sugya_key, "sugya"), "issue")


def machlokot_by_issue(s: Session, issue_key: str) -> list[KlItem]:
    return _current_of(s, _sources(s, "about_issue", issue_key, "issue"), "machloket")


def sugyot_by_machloket(s: Session, machloket_key: str) -> list[KlItem]:
    return _current_of(s, _targets(s, "raised_in", machloket_key, "machloket"), "sugya")


def positions_by_issue(s: Session, issue_key: str) -> list[KlItem]:
    """Current positions of an issue, without reading_only ones."""
    return _current_of(s, _sources(s, "position_on", issue_key, "issue"), "position")
