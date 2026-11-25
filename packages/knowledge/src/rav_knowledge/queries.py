"""Direct queries along the three axes: sugya (text + flow), issue (positions), machloket."""

from __future__ import annotations

from rav_ai.db.models import KlItem, KlLink
from sqlalchemy import select
from sqlalchemy.orm import Session


def _current(s: Session, key: str, kind: str) -> KlItem | None:
    return s.scalars(
        select(KlItem).where(KlItem.item_key == key, KlItem.kind == kind, KlItem.is_current)
    ).first()


def issues_by_sugya(s: Session, sugya_key: str) -> list[KlItem]:
    sg = _current(s, sugya_key, "sugya")
    if sg is None:
        return []
    return list(
        s.scalars(
            select(KlItem)
            .join(KlLink, KlLink.from_item_id == KlItem.id)
            .where(
                KlLink.rel == "raised_in",
                KlLink.to_item_id == sg.id,
                KlItem.kind == "issue",
                KlItem.is_current,
            )
            .order_by(KlItem.id)
        )
    )


def machlokot_by_issue(s: Session, issue_key: str) -> list[KlItem]:
    iss = _current(s, issue_key, "issue")
    if iss is None:
        return []
    return list(
        s.scalars(
            select(KlItem)
            .join(KlLink, KlLink.from_item_id == KlItem.id)
            .where(
                KlLink.rel == "about_issue",
                KlLink.to_item_id == iss.id,
                KlItem.kind == "machloket",
                KlItem.is_current,
            )
            .order_by(KlItem.id)
        )
    )


def sugyot_by_machloket(s: Session, machloket_key: str) -> list[KlItem]:
    m = _current(s, machloket_key, "machloket")
    if m is None:
        return []
    return list(
        s.scalars(
            select(KlItem)
            .join(KlLink, KlLink.to_item_id == KlItem.id)
            .where(
                KlLink.rel == "raised_in",
                KlLink.from_item_id == m.id,
                KlItem.kind == "sugya",
                KlItem.is_current,
            )
            .order_by(KlItem.id)
        )
    )
