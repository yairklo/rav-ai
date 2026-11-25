"""machloket = a full item (not a view): created automatically as a candidate for every issue
with two or more positions, and verified like any other item (K-check: every position in
``sides`` belongs to the machloket's issue)."""

from __future__ import annotations

from rav_ai.db.models import KlItem, KlLink
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from .checks import CheckContext, run_checks
from .queries import positions_by_issue
from .schemas import SCHEMA_VERSION, Machloket
from .store import add_link, sha, write_item

Issue_ = aliased(KlItem)
_GEMARA_TIERS = {"canonical", None}


def _positions(s: Session, issue: KlItem) -> list[KlItem]:
    """Current, non-rejected positions of the issue (any version of it); reading_only ones
    (a lens's reading, not a position) are not part of a machloket."""
    return [p for p in positions_by_issue(s, issue.item_key) if p.status != "rejected"]


def _kind(positions: list[KlItem]) -> str:
    tiers = {(p.body.get("holder") or {}).get("tier") for p in positions}
    if tiers <= _GEMARA_TIERS:
        return "gemara"
    return "rishonim" if tiers <= {"rishonim", "canonical", None} else "poskim"


def build_machloket_candidate(
    s: Session, issue: KlItem, job_id: int | None = None
) -> KlItem | None:
    pos = _positions(s, issue)
    sides: dict[str, list[str]] = {}
    for p in pos:
        sides.setdefault(p.body.get("side_label") or p.item_key, []).append(p.item_key)
    if len(sides) < 2:
        return None
    sugyot = list(
        dict.fromkeys(
            s.scalars(
                select(KlItem.item_key)
                .join(KlLink, KlLink.to_item_id == KlItem.id)
                .join(Issue_, Issue_.id == KlLink.from_item_id)
                .where(Issue_.item_key == issue.item_key, KlLink.rel == "raised_in")
                .order_by(KlItem.id)
            )
        )
    )
    body = Machloket(
        issue_id=issue.item_key,
        sides=list(sides.values()),
        sugya_ids=sugyot,
        machloket_kind=_kind(pos),
    )
    ctx = CheckContext(
        kind="machloket", model=body, position_issue={p.item_key: issue.item_key for p in pos}
    )
    report = run_checks(ctx)
    item = write_item(
        s,
        kind="machloket",
        item_key=f"machloket:{issue.item_key}",
        scope_key=issue.scope_key,
        body=body.model_dump(mode="json"),
        input_hash=sha("machloket", SCHEMA_VERSION, [(p.item_key, p.version) for p in pos]),
        job_id=job_id,
        model=None,
        prompt_version="auto",
        prompt_hash=None,
        schema_version=SCHEMA_VERSION,
        status="generated" if report.ok else "rejected",
        work_id=issue.work_id,
        title_he=issue.title_he,
        confidence=1.0 if report.ok else 0.0,
        check_report=report.to_json(),
        rejected_by=None if report.ok else "auto",
        can_display_public=all(p.can_display_public for p in pos) and issue.can_display_public,
        can_train=all(p.can_train for p in pos) and issue.can_train,
        source_version_ids=sorted({v for p in pos for v in (p.source_version_ids or [])}),
        search_text_he=issue.title_he,
    )
    add_link(s, item.id, issue.id, "about_issue")
    for p in pos:
        add_link(s, item.id, p.id, "discusses")
    for key in sugyot:
        sg = s.scalars(select(KlItem).where(KlItem.item_key == key, KlItem.is_current)).first()
        if sg:
            add_link(s, item.id, sg.id, "raised_in")
    return item
