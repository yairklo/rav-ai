"""Read side: text by canonical ref, license-gated, plus the layered neighbourhood of a ref."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

# Higher = later layer. Used to split linked refs into "below" (earlier) and "above" (later).
PERIOD_RANK = {
    "tanakh": 0,
    "second_temple": 1,
    "tannaim": 2,
    "mishnah": 2,
    "tosefta": 2,
    "amoraim": 3,
    "talmud": 3,
    "geonim": 4,
    "rishonim": 5,
    "acharonim": 6,
    "contemporary": 7,
    "unknown": 5,
}


def deployment_mode() -> str:
    return os.environ.get("RAV_DEPLOYMENT_MODE", "personal")  # personal | public


def may_display(can_index: bool, can_display_public: bool, mode: str | None = None) -> bool:
    """Display gate. public -> needs can_display_public; personal -> needs can_index."""
    return can_display_public if (mode or deployment_mode()) == "public" else can_index


@dataclass
class TextRow:
    ref: str
    work: str
    period: str | None
    language: str
    version_title: str
    license: str | None
    can_index: bool
    can_display_public: bool
    can_train: bool
    text_raw: str
    is_primary: bool

    def shown(self, mode: str | None = None) -> str:
        if may_display(self.can_index, self.can_display_public, mode):
            return self.text_raw
        return f"[withheld: license '{self.license or 'none'}' does not permit display]"


def get_texts(session: Session, ref: str) -> list[TextRow]:
    rows = session.execute(
        text("""
        SELECT s.ref, w.title, w.period, v.language, v.version_title, v.license, v.can_index,
               v.can_display_public, v.can_train, t.text_raw, v.is_primary
        FROM segments s JOIN segment_text t ON t.segment_id = s.id
        JOIN versions v ON v.id = t.version_id JOIN works w ON w.id = s.work_id
        WHERE s.ref = :ref
        ORDER BY v.is_primary DESC, v.priority DESC NULLS LAST, v.version_title"""),
        {"ref": ref},
    ).all()
    return [TextRow(*r) for r in rows]


@dataclass
class Neighbor:
    ref: str
    work: str
    period: str
    edge_type: str
    origin: str
    snippet: str


def neighbors(session: Session, ref: str, limit_per_group: int = 10_000):
    """Linked segments split into layers above/below/same, each grouped by type and period."""
    me = session.execute(
        text("""SELECT s.id, w.period FROM segments s JOIN works w ON w.id=s.work_id
                                 WHERE s.ref=:r"""),
        {"r": ref},
    ).first()
    if me is None:
        return None
    sid, my_period = me
    my_rank = PERIOD_RANK.get(my_period or "unknown", 5)
    rows = session.execute(
        text("""
        SELECT o.ref, w.title, COALESCE(w.period,'unknown'), e.edge_type, e.origin,
               (SELECT t.text_a FROM segment_text t JOIN versions v ON v.id=t.version_id
                 WHERE t.segment_id=o.id ORDER BY v.is_primary DESC, v.priority DESC NULLS LAST
                 LIMIT 1),
               (SELECT v.can_index OR v.can_display_public FROM segment_text t
                  JOIN versions v ON v.id=t.version_id WHERE t.segment_id=o.id
                  ORDER BY v.is_primary DESC, v.priority DESC NULLS LAST LIMIT 1),
               (SELECT v.can_display_public FROM segment_text t
                  JOIN versions v ON v.id=t.version_id WHERE t.segment_id=o.id
                  ORDER BY v.is_primary DESC, v.priority DESC NULLS LAST LIMIT 1),
               o.sort_key, o.work_id
        FROM (SELECT segment_b_id AS other, edge_type, origin FROM edges WHERE segment_a_id=:i
              UNION ALL
              SELECT segment_a_id, edge_type, origin FROM edges WHERE segment_b_id=:i) e
        JOIN segments o ON o.id = e.other JOIN works w ON w.id = o.work_id
        ORDER BY o.work_id, o.node_index, o.sort_key, e.edge_type, o.ref
        LIMIT :lim"""),
        {"i": sid, "lim": limit_per_group},
    ).all()
    out: dict[str, dict[str, dict[str, list[Neighbor]]]] = {
        "above": defaultdict(lambda: defaultdict(list)),
        "below": defaultdict(lambda: defaultdict(list)),
        "same": defaultdict(lambda: defaultdict(list)),
    }
    for oref, work, period, etype, origin, snippet, idx_ok, pub_ok, _sk, _wid in rows:
        rank = PERIOD_RANK.get(period, 5)
        side = "above" if rank > my_rank else "below" if rank < my_rank else "same"
        if snippet is None:
            shown = ""
        elif may_display(bool(idx_ok), bool(pub_ok)):
            shown = snippet
        else:
            shown = "[withheld: license]"
        out[side][etype or "(untyped)"][period].append(
            Neighbor(oref, work, period, etype, origin, shown)
        )
    return out, my_period
