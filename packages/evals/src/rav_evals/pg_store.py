"""Postgres-backed ``SourceStore`` over the corpus tables (segments, segment_text, versions).

V1 needs "does this ref exist", V2 needs the text in the same form the verifier compares:
``segment_text.text_b`` is textnorm level B, ``plene="double"``, ``markup="sefaria"`` - the
exact policy of ``rav_evals.normalization.normalize``.

Only versions with ``can_index`` are used; a ref whose every text is non-indexable is treated as
absent (its text must not back a quote). When a ref has several indexable versions the best one
wins: primary first, then higher priority, then Hebrew, then lowest version id (deterministic).
"""

from __future__ import annotations

from rav_ai.db.models import Segment, SegmentText, Version
from sqlalchemy import case, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class PostgresSourceStore:
    def __init__(self, engine: Engine):
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def get_text(self, ref: str) -> str | None:
        with self._sessions() as s:
            return self._best_text(s, ref)

    @staticmethod
    def _best_text(s: Session, ref: str) -> str | None:
        q = (
            select(SegmentText.text_b)
            .join(Segment, Segment.id == SegmentText.segment_id)
            .join(Version, Version.id == SegmentText.version_id)
            .where(Segment.ref == ref, Version.can_index.is_(True))
            .order_by(
                Version.is_primary.desc(),
                Version.priority.desc().nulls_last(),
                case((Version.language == "he", 0), else_=1),
                Version.id,
            )
            .limit(1)
        )
        return s.scalars(q).first()
