"""Canonical schema (PostgreSQL + pgvector extension).

Layering: authors -> works -> versions -> segments -> segment_text, plus edges between segments.
License flags live on `versions` and are *enforced at display/export time*, never at ingest.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite in unit tests).
JsonType = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Snapshot(Base):
    """One import run of an export (dated). Diffs are computed against the previous snapshot."""

    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)  # "sefaria"
    export_label: Mapped[str] = mapped_column(String(64))  # last_export.txt value
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|complete
    stats: Mapped[dict | None] = mapped_column(JSONB)


class SnapshotChange(Base):
    """Per-segment-text diff vs. the previous state (hash based)."""

    __tablename__ = "snapshot_changes"
    __table_args__ = (Index("ix_snapshot_changes_snap", "snapshot_id", "change"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"))
    version_id: Mapped[int] = mapped_column(ForeignKey("versions.id"))
    ref: Mapped[str] = mapped_column(String(512))
    change: Mapped[str] = mapped_column(String(8))  # added|changed|removed
    old_hash: Mapped[str | None] = mapped_column(String(64))
    new_hash: Mapped[str | None] = mapped_column(String(64))


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(128), unique=True)  # Sefaria slug or derived
    name: Mapped[str] = mapped_column(String(256))
    he_name: Mapped[str | None] = mapped_column(String(256))
    period: Mapped[str | None] = mapped_column(String(32), index=True)  # tannaim, rishonim, ...
    birth_year: Mapped[int | None] = mapped_column(Integer)
    death_year: Mapped[int | None] = mapped_column(Integer)
    years_note: Mapped[str | None] = mapped_column(String(256))


class Work(Base):
    __tablename__ = "works"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(256), unique=True)  # Sefaria English title
    he_title: Mapped[str | None] = mapped_column(String(256))
    author_id: Mapped[int | None] = mapped_column(ForeignKey("authors.id"), index=True)
    category: Mapped[str] = mapped_column(String(128), index=True)  # top-level category
    category_path: Mapped[list] = mapped_column(JSONB)
    era_code: Mapped[str | None] = mapped_column(String(16))
    period: Mapped[str | None] = mapped_column(String(32), index=True)
    comp_year_start: Mapped[int | None] = mapped_column(Integer)
    comp_year_end: Mapped[int | None] = mapped_column(Integer)
    dependence: Mapped[str | None] = mapped_column(String(32))  # Commentary|Targum|...
    base_work_id: Mapped[int | None] = mapped_column(ForeignKey("works.id"), index=True)
    base_work_titles: Mapped[list | None] = mapped_column(
        JSONB(none_as_null=True)
    )  # all declared base texts
    section_names: Mapped[list] = mapped_column(JSONB)
    schema: Mapped[dict] = mapped_column(JSONB)  # Sefaria schema, verbatim


class Version(Base):
    __tablename__ = "versions"
    __table_args__ = (UniqueConstraint("work_id", "source", "language", "version_title"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("works.id"), index=True)
    source: Mapped[str] = mapped_column(String(32))  # sefaria
    language: Mapped[str] = mapped_column(String(8))
    version_title: Mapped[str] = mapped_column(String(512))
    source_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(64), index=True)  # as exported
    can_index: Mapped[bool] = mapped_column(Boolean)
    can_display_public: Mapped[bool] = mapped_column(Boolean)
    can_train: Mapped[bool] = mapped_column(Boolean)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(String(32))
    word_count: Mapped[int] = mapped_column(BigInteger, default=0)
    file_hash: Mapped[str | None] = mapped_column(String(64))  # sha256 of the export file
    first_snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"))
    last_snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Segment(Base):
    """Smallest addressable unit; identity = canonical Sefaria ref."""

    __tablename__ = "segments"
    __table_args__ = (Index("ix_segments_work_sort", "work_id", "node_index", "sort_key"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ref: Mapped[str] = mapped_column(String(512), unique=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("works.id"))
    node_index: Mapped[int] = mapped_column(SmallInteger, default=0)
    node_path: Mapped[list] = mapped_column(JSONB)  # complex-schema node titles
    section_path: Mapped[list] = mapped_column(JSONB)  # address labels, e.g. ["21b", "5"]
    indices: Mapped[list] = mapped_column(JSONB)  # 0-based array positions
    section_names: Mapped[list] = mapped_column(JSONB)
    sort_key: Mapped[str] = mapped_column(String(128))  # zero-padded indices: document order
    parent_ref: Mapped[str | None] = mapped_column(String(512))  # e.g. "Shabbat 21b"


class SegmentText(Base):
    """Text of one segment in one version: raw (verbatim) + two derived normalized forms."""

    __tablename__ = "segment_text"
    __table_args__ = (
        UniqueConstraint("segment_id", "version_id"),
        Index("ix_segment_text_version", "version_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"))
    version_id: Mapped[int] = mapped_column(ForeignKey("versions.id"))
    text_raw: Mapped[str] = mapped_column(Text)  # byte-exact; never rewritten
    raw_hash: Mapped[str] = mapped_column(String(64), index=True)  # sha256(text_raw utf-8)
    text_a: Mapped[str] = mapped_column(Text)  # no niqqud/te'amim/markup, unified glyphs
    text_b: Mapped[str] = mapped_column(Text)  # A + no punctuation, search-oriented
    offsets_a: Mapped[bytes] = mapped_column(LargeBinary)  # see rav_ai.normalize.OffsetMap
    offsets_b: Mapped[bytes] = mapped_column(LargeBinary)
    word_count: Mapped[int] = mapped_column(Integer, default=0)


class Edge(Base):
    """Relation between two segments. Stored once (a,b as exported); query either side."""

    __tablename__ = "edges"
    __table_args__ = (
        Index("ix_edges_a", "segment_a_id"),
        Index("ix_edges_b", "segment_b_id"),
        Index("ix_edges_type", "edge_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_a_id: Mapped[int] = mapped_column(ForeignKey("segments.id"))
    segment_b_id: Mapped[int] = mapped_column(ForeignKey("segments.id"))
    a_start: Mapped[int | None] = mapped_column(Integer)  # char range in raw text, if known
    a_end: Mapped[int | None] = mapped_column(Integer)
    b_start: Mapped[int | None] = mapped_column(Integer)
    b_end: Mapped[int | None] = mapped_column(Integer)
    edge_type: Mapped[str] = mapped_column(String(64))  # Sefaria type verbatim ('' = untyped)
    origin: Mapped[str] = mapped_column(String(16))  # sefaria | heuristic | llm
    confidence: Mapped[float | None] = mapped_column(Float)
    a_span: Mapped[int] = mapped_column(Integer, default=1)  # segments in original endpoint A
    b_span: Mapped[int] = mapped_column(Integer, default=1)
    range_collapsed: Mapped[bool] = mapped_column(Boolean, default=False)
    citation_a: Mapped[str | None] = mapped_column(String(512))
    citation_b: Mapped[str | None] = mapped_column(String(512))
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"))


class UnresolvedLink(Base):
    """Links that could not be resolved to existing segments (kept for the gap report)."""

    __tablename__ = "unresolved_links"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"))
    citation_a: Mapped[str] = mapped_column(String(512))
    citation_b: Mapped[str] = mapped_column(String(512))
    edge_type: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(48), index=True)
    side: Mapped[str] = mapped_column(String(4))  # a|b|both


# ---- tables created now, populated by later stages -------------------------------------


class QuoteSpan(Base):
    """A span in a later text that quotes a span of an earlier text."""

    __tablename__ = "quote_spans"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), index=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("versions.id"))
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    source_segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), index=True)
    source_start: Mapped[int | None] = mapped_column(Integer)
    source_end: Mapped[int | None] = mapped_column(Integer)
    kind: Mapped[str | None] = mapped_column(String(24))  # verbatim|abridged|paraphrase|reference
    confidence: Mapped[float | None] = mapped_column(Float)
    origin: Mapped[str] = mapped_column(String(16))
    raw_hash: Mapped[str | None] = mapped_column(String(64))  # input hash for staleness


class Annotation(Base):
    """Span-level annotation, e.g. abbreviation-expansion candidates (never rewrites text)."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), index=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("versions.id"))
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(24), index=True)  # abbrev|person|work_mention|...
    candidates: Mapped[list | None] = mapped_column(JSONB)
    chosen: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    origin: Mapped[str] = mapped_column(String(16))
    raw_hash: Mapped[str | None] = mapped_column(String(64))


class Card(Base):
    """AI analysis card of a segment; may rest on other cards (dependency edges)."""

    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), index=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("versions.id"))
    card: Mapped[dict] = mapped_column(JSONB)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    input_hash: Mapped[str | None] = mapped_column(String(64))
    stale: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CardDependency(Base):
    """`card_id` rests on `depends_on_card_id` (invalidate downstream when upstream is stale)."""

    __tablename__ = "card_dependencies"

    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"), primary_key=True)
    depends_on_card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"), primary_key=True)


class Run(Base):
    """One system run (a question answered by a model), kept for evals and future training data.

    Owned by the evals package logically, but defined on the shared Base so that Alembic
    migrates it together with the corpus.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    # Input
    question_text: Mapped[str] = mapped_column(Text)
    question_id: Mapped[str | None] = mapped_column(String(128), index=True)  # golden id, if any
    golden_version: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128), index=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    prompt_hash: Mapped[str | None] = mapped_column(String(64), index=True)  # sha256 of the prompt

    # What happened
    messages: Mapped[list[Any]] = mapped_column(
        JsonType, default=list, server_default=text("'[]'")
    )  # full transcript (harness runs)
    retrieved_sources: Mapped[list[Any]] = mapped_column(JsonType, default=list)  # refs (+scores)
    tool_calls: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    answer: Mapped[dict[str, Any] | None] = mapped_column(JsonType)  # Answer.model_dump()
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    steps: Mapped[int] = mapped_column(Integer, default=0, server_default="0")  # LLM calls
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float | None] = mapped_column(Float)  # None = pricing unknown

    # Verification
    verification: Mapped[dict[str, Any] | None] = mapped_column(JsonType)  # VerificationReport
    verification_passed: Mapped[bool | None] = mapped_column(Boolean, index=True)

    # Human/LLM review, filled later; gate for using the row as training data
    review_label: Mapped[str | None] = mapped_column(String(32))
    review_notes: Mapped[str | None] = mapped_column(Text)
