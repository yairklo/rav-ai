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
    CheckConstraint,
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
    book_a: Mapped[str | None] = mapped_column(String(512), index=True)  # Sefaria 'Text 1'
    book_b: Mapped[str | None] = mapped_column(String(512), index=True)  # Sefaria 'Text 2'


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
    # knowledge layer (migration 0005)
    status: Mapped[str] = mapped_column(String(16), default="generated", server_default="generated")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    method: Mapped[str | None] = mapped_column(String(48))  # e.g. s0.acronym, s0b.llm_choice
    method_version: Mapped[str | None] = mapped_column(String(32))
    context_domain: Mapped[str | None] = mapped_column(String(16))  # talmud|halakha|responsa|...
    context: Mapped[dict | None] = mapped_column(JSONB)
    entity_id: Mapped[int | None] = mapped_column(ForeignKey("kl_entities.id"))
    job_id: Mapped[int | None] = mapped_column(ForeignKey("kl_jobs.id"), index=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=text("now()")
    )


Index(
    "uq_annotations_current_span",
    Annotation.segment_id,
    Annotation.version_id,
    Annotation.span_start,
    Annotation.span_end,
    Annotation.kind,
    unique=True,
    postgresql_where=text("is_current"),
)


class Card(Base):
    """DEPRECATED: superseded by the knowledge layer (``kl_items``); kept until migrated.

    AI analysis card of a segment; may rest on other cards (dependency edges).
    """

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
    """DEPRECATED (see ``Card``). `card_id` rests on `depends_on_card_id` (stale propagates)."""

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
    knowledge_items: Mapped[list[Any]] = mapped_column(
        JsonType, default=list, server_default=text("'[]'")
    )  # [(item_key, version)] of knowledge-layer items the run relied on

    # Verification
    verification: Mapped[dict[str, Any] | None] = mapped_column(JsonType)  # VerificationReport
    verification_passed: Mapped[bool | None] = mapped_column(Boolean, index=True)

    # Human/LLM review, filled later; gate for using the row as training data
    review_label: Mapped[str | None] = mapped_column(String(32))
    review_notes: Mapped[str | None] = mapped_column(Text)


# ---- knowledge layer (migration 0005) --------------------------------------------------
# The text is the authority; these tables are a map over it. Every claim hangs on a span of
# raw text (kl_evidence); an item is never a citable source.

KL_KINDS = (
    "sugya",
    "sugya_summary",
    "commentary_note",
    "issue",
    "position",
    "machloket",
    "open_question",
    "chakira",
    "source_claim",
    "girsa_variant",
    "method_principle",
    "topic_card",
    "mishnah_card",
)
KL_ITEM_STATUS = ("generated", "verified", "rejected")
KL_JOB_STATUS = ("running", "done", "failed", "aborted", "promoted")
KL_TASK_STATE = ("pending", "running", "done", "failed", "skipped")
KL_LINK_RELS = (
    "part_of",
    "summarizes",
    "annotates",
    "depends_on",
    "discusses",
    "continues",
    "sub_issue_of",
    "position_on",
    "about_issue",
    "raised_in",
)
KL_RELATIONS = (
    "objects_to",
    "answers",
    "supports",
    "disputes",
    "rules_like",
    "qualifies",
    "reads_as",
    "aligns_with_side",
    "rejects_reading",
    "claims_error_in",
    "depends_on_girsa",
    "parallel_to",
    "same_as",
    "derived_from",
    # din kdima: from (the one that goes first) -> to (the displaced); a ruling, not shaat_hadchak
    "takes_precedence_over",
)
KL_EVIDENCE_ROLES = (
    "states",
    "supports",
    "attributes",
    "adopts",
    "disputes",
    "interprets",
    "proposes",
    "claims_error",
    "witness",
)
KL_TIERS = (
    "canonical",
    "rishonim",
    "codes",
    "nosei_kelim",
    "acharonim",
    "lamdanut",
    "later_responsa",
    "sugya_compendium",
    "modern_secondary",
    "girsa_witness",
)


KL_RECEPTION_CONTEXTS = (
    "iyun",
    "lemaase",
    "snif",
    "shaat_hadchak",
    "bediavad",
    "public_instruction",
)
KL_RECEPTION_CONTEXT_CHECK = (
    "context IN ('iyun', 'lemaase', 'snif', 'shaat_hadchak', 'bediavad', 'public_instruction') "
    "OR context LIKE 'minhag:%'"
)


def _in(col: str, values: tuple[str, ...], *, nullable: bool = False) -> str:
    vals = ", ".join(f"'{v}'" for v in values)
    return f"{col} IN ({vals})" + (f" OR {col} IS NULL" if nullable else "")


class KlEntity(Base):
    """A person/work/place/concept that sources talk about (holder of a view, author, ...)."""

    __tablename__ = "kl_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[str] = mapped_column(String(16), default="person")  # person|work|place|concept
    name_he: Mapped[str | None] = mapped_column(String(256))
    name_en: Mapped[str | None] = mapped_column(String(256))
    author_id: Mapped[int | None] = mapped_column(ForeignKey("authors.id"), index=True)
    work_id: Mapped[int | None] = mapped_column(ForeignKey("works.id"), index=True)
    period: Mapped[str | None] = mapped_column(String(32))
    aliases: Mapped[list | None] = mapped_column(JSONB)
    note: Mapped[str | None] = mapped_column(Text)


class KlWorkAlias(Base):
    """Alternative names of a work as sources cite it (e.g. "הר"ן" -> Ran on Shabbat)."""

    __tablename__ = "kl_work_aliases"
    __table_args__ = (UniqueConstraint("work_id", "alias"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("works.id"), index=True)
    alias: Mapped[str] = mapped_column(String(256), index=True)
    kind: Mapped[str | None] = mapped_column(String(16))  # short|citation|nickname


class KlJob(Base):
    __tablename__ = "kl_jobs"
    __table_args__ = (
        CheckConstraint(_in("status", KL_JOB_STATUS), name="ck_kl_jobs_status"),
        Index("ix_kl_jobs_stage_scope", "stage", "scope_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    stage: Mapped[str] = mapped_column(String(16))
    scope_key: Mapped[str] = mapped_column(String(256))
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    schema_version: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="running")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    stats: Mapped[dict | None] = mapped_column(JSONB)
    note: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KlTask(Base):
    __tablename__ = "kl_tasks"
    __table_args__ = (
        UniqueConstraint("job_id", "task_key"),
        CheckConstraint(_in("state", KL_TASK_STATE), name="ck_kl_tasks_state"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("kl_jobs.id"), index=True)
    task_key: Mapped[str] = mapped_column(String(256))
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(16), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    result: Mapped[dict | None] = mapped_column(JSONB)  # parsed LLM output (resume without re-call)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KlItem(Base):
    __tablename__ = "kl_items"
    __table_args__ = (
        UniqueConstraint("item_key", "version"),
        UniqueConstraint("item_key", "input_hash"),
        CheckConstraint(_in("kind", KL_KINDS), name="ck_kl_items_kind"),
        CheckConstraint(_in("status", KL_ITEM_STATUS), name="ck_kl_items_status"),
        Index(
            "uq_kl_items_current",
            "item_key",
            unique=True,
            postgresql_where=text("is_current"),
        ),
        Index("ix_kl_items_scope_kind", "scope_key", "kind"),
        Index(
            "ix_kl_items_search",
            text("to_tsvector('simple', coalesce(search_text_he, ''))"),
            postgresql_using="gin",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(String(24))
    item_key: Mapped[str] = mapped_column(String(256))
    version: Mapped[int] = mapped_column(Integer, default=1)
    scope_key: Mapped[str] = mapped_column(String(256))
    work_id: Mapped[int | None] = mapped_column(ForeignKey("works.id"), index=True)
    start_segment_id: Mapped[int | None] = mapped_column(ForeignKey("segments.id"))
    end_segment_id: Mapped[int | None] = mapped_column(ForeignKey("segments.id"))
    title_he: Mapped[str | None] = mapped_column(String(512))
    body: Mapped[dict] = mapped_column(JSONB)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="generated")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    stale: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    confidence: Mapped[float | None] = mapped_column(Float)
    input_hash: Mapped[str] = mapped_column(String(64))
    job_id: Mapped[int | None] = mapped_column(ForeignKey("kl_jobs.id"), index=True)
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    source_version_ids: Mapped[list | None] = mapped_column(JSONB)
    can_display_public: Mapped[bool] = mapped_column(Boolean, default=True)
    can_train: Mapped[bool] = mapped_column(Boolean, default=True)
    check_report: Mapped[dict | None] = mapped_column(JSONB)
    rejected_by: Mapped[str | None] = mapped_column(String(16))  # auto|human
    search_text_he: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KlItemSegment(Base):
    __tablename__ = "kl_item_segments"
    __table_args__ = (
        Index("ix_kl_item_segments_segment", "segment_id"),
        CheckConstraint(_in("role", ("member", "anchor", "commentary")), name="ck_kl_iseg_role"),
    )

    item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"), primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(12), primary_key=True, default="member")
    ord: Mapped[int] = mapped_column(Integer, default=0)


class KlEvidence(Base):
    __tablename__ = "kl_evidence"
    __table_args__ = (
        Index("ix_kl_evidence_segment", "segment_id"),
        Index("ix_kl_evidence_item", "item_id"),
        CheckConstraint(_in("role", KL_EVIDENCE_ROLES), name="ck_kl_evidence_role"),
        CheckConstraint(
            _in("locate_status", ("located", "not_found", "ambiguous", "out_of_scope")),
            name="ck_kl_evidence_locate",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"))
    claim_id: Mapped[str | None] = mapped_column(String(64))  # id of the claim inside body
    segment_id: Mapped[int | None] = mapped_column(ForeignKey("segments.id"))
    version_id: Mapped[int | None] = mapped_column(ForeignKey("versions.id"))
    span_start: Mapped[int | None] = mapped_column(Integer)  # char offsets in RAW text
    span_end: Mapped[int | None] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)
    quote_prefix: Mapped[str | None] = mapped_column(String(32))  # form B, up to 32 chars
    quote_suffix: Mapped[str | None] = mapped_column(String(32))
    raw_hash: Mapped[str | None] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16), default="supports")
    locate_status: Mapped[str] = mapped_column(String(16), default="located")
    entail_status: Mapped[str | None] = mapped_column(String(16))


class KlLink(Base):
    __tablename__ = "kl_links"
    __table_args__ = (
        UniqueConstraint("from_item_id", "to_item_id", "rel"),
        Index("ix_kl_links_to", "to_item_id"),
        CheckConstraint(_in("rel", KL_LINK_RELS), name="ck_kl_links_rel"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    from_item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"))
    to_item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"))
    rel: Mapped[str] = mapped_column(String(16))


class KlRelation(Base):
    __tablename__ = "kl_relations"
    __table_args__ = (
        Index("ix_kl_relations_from", "from_item_id"),
        Index("ix_kl_relations_to", "to_item_id"),
        CheckConstraint(_in("rel", KL_RELATIONS), name="ck_kl_relations_rel"),
        CheckConstraint(
            _in(
                "attribution",
                ("direct", "reported", "explicit", "inferred_by_source", "inferred_by_system"),
                nullable=True,
            ),
            name="ck_kl_relations_attribution",
        ),
        CheckConstraint(
            _in(
                "stance",
                ("asserts", "suggests", "considers", "rejects", "reports_others", "as_snif"),
                nullable=True,
            ),
            name="ck_kl_relations_stance",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    from_item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"))
    from_claim: Mapped[str | None] = mapped_column(String(64))
    to_item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"))
    to_claim: Mapped[str | None] = mapped_column(String(64))
    rel: Mapped[str] = mapped_column(String(24))
    asserted_by_entity: Mapped[int | None] = mapped_column(ForeignKey("kl_entities.id"))
    asserted_by_work: Mapped[int | None] = mapped_column(ForeignKey("works.id"))
    attribution: Mapped[str | None] = mapped_column(String(24))
    stance: Mapped[str | None] = mapped_column(String(16))
    evidence_ids: Mapped[list | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))


class KlSource(Base):
    """Which works feed the layer, and in what role. ``weight_tier`` is only a tie-breaker for
    selection and display - never halakhic weight or a reliability score."""

    __tablename__ = "kl_sources"
    __table_args__ = (
        CheckConstraint(_in("tier", KL_TIERS), name="ck_kl_sources_tier"),
        CheckConstraint(
            _in("role", ("primary", "lens", "witness", "reference_only")), name="ck_kl_sources_role"
        ),
        CheckConstraint(
            _in(
                "license_tier",
                ("PUBLIC_COMMERCIAL", "NON_COMMERCIAL_ONLY", "PERSONAL_ONLY_RESTRICTED"),
            ),
            name="ck_kl_sources_license_tier",
        ),
        CheckConstraint(
            _in("weight_tier", ("core", "major", "extended", "long_tail"), nullable=True),
            name="ck_kl_sources_weight_tier",
        ),
    )

    work_id: Mapped[int] = mapped_column(ForeignKey("works.id"), primary_key=True)
    tier: Mapped[str] = mapped_column(String(24))
    role: Mapped[str] = mapped_column(String(16))
    origin: Mapped[str | None] = mapped_column(String(64))
    license_tier: Mapped[str] = mapped_column(String(32), default="PUBLIC_COMMERCIAL")
    license_basis: Mapped[str | None] = mapped_column(Text)
    weight_tier: Mapped[str | None] = mapped_column(String(12))
    # closed list validated in pydantic (rav_knowledge.sources.AUTHORITY_AREAS), NULL = not
    # classified; information only (like weight_tier), no effect on selection
    authority_areas: Mapped[list | None] = mapped_column(JSONB)
    added_by: Mapped[str | None] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text)


class KlWitness(Base):
    """Register of textual witnesses so a ``girsa_variant`` points at an identified witness."""

    __tablename__ = "kl_witnesses"
    __table_args__ = (
        CheckConstraint(
            _in("kind", ("manuscript", "print", "genizah", "rishon_quote", "edition")),
            name="ck_kl_witnesses_kind",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    siglum: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(16))
    name_he: Mapped[str | None] = mapped_column(String(256))
    date_from: Mapped[int | None] = mapped_column(Integer)
    date_to: Mapped[int | None] = mapped_column(Integer)
    tradition: Mapped[str | None] = mapped_column(String(64))
    holding: Mapped[str | None] = mapped_column(String(256))
    source_url: Mapped[str | None] = mapped_column(Text)
    license_basis: Mapped[str | None] = mapped_column(Text)
    can_index: Mapped[bool] = mapped_column(Boolean, default=False)
    can_display_public: Mapped[bool] = mapped_column(Boolean, default=False)


class KlReception(Base):
    """Computed cache of how a claim is received in a context. Never manual input, never a
    number; rebuilt when the policy changes. ``rejected`` is allowed only for context=lemaase."""

    __tablename__ = "kl_reception"
    __table_args__ = (
        UniqueConstraint("target_item_id", "target_claim", "context", "policy_version"),
        CheckConstraint(
            _in("weight_label", ("primary", "alternative", "snif_only", "rejected", "disputed")),
            name="ck_kl_reception_label",
        ),
        CheckConstraint(
            "weight_label <> 'rejected' OR context = 'lemaase'", name="ck_kl_reception_rejected"
        ),
        CheckConstraint(KL_RECEPTION_CONTEXT_CHECK, name="ck_kl_reception_context"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    target_item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"))
    target_claim: Mapped[str] = mapped_column(String(64), default="")
    # iyun|lemaase|snif|shaat_hadchak|bediavad|public_instruction|minhag:<x>
    context: Mapped[str] = mapped_column(String(48))
    weight_label: Mapped[str] = mapped_column(String(16))
    basis: Mapped[dict | None] = mapped_column(JSONB)
    policy_version: Mapped[str] = mapped_column(String(32))
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KlCoverage(Base):
    """Distinguishes "this work has nothing on it" from "we did not ingest this work"."""

    __tablename__ = "kl_coverage"
    __table_args__ = (
        CheckConstraint(
            _in("status", ("covered", "not_in_work", "not_ingested", "unknown")),
            name="ck_kl_coverage_status",
        ),
    )

    scope_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("works.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="unknown")
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KlOverride(Base):
    """Human corrections; applied after every task so a re-run cannot undo them."""

    __tablename__ = "kl_overrides"
    __table_args__ = (
        CheckConstraint(
            _in("action", ("approve", "edit", "reject", "set_boundary", "set_abbrev")),
            name="ck_kl_overrides_action",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    target_key: Mapped[str] = mapped_column(String(256), index=True)  # item_key or annotation key
    action: Mapped[str] = mapped_column(String(16))
    corrected: Mapped[dict | None] = mapped_column(JSONB)
    note: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KlReview(Base):
    __tablename__ = "kl_reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("kl_items.id"), index=True)
    reviewer: Mapped[str | None] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(16))  # approve|reject|edit|flag
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
