"""Database access for the knowledge layer: scopes, input segments, items, evidence, links."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from rav_ai.db.models import (
    Author,
    Edge,
    KlEvidence,
    KlItem,
    KlItemSegment,
    KlLink,
    KlOverride,
    KlSource,
    Segment,
    SegmentText,
    Version,
    Work,
)
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .checks import SegInfo
from .config import KLConfig
from .evidence import SegText, locate
from .licensing import admit, admit_aid


def sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(
            (
                p if isinstance(p, str) else json.dumps(p, sort_keys=True, ensure_ascii=False)
            ).encode()
        )
        h.update(b"\x1f")
    return h.hexdigest()


# ---- scopes ------------------------------------------------------------------------------


@dataclass
class Page:
    label: str  # "21b"
    parent_ref: str  # "Shabbat 21b"
    segment_ids: list[int]


@dataclass
class Scope:
    key: str
    work_id: int
    work_title: str
    category: str
    pages: list[Page]

    @property
    def segment_ids(self) -> list[int]:
        return [i for p in self.pages for i in p.segment_ids]


_SCOPE_RE = re.compile(r"^(?P<work>.+?)\s+(?P<a>[0-9]+[ab]?)(?:\s*-\s*(?P<b>[0-9]+[ab]?))?$")


def _page_sort(label: str) -> tuple[int, int]:
    m = re.match(r"(\d+)([ab]?)", label)
    return (int(m.group(1)), 0 if m.group(2) in ("", "a") else 1) if m else (0, 0)


def load_scope(s: Session, key: str) -> Scope:
    """ "Shabbat 21a-24a" -> every page (amud / siman) from 21a to 24a inclusive."""
    m = _SCOPE_RE.match(key.strip())
    if not m:
        raise ValueError(f"cannot parse scope {key!r} (expected '<Work> <from>[-<to>]')")
    work = s.scalars(select(Work).where(Work.title == m["work"])).first()
    if work is None:
        raise ValueError(f"unknown work {m['work']!r}")
    lo, hi = _page_sort(m["a"]), _page_sort(m["b"] or m["a"])
    rows = s.execute(
        select(Segment.id, Segment.parent_ref, Segment.sort_key)
        .where(Segment.work_id == work.id, Segment.parent_ref.is_not(None))
        .order_by(Segment.sort_key, Segment.id)
    ).all()
    pages: dict[str, Page] = {}
    for sid, parent, _sk in rows:
        label = parent.rsplit(" ", 1)[-1]
        if lo <= _page_sort(label) <= hi:
            pages.setdefault(label, Page(label, parent, [])).segment_ids.append(sid)
    return Scope(key, work.id, work.title, work.category, list(pages.values()))


# ---- input segments -----------------------------------------------------------------------


@dataclass
class InputSeg:
    info: SegInfo
    text: SegText
    version: Version
    source: KlSource | None
    is_aid: bool = False


@dataclass
class LoadedInput:
    segs: list[InputSeg] = field(default_factory=list)
    aids: list[InputSeg] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)  # reason -> count (goes to job stats)

    def by_label(self) -> dict[str, InputSeg]:
        return {x.info.label: x for x in self.segs}

    @property
    def raw_hashes(self) -> list[str]:
        return [x.text.raw_hash for x in self.segs + self.aids]

    @property
    def version_ids(self) -> list[int]:
        return sorted({x.version.id for x in self.segs + self.aids})


def _best_he_versions(s: Session, segment_ids: list[int]):
    q = (
        select(SegmentText, Version, Segment, Work)
        .join(Version, Version.id == SegmentText.version_id)
        .join(Segment, Segment.id == SegmentText.segment_id)
        .join(Work, Work.id == Segment.work_id)
        .where(SegmentText.segment_id.in_(segment_ids), Version.language == "he")
        .order_by(
            Segment.sort_key,
            Segment.id,
            Version.is_primary.desc(),
            Version.priority.desc().nulls_last(),
            Version.id,
        )
    )
    return s.execute(q).all()


def _make(
    st: SegmentText, v: Version, seg: Segment, w: Work, src, label: str, order: int, author
) -> InputSeg:
    info = SegInfo(
        label=label,
        segment_id=seg.id,
        ref=seg.ref,
        work_id=w.id,
        work_title=w.title,
        author_slug=author.slug if author else None,
        tier=src.tier if src else None,
        source_role=src.role if src else None,
        period=(author.period if author else None) or w.period,
        order=order,
    )
    text = SegText(
        seg.id,
        v.id,
        seg.ref,
        st.raw_hash,
        st.text_a,
        st.text_b,
        st.offsets_b,
        work_id=w.id,
        offsets_a=st.offsets_a,
        text_raw=st.text_raw,
    )
    return InputSeg(info, text, v, src)


def load_input(
    s: Session, segment_ids: list[int], cfg: KLConfig, *, with_aids: bool = False
) -> LoadedInput:
    """Segments in document order with the preferred indexable Hebrew version, license-gated."""
    out = LoadedInput()
    if not segment_ids:
        return out
    seen: set[int] = set()
    sources = {x.work_id: x for x in s.scalars(select(KlSource))}
    authors = {a.id: a for a in s.scalars(select(Author))}
    n = 0
    for st, v, seg, w in _best_he_versions(s, segment_ids):
        if seg.id in seen:
            continue
        ok, why = admit(sources.get(w.id), v, cfg)
        if not ok:
            # try the next version of the same segment; count only when none is admitted
            out.excluded[why] = out.excluded.get(why, 0) + 1
            continue
        seen.add(seg.id)
        n += 1
        out.segs.append(
            _make(st, v, seg, w, sources.get(w.id), f"S{n}", n, authors.get(w.author_id))
        )
    if with_aids:
        out.aids = load_aids(s, [x.info.segment_id for x in out.segs], cfg, sources, authors, out)
    return out


def load_aids(s, seg_ids, cfg: KLConfig, sources, authors, out: LoadedInput) -> list[InputSeg]:
    """Rashi (always) and Steinsaltz (flag) as reading aids; linked by Sefaria edges."""
    prefixes = ["Rashi on "]
    if cfg.use_steinsaltz_aid:
        prefixes.append(cfg.steinsaltz_prefix)
    rows = s.execute(
        select(Edge.segment_a_id, Edge.segment_b_id).where(
            (Edge.segment_a_id.in_(seg_ids)) | (Edge.segment_b_id.in_(seg_ids))
        )
    ).all()
    base = set(seg_ids)
    other = sorted({b if a in base else a for a, b in rows} - base)
    if not other:
        return []
    aids: list[InputSeg] = []
    seen: set[int] = set()
    k = 0
    for st, v, seg, w in _best_he_versions(s, other):
        if seg.id in seen or not any(w.title.startswith(p) for p in prefixes):
            continue
        ok, why = admit_aid(sources.get(w.id), v, cfg)
        if not ok:
            out.excluded[f"aid:{why}"] = out.excluded.get(f"aid:{why}", 0) + 1
            continue
        seen.add(seg.id)
        k += 1
        a = _make(st, v, seg, w, sources.get(w.id), f"A{k}", 0, authors.get(w.author_id))
        a.is_aid = True
        aids.append(a)
    return aids


def uses_steinsaltz(inp: LoadedInput, cfg: KLConfig) -> bool:
    return any(
        a.info.work_title and a.info.work_title.startswith(cfg.steinsaltz_prefix) for a in inp.aids
    )


# ---- items --------------------------------------------------------------------------------


def current_item(s: Session, item_key: str) -> KlItem | None:
    return s.scalars(select(KlItem).where(KlItem.item_key == item_key, KlItem.is_current)).first()


def write_item(
    s: Session,
    *,
    kind: str,
    item_key: str,
    scope_key: str,
    body: dict,
    input_hash: str,
    job_id: int | None,
    model: str | None,
    prompt_version: str | None,
    prompt_hash: str | None,
    schema_version: int,
    can_display_public: bool,  # no default: licensing is fail-closed, the caller decides
    can_train: bool,
    status: str = "generated",
    work_id: int | None = None,
    start_segment_id: int | None = None,
    end_segment_id: int | None = None,
    title_he: str | None = None,
    confidence: float | None = None,
    source_version_ids: list[int] | None = None,
    check_report: dict | None = None,
    rejected_by: str | None = None,
    search_text_he: str | None = None,
    shadow: bool = False,
    make_current: bool = False,
) -> KlItem:
    """Insert a new version (idempotent per (item_key, input_hash)).

    is_current rules: the first output for a key is current; later outputs (new prompt/model/
    input) are written NON-current until ``promote``; a verified current item is never replaced;
    ``shadow`` (second-model runs) is never current. ``make_current`` (human override versions)
    demotes the current version and takes its place.
    """
    existing = s.scalars(
        select(KlItem).where(KlItem.item_key == item_key, KlItem.input_hash == input_hash)
    ).first()
    if existing is not None:
        return existing
    cur = current_item(s, item_key)
    if make_current and cur is not None:
        cur.is_current = False
        s.flush()
        cur = None
    version = (
        s.scalar(select(func.max(KlItem.version)).where(KlItem.item_key == item_key)) or 0
    ) + 1
    item = KlItem(
        kind=kind,
        item_key=item_key,
        version=version,
        scope_key=scope_key,
        work_id=work_id,
        start_segment_id=start_segment_id,
        end_segment_id=end_segment_id,
        title_he=title_he,
        body=body,
        schema_version=schema_version,
        status=status,
        is_current=(make_current or (cur is None and not shadow and status != "rejected")),
        confidence=confidence,
        input_hash=input_hash,
        job_id=job_id,
        model=model,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        source_version_ids=source_version_ids,
        can_display_public=can_display_public,
        can_train=can_train,
        check_report=check_report,
        rejected_by=rejected_by,
        search_text_he=search_text_he,
    )
    s.add(item)
    s.flush()
    return item


def add_segments(s: Session, item: KlItem, rows: list[tuple[int, str]]) -> None:
    for i, (sid, role) in enumerate(rows):
        s.merge(KlItemSegment(item_id=item.id, segment_id=sid, role=role, ord=i))


def copy_evidence(s: Session, src_item_id: int, dst_item_id: int) -> int:
    """Copy the evidence rows of one item version to another and locate every quote again
    against the current text (status / spans refreshed; a quote that is gone stays not_found)."""
    n = 0
    for e in s.scalars(select(KlEvidence).where(KlEvidence.item_id == src_item_id)):
        st = (
            s.scalars(
                select(SegmentText).where(
                    SegmentText.segment_id == e.segment_id, SegmentText.version_id == e.version_id
                )
            ).first()
            if e.segment_id is not None and e.version_id is not None
            else None
        )
        row = {
            c.name: getattr(e, c.name) for c in KlEvidence.__table__.columns if c.name != "id"
        } | {"item_id": dst_item_id}
        if st is not None:
            loc = locate(
                e.quote,
                SegText(
                    st.segment_id,
                    st.version_id,
                    "",
                    st.raw_hash,
                    st.text_a,
                    st.text_b,
                    st.offsets_b,
                ),
            )
            row.update(
                span_start=loc.span_start,
                span_end=loc.span_end,
                quote_prefix=loc.quote_prefix,
                quote_suffix=loc.quote_suffix,
                raw_hash=loc.raw_hash,
                locate_status=loc.status,
            )
        s.add(KlEvidence(**row))
        n += 1
    return n


def add_link(s: Session, frm: int, to: int, rel: str) -> None:
    if not s.scalars(
        select(KlLink.id).where(
            KlLink.from_item_id == frm, KlLink.to_item_id == to, KlLink.rel == rel
        )
    ).first():
        s.add(KlLink(from_item_id=frm, to_item_id=to, rel=rel))


def add_evidence(s: Session, item_id: int, rows: list[dict]) -> None:
    for r in rows:
        s.add(KlEvidence(item_id=item_id, **r))


def mark_stale_dependents(s: Session, item_ids: list[int]) -> int:
    """Recursive stale through kl_links(depends_on). Returns the number of items marked."""
    seen: set[int] = set()
    frontier = list(item_ids)
    while frontier:
        rows = s.scalars(
            select(KlLink.from_item_id).where(
                KlLink.rel == "depends_on", KlLink.to_item_id.in_(frontier)
            )
        ).all()
        frontier = [r for r in rows if r not in seen and r not in item_ids]
        seen.update(frontier)
    if seen:
        s.execute(update(KlItem).where(KlItem.id.in_(seen)).values(stale=True))
    return len(seen)


def overrides_for(s: Session, keys: list[str]) -> dict[str, KlOverride]:
    """Latest override per target key."""
    if not keys:
        return {}
    out: dict[str, KlOverride] = {}
    for o in s.scalars(
        select(KlOverride).where(KlOverride.target_key.in_(keys)).order_by(KlOverride.id)
    ):
        out[o.target_key] = o
    return out
