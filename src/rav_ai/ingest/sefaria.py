"""Ingest the Sefaria export (Hebrew/Aramaic versions) as a dated snapshot, with hash diffs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.engine import Connection, Engine

from rav_ai.config import LicensePolicy
from rav_ai.db.models import (
    Author,
    Segment,
    SegmentText,
    Snapshot,
    SnapshotChange,
    Version,
    Work,
)
from rav_ai.ingest.order import layer_of
from rav_ai.normalize import normalize_a, normalize_b
from rav_ai.sefaria.layout import parse_version_object, schema_object_name
from rav_ai.sefaria.periods import derive_period
from rav_ai.sefaria.structure import iter_units

SOURCE = "sefaria"
PAGE = 2500
Log = Callable[[str], None]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sort_key(indices) -> str:
    return ".".join(f"{i:06d}" for i in indices)


def parent_ref(ref: str, labels) -> str | None:
    if len(labels) < 2:
        return None
    return ref[: ref.rindex(":")] if ":" in ref else None


@dataclass
class BookFiles:
    title: str
    category_path: tuple[str, ...]
    schema_path: Path
    version_paths: list[Path]


def discover_books(raw_dir: Path, language_dir: str = "Hebrew") -> dict[str, BookFiles]:
    books: dict[str, BookFiles] = {}
    for p in sorted((raw_dir / "json").rglob("*.json")):
        v = parse_version_object(p.relative_to(raw_dir).as_posix())
        if v is None or v.is_merged or v.language_dir != language_dir:
            continue
        sp = raw_dir / schema_object_name(v.book)
        if not sp.exists():
            continue
        b = books.setdefault(v.book, BookFiles(v.book, v.category_path, sp, []))
        b.version_paths.append(p)
    return books


# --- snapshots ----------------------------------------------------------------------


def export_label(raw_dir: Path) -> str:
    f = raw_dir / "last_export.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else "unknown"


def start_snapshot(engine: Engine, raw_dir: Path) -> Snapshot:
    """Reuse an unfinished snapshot of the same export (resume), else open a new one."""
    label = export_label(raw_dir)
    with engine.begin() as conn:
        row = conn.execute(
            select(Snapshot)
            .where(
                Snapshot.source == SOURCE,
                Snapshot.export_label == label,
                Snapshot.status == "running",
            )
            .order_by(Snapshot.id.desc())
        ).first()
        if row:
            return row
        res = conn.execute(
            insert(Snapshot)
            .values(
                source=SOURCE,
                export_label=label,
                taken_at=datetime.now(UTC),
                status="running",
                stats={},
            )
            .returning(Snapshot.__table__)
        )
        return res.first()


def finish_snapshot(engine: Engine, snap_id: int, stats: dict) -> None:
    with engine.begin() as conn:
        old = conn.execute(select(Snapshot.stats).where(Snapshot.id == snap_id)).scalar() or {}
        conn.execute(
            update(Snapshot)
            .where(Snapshot.id == snap_id)
            .values(status="complete", stats={**old, **stats})
        )


# --- authors / works ------------------------------------------------------------------


def _upsert_author(conn: Connection, schema_doc: dict, period: str) -> int | None:
    authors = schema_doc.get("authors") or []
    if not authors:
        return None
    a = authors[0]
    slug = a.get("slug") or a.get("en") or a.get("he")
    if not slug:
        return None
    row = conn.execute(select(Author.id).where(Author.slug == slug)).first()
    if row:
        return row[0]
    return conn.execute(
        insert(Author)
        .values(
            slug=slug,
            name=a.get("en") or a.get("he") or slug,
            he_name=a.get("he"),
            period=period,
        )
        .returning(Author.id)
    ).scalar_one()


def upsert_work(conn: Connection, book: BookFiles) -> tuple[int, dict]:
    with open(book.schema_path, encoding="utf-8") as f:
        doc = json.load(f)
    cats = doc.get("categories") or list(book.category_path)
    period = derive_period(cats, doc.get("era"), doc.get("dependence"))
    comp = doc.get("compDate") or []
    vals = dict(
        he_title=doc.get("heTitle"),
        author_id=_upsert_author(conn, doc, period),
        category=cats[0] if cats else "Uncategorized",
        category_path=cats,
        era_code=doc.get("era"),
        period=period,
        comp_year_start=comp[0] if comp and isinstance(comp[0], int) else None,
        comp_year_end=comp[-1] if comp and isinstance(comp[-1], int) else None,
        dependence=doc.get("dependence"),
        base_work_titles=[b.get("en") for b in doc.get("base_text_titles") or []] or None,
        section_names=doc.get("sectionNames") or [],
        schema=doc["schema"],
    )
    row = conn.execute(select(Work.id).where(Work.title == book.title)).first()
    if row:
        conn.execute(update(Work).where(Work.id == row[0]).values(**vals))
        return row[0], doc["schema"]
    wid = conn.execute(
        insert(Work).values(title=book.title, **vals).returning(Work.id)
    ).scalar_one()
    return wid, doc["schema"]


def link_base_works(engine: Engine) -> int:
    n = 0
    with engine.begin() as conn:
        ids = dict(conn.execute(select(Work.title, Work.id)).all())
        for wid, titles in conn.execute(
            select(Work.id, Work.base_work_titles).where(Work.base_work_titles.is_not(None))
        ).all():
            base = next((ids[t] for t in (titles or []) if t in ids), None)
            if base:
                conn.execute(update(Work).where(Work.id == wid).values(base_work_id=base))
                n += 1
    return n


# --- versions -------------------------------------------------------------------------


def apply_policy(policy: LicensePolicy, lic: str | None) -> dict:
    p = policy.evaluate(lic)
    return dict(
        can_index=p.can_index, can_display_public=p.can_display_public, can_train=p.can_train
    )


def recompute_flags(engine: Engine, policy: LicensePolicy) -> int:
    """Re-apply license policy to all stored versions (no re-import needed)."""
    n = 0
    with engine.begin() as conn:
        for vid, lic in conn.execute(select(Version.id, Version.license)).all():
            conn.execute(
                update(Version).where(Version.id == vid).values(**apply_policy(policy, lic))
            )
            n += 1
    return n


def _load_segment_ids(conn: Connection, work_id: int) -> dict[str, int]:
    return dict(
        conn.execute(select(Segment.ref, Segment.id).where(Segment.work_id == work_id)).all()
    )


def ingest_version(
    conn: Connection,
    snap_id: int,
    work_id: int,
    schema: dict,
    title: str,
    vpath: Path,
    policy: LicensePolicy,
    fhash: str,
    renormalize: bool = False,
) -> dict:
    """Insert/diff one version file. Returns change counts.

    renormalize=True also recomputes the derived forms (A/B/offsets) of unchanged segments in
    place (use after a textnorm upgrade); the raw text and hashes are untouched.
    """
    with open(vpath, encoding="utf-8") as f:
        doc = json.load(f)
    vtitle = doc.get("versionTitle") or vpath.stem
    lang = doc.get("language") or "he"
    meta = dict(
        source_url=doc.get("versionSource"),
        license=doc.get("license"),
        **apply_policy(policy, doc.get("license")),
        is_primary=bool(doc.get("isPrimary")),
        priority=float(doc["priority"]) if isinstance(doc.get("priority"), int | float) else None,
        status=doc.get("status"),
        file_hash=fhash,
        last_snapshot_id=snap_id,
    )
    ver = conn.execute(
        select(Version.id).where(
            Version.work_id == work_id,
            Version.source == SOURCE,
            Version.language == lang,
            Version.version_title == vtitle,
        )
    ).first()
    if ver:
        vid = ver[0]
        conn.execute(update(Version).where(Version.id == vid).values(**meta))
        old = {
            r: (i, h)
            for r, i, h in conn.execute(
                select(Segment.ref, SegmentText.id, SegmentText.raw_hash)
                .join(SegmentText, SegmentText.segment_id == Segment.id)
                .where(SegmentText.version_id == vid, Segment.work_id == work_id)
            )
        }
    else:
        vid = conn.execute(
            insert(Version)
            .values(
                work_id=work_id,
                source=SOURCE,
                language=lang,
                version_title=vtitle,
                first_snapshot_id=snap_id,
                word_count=0,
                **meta,
            )
            .returning(Version.id)
        ).scalar_one()
        old = {}

    units, dup = {}, 0
    for u in iter_units(title, schema, doc.get("text")):
        if u.ref in units:
            dup += 1
            continue
        units[u.ref] = (u, sha256_bytes(u.text.encode("utf-8")))

    added = [r for r in units if r not in old]
    changed = [r for r, (_, h) in units.items() if r in old and old[r][1] != h]
    removed = [r for r in old if r not in units]
    rederive = [r for r in units if r in old and r not in changed] if renormalize else []
    todo = added + changed + rederive

    if todo:
        seg_ids = _load_segment_ids(conn, work_id)
        new_segs = [r for r in todo if r not in seg_ids]
        for i in range(0, len(new_segs), PAGE):
            rows = []
            for r in new_segs[i : i + PAGE]:
                u = units[r][0]
                rows.append(
                    dict(
                        ref=r,
                        work_id=work_id,
                        node_index=u.node_index,
                        node_path=list(u.node_path),
                        section_path=list(u.labels),
                        indices=list(u.indices),
                        section_names=list(u.section_names),
                        sort_key=sort_key(u.indices),
                        parent_ref=parent_ref(r, u.labels),
                    )
                )
            conn.execute(insert(Segment), rows)
        if new_segs:
            seg_ids = _load_segment_ids(conn, work_id)
        if changed or rederive:
            conn.execute(
                delete(SegmentText).where(
                    SegmentText.id.in_([old[r][0] for r in (*changed, *rederive)])
                )
            )
        for i in range(0, len(todo), PAGE):
            rows = []
            for r in todo[i : i + PAGE]:
                u, h = units[r]
                a, b = normalize_a(u.text), normalize_b(u.text)
                rows.append(
                    dict(
                        segment_id=seg_ids[r],
                        version_id=vid,
                        text_raw=u.text,
                        raw_hash=h,
                        text_a=a.text,
                        text_b=b.text,
                        offsets_a=a.offsets.to_bytes(),
                        offsets_b=b.offsets.to_bytes(),
                        word_count=len(b.text.split()),
                    )
                )
            conn.execute(insert(SegmentText), rows)
    if removed:
        conn.execute(delete(SegmentText).where(SegmentText.id.in_([old[r][0] for r in removed])))

    def chg(r, kind, old_h, new_h):
        return dict(
            snapshot_id=snap_id, version_id=vid, ref=r, change=kind, old_hash=old_h, new_hash=new_h
        )

    changes = (
        (
            [chg(r, "added", None, units[r][1]) for r in added]
            if old
            else []  # first import: counts only
        )
        + [chg(r, "changed", old[r][1], units[r][1]) for r in changed]
        + [chg(r, "removed", old[r][1], None) for r in removed]
    )
    for i in range(0, len(changes), 5000):
        conn.execute(insert(SnapshotChange), changes[i : i + 5000])

    wc = conn.execute(
        select(func.coalesce(func.sum(SegmentText.word_count), 0)).where(
            SegmentText.version_id == vid
        )
    ).scalar_one()
    conn.execute(update(Version).where(Version.id == vid).values(word_count=wc))
    return dict(
        version_id=vid,
        added=len(added),
        changed=len(changed),
        removed=len(removed),
        unchanged=len(units) - len(added) - len(changed),
        duplicate_refs=dup,
    )


def ingest_all(
    engine: Engine,
    raw_dir: Path,
    policy: LicensePolicy,
    snap_id: int,
    only_books: set[str] | None = None,
    log: Log = print,
    renormalize: bool = False,
) -> dict:
    books = discover_books(raw_dir)
    if only_books:
        books = {t: b for t, b in books.items() if t in only_books}
    totals: Counter = Counter()
    errors: list[str] = []
    ordered = sorted(
        books.items(),
        key=lambda kv: (
            layer_of(kv[0], kv[1].category_path),
            not kv[0].startswith("Shulchan Arukh"),  # base text before its commentators
            kv[0],
        ),
    )
    for i, (title, book) in enumerate(ordered, 1):
        try:
            with engine.begin() as conn:
                wid, schema = upsert_work(conn, book)
            for vp in book.version_paths:
                fh = file_sha256(vp)
                with engine.begin() as conn:
                    prior = conn.execute(
                        select(Version.id, Version.file_hash, Version.language).where(
                            Version.work_id == wid, Version.source == SOURCE
                        )
                    ).all()
                    # unchanged file + already known -> only mark as seen
                    hit = None if renormalize else next((p for p in prior if p[1] == fh), None)
                    if hit:
                        conn.execute(
                            update(Version)
                            .where(Version.id == hit[0])
                            .values(last_snapshot_id=snap_id)
                        )
                        totals["versions_unchanged_skipped"] += 1
                        continue
                    res = ingest_version(
                        conn, snap_id, wid, schema, title, vp, policy, fh, renormalize
                    )
                for k in ("added", "changed", "removed", "unchanged", "duplicate_refs"):
                    totals[k] += res[k]
                totals["versions_processed"] += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{title}: {type(e).__name__}: {e}")
            log(f"ERROR {title}: {e}")
        if i % 25 == 0 or i == len(books):
            log(f"  books {i}/{len(books)}  segments+ {totals['added']:,}")
    base_links = link_base_works(engine)
    out = dict(totals)
    out.update(books=len(books), base_works_linked=base_links, errors=errors)
    return out


def text_sql_ping(engine: Engine) -> bool:
    with engine.connect() as c:
        return c.execute(text("select 1")).scalar() == 1
