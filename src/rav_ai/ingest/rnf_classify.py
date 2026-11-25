"""Classify `ref_not_found` link endpoints against the work's structure and the ingested text.

Nothing here resolves anything: it only says *why* an endpoint is unresolved, so a gap in the
Hebrew text is not confused with a resolver bug or a bad link. Classes:

- ``partial_hebrew_text``: the ref is plausible for the work (node in the schema) but the
  ingested Hebrew version has no text there: whole node absent, a hole inside the text, text
  starting later, or a ref past the last segment of a work that is not a complete base text.
- ``numbering_mismatch``: the ref cannot fit the work's structure at all (more levels than the
  schema, or an address of the wrong type, e.g. ``12a`` in an integer-addressed node).
- ``resolver_bug``: the text exists and the cited range covers it, but the (old) resolver
  failed: a range with an endpoint in a hole.
- ``other``: everything else, with a detail string (unparseable, node not in schema, ref past
  the end of a complete base text such as Tanakh/Mishnah/Bavli, i.e. an invalid link target).

The detail is stored as ``ref_not_found:<class>`` only in reports; the DB reason is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rav_ai.sefaria.linkrefs import WorkIndex, labels_key, parse_citation
from rav_ai.sefaria.structure import leaves

PARTIAL = "partial_hebrew_text"
NUMBERING = "numbering_mismatch"
BUG = "resolver_bug"
OTHER = "other"
CLASSES = (PARTIAL, NUMBERING, BUG, OTHER)

_TALMUD = re.compile(r"^[0-9]+[ab]$")
_INT = re.compile(r"^[0-9]+$")


@dataclass
class WorkProfile:
    """Everything the classifier needs to know about one work."""

    index: WorkIndex
    nodes: dict[str, list[str]]  # schema node path -> addressTypes of that leaf
    complete_base: bool  # Tanakh / Mishnah / Bavli base text: a gap there is not a Hebrew gap

    @classmethod
    def build(cls, index: WorkIndex, schema: dict, category_path: list, dependence) -> WorkProfile:
        nodes = {
            ", ".join(leaf.node_path): list(leaf.node.get("addressTypes") or [])
            for leaf in leaves(schema)
        }
        cats = set(category_path or [])
        complete = dependence is None and bool(cats & {"Tanakh", "Mishnah", "Bavli"})
        return cls(index, nodes, complete)


def _first_missing(idx: WorkIndex, node: str, tup: tuple[str, ...]) -> int | None:
    for k in range(len(tup)):
        if (node, tup[: k + 1]) not in idx._prefix:
            return k
    return None


def _gap_kind(idx: WorkIndex, node: str, tup: tuple[str, ...], k: int) -> str | None:
    """beyond | before | hole for the first missing label, relative to its existing siblings."""
    prefix = tup[:k]
    sib = {
        labels[k]
        for _, n, labels, _ in idx.entries
        if n == node and labels[:k] == prefix and len(labels) > k
    }
    keys = [labels_key((s,)) for s in sib]
    q = labels_key((tup[k],))
    if q is None or not keys or None in keys:
        return None
    lo, hi = min(keys), max(keys)  # type: ignore[type-var]
    return "beyond" if q > hi else "before" if q < lo else "hole"  # type: ignore[operator]


def _address_mismatch(types: list[str], labels: tuple[str, ...]) -> bool:
    for i, (t, lab) in enumerate(zip(types, labels, strict=False)):
        if t == "Talmud" and not (_TALMUD.match(lab) or (i == 0 and _INT.match(lab))):
            return True  # a bare daf ("117", both sides) is legal at the first level
        if t == "Integer" and not _INT.match(lab):
            return True
    return False


def classify(profile: WorkProfile, citation: str) -> tuple[str, str]:
    """(class, detail) for an endpoint the resolver could not find."""
    idx = profile.index
    p = parse_citation(idx.title, citation)
    if p is None:
        return OTHER, "unparseable"
    if (p.node, ()) not in idx._prefix:
        if p.node in profile.nodes:
            return PARTIAL, "node_absent_in_hebrew"
        return OTHER, "node_not_in_schema"
    types = profile.nodes.get(p.node, [])
    for label_set in (p.start, p.end or ()):
        if types and len(label_set) > len(types):
            return NUMBERING, "deeper_than_structure"
        if types and _address_mismatch(types, label_set):
            return NUMBERING, "address_type"
    if p.end is not None and _hole_range_resolvable(idx, p):
        return BUG, "range_endpoint_in_hole"
    for which, tup in (("start", p.start), ("end", p.end or ())):
        k = _first_missing(idx, p.node, tup)
        if k is None:
            continue
        kind = _gap_kind(idx, p.node, tup, k)
        if kind is None:
            return OTHER, "nonnumeric_label"
        if profile.complete_base:
            return OTHER, "invalid_ref_in_complete_text"
        return PARTIAL, f"{kind}_{which}"
    return OTHER, "unclassified"


def _hole_range_resolvable(idx: WorkIndex, p) -> bool:
    return bool(idx._resolve_hole_range(p))


# ---- over the database -------------------------------------------------------------------

_PRIORITY = {BUG: 0, NUMBERING: 1, PARTIAL: 2, OTHER: 3}
_UNCLASSIFIED_SQL = """
SELECT id, citation_a, citation_b, book_a, book_b FROM unresolved_links
WHERE reason = 'ref_not_found'
  AND NOT (book_a ~ 'Zohar' OR book_b ~ 'Zohar'
           OR book_a ~ '^Rav Hirsch on Torah' OR book_b ~ '^Rav Hirsch on Torah')"""


@dataclass
class Classified:
    link_id: int
    citation: str  # the failing endpoint
    book: str
    cls: str
    detail: str


def classify_unresolved(engine, log=print) -> list[Classified]:
    """Classify every `ref_not_found` link that is not already explained (Zohar, Rav Hirsch)."""
    from sqlalchemy import text

    from rav_ai.ingest.links import work_prefix

    with engine.connect() as c:
        rows = c.execute(text(_UNCLASSIFIED_SQL)).all()
        works = dict(c.execute(text("SELECT title, id FROM works")).all())
    by_work: dict[tuple[int, str], set[tuple[str, str]]] = {}
    for _, ca, cb, ba, bb in rows:
        for cit, book in ((ca, ba), (cb, bb)):
            wid, it = works.get(book), book
            if wid is None:
                parent = work_prefix(book, works)
                if parent is None:
                    continue
                wid, it = works[parent], parent
            by_work.setdefault((wid, it), set()).add((cit, book))
    verdict: dict[tuple[str, str], tuple[str, str]] = {}
    for n, ((wid, it), eps) in enumerate(sorted(by_work.items())):
        with engine.connect() as c:
            segs = c.execute(
                text("""SELECT ref, node_index, indices, node_path, section_path, id
                        FROM segments WHERE work_id = :w"""),
                {"w": wid},
            ).all()
            sch, cp, dep = c.execute(
                text("SELECT schema, category_path, dependence FROM works WHERE id = :w"),
                {"w": wid},
            ).one()
        prof = WorkProfile.build(WorkIndex.build(it, segs), sch, cp, dep)
        for cit, book in eps:
            if prof.index.resolve(cit) == []:  # this endpoint is the failing one
                verdict[(cit, book)] = classify(prof, cit)
        if n % 200 == 0:
            log(f"classified works {n}/{len(by_work)}")
    out = []
    for lid, ca, cb, ba, bb in rows:
        cands = [(k, verdict[k]) for k in ((ca, ba), (cb, bb)) if k in verdict]
        if not cands:
            out.append(Classified(lid, ca, ba or "", OTHER, "no_failing_endpoint_found"))
            continue
        (cit, book), (cls, detail) = min(cands, key=lambda x: _PRIORITY[x[1][0]])
        out.append(Classified(lid, cit, book, cls, detail))
    return out


def stratified_sample(rows: list[Classified], seed: int = 20261125) -> list[Classified]:
    """20 links for each of the 30 books with most links, plus 200 random links."""
    import random
    from collections import Counter, defaultdict

    rng = random.Random(seed)
    by_book: dict[str, list[Classified]] = defaultdict(list)
    for r in rows:
        by_book[r.book].append(r)
    top = [b for b, _ in Counter({b: len(v) for b, v in by_book.items()}).most_common(30)]
    out = [r for b in top for r in rng.sample(by_book[b], min(20, len(by_book[b])))]
    return out + rng.sample(rows, min(200, len(rows)))


def distribution_md(title: str, rows: list[Classified], examples: int = 2) -> list[str]:
    from collections import Counter

    n = Counter((r.cls, r.detail) for r in rows)
    ex: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        ex.setdefault((r.cls, r.detail), [])
        if len(ex[(r.cls, r.detail)]) < examples:
            ex[(r.cls, r.detail)].append(r.citation)
    md = [f"### {title} ({len(rows):,} links)", "", "| class | detail | links | examples |"]
    md += ["|---|---|---:|---|"]
    for (cls, detail), v in sorted(n.items(), key=lambda x: (CLASSES.index(x[0][0]), -x[1])):
        md.append(
            f"| {cls} | {detail} | {v:,} | {'; '.join(f'`{e}`' for e in ex[(cls, detail)])} |"
        )
    return [*md, ""]
