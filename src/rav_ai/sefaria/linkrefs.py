"""Resolve Sefaria link citations (segments, sections, ranges) to concrete refs of a work."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

_ADDR = r"[0-9]+[ab]?(?:[:.][0-9]+[ab]?)*"
_DAF_RE = re.compile(r"^[0-9]+[ab]$")
_REST_RE = re.compile(rf"^(?:, (?P<node>.+?))?(?: (?P<start>{_ADDR})(?:-(?P<end>{_ADDR}))?)?$")


_NUM_RE = re.compile(r"^([0-9]+)([ab]?)$")
_INF = float("inf")


def label_pos(label: str) -> int | None:
    """Numeric position of one address label: '7' -> 7, '33a' -> 66, '33b' -> 67 (daf order)."""
    m = _NUM_RE.match(label)
    if not m:
        return None
    return int(m[1]) * 2 + (m[2] == "b") if m[2] else int(m[1])


def labels_key(labels: tuple[str, ...]) -> tuple[int, ...] | None:
    out = [label_pos(x) for x in labels]
    return None if None in out else tuple(out)  # type: ignore[arg-type]


def _label_kind(label: str) -> str | None:
    if _DAF_RE.match(label):
        return "talmud"
    return "int" if label.isdigit() else None


def is_complete_base(category_path, dependence) -> bool:
    """Tanakh / Mishnah / Bavli base text (not a commentary): a gap there is not a Hebrew gap."""
    return dependence is None and bool(set(category_path or []) & {"Tanakh", "Mishnah", "Bavli"})


@dataclass
class ParsedCitation:
    node: str  # ", "-joined node titles ('' for the root/default node)
    start: tuple[str, ...]
    end: tuple[str, ...] | None


def parse_citation(title: str, citation: str) -> ParsedCitation | None:
    if citation != title and not citation.startswith(title):
        return None
    m = _REST_RE.match(citation[len(title) :])
    if not m:
        return None
    split = lambda a: tuple(a.replace(".", ":").split(":"))  # noqa: E731
    start = split(m["start"]) if m["start"] else ()
    end = split(m["end"]) if m["end"] else None
    if end is not None and len(end) < len(start):  # "1:1-3" -> end "1:3"
        end = start[: len(start) - len(end)] + end
    return ParsedCitation(m["node"] or "", start, end)


@dataclass
class WorkIndex:
    """Document-ordered refs of one work (union over its versions)."""

    title: str
    entries: list[tuple[str, str, tuple[str, ...], int]] = field(default_factory=list)
    # entries: (ref, joined node path, labels, segment id), in document order
    _prefix: dict[tuple[str, tuple[str, ...]], tuple[int, int]] = field(default_factory=dict)
    # per node: (position of its first entry, numeric key per entry); only for orderable nodes
    _order: dict[str, tuple[int, list[tuple[int, ...]]]] = field(default_factory=dict)
    # False for a complete base text (Tanakh/Mishnah/Bavli): a missing ref there is never a hole
    allow_hole_ranges: bool = True
    _kinds: dict[tuple[str, int], set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, title: str, rows) -> WorkIndex:
        """rows: iterable of (ref, node_index, indices, node_path, labels, segment_id)."""
        uniq = {}
        for ref, node_index, indices, node_path, labels, sid in rows:
            uniq.setdefault(
                ref, (node_index, tuple(indices), ", ".join(node_path), tuple(labels), sid)
            )
        ordered = sorted(uniq.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        idx = cls(title)
        for pos, (ref, (_, _, node, labels, sid)) in enumerate(ordered):
            idx.entries.append((ref, node, labels, sid))
            for n in range(len(labels) + 1):
                key = (node, labels[:n])
                lo, _hi = idx._prefix.get(key, (pos, pos))
                idx._prefix[key] = (lo, pos)
            if labels and _DAF_RE.match(labels[0]):
                # Sefaria: a bare daf ("Shabbat 117") means both sides (117a and 117b)
                for n in range(1, len(labels) + 1):
                    key = (node, (labels[0][:-1], *labels[1:n]))
                    lo, _hi = idx._prefix.get(key, (pos, pos))
                    idx._prefix[key] = (lo, pos)
        idx._build_order()
        return idx

    def _build_order(self) -> None:
        """Numeric keys per node, only where entries are contiguous and in numeric order."""
        groups: dict[str, list[int]] = {}
        for pos, entry in enumerate(self.entries):
            groups.setdefault(entry[1], []).append(pos)
        for node, poss in groups.items():
            keys = [labels_key(self.entries[p][2]) for p in poss]
            if any(k is None for k in keys) or poss[-1] - poss[0] + 1 != len(poss):
                continue
            if any(a > b for a, b in zip(keys, keys[1:], strict=False)):  # type: ignore[operator]
                continue
            self._order[node] = (poss[0], keys)  # type: ignore[assignment]

    def resolve(self, citation: str) -> list[int] | None:
        """Segment ids covered by `citation`; None if unparseable, [] if parsed but absent."""
        p = parse_citation(self.title, citation)
        if p is None:
            return None
        first = self._prefix.get((p.node, p.start))
        last = self._prefix.get((p.node, p.end)) if p.end is not None else None
        if first is not None and (p.end is None or last is not None):
            lo, hi = first
            if last is not None:
                hi = max(last[1], lo)
            return [e[3] for e in self.entries[lo : hi + 1]]
        if p.end is not None:
            return self._resolve_hole_range(p)
        return []

    def _label_kinds(self, node: str, level: int) -> set[str]:
        """Address kinds ('talmud' | 'int') of the existing labels of `node` at `level`."""
        cache = self._kinds
        if (node, level) not in cache:
            cache[(node, level)] = {
                k
                for _, n, labels, _ in self.entries
                if n == node and len(labels) > level and (k := _label_kind(labels[level]))
            }
        return cache[(node, level)]

    def _fits_node(self, node: str, labels: tuple[str, ...]) -> bool:
        """Every label has the address kind the node uses at that level (12a in an integer node
        does not: label_pos('12a') = 24 would otherwise land in chapter 24)."""
        for level, lab in enumerate(labels):
            kinds = self._label_kinds(node, level)
            if _label_kind(lab) not in kinds:
                return False
        return True

    def _interior_hole(self, node: str, keys: list, tup: tuple[str, ...]) -> bool:
        """True if `tup` is absent but sits *between* existing siblings under one parent: at its
        first missing level there is an existing sibling before it and one after it. A label
        after the last (or before the first) sibling is the end of the text, not a hole."""
        k = next((i for i in range(len(tup)) if (node, tup[: i + 1]) not in self._prefix), None)
        if k is None:
            return True  # exists
        key = labels_key(tup[: k + 1])
        if key is None:
            return False
        parent = key[:k]
        start = bisect_left(keys, parent)
        end = bisect_right(keys, (*parent, _INF))
        before = bisect_left(keys, key) > start
        after = bisect_right(keys, (*key, _INF)) < end
        return before and after

    def _resolve_hole_range(self, p: ParsedCitation) -> list[int]:
        """A range with an endpoint in an *interior hole* of the text (a segment that has no
        Hebrew text between two that do, e.g. 'Rashi on Shevuot 30a:1-36b:4'): the existing
        segments inside the cited range.

        Conservative: an endpoint counts as a hole only if, at the level where it is missing,
        it has an existing sibling before and after it under the same parent; one past the last
        (or before the first) sibling, e.g. 'Mishnah Chagigah 2:1-10', is left unresolved, as is
        an address of the wrong kind for the node (a daf in an integer-numbered node), and
        every work flagged as a complete base text (Tanakh, Mishnah, Bavli)."""
        o = self._order.get(p.node)
        ks = labels_key(p.start)
        ke = labels_key(p.end) if p.end is not None else None
        if not self.allow_hole_ranges or o is None or ks is None or ke is None or ks > ke:
            return []
        assert p.end is not None
        if not (self._fits_node(p.node, p.start) and self._fits_node(p.node, p.end)):
            return []
        base, keys = o
        first_labels = self.entries[base][2]
        if first_labels and _DAF_RE.match(first_labels[0]):
            if not (_DAF_RE.match(p.start[0]) and _DAF_RE.match(p.end[0])):
                return []  # bare daf ("117") mixed with a/b addressing: not handled here
        for tup in (p.start, p.end):
            if not self._interior_hole(p.node, keys, tup):
                return []
        lo = bisect_left(keys, ks)
        hi = bisect_right(keys, (*ke, _INF)) - 1
        if lo > hi or lo >= len(keys):
            return []
        return [e[3] for e in self.entries[base + lo : base + hi + 1]]
