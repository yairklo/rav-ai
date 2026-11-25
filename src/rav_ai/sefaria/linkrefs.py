"""Resolve Sefaria link citations (segments, sections, ranges) to concrete refs of a work."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ADDR = r"[0-9]+[ab]?(?:[:.][0-9]+[ab]?)*"
_DAF_RE = re.compile(r"^[0-9]+[ab]$")
_REST_RE = re.compile(rf"^(?:, (?P<node>.+?))?(?: (?P<start>{_ADDR})(?:-(?P<end>{_ADDR}))?)?$")


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
        return idx

    def resolve(self, citation: str) -> list[int] | None:
        """Segment ids covered by `citation`; None if unparseable, [] if parsed but absent."""
        p = parse_citation(self.title, citation)
        if p is None:
            return None
        first = self._prefix.get((p.node, p.start))
        if first is None:
            return []
        lo, hi = first
        if p.end is not None:
            last = self._prefix.get((p.node, p.end))
            if last is None:
                return []
            hi = max(last[1], lo)
        return [e[3] for e in self.entries[lo : hi + 1]]
