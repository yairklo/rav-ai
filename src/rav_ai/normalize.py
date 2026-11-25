"""Thin wrapper over the shared ``textnorm`` library (the only normalization implementation).

Stored forms (Sefaria markup removed first, offsets always point into the untouched raw text):
  A = textnorm level "A"            (no niqqud/te'amim, final letters folded, glyphs unified)
  B = textnorm level "B", plene="double"  (A + no punctuation + doubled vav/yod collapsed)
No ``plene="vav"`` (lossy), no abbreviation expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

from textnorm import decode_spans, encode_spans, normalize

Spans = tuple[tuple[int, int], ...]


class OffsetMap:
    """Normalized char index -> raw char range, (de)serializable via textnorm.offsets."""

    def __init__(self, spans: Spans):
        self.spans = spans

    def to_raw(self, ns: int, ne: int) -> tuple[int, int]:
        """Raw [start, end) covering normalized [ns, ne) (0 <= ns < ne <= len)."""
        if not 0 <= ns < ne <= len(self.spans):
            raise ValueError(f"range ({ns}, {ne}) outside normalized text of {len(self.spans)}")
        return self.spans[ns][0], self.spans[ne - 1][1]

    def to_bytes(self) -> bytes:
        return encode_spans(self.spans)

    @classmethod
    def from_bytes(cls, data: bytes) -> OffsetMap:
        return cls(decode_spans(data))


@dataclass
class Normalized:
    text: str
    offsets: OffsetMap


def normalize_a(raw: str) -> Normalized:
    r = normalize(raw, "A", markup="sefaria")
    return Normalized(r.text, OffsetMap(r.spans))


def normalize_b(raw: str) -> Normalized:
    r = normalize(raw, "B", plene="double", markup="sefaria")
    return Normalized(r.text, OffsetMap(r.spans))
