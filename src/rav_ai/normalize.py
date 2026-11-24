"""Normalization of raw Sefaria text into two derived forms, with offset maps back to the raw text.

Raw text is never modified. Forms:

A ("surface"):  HTML tags/entities/footnotes removed (``<br>`` -> space), niqqud + cantillation
                + bidi/zero-width marks removed, glyph variants unified (geresh/gershayim/quote
                variants -> ' and ", maqaf/dashes -> '-', presentation forms and ligatures
                decomposed, odd whitespace -> space), whitespace collapsed and trimmed.
                Spelling (ktiv male/haser) is NOT touched; abbreviations are NOT expanded.
B ("search"):   A, plus all punctuation removed (apostrophes/quotes deleted so ``רמב"ם`` ->
                ``רמבם``; maqaf/dashes become spaces). Whitespace-tokenizable.

Offset map: ordered chunks ``(n_start, r_start, n_len, r_len)``. A chunk with n_len == r_len is a
1:1 copy (char n_start+k <- raw r_start+k). Otherwise every normalized char of the chunk maps to
the whole raw span [r_start, r_start+r_len). Deleted raw text appears only as gaps between chunks.
``OffsetMap.to_raw(ns, ne)`` returns the raw half-open range covering normalized [ns, ne).
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from bisect import bisect_right
from dataclasses import dataclass

Chunk = tuple[int, int, int, int]  # n_start, r_start, n_len, r_len

_FOOTNOTE = r'<sup class="footnote-marker">.*?</sup><i class="footnote">.*?</i>'
_COMMON = [
    ("fn", _FOOTNOTE),
    ("br", r"<br\s*/?>"),
    ("tag", r"</?[A-Za-z][^>]*>"),
    ("ent", r"&(?:[A-Za-z]+|#[0-9]+|#[xX][0-9a-fA-F]+);"),
    ("marks", "[֑-ְ֯-ׇֽֿׁׂׅׄ]+"),
    ("invis", "[​-‏‪-‮⁠﻿ـ­]+"),
    ("pres", "[יִ-ﭏ]"),
    ("lig", "[װ-ײ]"),
    ("sq", "['‘’׳`´]"),
    ("dq", '["“”„״«»]'),
    ("dash", "[‐-―־−-]"),
    ("ws", r"[^\S ]"),
]
_PAT_A = re.compile("|".join(f"(?P<{k}>{v})" for k, v in _COMMON), re.S)
_PAT_B = re.compile("|".join(f"(?P<{k}>{v})" for k, v in [*_COMMON, ("punct", r"[^\w\s]|_")]), re.S)
_LIG = {"װ": "וו", "ױ": "וי", "ײ": "יי"}


def _html_unescape(s: str) -> str:
    import html

    return html.unescape(s)


def _repl_a(m: re.Match) -> str:
    k = m.lastgroup
    s = m.group()
    if k in ("fn", "tag", "marks", "invis"):
        return ""
    if k == "br":
        return " "
    if k == "ent":
        u = _html_unescape(s)
        return " " if u.isspace() else ("" if u == s else u)
    if k == "pres":
        d = unicodedata.normalize("NFKD", s)
        return "".join(c for c in d if not unicodedata.combining(c))
    if k == "lig":
        return _LIG[s]
    if k == "sq":
        return "'"
    if k == "dq":
        return '"'
    if k == "dash":
        return "-"
    if k == "ws":
        return " "
    return s


def _repl_b(m: re.Match) -> str:
    k = m.lastgroup
    if k in ("sq", "dq", "punct"):
        return ""
    if k == "dash":
        return " "
    if k == "ent":
        u = _repl_a(m)
        return u if (not u or u.isspace() or u.isalnum()) else ""
    return _repl_a(m)


def _add(chunks: list[Chunk], n: int, r: int, nlen: int, rlen: int) -> None:
    if nlen == rlen and chunks:
        pn, pr, pnl, prl = chunks[-1]
        if pnl == prl and pn + pnl == n and pr + prl == r:
            chunks[-1] = (pn, pr, pnl + nlen, prl + rlen)
            return
    chunks.append((n, r, nlen, rlen))


def _pass(raw: str, pat: re.Pattern, repl) -> tuple[str, list[Chunk]]:
    out: list[str] = []
    chunks: list[Chunk] = []
    n = pos = 0
    for m in pat.finditer(raw):
        s, e = m.span()
        if s > pos:
            _add(chunks, n, pos, s - pos, s - pos)
            out.append(raw[pos:s])
            n += s - pos
        rep = repl(m)
        if rep:
            _add(chunks, n, s, len(rep), e - s)
            out.append(rep)
            n += len(rep)
        pos = e
    if pos < len(raw):
        _add(chunks, n, pos, len(raw) - pos, len(raw) - pos)
        out.append(raw[pos:])
    return "".join(out), chunks


def _emit(out: list[Chunk], n: int, r: int, nl: int, rl: int, a: int, b: int) -> None:
    """Append the part [a, b) (old normalized coords) of chunk (n, r, nl, rl)."""
    if nl == rl:
        _add(out, 0, r + (a - n), b - a, b - a)
    else:
        out.append((0, r, b - a, rl))


_WS_RE = re.compile(r"^ +| +$| {2,}")


def _collapse_ws(text: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    dels: list[tuple[int, int]] = []
    for m in _WS_RE.finditer(text):
        s, e = m.span()
        if m.group().strip() == "" and (s == 0 or e == len(text)):
            dels.append((s, e))
        else:
            dels.append((s + 1, e))
    if not dels:
        return text, chunks
    new_chunks: list[Chunk] = []
    di = 0
    for n, r, nl, rl in chunks:
        ce = n + nl
        cur = n
        while di < len(dels) and dels[di][0] < ce:
            ds, de = dels[di]
            if ds > cur:
                _emit(new_chunks, n, r, nl, rl, cur, ds)
            cur = max(cur, min(de, ce))
            if de <= ce:
                di += 1
            else:
                break
        if cur < ce:
            _emit(new_chunks, n, r, nl, rl, cur, ce)
    # renumber normalized starts contiguously (merge 1:1 chunks that became adjacent)
    final: list[Chunk] = []
    n = 0
    for _n, r, nl, rl in new_chunks:
        _add(final, n, r, nl, rl)
        n += nl
    kept = []
    last = 0
    for ds, de in dels:
        kept.append(text[last:ds])
        last = de
    kept.append(text[last:])
    return "".join(kept), final


@dataclass
class Normalized:
    text: str
    offsets: OffsetMap


def normalize_a(raw: str) -> Normalized:
    t, ch = _pass(raw, _PAT_A, _repl_a)
    t, ch = _collapse_ws(t, ch)
    return Normalized(t, OffsetMap(ch))


def normalize_b(raw: str) -> Normalized:
    t, ch = _pass(raw, _PAT_B, _repl_b)
    t, ch = _collapse_ws(t, ch)
    return Normalized(t, OffsetMap(ch))


def _put_varint(buf: bytearray, v: int) -> None:
    while v >= 0x80:
        buf.append((v & 0x7F) | 0x80)
        v >>= 7
    buf.append(v)


class OffsetMap:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._starts = [c[0] for c in chunks]

    # --- mapping -------------------------------------------------------------------
    def to_raw(self, ns: int, ne: int) -> tuple[int, int]:
        """Raw [start, end) covering normalized [ns, ne). Requires 0 <= ns < ne <= len."""
        if ne <= ns:
            raise ValueError("empty range")
        return self._raw_start(ns), self._raw_end(ne - 1)

    def _chunk(self, i: int) -> Chunk:
        k = bisect_right(self._starts, i) - 1
        if k < 0 or i >= self.chunks[k][0] + self.chunks[k][2]:
            raise IndexError(i)
        return self.chunks[k]

    def _raw_start(self, i: int) -> int:
        n, r, nl, rl = self._chunk(i)
        return r + (i - n) if nl == rl else r

    def _raw_end(self, i: int) -> int:
        n, r, nl, rl = self._chunk(i)
        return r + (i - n) + 1 if nl == rl else r + rl

    # --- storage -------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        """zlib(varints: [raw gap, n_len, r_len] per chunk)."""
        buf = bytearray()
        prev_r_end = 0
        for _n, r, nl, rl in self.chunks:
            _put_varint(buf, r - prev_r_end)
            _put_varint(buf, nl)
            _put_varint(buf, rl)
            prev_r_end = r + rl
        return zlib.compress(bytes(buf), 6)

    @classmethod
    def from_bytes(cls, data: bytes) -> OffsetMap:
        raw = zlib.decompress(data)
        vals, shift, cur = [], 0, 0
        for b in raw:
            cur |= (b & 0x7F) << shift
            if b & 0x80:
                shift += 7
            else:
                vals.append(cur)
                cur, shift = 0, 0
        chunks: list[Chunk] = []
        n = r_end = 0
        for j in range(0, len(vals), 3):
            gap, nl, rl = vals[j : j + 3]
            r = r_end + gap
            chunks.append((n, r, nl, rl))
            n += nl
            r_end = r + rl
        return cls(chunks)
