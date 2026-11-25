"""Deterministic evidence location. The LLM returns quotes; the code finds them.

quote --rav_evals.normalization.normalize--> level-B form --find in segment_text.text_b-->
(start, end) in text_b --OffsetMap.to_raw--> span in the RAW text, plus a 32-char prefix/suffix
(form B) so the span can be found again after the text changes (``reanchor``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rav_ai.normalize import OffsetMap
from rav_evals.normalization import normalize

_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")
CONTEXT_CHARS = 32


@dataclass
class SegText:
    """One segment in the version chosen as input."""

    segment_id: int
    version_id: int
    ref: str
    raw_hash: str
    text_a: str
    text_b: str
    offsets_b: bytes
    work_id: int | None = None
    offsets_a: bytes = b""
    text_raw: str = ""


@dataclass
class Located:
    status: str  # located | not_found | ambiguous | out_of_scope
    segment_id: int | None = None
    version_id: int | None = None
    span_start: int | None = None  # RAW text offsets
    span_end: int | None = None
    quote_prefix: str | None = None
    quote_suffix: str | None = None
    raw_hash: str | None = None


def fragments(quote: str) -> list[str]:
    """Normalized fragments of a quote; "..." separates ordered fragments."""
    frags = [normalize(f).strip() for f in _ELLIPSIS.split(quote)]
    return [f for f in frags if f]


def _word_starts(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r"\S+", text)]


def _occurrences(frag: str, text: str, start_from: int = 0) -> list[int]:
    """Start offsets of ``frag`` in ``text`` that begin and end on a word boundary."""
    rx = re.compile(rf"(?<!\w){re.escape(frag)}(?!\w)")
    out, pos = [], start_from
    while True:
        m = rx.search(text, pos)
        if m is None:
            return out
        out.append(m.start())
        pos = m.start() + 1


def _complete(frags: list[str], first: int, text: str) -> int | None:
    """End offset of the ordered fragments starting at ``first`` (greedy), or None."""
    cur = first + len(frags[0])
    for f in frags[1:]:
        nxt = _occurrences(f, text, cur)
        if not nxt:
            return None
        cur = nxt[0] + len(f)
    return cur


def is_short(frags: list[str], min_words: int, min_chars: int) -> bool:
    joined = " ".join(frags)
    return len(joined.split()) < min_words or len(joined) < min_chars


def find_span(
    quote: str,
    text_b: str,
    *,
    w_start: int | None = None,
    start_from: int = 0,
    min_words: int = 2,
    min_chars: int = 6,
) -> tuple[int, int, bool] | None:
    """(start, end, unique) of the quote in ``text_b`` or None.

    Every fragment must begin and end on a word boundary ("רב" is not found inside "רבא");
    fragments must occur in order. ``unique`` is False for a quote shorter than the threshold
    (fewer than ``min_words`` words or ``min_chars`` chars) and when more than one ordered
    occurrence of the whole quote exists. ``w_start`` (numbered anchors) picks the occurrence
    at/after that word index and counts as unique.
    """
    frags = fragments(quote)
    if not frags:
        return None
    pos = start_from
    if w_start is not None:
        ws = _word_starts(text_b)
        if 0 <= w_start < len(ws):
            pos = max(pos, ws[w_start])
    firsts = _occurrences(frags[0], text_b, pos)
    if not firsts and w_start is not None:  # the index was wrong: fall back to a plain search
        firsts = _occurrences(frags[0], text_b, start_from)
        w_start = None
    hits = [(f, e) for f in firsts if (e := _complete(frags, f, text_b)) is not None]
    if not hits:
        return None
    first, end = hits[0]
    unique = (w_start is not None or len(hits) == 1) and not is_short(frags, min_words, min_chars)
    return first, end, unique


def locate(
    quote: str,
    seg: SegText | None,
    *,
    w_start: int | None = None,
    min_words: int = 2,
    min_chars: int = 6,
) -> Located:
    """Locate a quote in the text of ``seg`` (None = the label was not in the input)."""
    if seg is None:
        return Located("out_of_scope")
    hit = find_span(quote, seg.text_b, w_start=w_start, min_words=min_words, min_chars=min_chars)
    if hit is None:
        return Located("not_found", seg.segment_id, seg.version_id, raw_hash=seg.raw_hash)
    s, e, unique = hit
    rs, re_ = OffsetMap.from_bytes(seg.offsets_b).to_raw(s, e)
    return Located(
        "located" if unique else "ambiguous",
        seg.segment_id,
        seg.version_id,
        rs,
        re_,
        seg.text_b[max(0, s - CONTEXT_CHARS) : s],
        seg.text_b[e : e + CONTEXT_CHARS],
        seg.raw_hash,
    )


def reanchor(
    quote: str,
    prefix: str | None,
    suffix: str | None,
    old_raw_hash: str | None,
    seg: SegText,
) -> tuple[Located | None, bool]:
    """Re-find a stored span after the segment text may have changed.

    Returns (located, stale). If ``raw_hash`` is unchanged nothing is done. Otherwise the quote
    is searched again; among several hits the one whose prefix/suffix match wins. ``stale`` is
    True only when the quote cannot be found at all.
    """
    if old_raw_hash is not None and old_raw_hash == seg.raw_hash:
        return None, False
    frags = fragments(quote)
    if not frags:
        return None, True
    # all occurrences of the first fragment, then complete the rest in order
    cands: list[tuple[int, int]] = []
    pos = 0
    while True:
        hit = find_span(quote, seg.text_b, start_from=pos)
        if hit is None:
            break
        cands.append((hit[0], hit[1]))
        pos = hit[0] + 1
    if not cands:
        return None, True

    def score(c: tuple[int, int]) -> int:
        s, e = c
        sc = 0
        if prefix and seg.text_b[max(0, s - len(prefix)) : s] == prefix:
            sc += 1
        if suffix and seg.text_b[e : e + len(suffix)] == suffix:
            sc += 1
        return sc

    best = max(cands, key=score)
    s, e = best
    rs, re_ = OffsetMap.from_bytes(seg.offsets_b).to_raw(s, e)
    return (
        Located(
            "located" if len(cands) == 1 or score(best) > 0 else "ambiguous",
            seg.segment_id,
            seg.version_id,
            rs,
            re_,
            seg.text_b[max(0, s - CONTEXT_CHARS) : s],
            seg.text_b[e : e + CONTEXT_CHARS],
            seg.raw_hash,
        ),
        False,
    )
