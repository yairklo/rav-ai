"""Normalization of Hebrew/Aramaic text with a character-level offset map.

Every output character carries ``(start, end)``: the half-open range of code points in the
*original* string that produced it. Offsets are Python ``str`` indices (code points).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

Level = Literal["A", "B", "a", "b"]
Plene = Literal["none", "double", "vav"]

_FINALS = str.maketrans("ךםןףץ", "כמנפצ")


# Hebrew points / cantillation (combining marks) that attach to the preceding letter.
def _is_hebrew_mark(cp: int) -> bool:
    return (0x0591 <= cp <= 0x05BD or cp == 0x05BF or cp in (0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7)
            or cp == 0x034F)  # fmt: skip


_GERESH = set("'׳‘’‛′ʼ`´ʹ")
_GERSHAYIM = set('"״“”„‟″')
_DASHES = set("-־‐‑‒–—―−﹣－")

_HEB_LETTER = range(0x05D0, 0x05EB)


def _is_letter(ch: str) -> bool:
    return ord(ch) in _HEB_LETTER


@dataclass(frozen=True)
class NormResult:
    """Normalized text plus offset map back to the original."""

    original: str
    text: str
    spans: tuple[tuple[int, int], ...]
    level: str

    def __post_init__(self) -> None:
        assert len(self.text) == len(self.spans)

    def map_span(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        """Map a half-open range of the normalized text to the original text.

        The result runs from the first original char of ``text[norm_start]`` to the end of
        ``text[norm_end - 1]`` (including combining marks attached to it), so it is the
        smallest original slice that contains every char that produced the range. It may also
        contain original chars that were dropped in between (diacritics, controls).
        An empty range maps to an empty range at the corresponding original position.
        """
        n = len(self.text)
        if not (0 <= norm_start <= norm_end <= n):
            raise ValueError(f"span ({norm_start}, {norm_end}) outside normalized text of {n}")
        if norm_start == norm_end:
            pos = (
                self.spans[norm_start][0]
                if norm_start < n
                else (self.spans[-1][1] if n else len(self.original))
            )
            return pos, pos
        return self.spans[norm_start][0], self.spans[norm_end - 1][1]

    def original_slice(self, norm_start: int, norm_end: int) -> str:
        s, e = self.map_span(norm_start, norm_end)
        return self.original[s:e]


def _stage_a(text: str) -> list[list]:
    """Per-character pass. Returns mutable [char, start, end] items."""
    items: list[list] = []
    for i, raw in enumerate(text):
        cp = ord(raw)
        if 0xFB1D <= cp <= 0xFB4F:
            expanded = unicodedata.normalize("NFKD", raw)
        elif cp > 127 and raw not in _GERESH and raw not in _GERSHAYIM:
            expanded = unicodedata.normalize("NFKC", raw)
        else:
            expanded = raw
        for ch in expanded:
            c = ord(ch)
            if _is_hebrew_mark(c):
                if items and items[-1][2] <= i + 1:
                    items[-1][2] = i + 1  # attach mark to its base letter's span
                continue
            if c == 0x05C0:  # paseq
                continue
            cat = unicodedata.category(ch)
            if cat in ("Cf", "Cc", "Cs", "Co") and not ch.isspace():
                continue
            if ch.isspace():
                ch = " "
            elif ch in _GERESH:
                ch = "'"
            elif ch in _GERSHAYIM:
                ch = '"'
            elif ch in _DASHES:
                ch = "-"
            elif c == 0x05C3:  # sof pasuq
                ch = ":"
            elif cat == "Mn" and 0x0300 <= c <= 0x036F:
                # stray Latin combining mark: keep attached, drop
                continue
            else:
                ch = ch.translate(_FINALS)
                if not _is_letter(ch):
                    low = ch.lower()
                    ch = low if len(low) == 1 else ch
            items.append([ch, i, i + 1])
    return items


def _stage_b_punct(items: list[list]) -> list[list]:
    out: list[list] = []
    for it in items:
        ch = it[0]
        if ch in ("'", '"'):
            continue  # geresh/gershayim/quotes vanish without splitting words (ר"ל -> רל)
        if ch == " ":
            out.append(it)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S")):
            out.append([" ", it[1], it[2]])
        else:
            out.append(it)
    return out


def _words(items: list[list]):
    """Yield (lo, hi) index ranges of maximal Hebrew-letter runs."""
    lo = None
    for k, it in enumerate(items):
        if _is_letter(it[0]):
            if lo is None:
                lo = k
        elif lo is not None:
            yield lo, k
            lo = None
    if lo is not None:
        yield lo, len(items)


def _plene(items: list[list], mode: str) -> list[list]:
    if mode == "none":
        return items
    drop: set[int] = set()
    for lo, hi in _words(items):
        w = [items[k][0] for k in range(lo, hi)]
        for j in range(1, len(w)):
            # collapse doubled vav/yod: מצווה -> מצוה, ייתכן -> יתכן
            if w[j] == w[j - 1] and w[j] in "וי":
                drop.add(lo + j)
        if mode == "vav":
            kept = [j for j in range(len(w)) if lo + j not in drop]
            letters = [w[j] for j in kept]
            if len(letters) >= 4:
                for pos in range(1, len(letters) - 1):
                    if (
                        letters[pos] == "ו"
                        and letters[pos - 1] not in "וי"
                        and (letters[pos + 1] not in "וי")
                    ):
                        drop.add(lo + kept[pos])
    return [it for k, it in enumerate(items) if k not in drop]


def _collapse(items: list[list]) -> list[list]:
    out: list[list] = []
    for it in items:
        if it[0] == " ":
            if not out:
                continue
            if out[-1][0] == " ":
                out[-1][2] = max(out[-1][2], it[2])
                continue
        out.append(it)
    while out and out[-1][0] == " ":
        out.pop()
    return out


def normalize(text: str, level: Level = "A", *, plene: Plene = "double") -> NormResult:
    """Normalize ``text`` and return it with its offset map.

    Level A: strip niqqud, cantillation and control chars; fold final letters; unify hyphens/
    maqaf (-), geresh (') and gershayim/quotes ("), whitespace (single space, trimmed); lowercase
    non-Hebrew letters; NFKC-fold compatibility forms (presentation forms, NBSP...).

    Level B: A plus punctuation removal (geresh/gershayim/quotes vanish without splitting the
    word: ר"ל -> רל; other punctuation becomes a space) and plene/defective folding, chosen by
    ``plene``: "none"; "double" (default; collapse doubled ו/י: מצווה->מצוה, ייתכן->יתכן; no
    meaningful loss of precision); "vav" (also drop a medial ו in words of >=4 letters:
    כתוב->כתב, but also מצוה->מצה, so lossy).
    """
    lv = level.upper()
    if lv not in ("A", "B"):
        raise ValueError(f"level must be 'A' or 'B', got {level!r}")
    if plene not in ("none", "double", "vav"):
        raise ValueError(f"bad plene mode {plene!r}")
    items = _stage_a(text)
    if lv == "B":
        items = _stage_b_punct(items)
        items = _plene(items, plene)
    items = _collapse(items)
    return NormResult(
        original=text,
        text="".join(it[0] for it in items),
        spans=tuple((it[1], it[2]) for it in items),
        level=lv,
    )
