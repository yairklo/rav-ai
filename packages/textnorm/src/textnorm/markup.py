"""Removal of Sefaria's inline HTML, with a map back to original code-point offsets.

Sefaria text embeds ``<b>``, ``<i data-commentator=...>`` (sometimes malformed), ``<br>``,
HTML entities and footnotes (``<sup class="footnote-marker">..</sup><i class="footnote">..</i>``).
``strip_sefaria_markup`` returns the visible text plus, for each visible char, the original
``(start, end)`` range that produced it (an entity's decoded chars all map to the whole entity).
"""

from __future__ import annotations

import html
import re

_PATTERN = re.compile(
    r'(?P<fn><sup class="footnote-marker">.*?</sup><i class="footnote">.*?</i>)'
    r"|(?P<br><br\s*/?>)"
    r"|(?P<tag></?[A-Za-z][^>]*>)"
    r"|(?P<ent>&(?:[A-Za-z]+|#[0-9]+|#[xX][0-9a-fA-F]+);)",
    re.S,
)


def strip_sefaria_markup(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return (visible_text, origin) with ``origin[i]`` = original range of ``visible_text[i]``."""
    out: list[str] = []
    origin: list[tuple[int, int]] = []
    pos = 0
    for m in _PATTERN.finditer(text):
        s, e = m.span()
        if s > pos:
            out.append(text[pos:s])
            origin.extend((k, k + 1) for k in range(pos, s))
        kind = m.lastgroup
        if kind == "br":
            out.append(" ")
            origin.append((s, e))
        elif kind == "ent":
            dec = html.unescape(m.group())
            if dec != m.group():
                out.append(dec)
                origin.extend((s, e) for _ in dec)
            else:  # unknown entity: keep literally
                out.append(m.group())
                origin.extend((k, k + 1) for k in range(s, e))
        # fn / tag: dropped entirely
        pos = e
    if pos < len(text):
        out.append(text[pos:])
        origin.extend((k, k + 1) for k in range(pos, len(text)))
    return "".join(out), origin
