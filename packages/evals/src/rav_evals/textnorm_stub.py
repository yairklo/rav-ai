"""Temporary text normalizer. To be replaced by the shared ``textnorm`` library.

The verifier takes the normalizer as a parameter, so swapping is a one-line change.
"""

from __future__ import annotations

import html
import re
import unicodedata

_TAGS = re.compile(r"<[^>]+>")
_MARKS = re.compile(r"[֑-ֽֿ-ׇ]")  # cantillation + nikud (not maqaf)
_NON_WORD = re.compile(r"[\W_]+")
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = _TAGS.sub(" ", text)
    text = _MARKS.sub("", text)
    text = _NON_WORD.sub(" ", text)  # punctuation, maqaf, gershayim -> space
    return " ".join(text.translate(_FINALS).lower().split())
