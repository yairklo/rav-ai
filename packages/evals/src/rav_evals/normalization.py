"""The verifier's normalizer: the shared ``textnorm`` library with the policy under which
``segment_text.text_b`` is stored (level B, plene="double", Sefaria markup stripped).

Quote and source go through the same function, so comparing against stored text is exact.
"""

from __future__ import annotations

from textnorm import normalize as _normalize


def normalize(text: str) -> str:
    return _normalize(text, "B", plene="double", markup="sefaria").text
