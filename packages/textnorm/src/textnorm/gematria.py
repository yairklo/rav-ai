"""Hebrew numerals (gematria) <-> int."""

from __future__ import annotations

import re

_VALUES = {
    "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
    "י": 10, "כ": 20, "ך": 20, "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50,
    "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90,
    "ק": 100, "ר": 200, "ש": 300, "ת": 400,
}  # fmt: skip

_STRIP = re.compile(r"[֑-ׇ\"'׳״“”‘’.]")


def _parse_below_1000(letters: str) -> int | None:
    if not letters:
        return None
    total = 0
    prev = 10_000
    tav_run = 0
    for ch in letters:
        v = _VALUES.get(ch)
        if v is None:
            return None
        if v == 400:
            tav_run += 1
            if tav_run > 2 or prev not in (10_000, 400):
                return None
        elif v >= prev:  # must be strictly decreasing (except repeated tav)
            return None
        prev = v
        total += v
    if total >= 1000 or to_gematria(total, quotes=False) != letters:
        return None  # only the canonical spelling counts: "בא" is a word, "ג" is 3
    return total


def parse_numeral(token: str) -> int | None:
    """Parse a Hebrew numeral such as כ"א, תרע"א, ה'תשפ"ד or digits. None if not valid.

    Strict: letters must be in descending value order (so "אב" is not a number), at most two
    tavs, below 1000 unless a thousands letter + geresh prefix is present.
    """
    token = token.strip()
    if token.isdigit():
        return int(token)
    thousands = 0
    m = re.match(r"^([א-ט])['׳]\s?(.+)$", token)
    if m:
        thousands = _VALUES[m.group(1)] * 1000
        token = m.group(2)
    letters = _STRIP.sub("", token).translate(str.maketrans("ךםןףץ", "כמנפצ"))
    if not letters or not all(c in _VALUES for c in letters):
        return None
    v = _parse_below_1000(letters)
    return None if v is None else v + thousands


_UNITS = "אבגדהוזחט"
_TENS = "יכלמנסעפצ"
_HUNDREDS = "קרשת"


def to_gematria(n: int, *, quotes: bool = True) -> str:
    """Format 1..999 as a Hebrew numeral (טו/טז convention), with gershayim if quotes."""
    if not 1 <= n <= 999:
        raise ValueError(n)
    out = ""
    while n >= 400:
        out += "ת"
        n -= 400
    if n >= 100:
        out += _HUNDREDS[n // 100 - 1]
        n %= 100
    if n in (15, 16):
        out += "ט" + ("ו" if n == 15 else "ז")
        n = 0
    if n >= 10:
        out += _TENS[n // 10 - 1]
        n %= 10
    if n:
        out += _UNITS[n - 1]
    if quotes:
        out = out + "'" if len(out) == 1 else out[:-1] + '"' + out[-1]
    return out
