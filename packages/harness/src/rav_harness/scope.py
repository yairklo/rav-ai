"""Pilot scope: which refs the knowledge layer has ingested and checked (config/harness.toml
[scope]). Pure functions; the DB-backed gate lives in knowledge_tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from textnorm import find_citations

from .config import ScopeRule

_GEMATRIA = {
    **dict(zip("אבגדהוזחט", range(1, 10), strict=True)),
    **dict(zip("יכלמנסעפצ", range(10, 100, 10), strict=True)),
    **dict(zip("קרשת", (100, 200, 300, 400), strict=True)),
    "ך": 20,
    "ם": 40,
    "ן": 50,
    "ף": 80,
    "ץ": 90,
}
_QUOTES = str.maketrans("", "", "\"'״׳")


def hebrew_number(s: str) -> int | None:
    """תרעא -> 671, או תרע"א. None when not a numeral."""
    letters = s.translate(_QUOTES)
    if not letters or any(c not in _GEMATRIA for c in letters):
        return None
    return sum(_GEMATRIA[c] for c in letters)


_SA_BOOKS = {
    'שו"ע': "Shulchan Arukh",
    "שולחן ערוך": "Shulchan Arukh",
    "טור": "Tur",
    "בית יוסף": "Beit Yosef",
    'ב"י': "Beit Yosef",
}
_SA_RE = re.compile(
    r"(?P<book>שו[\"״]ע|שולחן ערוך|טור|בית יוסף|ב[\"״]י)\s*"
    r"(?:או[\"״]ח|אורח חיים)\s*(?:סי'|סימן)?\s*(?P<num>[א-ת\"״']{1,6})(?![א-ת])"
)


def extract_refs(question: str) -> list[str]:
    """Canonical refs the question mentions: textnorm citations plus Tur/BY/SA Orach Chayim."""
    refs: list[str] = []
    canonical = question.strip()
    if re.fullmatch(r"[A-Z][A-Za-z' ,\-]* \d+[ab]?(?::\d+)*", canonical):  # already a ref
        refs.append(canonical)
    for c in find_citations(question):
        if c.id and c.id not in refs:
            refs.append(c.id)
    for m in _SA_RE.finditer(question.replace("״", '"')):
        n = hebrew_number(m["num"])
        book = _SA_BOOKS.get(m["book"].replace("״", '"'))
        if n and book:
            ref = f"{book}, Orach Chayim {n}"
            if ref not in refs:
                refs.append(ref)
    return refs


_DAF_RE = re.compile(r"^(?P<work>.+?) (?P<n>\d+)(?P<side>[ab])?(?:[:.\-].*)?$")


def _daf_key(n: int, side: str | None) -> tuple[int, int]:
    return (n, 0 if side in (None, "", "a") else 1)


def _in_rule(work: str, n: int, side: str | None, rule: ScopeRule) -> bool:
    if rule.work != work:
        return False
    if rule.kind == "daf":
        lo = _split_daf(str(rule.lo))
        hi = _split_daf(str(rule.hi))
        if side is None:  # whole daf: in scope when either amud is
            return lo[0] <= n <= hi[0]
        return _daf_key(*lo) <= _daf_key(n, side) <= _daf_key(*hi)
    return int(rule.lo) <= n <= int(rule.hi)


def _split_daf(s: str) -> tuple[int, str | None]:
    m = re.fullmatch(r"(\d+)([ab])?", s)
    if not m:
        raise ValueError(f"bad daf {s!r}")
    return int(m[1]), m[2]


def ref_in_scope(ref: str, rules: tuple[ScopeRule, ...]) -> bool:
    m = _DAF_RE.match(ref.strip())
    if not m:
        return False
    return any(_in_rule(m["work"], int(m["n"]), m["side"], r) for r in rules)


def scope_key_in_scope(scope_key: str, rules: tuple[ScopeRule, ...]) -> bool:
    """ "Shabbat 21a-24a" (kl scope key): both ends must be inside the pilot scope."""
    m = re.match(r"^(?P<work>.+?) (?P<a>\d+[ab]?)(?:-(?P<b>\d+[ab]?))?$", scope_key.strip())
    if not m:
        return False
    work = m["work"]
    return all(
        any(_in_rule(work, *_split_daf(end), rule) for rule in rules)
        for end in (m["a"], m["b"] or m["a"])
    )


@dataclass
class ScopeGate:
    """The deterministic pre-loop result: stored in the run and shown to the model."""

    status: str  # in_scope | partial | out_of_scope
    refs: list[str] = field(default_factory=list)
    refs_in_scope: list[str] = field(default_factory=list)
    refs_out_of_scope: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    unchecked_core_books: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    practical: bool = False
    # work title -> name shown in the answer text (he_title when there is one)
    unchecked_display: dict[str, str] = field(default_factory=dict)
    # kl scope keys the gate confirmed as inside the pilot scope (from in-scope issues and
    # refs); empty when out_of_scope. Not the keys of every item the run touched.
    scope_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "refs": self.refs,
            "refs_in_scope": self.refs_in_scope,
            "refs_out_of_scope": self.refs_out_of_scope,
            "issues": self.issues,
            "unchecked_core_books": self.unchecked_core_books,
            "unchecked_display": self.unchecked_display,
            "scope_keys": self.scope_keys,
            "reasons": self.reasons,
            "practical": self.practical,
        }


def is_practical(question: str, pattern: re.Pattern) -> bool:
    """Practical (halakha lemaase) question: ``[knowledge.phrases] practical_regex``."""
    return bool(pattern.search(question))


def key_covers_ref(scope_key: str, ref: str) -> bool:
    """Does a kl scope key ("Shabbat 21a-24a") cover ``ref`` ("Shabbat 22b:3", "Shabbat 22")?"""
    km = re.match(r"^(?P<work>.+?) (?P<a>\d+[ab]?)(?:-(?P<b>\d+[ab]?))?$", scope_key.strip())
    rm = _DAF_RE.match(ref.strip())
    if not km or not rm or km["work"] != rm["work"]:
        return False
    lo, hi = _split_daf(km["a"]), _split_daf(km["b"] or km["a"])
    n, side = int(rm["n"]), rm["side"]
    if side is None:
        return lo[0] <= n <= hi[0]
    return _daf_key(*lo) <= _daf_key(n, side) <= _daf_key(*hi)
