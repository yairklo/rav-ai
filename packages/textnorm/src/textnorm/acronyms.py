"""Acronym / abbreviation candidate detection with dictionary look-up (never expands text)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .gematria import parse_numeral
from .normalize import normalize

_PREFIXES = "ובהכלמשד"
_ACRONYM_RE = re.compile(r'(?<![\w"\'])([א-ת]+(?:"[א-ת]+)+)(?![\w"])')
_TRUNC_RE = re.compile(r"(?<![\w\"'])([א-ת]{1,6}')(?![\w\"])")


@dataclass(frozen=True)
class Expansion:
    text: str
    domain: str  # most likely domain: talmud | halakha | responsa | general
    source: str
    prior: dict[str, float] = field(default_factory=dict, compare=False, hash=False)
    # prior[domain] in 0..1: how common this expansion is in each domain (hand-estimated)


@dataclass(frozen=True)
class AcronymCandidate:
    span: tuple[int, int]  # in the ORIGINAL text
    text: str  # original slice
    norm: str  # level-A form, e.g. ר"ל
    kind: str  # "acronym" (gershayim) | "abbreviation" (geresh truncation, e.g. סי')
    prefix: str  # stripped prefix letters (ו, ה, ב...) when matched via the dictionary
    expansions: tuple[Expansion, ...]  # possible expansions; empty if unknown
    could_be_numeral: bool  # e.g. כ"א, תרע"א also parse as Hebrew numbers (ambiguous)

    @property
    def known(self) -> bool:
        return bool(self.expansions)


def _key(s: str) -> str:
    return normalize(s, "A").text.replace('"', "")


Dictionary = dict[str, tuple[Expansion, ...]]


def load_dictionary(path: str | Path | None = None) -> Dictionary:
    """Load the acronym JSON (bundled by default) into ``{key: expansions}``.

    Duplicate keys (e.g. two entries for ר"י) are merged in file order.
    """
    if path is None:
        raw = resources.files("textnorm").joinpath("data/acronyms.json").read_text("utf8")
    else:
        raw = Path(path).read_text("utf8")
    out: dict[str, list[Expansion]] = {}
    for e in json.loads(raw)["entries"]:
        lst = out.setdefault(_key(e["abbr"]), [])
        for x in e["expansions"]:
            ex = Expansion(x["text"], x["domain"], e["source"], dict(x["prior"]))
            if ex not in lst:
                lst.append(ex)
    return {k: tuple(v) for k, v in out.items()}


@lru_cache(maxsize=1)
def _default() -> Dictionary:
    return load_dictionary()


def lookup(token: str, dictionary: Dictionary | None = None) -> tuple[str, tuple[Expansion, ...]]:
    """Return ``(prefix, expansions)`` for an acronym token, stripping up to 2 prefix letters."""
    d = _default() if dictionary is None else dictionary
    k = _key(token)
    if k in d:
        return "", d[k]
    for n in (1, 2):
        if len(k) - n >= 2 and all(c in _PREFIXES for c in k[:n]) and k[n:] in d:
            return k[:n], d[k[n:]]
    return "", ()


DOMAINS = ("talmud", "halakha", "responsa", "general")


def rank_candidates(
    abbr: str, context_domain: str, dictionary: Dictionary | None = None
) -> list[Expansion]:
    """Order the possible expansions of ``abbr`` by their prior in ``context_domain``.

    Only sorts: nothing is chosen or expanded in text. ``context_domain`` is one of ``DOMAINS``.
    Ties keep dictionary order. Unknown abbreviations give ``[]``; clitic prefixes are handled
    as in ``find_acronyms``.
    """
    if context_domain not in DOMAINS:
        raise ValueError(f"context_domain must be one of {DOMAINS}, got {context_domain!r}")
    _, exps = lookup(abbr, dictionary)
    return sorted(exps, key=lambda x: -x.prior.get(context_domain, 0.0))


def _numeral_like(tok: str) -> bool:
    """True if the token (optionally after 1-2 clitic prefix letters) is a valid Hebrew numeral."""
    if parse_numeral(tok) is not None:
        return True
    return any(
        tok[:n].isalpha()
        and all(c in _PREFIXES for c in tok[:n])
        and parse_numeral(tok[n:]) is not None
        for n in (1, 2)
    )


def find_acronyms(text: str, dictionary: Dictionary | None = None) -> list[AcronymCandidate]:
    """Find acronym candidates (words with gershayim, e.g. ר"ל, ע"כ, מ"מ) and geresh
    abbreviations (e.g. סי', וכו'). Geresh-ended words are reported only when they are in the
    dictionary or have at most two letters (they are otherwise indistinguishable from a closing
    single quote). Works on niqqud/variant-quote text; offsets refer to ``text``.
    """
    norm = normalize(text, "A")
    found: list[AcronymCandidate] = []
    for regex, kind in ((_ACRONYM_RE, "acronym"), (_TRUNC_RE, "abbreviation")):
        for m in regex.finditer(norm.text):
            tok = m.group(1)
            prefix, exps = lookup(tok, dictionary)
            if kind == "abbreviation" and not exps and len(tok) - 1 > 2:
                continue
            s, e = norm.map_span(m.start(1), m.end(1))
            found.append(
                AcronymCandidate(
                    span=(s, e),
                    text=text[s:e],
                    norm=tok,
                    kind=kind,
                    prefix=prefix,
                    expansions=exps,
                    could_be_numeral=_numeral_like(tok),
                )
            )
    found.sort(key=lambda c: c.span)
    return found
