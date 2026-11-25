"""Policy for Sefaria links: every one is a *candidate*, never evidence.

A Sefaria link (Mesorat HaShas, Ein Mishpat / Ner Mitzvah, commentary, anything else in the
bulk export) says "someone linked these two places". It is a pointer worth *checking*. It is
**not proof** of a halakhic connection between two passages, and it must never be used to
decide the identity of a discussion (nidon) or to establish that two sources discuss the same
question. Code that shows a link to a user or to a model must label it as a candidate; code
that decides identity must ignore edges entirely.

The rule is deliberately schema-free: ``origin == 'sefaria'`` means candidate, ``confidence``
stays NULL, and ``candidate_kind`` is derived from the verbatim ``edge_type`` here, not stored.
The harness (KL-2) and the evals (KL-3) import this module so the rule lives in one place.

``commentary``/``targum`` (and their super-commentary relatives) are *navigation only*: they
say where a text sits relative to its base text.  Mesorat HaShas and Ein Mishpat / Ner Mitzvah
get explicit kinds because they look like halakhic parallels and are the easiest to over-trust.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

SEFARIA_ORIGIN = "sefaria"

KIND_MESORAT_HASHAS = "mesorat_hashas"
KIND_EIN_MISHPAT = "ein_mishpat_ner_mitzvah"
KIND_NAVIGATION = "navigation"
KIND_OTHER = "sefaria_link"

# Verbatim Sefaria "Conection Type" strings (taken from the export, not guessed) -> kind.
CANDIDATE_EDGE_TYPES: dict[str, str] = {
    "mesorat hashas": KIND_MESORAT_HASHAS,
    "ein mishpat / ner mitsvah": KIND_EIN_MISHPAT,
    "ein mishpat": KIND_EIN_MISHPAT,
    "commentary": KIND_NAVIGATION,
    "targum": KIND_NAVIGATION,
    "super_commentary": KIND_NAVIGATION,
    "super commentary": KIND_NAVIGATION,
}

NAVIGATION_KINDS = frozenset({KIND_NAVIGATION})
EXPLICIT_KINDS = frozenset({KIND_MESORAT_HASHAS, KIND_EIN_MISHPAT})


def _get(edge: Any, key: str) -> Any:
    if isinstance(edge, Mapping):
        return edge.get(key)
    return getattr(edge, key, None)


def candidate_kind(edge: Any) -> str | None:
    """The candidate_kind of an edge (mapping or object with origin/edge_type), else None."""
    if not is_candidate(edge):
        return None
    return CANDIDATE_EDGE_TYPES.get((_get(edge, "edge_type") or "").strip(), KIND_OTHER)


def is_candidate(edge: Any) -> bool:
    """True for every Sefaria-origin edge. This is *not* a statement that the link is valid:
    it says the edge is a lead to verify, never proof. Other origins (heuristic, llm) have
    their own policy and are not covered by this function."""
    return _get(edge, "origin") == SEFARIA_ORIGIN


@dataclass(frozen=True)
class CandidateLink:
    edge_id: int
    segment_a_id: int
    segment_b_id: int
    edge_type: str
    candidate_kind: str
    citation_a: str | None
    citation_b: str | None
    candidate: bool = True  # always: a link is never proof


def candidate_links(
    conn: Connection, segment_ids: Iterable[int], kinds: Iterable[str] | None = None
) -> list[CandidateLink]:
    """Sefaria edges touching any of ``segment_ids`` (either side), as candidates.

    ``kinds`` filters on candidate_kind (e.g. ``{KIND_MESORAT_HASHAS}``); None = all kinds.
    """
    ids = sorted(set(segment_ids))
    if not ids:
        return []
    rows = conn.execute(
        text("""SELECT id, segment_a_id, segment_b_id, edge_type, origin, citation_a, citation_b
                FROM edges
                WHERE origin = :o AND (segment_a_id = ANY(:ids) OR segment_b_id = ANY(:ids))
                ORDER BY id"""),
        {"o": SEFARIA_ORIGIN, "ids": ids},
    ).mappings()
    want = set(kinds) if kinds is not None else None
    out = []
    for r in rows:
        kind = candidate_kind(r)
        if kind is None or (want is not None and kind not in want):
            continue
        out.append(
            CandidateLink(
                r["id"], r["segment_a_id"], r["segment_b_id"], r["edge_type"], kind,
                r["citation_a"], r["citation_b"],
            )
        )  # fmt: skip
    return out
