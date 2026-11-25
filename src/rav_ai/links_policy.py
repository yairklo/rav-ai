"""Policy for Sefaria links: every one is a *candidate*, never evidence.

A Sefaria link (Mesorat HaShas, Ein Mishpat / Ner Mitzvah, commentary, anything else in the
bulk export) says "someone linked these two places". It is a pointer worth *checking*. It is
**not proof** of a halakhic connection between two passages, and it must never be used to
decide the identity of a discussion (nidon) or to establish that two sources discuss the same
question. Code that shows a link to a user or to a model must label it as a candidate; code
that decides identity must ignore edges entirely.

The rule is deliberately schema-free and **fail-closed**: in the ``edges`` table every edge
whose ``origin`` is not exactly ``'human'`` is a candidate (sefaria, heuristic, llm, an unknown
value, NULL, ``''``). ``confidence`` stays NULL and ``candidate_kind`` is derived from the
verbatim ``edge_type`` here, not stored. The harness (KL-2) and the evals (KL-3) import this
module so the rule lives in one place: ``candidate_label(origin)`` is the text they show next
to a link (replacing any local stub / SEFARIA_LABEL).

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
HUMAN_ORIGIN = "human"  # the only origin that is not a candidate

KIND_MESORAT_HASHAS = "mesorat_hashas"
KIND_EIN_MISHPAT = "ein_mishpat_ner_mitzvah"
KIND_NAVIGATION = "navigation"
KIND_OTHER = "sefaria_link"
KIND_AUTO = "auto_link"  # heuristic / llm / unknown origin

LABEL_SEFARIA = "קישור של ספריא: מועמד, לא ראיה"
LABEL_AUTO = "קישור אוטומטי: מועמד, לא ראיה"

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


def _origin(edge: Any) -> str | None:
    """The origin of ``edge``: a str/None, a Mapping with 'origin', or an object with .origin."""
    if edge is None or isinstance(edge, str):
        return edge
    if isinstance(edge, Mapping):
        return edge.get("origin")
    if hasattr(edge, "origin"):
        return edge.origin
    raise TypeError(f"is_candidate: unsupported type {type(edge).__name__}")


def is_candidate(edge: Any) -> bool:
    """Fail-closed: True unless the origin is exactly ``'human'``.

    ``edge`` is an origin (``str | None``), a Mapping with ``origin``, or an object with
    ``.origin``; any other type raises TypeError (never a silent False). True says the edge is
    a lead to verify, never proof; it is not a statement that the link is valid."""
    origin = _origin(edge)
    if origin is not None and not isinstance(origin, str):
        raise TypeError(f"origin must be str or None, got {type(origin).__name__}")
    return origin != HUMAN_ORIGIN


def candidate_kind(edge: Any) -> str | None:
    """The candidate_kind of a candidate edge, None only for a human edge.

    Sefaria edges map their (stripped, lowercased) ``edge_type``; every other candidate origin
    is ``auto_link``."""
    if not is_candidate(edge):
        return None
    if _origin(edge) != SEFARIA_ORIGIN:
        return KIND_AUTO
    et = edge.get("edge_type") if isinstance(edge, Mapping) else getattr(edge, "edge_type", None)
    return CANDIDATE_EDGE_TYPES.get((et or "").strip().lower(), KIND_OTHER)


def candidate_label(origin: Any) -> str:
    """Display label for a link of this origin: always says "candidate, not evidence"."""
    return LABEL_SEFARIA if _origin(origin) == SEFARIA_ORIGIN else LABEL_AUTO


@dataclass(frozen=True)
class CandidateLink:
    edge_id: int
    segment_a_id: int
    segment_b_id: int
    edge_type: str
    candidate_kind: str
    citation_a: str | None
    citation_b: str | None
    origin: str | None = None
    candidate: bool = True  # always: a link is never proof


def candidate_links(
    conn: Connection, segment_ids: Iterable[int], kinds: Iterable[str] | None = None
) -> list[CandidateLink]:
    """Non-human edges touching any of ``segment_ids`` (either side), as candidates.

    Any origin but ``'human'`` is returned (sefaria, heuristic, llm, unknown, NULL).

    ``kinds`` filters on candidate_kind (e.g. ``{KIND_MESORAT_HASHAS}``); None = all kinds.
    """
    ids = sorted(set(segment_ids))
    if not ids:
        return []
    rows = conn.execute(
        text("""SELECT id, segment_a_id, segment_b_id, edge_type, origin, citation_a, citation_b
                FROM edges
                WHERE origin IS DISTINCT FROM :o
                  AND (segment_a_id = ANY(:ids) OR segment_b_id = ANY(:ids))
                ORDER BY id"""),
        {"o": HUMAN_ORIGIN, "ids": ids},
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
                r["citation_a"], r["citation_b"], r["origin"],
            )
        )  # fmt: skip
    return out
