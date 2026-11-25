"""What the verifier needs to know about the knowledge layer (V5-V10).

The checks read the ``kl_*`` tables (and ``edges``); when a table is absent (a database before
migration 0005, or a recorded fixture that did not snapshot it) the check is *skipped* and the
skip is reported, never silently passed.

``InMemoryKnowledgeLookup`` is the snapshot format stored in recorded fixtures; the Postgres
implementation reads the live tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

try:  # the shared link policy (ingest PR 3); never trusted to be MORE open than ours
    from rav_ai.links_policy import is_candidate as _policy_is_candidate
except ImportError:  # pragma: no cover - depends on which branch is merged first
    _policy_is_candidate = None

# origins of a knowledge-layer relation / link (its own domain, not the edges policy)
KL_ORIGIN_RANK = {"llm": 0, "judge": 1, "human": 2}
KL_TABLES = ("kl_relations", "kl_links")

# coverage statuses that mean "we did not check this work for this scope"
UNCHECKED_COVERAGE = ("not_ingested", "unknown")


def is_candidate(edge: Any) -> bool:
    """Fail closed: an ``edges`` row is a candidate (a lead to verify, never proof) unless a
    human made it - sefaria, heuristic, llm, an unknown origin and NULL are all candidates.

    The shared policy can only ADD candidates here, never remove one."""
    # TODO(links_policy): align with rav_ai.links_policy once it is fail-closed (ingest PR 3)
    origin = edge.get("origin") if isinstance(edge, dict) else getattr(edge, "origin", None)
    policy = bool(_policy_is_candidate(edge)) if _policy_is_candidate is not None else False
    return policy or origin != "human"


class KnowledgeLookup(Protocol):
    def has_table(self, name: str) -> bool: ...

    def work_role(self, ref: str) -> str | None:
        """kl_sources.role (primary|lens|witness|reference_only) of the work holding ``ref``;
        None when the work is not registered."""
        ...

    def edge(self, edge_id: int) -> dict[str, Any] | None:
        """{"origin", "edge_type", "a_ref", "b_ref"} of an ``edges`` row, None if absent."""
        ...

    def kl_row_exists(self, table: str, row_id: int) -> bool:
        """Does ``kl_relations`` / ``kl_links`` have this id?"""
        ...

    def work_of(self, ref: str) -> str | None:
        """Title of the work holding ``ref`` (None if the ref is unknown)."""
        ...

    def witness_exists(self, witness_id: int) -> bool: ...

    def unchecked_core_works(self, scope_keys: Iterable[str]) -> dict[str, set[str]]:
        """Core works (kl_sources.weight_tier='core') NOT checked for any of the scopes:
        {title: {title, he_title, ...names}}."""
        ...

    def checked_core_names(self, scope_keys: Iterable[str]) -> set[str]:
        """All names of core works that WERE checked for every scope (V8 fidelity)."""
        ...


class InMemoryKnowledgeLookup:
    """Snapshot form (fixtures, tests)::

    {"tables": ["kl_sources", "kl_coverage", "kl_witnesses", "edges"],
     "ref_work": {"Ref 1:1": "Work"},
     "sources": {"Work": {"role": "primary", "weight_tier": "core", "names": ["..."]}},
     "coverage": {"Scope 1a-2b": {"Work": "covered"}},
     "edges": {"17": {"origin": "sefaria", "edge_type": "mesorat hashas",
                      "a_ref": "Ref 1:1", "b_ref": "Other 2:3"}},
     "kl_relations": [4], "kl_links": [],
     "witnesses": [3]}
    """

    def __init__(self, data: dict[str, Any] | None = None):
        d = data or {}
        self.tables = set(d.get("tables", []))
        self.ref_work: dict[str, str] = dict(d.get("ref_work", {}))
        self.sources: dict[str, dict[str, Any]] = dict(d.get("sources", {}))
        self.coverage: dict[str, dict[str, str]] = dict(d.get("coverage", {}))
        self.edges = {int(k): v for k, v in d.get("edges", {}).items()}
        self.witnesses = {int(w) for w in d.get("witnesses", [])}
        self.kl_rows = {t: {int(i) for i in d.get(t, [])} for t in KL_TABLES}

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def work_role(self, ref: str) -> str | None:
        work = self.ref_work.get(ref)
        src = self.sources.get(work) if work else None
        return src.get("role") if src else None

    def edge(self, edge_id: int) -> dict[str, Any] | None:
        return self.edges.get(edge_id)

    def kl_row_exists(self, table: str, row_id: int) -> bool:
        return row_id in self.kl_rows.get(table, set())

    def work_of(self, ref: str) -> str | None:
        return self.ref_work.get(ref)

    def witness_exists(self, witness_id: int) -> bool:
        return witness_id in self.witnesses

    def unchecked_core_works(self, scope_keys: Iterable[str]) -> dict[str, set[str]]:
        keys = list(scope_keys)
        out: dict[str, set[str]] = {}
        for title, src in self.sources.items():
            if src.get("weight_tier") != "core":
                continue
            statuses = [self.coverage.get(k, {}).get(title, "unknown") for k in keys]
            if any(st in UNCHECKED_COVERAGE for st in statuses):
                out[title] = {title, *src.get("names", [])}
        return out

    def checked_core_names(self, scope_keys: Iterable[str]) -> set[str]:
        unchecked = self.unchecked_core_works(scope_keys)
        return {
            n
            for title, src in self.sources.items()
            if src.get("weight_tier") == "core" and title not in unchecked
            for n in (title, *src.get("names", []))
        }


class PostgresKnowledgeLookup:
    def __init__(self, engine: Engine):
        self._engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)
        self._tables = set(inspect(engine).get_table_names())

    def has_table(self, name: str) -> bool:
        return name in self._tables

    def work_role(self, ref: str) -> str | None:
        from rav_ai.db.models import KlSource, Segment

        with self._sessions() as s:
            return s.scalars(
                select(KlSource.role)
                .join(Segment, Segment.work_id == KlSource.work_id)
                .where(Segment.ref == ref)
            ).first()

    def edge(self, edge_id: int) -> dict[str, Any] | None:
        from rav_ai.db.models import Edge, Segment

        with self._sessions() as s:
            e = s.get(Edge, edge_id)
            if e is None:
                return None
            a, b = s.get(Segment, e.segment_a_id), s.get(Segment, e.segment_b_id)
            return {
                "origin": e.origin,
                "edge_type": e.edge_type,
                "a_ref": a.ref if a else None,
                "b_ref": b.ref if b else None,
            }

    def kl_row_exists(self, table: str, row_id: int) -> bool:
        from rav_ai.db.models import KlLink, KlRelation

        model = {"kl_relations": KlRelation, "kl_links": KlLink}.get(table)
        if model is None:
            return False
        with self._sessions() as s:
            return s.get(model, row_id) is not None

    def work_of(self, ref: str) -> str | None:
        from rav_ai.db.models import Segment, Work

        with self._sessions() as s:
            return s.scalars(
                select(Work.title)
                .join(Segment, Segment.work_id == Work.id)
                .where(Segment.ref == ref)
            ).first()

    def witness_exists(self, witness_id: int) -> bool:
        from rav_ai.db.models import KlWitness

        with self._sessions() as s:
            return s.get(KlWitness, witness_id) is not None

    def unchecked_core_works(self, scope_keys: Iterable[str]) -> dict[str, set[str]]:
        from rav_ai.db.models import KlCoverage, KlSource, Work

        keys = list(scope_keys)
        out: dict[str, set[str]] = {}
        with self._sessions() as s:
            core = s.execute(
                select(Work.id, Work.title, Work.he_title)
                .join(KlSource, KlSource.work_id == Work.id)
                .where(KlSource.weight_tier == "core")
                .order_by(Work.id)
            ).all()
            cov = {
                (c.scope_key, c.work_id): c.status
                for c in s.scalars(select(KlCoverage).where(KlCoverage.scope_key.in_(keys)))
            }
        for wid, title, he in core:
            if any(cov.get((k, wid), "unknown") in UNCHECKED_COVERAGE for k in keys):
                out[title] = {title, *([he] if he else [])}
        return out

    def checked_core_names(self, scope_keys: Iterable[str]) -> set[str]:
        from rav_ai.db.models import KlSource, Work

        unchecked = self.unchecked_core_works(scope_keys)
        with self._sessions() as s:
            rows = s.execute(
                select(Work.title, Work.he_title)
                .join(KlSource, KlSource.work_id == Work.id)
                .where(KlSource.weight_tier == "core")
            ).all()
        return {n for t, he in rows if t not in unchecked for n in (t, he) if n}

    def snapshot(self, out: Any, scope_keys: list[str]) -> dict[str, Any]:
        """The rows the V5-V10 checks read for one run, as an InMemoryKnowledgeLookup dict
        (so a recorded fixture replays without a database)."""
        from rav_ai.db.models import KlCoverage, KlSource, Segment, Work

        a = out.answer
        refs = sorted(a.cited_refs() | set(out.retrieved_refs))
        edge_ids = sorted({r.id for c in a.claims for r in c.relations if r.table == "edges"})
        wit = sorted({x.witness_id for c in a.claims for x in c.citations if x.witness_id})
        snap: dict[str, Any] = {
            "tables": sorted(
                self._tables & {"kl_sources", "kl_coverage", "kl_witnesses", "edges", *KL_TABLES}
            ),
            "ref_work": {},
            "sources": {},
            "coverage": {},
            "edges": {},
            "witnesses": [
                w for w in wit if self.has_table("kl_witnesses") and self.witness_exists(w)
            ],
        }
        with self._sessions() as s:
            for ref, title in s.execute(
                select(Segment.ref, Work.title)
                .join(Work, Work.id == Segment.work_id)
                .where(Segment.ref.in_(refs))
            ):
                snap["ref_work"][ref] = title
            if self.has_table("kl_sources"):
                for src, title, he in s.execute(
                    select(KlSource, Work.title, Work.he_title).join(
                        Work, Work.id == KlSource.work_id
                    )
                ).all():
                    if title in snap["ref_work"].values() or src.weight_tier == "core":
                        snap["sources"][title] = {
                            "role": src.role,
                            "weight_tier": src.weight_tier,
                            "names": [he] if he else [],
                        }
            if self.has_table("kl_coverage") and scope_keys:
                for c, title in s.execute(
                    select(KlCoverage, Work.title)
                    .join(Work, Work.id == KlCoverage.work_id)
                    .where(KlCoverage.scope_key.in_(scope_keys))
                ).all():
                    snap["coverage"].setdefault(c.scope_key, {})[title] = c.status
        for eid in edge_ids:
            e = self.edge(eid)
            if e:
                snap["edges"][str(eid)] = e
        for t in KL_TABLES:
            ids = sorted({r.id for c in a.claims for r in c.relations if r.table == t})
            snap[t] = [i for i in ids if self.has_table(t) and self.kl_row_exists(t, i)]
        return snap
