"""Deterministic answer verifier with a structured failure report.

V1-V4: citations (ref exists and was retrieved, quote in source, attribution).
V5-V10 (KL-3, knowledge layer; a *measurement*, mirroring the harness's run-time guard):

- V5 ``secondary_as_primary``: a citation marked ``source_role="primary"`` whose work is a lens
  (``kl_sources.role='lens'``).
- V6 ``candidate_as_proof``: for EVERY candidate relation of a claim, both endpoints must have
  been retrieved and cited in that claim (other relations do not excuse it).
  ``relation_unknown``: the relation's row does not exist. Candidacy of an ``edges`` row comes
  from its stored origin only (fail closed: anything but ``human``); ``kl_relations`` /
  ``kl_links`` are not links - their ``evidence_origin`` (llm < judge < human) is a separate
  metric. ``rel.candidate=True`` from the answer can only add a candidate.
- V7 ``witness_missing``: a textual-variant / scribal-error claim without a ``witness_id``,
  unless the claim attributes it explicitly to another holder (a citation with an
  ``attribution`` from a work other than the one said to be corrupt); ``witness_unknown`` if
  the id is not in ``kl_witnesses``.
- V8 ``coverage_notice_missing``: a core work not checked for the scope (``kl_coverage``) that
  the answer does not list in ``unchecked_core_books``; ``coverage_notice_wrong`` for a listed
  core work that *was* checked; ``scope_mismatch`` when the answer calls itself out_of_scope for
  an in-scope question (V8 then runs as usual). The scope comes from the question / run; for an
  out-of-scope question V8 requires the out-of-scope notice (``out_of_scope_notice_missing``).
- V9 ``kibalnu_undocumented``: acceptance / custom language with no citation.
- V10 ``sa_decisive_phrasing``: decisive phrasing about the Shulchan Arukh in the system's voice.

A check that needs a missing ``kl_*`` table (or no knowledge lookup at all) is skipped and listed
in ``skipped``; when there was something to check, the check id also goes to ``skipped_checks``
(``regress`` treats a skipped hard gate as a failure).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from .config import Markers, load_config
from .kl_lookup import KnowledgeLookup, is_candidate
from .normalization import normalize as default_normalize
from .schema import Answer, Citation, Claim, ClaimType

Normalizer = Callable[[str], str]


class SourceStore(Protocol):
    """Fetch source text by canonical ref. ``pg_store.PostgresSourceStore`` is the real one."""

    def get_text(self, ref: str) -> str | None:
        """Return the text for ``ref`` or None if no such ref exists."""
        ...


class InMemorySourceStore:
    """Fake store for tests and local runs."""

    def __init__(self, texts: dict[str, str]):
        self._texts = dict(texts)

    def get_text(self, ref: str) -> str | None:
        return self._texts.get(ref)


@dataclass(frozen=True)
class AttributionVerdict:
    status: Literal["ok", "fail", "skipped"]
    reason: str = ""


class AttributionChecker(Protocol):
    """V4: check ``citation.attribution`` when the source text itself quotes an earlier
    source (e.g. the Mishna Berura quoting the Magen Avraham)."""

    def check(self, citation: Citation, source_text: str) -> AttributionVerdict: ...


class StubAttributionChecker:
    """Placeholder: never fails, reports every attributed citation as skipped."""

    def check(self, citation: Citation, source_text: str) -> AttributionVerdict:
        return AttributionVerdict("skipped", "V4 not implemented")


class Failure(BaseModel):
    check: Literal["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"]
    code: str
    reason: str
    claim_index: int | None = None
    citation_index: int | None = None
    ref: str | None = None


class VerificationReport(BaseModel):
    failures: list[Failure] = []
    skipped: list[str] = []  # e.g. "V4: claim 2 citation 0 (V4 not implemented)"
    skipped_checks: list[str] = []  # checks skipped although there was content to check
    citations_checked: int = 0
    kl_relation_origins: dict[str, int] = {}  # evidence_origin of kl_* relations (metric)

    @property
    def passed(self) -> bool:
        return not self.failures

    def failure_codes(self) -> list[str]:
        return sorted({f"{f.check}:{f.code}" for f in self.failures})

    def count(self, check: str) -> int:
        return sum(f.check == check for f in self.failures)


_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")


def _quote_in_source(quote: str, source: str, norm: Normalizer) -> bool:
    """True if the quote appears in the source; ``...`` splits it into ordered fragments."""
    haystack = norm(source)
    pos = 0
    fragments = [norm(f) for f in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    for frag in fragments:
        idx = haystack.find(frag, pos)
        if idx < 0:
            return False
        pos = idx + len(frag)
    return True


def verify_answer(
    answer: Answer,
    retrieved_refs: Iterable[str],
    store: SourceStore,
    *,
    normalize: Normalizer = default_normalize,
    attribution_checker: AttributionChecker | None = None,
    knowledge: KnowledgeLookup | None = None,
    scope_keys: Iterable[str] = (),
    markers: Markers | None = None,
    question_in_scope: bool | None = None,
) -> VerificationReport:
    retrieved = list(retrieved_refs)
    report = _verify_citations(
        answer, retrieved, store, normalize=normalize, attribution_checker=attribution_checker
    )
    keys = list(scope_keys)
    verify_knowledge(
        answer,
        report,
        knowledge=knowledge,
        scope_keys=keys,
        markers=markers or load_config().markers,
        retrieved_refs=retrieved,
        question_in_scope=(True if keys else None)
        if question_in_scope is None
        else question_in_scope,
    )
    return report


def _verify_citations(
    answer: Answer,
    retrieved_refs: Iterable[str],
    store: SourceStore,
    *,
    normalize: Normalizer,
    attribution_checker: AttributionChecker | None,
) -> VerificationReport:
    retrieved = set(retrieved_refs)
    checker = attribution_checker or StubAttributionChecker()
    report = VerificationReport()

    if not answer.disclaimer.strip():
        report.failures.append(
            Failure(
                check="V3",
                code="missing_disclaimer",
                reason="answer lacks the 'system opinion, not a ruling' disclaimer",
            )
        )

    for ci, claim in enumerate(answer.claims):
        # V3: structural requirements per claim type
        if claim.type is ClaimType.QUOTE and not claim.citations:
            report.failures.append(
                Failure(
                    check="V3",
                    code="quote_without_citation",
                    reason="claim of type quote has no citation",
                    claim_index=ci,
                )
            )
        if claim.type in (ClaimType.CHIDUSH, ClaimType.SYSTEM_CONCLUSION) and (
            not claim.flagged_as_system
        ):
            report.failures.append(
                Failure(
                    check="V3",
                    code="unflagged_system_claim",
                    reason=f"claim of type {claim.type} is not explicitly flagged as the system's",
                    claim_index=ci,
                )
            )

        for ki, cit in enumerate(claim.citations):
            report.citations_checked += 1
            where = {"claim_index": ci, "citation_index": ki, "ref": cit.ref}

            # V1: ref exists and was retrieved in this run
            source = store.get_text(cit.ref)
            if source is None:
                report.failures.append(
                    Failure(
                        check="V1",
                        code="ref_not_found",
                        reason=f"ref {cit.ref!r} does not exist in the corpus",
                        **where,
                    )
                )
            if cit.ref not in retrieved:
                report.failures.append(
                    Failure(
                        check="V1",
                        code="ref_not_retrieved",
                        reason=f"ref {cit.ref!r} was not among the sources retrieved in this run",
                        **where,
                    )
                )
            if source is None:
                continue

            # V2: quoted text appears in the source
            if not _quote_in_source(cit.quoted_text, source, normalize):
                report.failures.append(
                    Failure(
                        check="V2",
                        code="quote_not_in_source",
                        reason="quoted text does not appear in the source after normalization",
                        **where,
                    )
                )

            # V4: attribution
            if cit.attribution:
                verdict = checker.check(cit, source)
                if verdict.status == "fail":
                    report.failures.append(
                        Failure(
                            check="V4", code="attribution_mismatch", reason=verdict.reason, **where
                        )
                    )
                elif verdict.status == "skipped":
                    report.skipped.append(f"V4: claim {ci} citation {ki} ({verdict.reason})")
    return report


# ---- V5-V10 -------------------------------------------------------------------------------

_GERSHAYIM = str.maketrans({"\u05f4": '"', "\u05f3": "'", "\u201d": '"', "\u201c": '"'})


def _norm_marks(text: str) -> str:
    return text.translate(_GERSHAYIM)


def has_marker(text: str, markers: Iterable[str]) -> bool:
    t = _norm_marks(text)
    return any(_norm_marks(m) in t for m in markers if m)


def system_voice(claim: Claim) -> bool:
    """The system speaks for itself: its own claim types, an explicit flag, or no source."""
    return (
        claim.type in (ClaimType.SYSTEM_CONCLUSION, ClaimType.CHIDUSH)
        or claim.flagged_as_system
        or not claim.citations
    )


GIRSA_RELS = ("claims_error_in", "depends_on_girsa")


def is_girsa_claim(claim: Claim, markers: Markers) -> bool:
    """The claim ASSERTS a textual variant / scribal error (a structured relation, or a marker
    without a denial phrase such as "אין לומר")."""
    if any(r.rel in GIRSA_RELS for r in claim.relations):
        return True
    return has_marker(claim.text, markers.girsa_error) and not has_marker(
        claim.text, markers.girsa_negation
    )


def _work(ref: str, knowledge: KnowledgeLookup | None) -> str:
    w = knowledge.work_of(ref) if knowledge is not None else None
    return w or re.sub(r"\s+\S*\d\S*$", "", ref).strip()


def attributed_to_other(claim: Claim, knowledge: KnowledgeLookup | None) -> bool:
    """The variant / error is explicitly reported in someone else's name: a citation with an
    attribution, from a work other than every work the claim quotes unattributed (the text
    said to be corrupt)."""
    targets = {
        _work(c.ref, knowledge) for c in claim.citations if not (c.attribution or "").strip()
    }
    return any(
        (c.attribution or "").strip() and _work(c.ref, knowledge) not in targets
        for c in claim.citations
    )


def known_witnesses(claim: Claim, knowledge: KnowledgeLookup | None) -> list[int]:
    """witness ids of the claim that exist in kl_witnesses (none without a lookup)."""
    ids = [c.witness_id for c in claim.citations if c.witness_id is not None]
    if knowledge is None or not knowledge.has_table("kl_witnesses"):
        return []
    return [w for w in ids if knowledge.witness_exists(w)]


def relation_is_candidate(rel, knowledge: KnowledgeLookup | None) -> bool | None:
    """True / False, or None when it cannot be decided (no lookup, table missing) -> skipped.
    Raises LookupError when the row does not exist (``relation_unknown``)."""
    if knowledge is None or not knowledge.has_table(rel.table):
        return None
    if rel.table == "edges":
        e = knowledge.edge(rel.id)
        if e is None:
            raise LookupError(rel.id)
        return rel.candidate or is_candidate(e)  # the answer's own origin is not trusted
    if not knowledge.kl_row_exists(rel.table, rel.id):
        raise LookupError(rel.id)
    return bool(rel.candidate)


def _endpoints(rel, knowledge: KnowledgeLookup | None) -> list[str]:
    if rel.table == "edges" and knowledge is not None:
        e = knowledge.edge(rel.id) or {}
        ends = [r for r in (e.get("a_ref"), e.get("b_ref")) if r]
        if ends:
            return ends
    return list(rel.evidence_refs)


def verify_knowledge(
    answer: Answer,
    report: VerificationReport,
    *,
    knowledge: KnowledgeLookup | None,
    scope_keys: list[str],
    markers: Markers,
    retrieved_refs: Iterable[str] = (),
    question_in_scope: bool | None = True,
) -> VerificationReport:
    """``question_in_scope``: True (the question/run has scope keys), False (the question is
    out of scope), None (no scope information: V8 is skipped)."""
    retrieved = set(retrieved_refs)

    def fail(check: str, code: str, reason: str, **where) -> None:
        report.failures.append(Failure(check=check, code=code, reason=reason, **where))

    def skip(check: str, why: str, *, hard: bool) -> None:
        report.skipped.append(f"{check}: {why}")
        if hard and check not in report.skipped_checks:
            report.skipped_checks.append(check)

    has_citations = any(c.citations for c in answer.claims)
    kl_sources = knowledge is not None and knowledge.has_table("kl_sources")
    if not kl_sources:
        skip("V5", "kl_sources table not available", hard=has_citations)
    witnesses = knowledge is not None and knowledge.has_table("kl_witnesses")
    unregistered: set[str] = set()

    for ci, claim in enumerate(answer.claims):
        # V5: a lens work cited as if it were the primary source
        if kl_sources:
            for ki, cit in enumerate(claim.citations):
                role = knowledge.work_role(cit.ref)
                if role is None:
                    unregistered.add(cit.ref)
                elif role == "lens" and cit.source_role == "primary":
                    fail(
                        "V5",
                        "secondary_as_primary",
                        f"{cit.ref!r} is a lens work (kl_sources.role=lens) cited as primary",
                        claim_index=ci,
                        citation_index=ki,
                        ref=cit.ref,
                    )

        # V6: every candidate relation needs BOTH endpoints retrieved and cited in this claim
        cited = {c.ref for c in claim.citations}
        for ri, rel in enumerate(claim.relations):
            try:
                cand = relation_is_candidate(rel, knowledge)
            except LookupError:
                fail(
                    "V6",
                    "relation_unknown",
                    f"{rel.table} id {rel.id} does not exist",
                    claim_index=ci,
                )
                continue
            if rel.table != "edges":
                o = rel.evidence_origin or "unknown"
                report.kl_relation_origins[o] = report.kl_relation_origins.get(o, 0) + 1
            if cand is None:
                skip("V6", f"claim {ci} relation {ri} ({rel.table} not available)", hard=True)
                continue
            if not cand:
                continue
            ends = _endpoints(rel, knowledge)
            missing = [r for r in ends if r not in cited or r not in retrieved]
            if len(ends) < 2 or missing:
                fail(
                    "V6",
                    "candidate_as_proof",
                    f"candidate {rel.table} {rel.id}: endpoints not both retrieved and cited "
                    f"(missing {missing or 'endpoint refs'})",
                    claim_index=ci,
                )

        # V7: a variant / scribal-error claim needs a known witness unless attributed to another
        if is_girsa_claim(claim, markers) and not attributed_to_other(claim, knowledge):
            ids = [c.witness_id for c in claim.citations if c.witness_id is not None]
            if not ids:
                fail(
                    "V7",
                    "witness_missing",
                    "variant / scribal-error claim without witness_id (and not attributed)",
                    claim_index=ci,
                )
            elif witnesses:
                for w in ids:
                    if not knowledge.witness_exists(w):
                        fail(
                            "V7",
                            "witness_unknown",
                            f"witness_id {w} is not in kl_witnesses",
                            claim_index=ci,
                        )
            else:
                skip("V7", f"claim {ci} witness existence (no kl_witnesses)", hard=True)

        # V9: "we have received" / custom language must be documented
        if has_marker(claim.text, markers.kibalnu) and not claim.citations:
            fail(
                "V9",
                "kibalnu_undocumented",
                "acceptance / custom stated without a citation",
                claim_index=ci,
            )

        # V10: decisive phrasing about the Shulchan Arukh in the system's own words
        if claim.type is not ClaimType.QUOTE:
            mentions_sa = has_marker(claim.text, markers.sa_mentions) or any(
                c.ref.startswith(p) for c in claim.citations for p in markers.sa_ref_prefixes
            )
            if mentions_sa and has_marker(claim.text, markers.decisive):
                fail(
                    "V10",
                    "sa_decisive_phrasing",
                    "the Shulchan Arukh is presented as a final ruling in the system's voice",
                    claim_index=ci,
                )

    if unregistered:
        skip("V5", f"{len(unregistered)} cited ref(s) of unregistered works", hard=True)

    # V8: the scope is the question's / run's, never what the answer declares about itself
    if question_in_scope is None:
        report.skipped.append("V8: no scope information for this question / run")
        return report
    if question_in_scope is False:
        head = "\n".join([answer.disclaimer, *(c.text for c in answer.claims[:1])])
        if not has_marker(head, markers.out_of_scope_notice):
            fail(
                "V8",
                "out_of_scope_notice_missing",
                "out-of-scope question: the answer must open with the out-of-scope notice",
            )
        report.skipped.append("V8: coverage not checked (question out of scope)")
        return report
    if answer.scope == "out_of_scope":
        fail("V8", "scope_mismatch", "answer says out_of_scope for an in-scope question")
    if not scope_keys:
        report.skipped.append("V8: no scope keys for this run")
    elif knowledge is None or not (
        knowledge.has_table("kl_coverage") and knowledge.has_table("kl_sources")
    ):
        report.skipped.append("V8: kl_coverage / kl_sources not available")
    else:
        unchecked = knowledge.unchecked_core_works(scope_keys)
        listed = {_norm_marks(b).strip() for b in answer.unchecked_core_books}
        for title, names in sorted(unchecked.items()):
            if not listed & {_norm_marks(n).strip() for n in names}:
                fail(
                    "V8",
                    "coverage_notice_missing",
                    f"core work {title!r} was not checked for {scope_keys} and is not listed",
                )
        all_unchecked = {_norm_marks(n).strip() for ns in unchecked.values() for n in ns}
        checked_core = {_norm_marks(n).strip() for n in knowledge.checked_core_names(scope_keys)}
        for b in sorted(listed - all_unchecked):
            if b in checked_core:
                fail(
                    "V8",
                    "coverage_notice_wrong",
                    f"{b!r} is listed as unchecked but kl_coverage says it was checked",
                )
    return report
