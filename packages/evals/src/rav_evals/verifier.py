"""Deterministic answer verifier with a structured failure report.

V1-V4: citations (ref exists and was retrieved, quote in source, attribution).
V5-V10 (KL-3, knowledge layer; a *measurement*, mirroring the harness's run-time guard):

- V5 ``secondary_as_primary``: a citation marked ``source_role="primary"`` whose work is a lens
  (``kl_sources.role='lens'``).
- V6 ``candidate_as_proof``: every relation of a claim is a candidate link and none of its
  endpoints was cited.
- V7 ``witness_missing``: a textual-variant / scribal-error claim in the system's own voice
  without a ``witness_id`` (``witness_unknown`` if the id is not in ``kl_witnesses``).
- V8 ``coverage_notice_missing``: a core work not checked for the scope (``kl_coverage``) that
  the answer does not list in ``unchecked_core_books``; ``coverage_notice_wrong`` for a listed
  core work that *was* checked.
- V9 ``kibalnu_undocumented``: acceptance / custom language with no citation.
- V10 ``sa_decisive_phrasing``: decisive phrasing about the Shulchan Arukh in the system's voice.

A check that needs a missing ``kl_*`` table is skipped and listed in ``skipped``.
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
    citations_checked: int = 0

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
) -> VerificationReport:
    report = _verify_citations(
        answer, retrieved_refs, store, normalize=normalize, attribution_checker=attribution_checker
    )
    verify_knowledge(
        answer,
        report,
        knowledge=knowledge,
        scope_keys=list(scope_keys),
        markers=markers or load_config().markers,
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


def relation_is_candidate(rel, knowledge: KnowledgeLookup | None) -> bool | None:
    """True/False, or None when the relation's table is not available (check skipped)."""
    if rel.candidate or (rel.evidence_origin and is_candidate({"origin": rel.evidence_origin})):
        return True
    if knowledge is None:
        return False
    if not knowledge.has_table(rel.table):
        return None
    if rel.table == "edges":
        e = knowledge.edge(rel.id)
        return bool(e and is_candidate(e))
    return False


def verify_knowledge(
    answer: Answer,
    report: VerificationReport,
    *,
    knowledge: KnowledgeLookup | None,
    scope_keys: list[str],
    markers: Markers,
) -> VerificationReport:
    def fail(check: str, code: str, reason: str, **where) -> None:
        report.failures.append(Failure(check=check, code=code, reason=reason, **where))

    kl_sources = knowledge is not None and knowledge.has_table("kl_sources")
    if not kl_sources:
        report.skipped.append("V5: kl_sources table not available")
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

        # V6: only candidate links, and their endpoints never cited
        if claim.relations:
            verdicts = [relation_is_candidate(r, knowledge) for r in claim.relations]
            if any(v is None for v in verdicts):
                report.skipped.append(f"V6: claim {ci} (relation table not available)")
            elif all(verdicts):
                cited = {c.ref for c in claim.citations}
                ends = {ref for r in claim.relations for ref in r.evidence_refs}
                if not (cited & ends):
                    fail(
                        "V6",
                        "candidate_as_proof",
                        "every relation is a candidate link and no endpoint was cited",
                        claim_index=ci,
                    )

        # V7: textual variant / scribal error in the system's voice needs a witness
        if is_girsa_claim(claim, markers) and system_voice(claim):
            ids = [c.witness_id for c in claim.citations if c.witness_id is not None]
            if not ids:
                fail(
                    "V7",
                    "witness_missing",
                    "variant / scribal-error claim in the system's voice without witness_id",
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
            elif knowledge is not None:
                report.skipped.append(f"V7: claim {ci} witness existence (no kl_witnesses)")

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
        report.skipped.append(f"V5: {len(unregistered)} cited ref(s) of unregistered works")

    # V8: coverage notice
    scope = answer.scope
    if scope == "out_of_scope":
        report.skipped.append("V8: answer is out of scope")
    elif not scope_keys:
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
