"""Deterministic answer verifier (V1-V4) with a structured failure report."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from .schema import Answer, Citation, ClaimType
from .textnorm_stub import normalize as default_normalize

Normalizer = Callable[[str], str]


class SourceStore(Protocol):
    """Fetch source text by canonical ref. The real implementation will use the DB."""

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
    check: Literal["V1", "V2", "V3", "V4"]
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
