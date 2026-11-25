"""Misattribution of views to authors (target < 2%), from a manually rated sample.

Unit: an ``interpretation`` claim in an answer that attributes a view to an author. A stratified
sample (category x source_role) of reviewed runs is rated by a human as ``attribution_ok`` or
``wrong_attribution`` with a subtype. Reported: rate, Wilson 95% CI and n.

Gate: PASS when the point estimate is below the target AND the CI upper bound is too; FAIL when
the point estimate is at/above it; otherwise UNDECIDED ("grow the sample") with the n needed.
Auxiliary automatic signal: V4 and V5 failures.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

from .schema import Answer, ClaimType

WRONG_SUBTYPES = (
    "secondary_as_primary",
    "hava_amina_as_conclusion",  # an acharon writes "lekhora" and then rejects it
    "wrong_holder",
    "reading_stated_as_rishon",
)

Verdict = Literal["PASS", "FAIL", "UNDECIDED"]


def z_for(level: float) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2)


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    z = z_for(level)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def needed_n(rate: float, target: float, level: float = 0.95, cap: int = 100_000) -> int | None:
    """Smallest n whose Wilson upper bound is below ``target`` if the observed rate holds."""
    if rate >= target:
        return None
    for n in range(1, cap + 1):
        if wilson(round(rate * n), n, level)[1] < target:
            return n
    return None


@dataclass(frozen=True)
class GateResult:
    verdict: Verdict
    rate: float | None
    ci: tuple[float, float]
    n: int
    wrong: int
    target: float
    needed_n: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "label": {"PASS": "PASS", "FAIL": "FAIL", "UNDECIDED": "לא מוכרע, הגדל מדגם"}[
                self.verdict
            ],
            "rate": self.rate,
            "ci_low": self.ci[0],
            "ci_high": self.ci[1],
            "n": self.n,
            "wrong": self.wrong,
            "target": self.target,
            "needed_n": self.needed_n,
        }


def attribution_gate(wrong: int, n: int, target: float = 0.02, level: float = 0.95) -> GateResult:
    ci = wilson(wrong, n, level)
    if n == 0:
        return GateResult("UNDECIDED", None, ci, 0, 0, target, needed_n(0.0, target, level))
    rate = wrong / n
    if rate >= target:
        return GateResult("FAIL", rate, ci, n, wrong, target)
    if ci[1] > target:
        return GateResult("UNDECIDED", rate, ci, n, wrong, target, needed_n(rate, target, level))
    return GateResult("PASS", rate, ci, n, wrong, target)


# ---- units and sampling -------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionUnit:
    unit_id: str  # "<run_id>#<claim_index>"
    run_id: str
    claim_index: int
    category: str
    source_role: str  # secondary_lens if any citation is a lens, else primary
    text: str


def attribution_units(run_id: str, category: str, answer: Answer) -> list[AttributionUnit]:
    out = []
    for i, c in enumerate(answer.claims):
        if c.type is not ClaimType.INTERPRETATION or not c.citations:
            continue
        role = (
            "secondary_lens"
            if any(x.source_role == "secondary_lens" for x in c.citations)
            else "primary"
        )
        out.append(AttributionUnit(f"{run_id}#{i}", run_id, i, category, role, c.text))
    return out


def stratified_sample(units: Iterable[AttributionUnit], n: int, seed: int) -> list[AttributionUnit]:
    """Round-robin over (category, source_role) strata, each shuffled with ``seed``: every
    stratum is represented before any gets a second unit. Deterministic."""
    strata: dict[tuple[str, str], list[AttributionUnit]] = defaultdict(list)
    for u in sorted(units, key=lambda u: u.unit_id):
        strata[(u.category, u.source_role)].append(u)
    rng = random.Random(seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    out: list[AttributionUnit] = []
    keys = sorted(strata)
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


def misattribution_report(
    ratings: Iterable[dict[str, Any]], target: float = 0.02, level: float = 0.95
) -> dict[str, Any]:
    """``ratings``: answer-queue claim ratings ({unit_id, verdict, error_type, subtype, ...}).
    Only interpretation units count; the latest rating per unit wins (input order)."""
    latest: dict[str, dict[str, Any]] = {}
    for r in ratings:
        if r.get("claim_type", "interpretation") == "interpretation" and r.get("verdict"):
            latest[r["unit_id"]] = r
    wrong = [r for r in latest.values() if r.get("error_type") == "wrong_attribution"]
    by_sub: dict[str, int] = defaultdict(int)
    for r in wrong:
        by_sub[r.get("subtype") or "unspecified"] += 1
    gate = attribution_gate(len(wrong), len(latest), target, level)
    return {**gate.as_dict(), "by_subtype": dict(sorted(by_sub.items()))}
