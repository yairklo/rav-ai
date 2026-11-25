"""Deterministic KL-3 metrics over a set of evaluated questions; shared by ``regress`` (recorded
fixtures), ``run`` (live) and ``compare``.

Recorded fixtures live in ``packages/evals/fixtures/runs/*.json``:

- ``kind="run"``: question, tool transcript, answer (RunOutput), the text of every source read,
  and a snapshot of the knowledge-layer rows the checks need (``InMemoryKnowledgeLookup``).
- ``kind="rejected_claim"``: a claim a human rejected in the review UI, with its error type; it
  is a regression case (is it caught automatically?).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import EvalsConfig, load_config
from .golden import GoldenQuestion, GoldenSet
from .kl_lookup import InMemoryKnowledgeLookup, KnowledgeLookup
from .runner import QuestionResult, RunOutput, novel_protocol_failures, run_eval
from .schema import Answer, Claim
from .sycophancy import SycophancyOutcome, evaluate_one, sycophancy_metrics
from .verifier import InMemorySourceStore, SourceStore, verify_answer

V_CHECKS = tuple(f"V{i}" for i in range(1, 11))


class RecordedRun(BaseModel):
    fixture_id: str
    kind: Literal["run"] = "run"
    synthetic: bool = Field(
        default=False, description="True = hand-built seed, not a recorded live run."
    )
    question: GoldenQuestion
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    output: RunOutput
    sources: dict[str, str] = Field(default_factory=dict, description="ref -> text read")
    knowledge: dict[str, Any] | None = None


class RejectedClaim(BaseModel):
    fixture_id: str
    kind: Literal["rejected_claim"] = "rejected_claim"
    run_fixture: str
    claim_index: int
    claim: Claim
    error_type: str
    subtype: str | None = None
    note: str = ""
    rater_id: str = ""
    prompt_version: str | None = None
    auto_caught: bool = Field(
        default=False,
        description="The verifier flagged it when it was exported; regress fails if it stops.",
    )


def load_fixtures(d: Path) -> tuple[list[RecordedRun], list[RejectedClaim]]:
    runs, rejected = [], []
    for p in sorted(Path(d).glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("kind") == "rejected_claim":
            rejected.append(RejectedClaim.model_validate(data))
        else:
            runs.append(RecordedRun.model_validate(data))
    return runs, rejected


class Evaluated(BaseModel):
    result: QuestionResult
    sycophancy: dict[str, Any] | None = None
    novel_failures: list[str] | None = None


def evaluate_question(
    q: GoldenQuestion,
    out: RunOutput,
    store: SourceStore,
    knowledge: KnowledgeLookup | None,
    cfg: EvalsConfig,
) -> tuple[QuestionResult, SycophancyOutcome | None]:
    g = GoldenSet(version="single", questions=[q])
    res = run_eval(g, lambda _q: out, store, knowledge=knowledge, markers=cfg.markers).results[0]
    syc = evaluate_one(q, out, res.verification, cfg.markers) if res.verification else None
    return res, syc


def evaluate_recorded(rr: RecordedRun, cfg: EvalsConfig) -> tuple[QuestionResult, Any]:
    kn = InMemoryKnowledgeLookup(rr.knowledge) if rr.knowledge is not None else None
    return evaluate_question(rr.question, rr.output, InMemorySourceStore(rr.sources), kn, cfg)


def rejected_caught(rc: RejectedClaim, runs: dict[str, RecordedRun], cfg: EvalsConfig) -> bool:
    """Would the deterministic verifier flag this claim on its own?"""
    rr = runs.get(rc.run_fixture)
    sources = rr.sources if rr else {}
    kn = InMemoryKnowledgeLookup(rr.knowledge) if rr and rr.knowledge is not None else None
    retrieved = rr.output.retrieved_refs if rr else []
    a = Answer(claims=[rc.claim], disclaimer="x")
    rep = verify_answer(
        a, retrieved, InMemorySourceStore(sources), knowledge=kn, markers=cfg.markers
    )
    return not rep.passed


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(
    results: list[QuestionResult],
    syco: list[SycophancyOutcome],
    cfg: EvalsConfig | None = None,
    rejected: tuple[int, int] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    vcounts: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    ver_pass = ver_n = 0
    req_ok = req_n = side_ok = side_n = rub_ok = rub_n = 0
    v8_n = v8_ok = 0
    novel_ok = novel_n = 0
    cost = 0.0
    cost_known = True
    failed_items: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    for r in results:
        by_status[r.status] += 1
        if r.verification:
            ver_n += 1
            ver_pass += r.verification.passed
            for f in r.verification.failures:
                vcounts[f.check] += 1
                codes[f"{f.check}:{f.code}"] += 1
            if not any(s.startswith("V8:") for s in r.verification.skipped):
                v8_n += 1
                v8_ok += not any(f.check == "V8" for f in r.verification.failures)
        for c in r.checks:
            if c.status not in ("passed", "failed"):
                continue
            ok = c.status == "passed"
            if c.kind == "required_ref":
                req_n, req_ok = req_n + 1, req_ok + ok
            elif c.kind == "required_side":
                side_n, side_ok = side_n + 1, side_ok + ok
            elif c.gate:
                rub_n, rub_ok = rub_n + 1, rub_ok + ok
            if c.kind == "novel_protocol":
                novel_n, novel_ok = novel_n + 1, novel_ok + ok
        if r.output is not None:
            if r.output.cost_usd is None:
                cost_known = False
            else:
                cost += r.output.cost_usd
            if r.status == "failed":
                for it in r.output.knowledge_items:
                    failed_items[json.dumps(it, ensure_ascii=False)] += 1
    sm = sycophancy_metrics(syco, cfg)
    m: dict[str, Any] = {
        "questions": len(results),
        "by_status": dict(sorted(by_status.items())),
        "verifier_pass_rate": _rate(ver_pass, ver_n),
        **{v: vcounts.get(v, 0) for v in V_CHECKS},
        "verifier_failure_codes": dict(sorted(codes.items())),
        "secondary_as_primary": codes.get("V5:secondary_as_primary", 0),
        "candidate_as_proof": codes.get("V6:candidate_as_proof", 0),
        "source_recall": _rate(req_ok, req_n),
        "required_sides_rate": _rate(side_ok, side_n),
        "rubric_pass_rate": _rate(rub_ok, rub_n),
        "coverage_notice_fidelity": _rate(v8_ok, v8_n),
        "novel_pass_rate": _rate(novel_ok, novel_n),
        "accepted_false_error": sm["accepted_false_error"],
        "flip_without_source_rate": sm["flip_without_source_rate"],
        "sycophancy_pass_rate": _rate(sm["passed"], sm["questions"]),
        "sycophancy": sm,
        "cost_usd": cost if cost_known else None,
        "kl_items_in_failed": dict(failed_items.most_common()),
    }
    if rejected is not None:
        m["rejected_claims"] = rejected[1]
        m["rejected_caught_rate"] = _rate(*rejected)
    return m


# "higher" = a drop beyond rate_tolerance fails; "lower" = a rise beyond count_tolerance fails
DIRECTIONS: dict[str, str] = {
    "verifier_pass_rate": "higher",
    "source_recall": "higher",
    "required_sides_rate": "higher",
    "rubric_pass_rate": "higher",
    "coverage_notice_fidelity": "higher",
    "novel_pass_rate": "higher",
    "sycophancy_pass_rate": "higher",
    **{v: "lower" for v in V_CHECKS},
    "accepted_false_error": "lower",
    "flip_without_source_rate": "lower_rate",
}


def compare_to_baseline(
    metrics: dict[str, Any], baseline: dict[str, Any], cfg: EvalsConfig
) -> list[str]:
    failures = []
    for k in cfg.hard_zero:
        if (metrics.get(k) or 0) > 0:
            failures.append(f"hard gate {k}={metrics[k]} (must be 0)")
    for k, d in DIRECTIONS.items():
        cur, base = metrics.get(k), baseline.get(k)
        if cur is None or base is None:
            continue
        if d == "higher" and cur < base - cfg.rate_tolerance:
            failures.append(f"{k} dropped {base:.3f} -> {cur:.3f} (tolerance {cfg.rate_tolerance})")
        elif d == "lower" and cur > base + cfg.count_tolerance:
            failures.append(f"{k} rose {base} -> {cur} (tolerance {cfg.count_tolerance})")
        elif d == "lower_rate" and cur > base + cfg.rate_tolerance:
            failures.append(f"{k} rose {base:.3f} -> {cur:.3f}")
    return failures


BASELINE_KEYS = tuple(DIRECTIONS) + ("questions",)


def baseline_from(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics.get(k) for k in BASELINE_KEYS}


def regress(fixtures_dir: Path, baseline_path: Path, cfg: EvalsConfig | None = None) -> dict:
    cfg = cfg or load_config()
    runs, rejected = load_fixtures(fixtures_dir)
    results, syco = [], []
    for rr in runs:
        res, s = evaluate_recorded(rr, cfg)
        results.append(res)
        if s is not None:
            syco.append(s)
    by_id = {r.fixture_id: r for r in runs}
    caught_now = {rc.fixture_id: rejected_caught(rc, by_id, cfg) for rc in rejected}
    caught = sum(caught_now.values())
    metrics = compute_metrics(results, syco, cfg, rejected=(caught, len(rejected)))
    baseline = (
        json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        if Path(baseline_path).exists()
        else None
    )
    failures = compare_to_baseline(metrics, baseline or {}, cfg)
    failures += [
        f"rejected claim {rc.fixture_id} ({rc.error_type}) is no longer caught by the verifier"
        for rc in rejected
        if rc.auto_caught and not caught_now[rc.fixture_id]
    ]
    if baseline is None:
        failures.append(f"no baseline at {baseline_path} (run: rav-evals regress --update)")
    return {
        "passed": not failures,
        "failures": failures,
        "fixtures": len(runs),
        "rejected_fixtures": len(rejected),
        "metrics": metrics,
        "per_fixture": [
            {
                "fixture_id": rr.fixture_id,
                "status": res.status,
                "codes": res.verification.failure_codes() if res.verification else [],
            }
            for rr, res in zip(runs, results, strict=True)
        ],
    }


__all__ = [
    "Evaluated",
    "RecordedRun",
    "RejectedClaim",
    "baseline_from",
    "compare_to_baseline",
    "compute_metrics",
    "evaluate_question",
    "evaluate_recorded",
    "load_fixtures",
    "novel_protocol_failures",
    "regress",
]
