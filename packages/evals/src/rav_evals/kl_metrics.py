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

from pydantic import BaseModel, Field, model_validator

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
    split: Literal["dev", "heldout"] | None = None  # defaults to the question's split

    @model_validator(mode="after")
    def _split(self) -> RecordedRun:
        if self.split is None:
            self.split = self.question.split
        elif self.split != self.question.split:
            raise ValueError(f"split {self.split!r} != question split {self.question.split!r}")
        return self


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
    caught_codes: list[str] = Field(
        default_factory=list,
        description="Failure codes at export; regress requires every one to still fire.",
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
    syc = (
        evaluate_one(q, out, res.verification, cfg.markers, knowledge) if res.verification else None
    )
    return res, syc


def evaluate_recorded(rr: RecordedRun, cfg: EvalsConfig) -> tuple[QuestionResult, Any]:
    kn = InMemoryKnowledgeLookup(rr.knowledge) if rr.knowledge is not None else None
    return evaluate_question(rr.question, rr.output, InMemorySourceStore(rr.sources), kn, cfg)


def claim_codes(claim: Claim, rr: RecordedRun, cfg: EvalsConfig) -> list[str]:
    """Failure codes the verifier gives this one claim, in the context of its recorded run."""
    kn = InMemoryKnowledgeLookup(rr.knowledge) if rr.knowledge is not None else None
    rep = verify_answer(
        Answer(claims=[claim], disclaimer="x"),
        rr.output.retrieved_refs,
        InMemorySourceStore(rr.sources),
        knowledge=kn,
        scope_keys=rr.output.scope_keys or rr.question.scope_keys,
        markers=cfg.markers,
    )
    return rep.failure_codes()


def rejected_caught(
    rc: RejectedClaim, runs: dict[str, RecordedRun], cfg: EvalsConfig
) -> list[str] | None:
    """Codes the verifier flags this rejected claim with now; None if its run is missing."""
    rr = runs.get(rc.run_fixture)
    return None if rr is None else claim_codes(rc.claim, rr, cfg)


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(
    results: list[QuestionResult],
    syco: list[SycophancyOutcome],
    cfg: EvalsConfig | None = None,
    rejected: tuple[int, int] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    hard_v = {k for k in cfg.hard_zero if k in V_CHECKS}
    skipped_hard = 0
    oos_soft = 0
    kl_origins: Counter[str] = Counter()
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
        ver_n += 1  # an errored question counts in the denominator, as not passing
        if r.verification:
            ver_pass += r.verification.passed
            skipped_hard += len(hard_v & set(r.verification.skipped_checks))
            kl_origins.update(r.verification.kl_relation_origins)
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
            if c.kind == "out_of_scope_conclusion" and not ok:
                oos_soft += 1
        if r.output is not None:
            if r.output.cost_usd is None:
                cost_known = False
            else:
                cost += r.output.cost_usd
            if r.status == "failed":
                for it in r.output.knowledge_items:
                    if isinstance(it, dict) and it.get("tool") == "_meta":
                        continue
                    failed_items[json.dumps(it, ensure_ascii=False, sort_keys=True)] += 1
    sm = sycophancy_metrics(syco, cfg)
    m: dict[str, Any] = {
        "questions": len(results),
        "by_status": dict(sorted(by_status.items())),
        "errors": by_status.get("error", 0),
        "skipped_hard": skipped_hard,
        "out_of_scope_no_conclusion": oos_soft,  # soft failure: reported, not a gate
        "kl_relation_origins": dict(sorted(kl_origins.items())),
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
    "skipped_hard": "lower",
    "errors": "lower",
    "flip_without_source_rate": "lower_rate",
}


def compare_to_baseline(
    metrics: dict[str, Any], baseline: dict[str, Any], cfg: EvalsConfig
) -> list[str]:
    failures = []
    for k in cfg.hard_zero:
        if k not in metrics:
            failures.append(f"config error: hard_zero key {k!r} is not a metric")
        elif (metrics.get(k) or 0) > 0:
            failures.append(f"hard gate {k}={metrics[k]} (must be 0)")
    for k, d in DIRECTIONS.items():
        cur, base = metrics.get(k), baseline.get(k)
        if base is None:
            continue
        if cur is None:
            failures.append(f"{k} had a baseline value ({base}) and is now missing")
            continue
        if d == "higher" and cur < base - cfg.rate_tolerance:
            failures.append(f"{k} dropped {base:.3f} -> {cur:.3f} (tolerance {cfg.rate_tolerance})")
        elif d == "lower" and cur > base + cfg.count_tolerance:
            failures.append(f"{k} rose {base} -> {cur} (tolerance {cfg.count_tolerance})")
        elif d == "lower_rate" and cur > base + cfg.rate_tolerance:
            failures.append(f"{k} rose {base:.3f} -> {cur:.3f}")
    return failures


BASELINE_KEYS = tuple(DIRECTIONS) + ("questions",)


def baseline_from(res: dict[str, Any]) -> dict[str, Any]:
    """Baseline from a ``regress`` result: the metrics, plus the fixture ids and the count per
    category (a missing fixture or a smaller category fails later runs)."""
    m = res["metrics"]
    return {
        **{k: m.get(k) for k in BASELINE_KEYS},
        "fixture_ids": res["fixture_ids"],
        "categories": res["categories"],
        "synthetic_only": res["synthetic_fixtures"] == res["fixtures"],
    }


def regress(fixtures_dir: Path, baseline_path: Path, cfg: EvalsConfig | None = None) -> dict:
    cfg = cfg or load_config()
    runs, rejected = load_fixtures(fixtures_dir)
    results, syco = [], []
    failures: list[str] = []
    hard_v = [k for k in cfg.hard_zero if k in V_CHECKS]
    for rr in runs:
        res, s = evaluate_recorded(rr, cfg)
        results.append(res)
        if s is not None:
            syco.append(s)
        skipped = set(res.verification.skipped_checks) if res.verification else set()
        failures += [f"hard gate {v} skipped on {rr.fixture_id}" for v in hard_v if v in skipped]
    by_id = {r.fixture_id: r for r in runs}
    caught_now = {rc.fixture_id: rejected_caught(rc, by_id, cfg) for rc in rejected}
    caught = sum(bool(c) for c in caught_now.values())
    metrics = compute_metrics(results, syco, cfg, rejected=(caught, len(rejected)))
    categories = dict(sorted(Counter(r.question.category.value for r in runs).items()))
    synthetic = sum(r.synthetic for r in runs)
    metrics["synthetic_fixtures"] = synthetic
    baseline = (
        json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        if Path(baseline_path).exists()
        else None
    )
    if not runs:
        failures.append("no recorded fixtures (0 runs)")
    failures += compare_to_baseline(metrics, baseline or {}, cfg)
    if baseline is None:
        failures.append(f"no baseline at {baseline_path} (run: rav-evals regress --update)")
    else:
        for fid in baseline.get("fixture_ids", []):
            if fid not in by_id:
                failures.append(f"fixture {fid} is in the baseline but missing")
        for cat, n in baseline.get("categories", {}).items():
            if categories.get(cat, 0) < n:
                failures.append(f"category {cat}: {categories.get(cat, 0)} fixtures < {n}")
    for rc in rejected:
        now = caught_now[rc.fixture_id]
        if now is None:
            failures.append(f"rejected claim {rc.fixture_id}: run {rc.run_fixture} is missing")
            continue
        lost = [c for c in rc.caught_codes if c not in now]
        if lost:
            failures.append(f"rejected claim {rc.fixture_id} no longer caught by {lost}")
        elif rc.auto_caught and not now:
            failures.append(f"rejected claim {rc.fixture_id} ({rc.error_type}) is no longer caught")
    return {
        "passed": not failures,
        "failures": failures,
        "fixtures": len(runs),
        "fixture_ids": sorted(by_id),
        "categories": categories,
        "synthetic_fixtures": synthetic,
        "rejected_fixtures": len(rejected),
        "metrics": metrics,
        "per_fixture": [
            {
                "fixture_id": rr.fixture_id,
                "status": res.status,
                "synthetic": rr.synthetic,
                "codes": res.verification.failure_codes() if res.verification else [],
                "skipped_checks": res.verification.skipped_checks if res.verification else [],
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
