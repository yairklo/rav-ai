"""Eval runner: runs an answer function over a golden set, verifies, scores, aggregates."""

from __future__ import annotations

import json
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, computed_field

from .golden import GoldenQuestion, GoldenSet, RubricItem, is_placeholder
from .normalization import normalize as default_normalize
from .schema import Answer, ClaimType
from .verifier import (
    AttributionChecker,
    Normalizer,
    SourceStore,
    VerificationReport,
    verify_answer,
)

CheckStatus = Literal["passed", "failed", "skipped", "pending_judge"]
QuestionStatus = Literal["passed", "failed", "pending", "error"]


class RunOutput(BaseModel):
    """What an answer function returns for one question."""

    answer: Answer
    retrieved_refs: list[str] = Field(default_factory=list)
    model: str = "unknown"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    prompt_version: str | None = None
    prompt_hash: str | None = None  # sha256 of the full prompt, see models.prompt_hash
    # knowledge-layer items the answer relied on: [{item_id, version, tool, tier}], plus a
    # {"tool": "_meta", "meta": {knowledge, scope_gate}} entry (stored in runs.knowledge_items)
    knowledge_items: list[dict[str, Any]] = Field(default_factory=list)
    # knowledge / scope_keys come from KL-3 (#14); scope_status is the harness scope gate's result
    scope_status: Literal["in_scope", "partial", "out_of_scope"] | None = None


AnswerFn = Callable[[GoldenQuestion], RunOutput]


class JudgeVerdict(BaseModel):
    passed: bool
    reason: str = ""


class Judge(Protocol):
    """Connection point for an LLM judge (intentionally NOT implemented here).

    Called once per rubric item of kind ``llm``. Without a judge those items are reported
    as ``pending_judge``.
    """

    def judge(
        self, question: GoldenQuestion, output: RunOutput, item: RubricItem
    ) -> JudgeVerdict: ...


class CheckResult(BaseModel):
    id: str
    kind: str
    status: CheckStatus
    reason: str = ""


class QuestionResult(BaseModel):
    question_id: str
    category: str
    difficulty: str
    question_status: str
    model: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    verification: VerificationReport | None = None
    checks: list[CheckResult] = []
    output: RunOutput | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> QuestionStatus:
        """error: answer fn raised. failed: verifier or a check failed. pending: nothing failed
        but some items still await the judge (NOT counted as passed). passed: all else."""
        if self.error is not None or self.verification is None:
            return "error"
        if not self.verification.passed or any(c.status == "failed" for c in self.checks):
            return "failed"
        if any(c.status == "pending_judge" for c in self.checks):
            return "pending"
        return "passed"

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class EvalReport(BaseModel):
    golden_version: str
    golden_status: str
    started_at: str
    results: list[QuestionResult]

    def summary(self) -> dict[str, Any]:
        def rates(rs: list[QuestionResult]) -> dict[str, Any]:
            c = Counter(r.status for r in rs)
            decided = len(rs) - c["pending"]
            return {
                "questions": len(rs),
                "passed": c["passed"],
                "failed": c["failed"],
                "pending": c["pending"],
                "errors": c["error"],
                # pending counts as not passed
                "pass_rate": c["passed"] / len(rs) if rs else None,
                # pending questions left out of the denominator
                "pass_rate_excluding_pending": c["passed"] / decided if decided else None,
            }

        by_cat: dict[str, list[QuestionResult]] = defaultdict(list)
        codes: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        for r in self.results:
            by_cat[r.category].append(r)
            if r.verification:
                codes.update(f.check + ":" + f.code for f in r.verification.failures)
            statuses.update(c.status for c in r.checks)
        return {
            **rates(self.results),
            "draft_questions": sum(r.question_status == "draft" for r in self.results),
            "by_category": {c: rates(v) for c, v in sorted(by_cat.items())},
            "verifier_failure_codes": dict(sorted(codes.items())),
            "check_statuses": dict(sorted(statuses.items())),
        }

    def to_json(self) -> str:
        data = self.model_dump(mode="json")
        data["summary"] = self.summary()
        return json.dumps(data, ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# Eval report - golden set `{self.golden_version}` ({self.golden_status})",
            "",
            f"Started: {self.started_at}",
            "",
            f"- Questions: {s['questions']} (draft: {s['draft_questions']})",
            f"- Passed: {s['passed']}  |  Failed: {s['failed']}  |  Pending judge: {s['pending']}"
            f"  |  Errors: {s['errors']}",
            f"- Pass rate: {_pct(s['pass_rate'])} (pending counted as not passed); "
            f"excluding pending: {_pct(s['pass_rate_excluding_pending'])}",
            f"- Check statuses: {s['check_statuses'] or '-'}",
            f"- Verifier failure codes: {s['verifier_failure_codes'] or '-'}",
            "",
            "| Category | Questions | Passed | Failed | Pending | Errors | Pass rate "
            "| Excl. pending |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cat, v in s["by_category"].items():
            lines.append(
                f"| {cat} | {v['questions']} | {v['passed']} | {v['failed']} | {v['pending']} | "
                f"{v['errors']} | {_pct(v['pass_rate'])} | "
                f"{_pct(v['pass_rate_excluding_pending'])} |"
            )
        for r in self.results:
            lines += ["", f"## {r.question_id} - {r.status.upper()}"]
            lines.append(f"{r.category} / {r.difficulty} / {r.question_status}")
            if r.error:
                lines.append(f"- **Error:** {r.error.splitlines()[-1] if r.error else ''}")
            if r.verification:
                for f in r.verification.failures:
                    lines.append(f"- {f.check}:{f.code} - {f.reason}")
            for c in r.checks:
                if c.status != "passed":
                    lines.append(f"- check `{c.id}` {c.status}: {c.reason}")
        return "\n".join(lines) + "\n"

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        jp, mp = out / "report.json", out / "report.md"
        jp.write_text(self.to_json(), encoding="utf-8")
        mp.write_text(self.to_markdown(), encoding="utf-8")
        return jp, mp


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _has_placeholder(values: list[str | None]) -> bool:
    return any(v is None or is_placeholder(v) for v in values)


def _contains(text: str, term: str, norm: Normalizer) -> bool:
    return norm(term) in norm(text)


def _run_check(
    item: RubricItem,
    q: GoldenQuestion,
    out: RunOutput,
    verification: VerificationReport,
    norm: Normalizer,
    judge: Judge | None,
) -> CheckResult:
    def res(status: CheckStatus, reason: str = "") -> CheckResult:
        return CheckResult(id=item.id, kind=item.kind, status=status, reason=reason)

    text = out.answer.full_text()
    k = item.kind
    if k == "verifier_passes":
        return res(
            "passed" if verification.passed else "failed", ", ".join(verification.failure_codes())
        )
    if k == "llm":
        if judge is None:
            return res("pending_judge", "no judge configured")
        v = judge.judge(q, out, item)
        return res("passed" if v.passed else "failed", v.reason)
    if k == "cites_ref":
        if _has_placeholder([item.ref]):
            return res("skipped", "placeholder ref")
        ok = item.ref in out.answer.cited_refs()
        return res("passed" if ok else "failed", "" if ok else f"{item.ref!r} not cited")
    if k == "has_claim_type":
        try:
            wanted = {ClaimType(t) for t in item.claim_types}
        except ValueError:
            return res("skipped", f"unknown claim_types {item.claim_types!r}")
        if not wanted:
            return res("skipped", "no claim_types given")
        ok = any(c.type in wanted for c in out.answer.claims)
        return res(
            "passed" if ok else "failed", "" if ok else f"no claim of types {sorted(wanted)}"
        )
    if k in ("mentions_any", "mentions_all", "mentions_none"):
        if not item.terms or _has_placeholder(list(item.terms)):
            return res("skipped", "no terms / placeholder")
        hits = [t for t in item.terms if _contains(text, t, norm)]
        if k == "mentions_any":
            ok, why = bool(hits), "none of the terms found"
        elif k == "mentions_all":
            ok, why = (
                len(hits) == len(item.terms),
                f"missing: {[t for t in item.terms if t not in hits]}",
            )
        else:
            ok, why = not hits, f"forbidden terms present: {hits}"
        return res("passed" if ok else "failed", "" if ok else why)
    return res("skipped", f"unknown kind {k}")  # pragma: no cover


def _auto_checks(q: GoldenQuestion, out: RunOutput, norm: Normalizer) -> list[CheckResult]:
    results: list[CheckResult] = []
    cited = out.answer.cited_refs()
    for ref in q.required_refs:
        cid = f"required_ref:{ref}"
        if is_placeholder(ref):
            results.append(
                CheckResult(id=cid, kind="required_ref", status="skipped", reason="placeholder ref")
            )
        else:
            ok = ref in cited
            results.append(
                CheckResult(
                    id=cid,
                    kind="required_ref",
                    status="passed" if ok else "failed",
                    reason="" if ok else "required ref not cited",
                )
            )
    text = out.answer.full_text()
    for side in q.required_sides:
        cid = f"side:{side.label}"
        if not side.terms or is_placeholder(side.label) or _has_placeholder(list(side.terms)):
            results.append(
                CheckResult(
                    id=cid, kind="required_side", status="skipped", reason="placeholder side/terms"
                )
            )
            continue
        ok = any(_contains(text, t, norm) for t in side.terms)
        results.append(
            CheckResult(
                id=cid,
                kind="required_side",
                status="passed" if ok else "failed",
                reason="" if ok else "side not mentioned",
            )
        )
    return results


def run_eval(
    golden: GoldenSet,
    answer_fn: AnswerFn,
    store: SourceStore,
    *,
    judge: Judge | None = None,
    normalize: Normalizer = default_normalize,
    attribution_checker: AttributionChecker | None = None,
) -> EvalReport:
    results: list[QuestionResult] = []
    for q in golden.questions:
        base = dict(
            question_id=q.id,
            category=q.category.value,
            difficulty=q.difficulty,
            question_status=q.status,
        )
        t0 = time.perf_counter()
        try:
            out = answer_fn(q)
        except Exception:  # noqa: BLE001 - a failing answer fn must not abort the run
            results.append(QuestionResult(**base, error=traceback.format_exc()))
            continue
        latency = int((time.perf_counter() - t0) * 1000)
        verification = verify_answer(
            out.answer,
            out.retrieved_refs,
            store,
            normalize=normalize,
            attribution_checker=attribution_checker,
        )
        checks = _auto_checks(q, out, normalize)
        checks += [_run_check(i, q, out, verification, normalize, judge) for i in q.rubric]
        results.append(
            QuestionResult(
                **base,
                model=out.model,
                latency_ms=latency,
                verification=verification,
                checks=checks,
                output=out,
            )
        )
    return EvalReport(
        golden_version=golden.version,
        golden_status=golden.status,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        results=results,
    )
