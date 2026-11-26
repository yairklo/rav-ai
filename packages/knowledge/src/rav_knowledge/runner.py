"""Job / task runner: idempotent, resumable, budgeted; promote / rollback; overrides; invalidate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rav_ai.config import ROOT
from rav_ai.db.models import (
    Annotation,
    KlEvidence,
    KlItem,
    KlItemSegment,
    KlJob,
    KlOverride,
    KlTask,
    Segment,
    SegmentText,
)
from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .backends import StopRun
from .config import KLConfig, load_config
from .evidence import SegText, reanchor
from .stages import DAG, STAGES, RunContext, TaskSpec, stage_prompt
from .store import (
    add_segments,
    copy_evidence,
    current_item,
    load_scope,
    mark_stale_dependents,
    overrides_for,
    sha,
    write_item,
)

REPORT_DIR = ROOT / "reports" / "kl"


class PromoteRefused(Exception):
    pass


@dataclass
class RunReport:
    job_id: int | None
    stage: str
    scope_key: str
    status: str = "done"
    tasks: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    est_cost_usd: float | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    stopped: str | None = None  # why the run stopped early (usage limit, --limit, auth)
    resume: str | None = None  # the command that continues the same job


def _now() -> datetime:
    return datetime.now(UTC)


# ---- overrides ----------------------------------------------------------------------------


def _human_version(
    s: Session,
    it: KlItem,
    o: KlOverride,
    *,
    body: dict,
    start_id: int | None = None,
    end_id: int | None = None,
    segment_ids: list[int] | None = None,
    cfg: KLConfig | None = None,
) -> tuple[KlItem, bool]:
    """A human decision never edits an item: it is a NEW version of the same item_key (verified,
    no model, prompt_version='human'), made current; the previous one is kept, not current.
    Returns (item, created); re-applying the same override to the same version creates nothing."""
    h = sha(o.id, it.id)
    existed = s.scalars(
        select(KlItem.id).where(KlItem.item_key == it.item_key, KlItem.input_hash == h)
    ).first()
    title = body.get("title_he") or it.title_he
    new = write_item(
        s,
        kind=it.kind,
        item_key=it.item_key,
        scope_key=it.scope_key,
        body=body,
        input_hash=h,
        job_id=it.job_id,
        model=None,
        prompt_version="human",
        prompt_hash=None,
        schema_version=it.schema_version,
        can_display_public=it.can_display_public,
        can_train=it.can_train,
        status="verified",
        work_id=it.work_id,
        start_segment_id=start_id or it.start_segment_id,
        end_segment_id=end_id or it.end_segment_id,
        title_he=title,
        confidence=1.0,
        source_version_ids=it.source_version_ids,
        check_report=it.check_report,
        search_text_he=title if it.kind == "sugya" else it.search_text_he,
        make_current=True,
    )
    if existed is None:
        if segment_ids is None:
            segment_ids = list(
                s.scalars(
                    select(KlItemSegment.segment_id)
                    .where(KlItemSegment.item_id == it.id)
                    .order_by(KlItemSegment.ord)
                )
            )
            roles = dict(
                s.execute(
                    select(KlItemSegment.segment_id, KlItemSegment.role).where(
                        KlItemSegment.item_id == it.id
                    )
                ).all()
            )
        else:
            roles = {}
        add_segments(
            s,
            new,
            [
                (sid, roles.get(sid, "anchor" if i == 0 else "member"))
                for i, sid in enumerate(segment_ids)
            ],
        )
        c = cfg or load_config()
        copy_evidence(s, it.id, new.id, min_words=c.quote_min_words, min_chars=c.quote_min_chars)
        s.flush()
    return new, existed is None


def apply_overrides(
    s: Session,
    item_ids: list[int],
    annotation_ids: list[int] | None = None,
    cfg: KLConfig | None = None,
) -> int:
    """Human decisions win over what a task just produced. Returns the number of conflicts: a
    decision that the NEW output disagrees with (an output equal to the decision is no conflict;
    a decision already applied to this output is not counted again)."""
    conflicts = 0
    items = [s.get(KlItem, i) for i in item_ids]
    ovs = overrides_for(s, [i.item_key for i in items if i])
    for it in items:
        o = ovs.get(it.item_key) if it else None
        if o is None:
            continue
        if o.action == "reject":
            it.status, it.rejected_by, it.is_current = "rejected", "human", False
            conflicts += 1
        elif o.action == "edit" and isinstance(o.corrected, dict):
            if all(it.body.get(k) == v for k, v in o.corrected.items()):
                continue  # the output already says what the human decided
            _, created = _human_version(s, it, o, body={**it.body, **o.corrected}, cfg=cfg)
            conflicts += created
        elif o.action == "approve":
            if (o.corrected or {}).get("input_hash") == it.input_hash:
                it.status = "verified"
            else:
                conflicts += 1
        elif o.action == "set_boundary" and isinstance(o.corrected, dict):
            conflicts += _set_boundary(s, it, o, cfg)
    for aid in annotation_ids or []:
        a = s.get(Annotation, aid)
        if a is None:
            continue
        key = f"abbrev:{a.segment_id}:{a.span_start}:{a.span_end}"
        o = overrides_for(s, [key]).get(key)
        if o and o.action == "set_abbrev" and (o.corrected or {}).get("chosen"):
            a.chosen, a.status, a.origin = o.corrected["chosen"], "verified", "human"
            conflicts += 1
    s.flush()
    return conflicts


def _set_boundary(s: Session, it: KlItem, o: KlOverride, cfg: KLConfig | None = None) -> int:
    """corrected = {"start_ref": ..., "end_ref": ...} (either optional). Returns 1 on a new
    human version, 0 when the output already has these boundaries."""
    corr = o.corrected
    start = s.scalars(select(Segment).where(Segment.ref == corr.get("start_ref", ""))).first()
    end = s.scalars(select(Segment).where(Segment.ref == corr.get("end_ref", ""))).first()
    start = start or s.get(Segment, it.start_segment_id)
    end = end or s.get(Segment, it.end_segment_id)
    if (start.id, end.id) == (it.start_segment_id, it.end_segment_id):
        return 0
    ids = list(
        s.scalars(
            select(Segment.id)
            .where(
                Segment.work_id == it.work_id,
                Segment.sort_key >= start.sort_key,
                Segment.sort_key <= end.sort_key,
            )
            .order_by(Segment.sort_key, Segment.id)
        )
    )
    _, created = _human_version(
        s,
        it,
        o,
        body={**it.body, "start_ref": start.ref, "end_ref": end.ref},
        start_id=start.id,
        end_id=end.id,
        segment_ids=ids,
        cfg=cfg,
    )
    return int(created)


# ---- run ----------------------------------------------------------------------------------


def _params_json(cfg: KLConfig) -> dict:
    return cfg.params()


def _find_or_create_job(s: Session, rc: RunContext, cfg: KLConfig) -> KlJob:
    ph = rc.prompt.hash if rc.prompt else sha("s0", "method")
    pv = rc.prompt.version if rc.prompt else "deterministic"
    job = None
    for j in s.scalars(
        select(KlJob)
        .where(
            KlJob.stage == rc.stage,
            KlJob.scope_key == rc.scope.key,
            KlJob.status.in_(("running", "failed", "aborted")),
            KlJob.prompt_hash == ph,
            KlJob.model == rc.model,
        )
        .order_by(KlJob.id.desc())
    ):
        if j.params == _params_json(cfg):
            job = j
            break
    if job is None:
        job = KlJob(
            stage=rc.stage,
            scope_key=rc.scope.key,
            model=rc.model,
            prompt_version=pv,
            prompt_hash=ph,
            schema_version=1,
            params=_params_json(cfg),
            status="running",
        )
        s.add(job)
    else:  # resume: a stuck 'running' task is reset
        job.status = "running"
        s.execute(
            update(KlTask)
            .where(KlTask.job_id == job.id, KlTask.state == "running")
            .values(state="pending")
        )
    s.flush()
    return job


def _prior_done(s: Session, rc: RunContext, spec: TaskSpec, job_id: int) -> KlTask | None:
    return s.scalars(
        select(KlTask)
        .join(KlJob, KlJob.id == KlTask.job_id)
        .where(
            KlJob.stage == rc.stage,
            KlJob.scope_key == rc.scope.key,
            KlJob.id != job_id,
            KlTask.input_hash == spec.input_hash,
            KlTask.state.in_(("done", "skipped")),
        )
        .order_by(KlTask.id.desc())
    ).first()


def run_stage(
    engine: Engine,
    stage: str,
    scope_key: str,
    cfg: KLConfig,
    llm: Any,
    *,
    budget_usd: float | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> RunReport:
    """``limit``: at most this many NEW tasks (LLM work) in this invocation; then the job stops as
    'aborted' and the same command resumes it. A ``StopRun`` from the backend (usage limit, not
    logged in) rolls back the task's writes, puts it back to 'pending' and stops the loop."""
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} is reserved in the DAG but not implemented (PR KL-2+)")
    st = STAGES[stage]
    budget = budget_usd if budget_usd is not None else (cfg.budget_usd or None)
    if cfg.backend != "api":
        budget = None  # a USD budget is an API-path control; the subscription has its own limits
    with Session(engine) as s:
        scope = load_scope(s, scope_key)
        model = None if stage == "s0" else cfg.model_for(stage)
        rc = RunContext(cfg, llm, scope, stage, model, stage_prompt(stage))
        report = RunReport(None, stage, scope_key)
        if dry_run:
            specs = st.plan(rc, s)
            report.tasks = len(specs)
            report.est_input_tokens = sum(x.est_in for x in specs)
            report.est_output_tokens = sum(x.est_out for x in specs)
            if model:
                report.est_cost_usd = cfg.cost(
                    model, report.est_input_tokens, report.est_output_tokens
                )
            report.stats = dict(rc.stats)
            return report
        job = _find_or_create_job(s, rc, cfg)
        rc.job_id = job.id
        specs = st.plan(rc, s)
        for sp in specs:
            t = s.scalars(
                select(KlTask).where(KlTask.job_id == job.id, KlTask.task_key == sp.key)
            ).first()
            if t is None:
                s.add(KlTask(job_id=job.id, task_key=sp.key, input_hash=sp.input_hash))
            elif t.input_hash != sp.input_hash:  # input changed since the interrupted run
                t.input_hash, t.state, t.result = sp.input_hash, "pending", None
        s.commit()
        job_id = job.id
    report.job_id = job_id
    report.tasks = len(specs)
    spent = 0.0
    with Session(engine) as s:
        spent = float(s.get(KlJob, job_id).cost_usd or 0.0)
    aborted = False
    started = 0
    drain = getattr(llm, "drain_calls", None)
    for sp in specs:
        with Session(engine) as s:
            t = s.scalars(
                select(KlTask).where(KlTask.job_id == job_id, KlTask.task_key == sp.key)
            ).one()
            if t.state in ("done", "skipped"):
                rc.prior_results[sp.key] = t.result
                report.skipped += t.state == "skipped"
                report.done += t.state == "done"
                continue
            prior = _prior_done(s, rc, sp, job_id)
            if prior is not None:
                t.state, t.result = "skipped", prior.result
                rc.prior_results[sp.key] = prior.result
                t.updated_at = _now()
                s.commit()
                report.skipped += 1
                continue
            if sp.payload.get("merge") and any(
                x.key != sp.key and x.key not in rc.prior_results for x in specs
            ):
                t.state, t.last_error = "failed", "dependency: a window task did not finish"
                s.commit()
                report.failed += 1
                report.errors.append(f"{sp.key}: {t.last_error}")
                continue
            est = cfg.cost(rc.model, sp.est_in, sp.est_out) if rc.model else 0.0
            if budget and est is not None and spent + est > budget:
                aborted = True
                break
            if limit is not None and started >= limit:
                aborted = True
                report.stopped = f"--limit {limit} reached"
                break
            started += 1
            t.state, t.attempts = "running", t.attempts + 1
            s.commit()
        try:
            with Session(engine) as s:
                out = st.execute(rc, s, sp)
                conflicts = apply_overrides(s, out.item_ids, out.annotation_ids, cfg)
                t = s.scalars(
                    select(KlTask).where(KlTask.job_id == job_id, KlTask.task_key == sp.key)
                ).one()
                result = out.result
                calls = drain() if drain else []
                if calls and (result is None or isinstance(result, dict)):
                    result = {**(result or {}), "_llm": _task_llm_meta(llm, calls)}
                t.state, t.last_error, t.result = "done", None, result
                t.input_tokens, t.output_tokens = out.usage.input_tokens, out.usage.output_tokens
                t.cost_usd = out.usage.cost_usd
                t.updated_at = _now()
                s.commit()
            spent += out.usage.cost_usd
            report.done += 1
            report.llm_calls += out.usage.calls
            report.input_tokens += out.usage.input_tokens
            report.output_tokens += out.usage.output_tokens
            report.cost_usd += out.usage.cost_usd
            rc.prior_results[sp.key] = out.result
            for k, v in out.stats.items():
                rc.bump(k, v)
            if conflicts:
                rc.bump("override_conflicts", conflicts)
        except StopRun as e:  # the task's session was never committed: its writes are gone
            if drain:
                drain()
            with Session(engine) as s:
                t = s.scalars(
                    select(KlTask).where(KlTask.job_id == job_id, KlTask.task_key == sp.key)
                ).one()
                t.state, t.last_error = "pending", f"{type(e).__name__}: {e}"[:2000]
                t.updated_at = _now()
                s.commit()
            aborted = True
            report.stopped = f"{type(e).__name__}: {e}"
            break
        except Exception as e:  # noqa: BLE001 - a failed task must not stop the others
            if drain:
                drain()
            usage = getattr(e, "usage", None)
            with Session(engine) as s:
                t = s.scalars(
                    select(KlTask).where(KlTask.job_id == job_id, KlTask.task_key == sp.key)
                ).one()
                t.state, t.last_error = "failed", f"{type(e).__name__}: {e}"[:2000]
                if usage:
                    t.input_tokens, t.output_tokens = usage.input_tokens, usage.output_tokens
                    t.cost_usd = usage.cost_usd
                s.commit()
            if usage:
                spent += usage.cost_usd
                report.llm_calls += usage.calls
                report.input_tokens += usage.input_tokens
                report.output_tokens += usage.output_tokens
                report.cost_usd += usage.cost_usd
            report.failed += 1
            report.errors.append(f"{sp.key}: {e}")
    with Session(engine) as s:
        job = s.get(KlJob, job_id)
        job.status = "aborted" if aborted else ("failed" if report.failed else "done")
        agg = s.execute(
            select(
                func.coalesce(func.sum(KlTask.input_tokens), 0),
                func.coalesce(func.sum(KlTask.output_tokens), 0),
                func.coalesce(func.sum(KlTask.cost_usd), 0.0),
            ).where(KlTask.job_id == job_id)
        ).one()
        job.input_tokens, job.output_tokens, job.cost_usd = int(agg[0]), int(agg[1]), float(agg[2])
        job.stats = {**rc.stats, "llm_calls": report.llm_calls, "skipped": report.skipped}
        meta = getattr(llm, "job_meta", None)
        if meta and stage != "s0":
            job.stats = {**(job.stats or {}), **meta()}
        if report.stopped:
            job.stats = {**job.stats, "stopped": report.stopped}
        job.finished_at = _now()
        report.status = job.status
        report.stats = job.stats
        s.commit()
    if aborted:
        report.resume = f'rav-kl run {stage} "{scope_key}"' + (
            f" --limit {limit}" if limit is not None else ""
        )
        if cfg.backend != "api":
            report.resume += f"  (backend {cfg.backend}: same config / RAV_KL_BACKEND)"
    return report


def _task_llm_meta(llm: Any, calls: list[dict]) -> dict:
    """Per-task record of the backend calls (kl_tasks has no params/stats column)."""
    return {
        "backend": getattr(llm, "backend", "api"),
        "calls": len(calls),
        "models_reported": sorted({c["model_reported"] for c in calls if c.get("model_reported")}),
        "cost_kind": "subscription",
        "cli_cost_usd": round(sum(c.get("total_cost_usd") or 0.0 for c in calls), 6),
        "failed_calls": [c.get("error") for c in calls if not c.get("ok")],
    }


# ---- promote ------------------------------------------------------------------------------


def evaluate_gate(s: Session, job: KlJob, cfg: KLConfig) -> tuple[bool, dict]:
    """The gate from ``rav-kl report`` if one was written for this job, else only the K checks."""
    f = REPORT_DIR / f"job_{job.id}.json"
    if f.exists():
        g = json.loads(f.read_text(encoding="utf-8")).get("gate", {})
        return bool(g.get("passed")), {"source": "kl-report", **g}
    items = list(s.scalars(select(KlItem).where(KlItem.job_id == job.id)))
    items = [i for i in items if not (i.check_report or {}).get("shadow")]
    n_ann = s.scalar(
        select(func.count()).select_from(Annotation).where(Annotation.job_id == job.id)
    )
    if not items:
        ok = bool(n_ann)
        return ok, {"source": "k-checks", "items": 0, "annotations": n_ann}
    rejected = sum(1 for i in items if i.status == "rejected")
    bad = [
        i.item_key
        for i in items
        if i.status != "rejected" and not (i.check_report or {}).get("ok", True)
    ]
    ok = not bad and rejected / len(items) <= cfg.gate_max_rejected_fraction
    return ok, {"source": "k-checks", "items": len(items), "rejected": rejected, "failing": bad}


def promote(
    engine: Engine, job_id: int, *, force_note: str | None = None, cfg: KLConfig | None = None
) -> dict:
    """Make a job's output current (rollback = promote an earlier job)."""
    with Session(engine) as s:
        job = s.get(KlJob, job_id)
        if job is None:
            raise PromoteRefused(f"no job {job_id}")
        passed, gate = evaluate_gate(s, job, cfg or load_config())
        if job.status in ("failed", "aborted", "running") and not force_note:
            raise PromoteRefused(f"job {job_id} is {job.status}")
        if not passed and not force_note:
            raise PromoteRefused(f"gate failed: {gate}")
        if force_note is not None and not force_note.strip():
            raise PromoteRefused("--force needs a note")
        stats = {"promoted": 0, "blocked_verified": 0, "stale": 0, "superseded": 0}
        replaced: list[int] = []
        items = [
            i
            for i in s.scalars(select(KlItem).where(KlItem.job_id == job_id).order_by(KlItem.id))
            if i.status != "rejected" and not (i.check_report or {}).get("shadow")
        ]
        keys = []
        for it in items:
            keys.append(it.item_key)
            old = current_item(s, it.item_key)
            if old is not None and old.id == it.id:
                continue
            if old is not None:
                if old.status == "verified" and not force_note:
                    stats["blocked_verified"] += 1
                    continue
                old.is_current = False
                replaced.append(old.id)
                s.flush()
            it.is_current = True
            stats["promoted"] += 1
            s.flush()
        anns = list(s.scalars(select(Annotation).where(Annotation.job_id == job_id)))
        for a in anns:
            if a.is_current:
                continue
            s.execute(
                update(Annotation)
                .where(
                    Annotation.segment_id == a.segment_id,
                    Annotation.version_id == a.version_id,
                    Annotation.span_start == a.span_start,
                    Annotation.span_end == a.span_end,
                    Annotation.kind == a.kind,
                    Annotation.is_current,
                )
                .values(is_current=False)
            )
            a.is_current = True
            s.flush()
            stats["promoted"] += 1
        if job.stage == "s1":  # supersede boundaries at scope level
            seg_ids = load_scope(s, job.scope_key).segment_ids
            for old in s.scalars(
                select(KlItem).where(
                    KlItem.kind == "sugya",
                    KlItem.is_current,
                    KlItem.start_segment_id.in_(seg_ids),
                    KlItem.job_id != job_id,
                    KlItem.item_key.not_in(keys or [""]),
                )
            ):
                old.is_current = False
                replaced.append(old.id)
                stats["superseded"] += 1
        s.flush()
        stats["stale"] = mark_stale_dependents(s, replaced)
        s.execute(
            update(KlJob)
            .where(
                KlJob.stage == job.stage,
                KlJob.scope_key == job.scope_key,
                KlJob.status == "promoted",
            )
            .values(status="done")
        )
        job.status = "promoted"
        job.note = (job.note or "") + (f" force: {force_note}" if force_note else "")
        s.commit()
        return {"job": job_id, "gate": gate, "forced": bool(force_note), **stats}


# ---- invalidate / status / report ------------------------------------------------------------


def invalidate(engine: Engine, segment_ids: list[int] | None = None) -> dict:
    """Re-anchor evidence whose segment text changed; stale (recursively) only if not found."""
    out = {"checked": 0, "reanchored": 0, "stale_items": 0}
    with Session(engine) as s:
        q = (
            select(KlEvidence, SegmentText)
            .join(
                SegmentText,
                (SegmentText.segment_id == KlEvidence.segment_id)
                & (SegmentText.version_id == KlEvidence.version_id),
            )
            .where(KlEvidence.raw_hash.is_not(None), KlEvidence.raw_hash != SegmentText.raw_hash)
        )
        if segment_ids:
            q = q.where(KlEvidence.segment_id.in_(segment_ids))
        lost: set[int] = set()
        for ev, st in s.execute(q).all():
            out["checked"] += 1
            seg = SegText(
                st.segment_id, st.version_id, "", st.raw_hash, st.text_a, st.text_b, st.offsets_b
            )
            loc, stale = reanchor(ev.quote, ev.quote_prefix, ev.quote_suffix, ev.raw_hash, seg)
            if stale:
                ev.locate_status = "not_found"
                lost.add(ev.item_id)
            elif loc:
                ev.span_start, ev.span_end = loc.span_start, loc.span_end
                ev.quote_prefix, ev.quote_suffix, ev.raw_hash = (
                    loc.quote_prefix,
                    loc.quote_suffix,
                    loc.raw_hash,
                )
                ev.locate_status = loc.status
                out["reanchored"] += 1
        if lost:
            s.execute(update(KlItem).where(KlItem.id.in_(lost)).values(stale=True))
            out["stale_items"] = len(lost) + mark_stale_dependents(s, sorted(lost))
        s.commit()
    return out


def status(engine: Engine) -> list[dict]:
    with Session(engine) as s:
        rows = s.scalars(select(KlJob).order_by(KlJob.id.desc()).limit(50))
        res = []
        for j in rows:
            counts = dict(
                s.execute(
                    select(KlTask.state, func.count())
                    .where(KlTask.job_id == j.id)
                    .group_by(KlTask.state)
                ).all()
            )
            res.append(
                {
                    "job": j.id,
                    "stage": j.stage,
                    "scope": j.scope_key,
                    "status": j.status,
                    "model": j.model,
                    "tasks": counts,
                    "cost_usd": round(j.cost_usd or 0.0, 4),
                }
            )
        return res


def report(
    engine: Engine, job_id: int, out_dir: Path | None = None, cfg: KLConfig | None = None
) -> dict:
    """Grounding metrics per job -> reports/kl/job_<id>.json (+ the gate used by promote)."""
    with Session(engine) as s:
        job = s.get(KlJob, job_id)
        items = [
            i
            for i in s.scalars(select(KlItem).where(KlItem.job_id == job_id))
            if not (i.check_report or {}).get("shadow")
        ]
        ev = s.execute(
            select(KlEvidence.locate_status, func.count())
            .join(KlItem, KlItem.id == KlEvidence.item_id)
            .where(KlItem.job_id == job_id)
            .group_by(KlEvidence.locate_status)
        ).all()
        evc = {k: v for k, v in ev}
        total = sum(evc.values())
        located = evc.get("located", 0) + evc.get("ambiguous", 0)
        rejected = sum(1 for i in items if i.status == "rejected")
        codes: dict[str, int] = {}
        for i in items:
            for f in (i.check_report or {}).get("findings", []):
                codes[f["code"]] = codes.get(f["code"], 0) + 1
        cfg = cfg or load_config()
        frac = located / total if total else 1.0
        passed = (
            bool(items or job.stage in ("s0", "s0b"))
            and frac >= cfg.gate_min_located_fraction
            and (not items or rejected / len(items) <= cfg.gate_max_rejected_fraction)
        )
        rep = {
            "job": job_id,
            "stage": job.stage,
            "scope": job.scope_key,
            "items": len(items),
            "rejected": rejected,
            "evidence": evc,
            "located_fraction": round(frac, 4),
            "finding_codes": codes,
            "gate": {
                "passed": passed,
                "min_located_fraction": cfg.gate_min_located_fraction,
                "max_rejected_fraction": cfg.gate_max_rejected_fraction,
            },
        }
    d = out_dir or REPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"job_{job_id}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), "utf-8")
    return rep


_ = DAG
