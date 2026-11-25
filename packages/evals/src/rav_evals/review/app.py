"""``rav-evals review``: the rating tool. FastAPI + Jinja, RTL, no JS framework, keyboard only,
listening on 127.0.0.1 only. Ratings are files in the repo (``golden/kl/ratings``)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..attribution import WRONG_SUBTYPES, misattribution_report
from ..config import FIXTURES_DIR, KL_GOLDEN_DIR, REPORTS_DIR, EvalsConfig, load_config
from ..pairs import DIRECTIONAL, LABELS
from ..ratings import ANSWER_GRADES, ERROR_TYPES, RatingStore
from . import queues

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
ALLOWED_HOST = "127.0.0.1"


@dataclass
class ReviewContext:
    rater_id: str
    store: RatingStore
    cfg: EvalsConfig = field(default_factory=load_config)
    fixtures_dir: Path = FIXTURES_DIR
    pairs_path: Path = KL_GOLDEN_DIR / "pairs.draft.jsonl"
    reports_dir: Path = REPORTS_DIR
    engine: Any = None  # SQLAlchemy engine: KL items queue + pair texts (optional)
    texts: dict[str, str] = field(default_factory=dict)  # pair texts without a database
    _answers: list | None = None

    def answers(self) -> list[queues.AnswerEntry]:
        if self._answers is None:
            self._answers = queues.answer_queue(self.fixtures_dir, self.cfg)
        return self._answers

    def text_for(self, ref: str) -> str | None:
        if ref in self.texts:
            return self.texts[ref]
        if self.engine is None:
            return None
        from sqlalchemy import text

        with self.engine.connect() as c:
            return c.execute(
                text("""SELECT st.text_raw FROM segment_text st
                        JOIN segments s ON s.id = st.segment_id
                        JOIN versions v ON v.id = st.version_id
                        WHERE s.ref = :r AND v.can_index
                        ORDER BY v.is_primary DESC, (v.language = 'he') DESC, v.id LIMIT 1"""),
                {"r": ref},
            ).scalar()

    def session(self):
        from sqlalchemy.orm import Session

        return Session(self.engine)


class ClaimRating(BaseModel):
    run_id: str
    claim_index: int
    verdict: str = Field(pattern="^(ok|wrong|fix)$")
    error_type: str | None = None
    subtype: str | None = None
    note: str = ""
    seconds: float | None = None


class GradeRating(BaseModel):
    run_id: str
    grade: str
    note: str = ""
    seconds: float | None = None


class PairRating(BaseModel):
    pair_id: str
    label: str
    direction: str | None = None  # ab | ba, for directional labels
    confidence: int = Field(default=2, ge=1, le=3)
    note: str = ""
    recheck: bool = False
    seconds: float | None = None


class ItemAction(BaseModel):
    action: str = Field(pattern="^(approve|reject|fix)$")
    claim_id: str = ""
    note: str = ""
    seconds: float | None = None


def _minutes(events: list[dict[str, Any]]) -> float | None:
    secs = [e["seconds"] for e in events if e.get("seconds")]
    return sum(secs) / len(secs) / 60 if secs else None


def create_app(ctx: ReviewContext) -> FastAPI:
    app = FastAPI(title="rav-evals review")
    app.state.ctx = ctx

    def render(request: Request, name: str, **kw) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, name, {"rater": ctx.rater_id, **kw})

    # ---- index / progress -------------------------------------------------------------

    def progress() -> dict[str, Any]:
        ans_events = list(ctx.store.events("answers", ctx.rater_id))
        claims = [
            {**e, "claim_type": "interpretation"} for e in ans_events if e.get("kind") == "claim"
        ]
        units = {e.run.fixture_id: e for e in ctx.answers()}
        interp = [
            c
            for c in claims
            if c["run_id"] in units
            and units[c["run_id"]].run.output.answer.claims[c["claim_index"]].type.value
            == "interpretation"
        ]
        mis = misattribution_report(
            interp, ctx.cfg.misattribution_max_rate, ctx.cfg.misattribution_ci_level
        )
        pairs = queues.pair_queue(ctx.pairs_path)
        rated_pairs = {k for (_r, k) in ctx.store.latest("pairs", "pair_id", ctx.rater_id)}
        item_events = list(ctx.store.events("items", ctx.rater_id))
        return {
            "answers": {
                "runs": len(units),
                "graded": len({e["run_id"] for e in ans_events if e.get("kind") == "grade"}),
                "claims_rated": len({(c["run_id"], c["claim_index"]) for c in claims}),
                "minutes_per_item": _minutes(ans_events),
                "misattribution": mis,
            },
            "pairs": {
                "total": len(pairs),
                "rated": len(rated_pairs & {p["pair_id"] for p in pairs}),
                "target": ctx.cfg.pairs_n,
                "minutes_per_item": _minutes(list(ctx.store.events("pairs", ctx.rater_id))),
            },
            "kl_items": {
                "reviewed": len(item_events),
                "queue": len(_item_ids()),
                "minutes_per_item": _minutes(item_events),
            },
        }

    def _item_ids() -> list[int]:
        if ctx.engine is None:
            return []
        from sqlalchemy import inspect

        if "kl_items" not in inspect(ctx.engine).get_table_names():
            return []
        with ctx.session() as s:
            return queues.item_queue(s, ctx.cfg)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return render(request, "index.html", p=progress(), gates=ctx.cfg.gates())

    @app.get("/api/progress")
    def api_progress():
        return progress()

    # ---- (a) answers --------------------------------------------------------------------

    @app.get("/answers", response_class=HTMLResponse)
    def answers(request: Request, i: int = 0):
        q = ctx.answers()
        if not q:
            return render(request, "empty.html", queue="answers", why="no recorded runs")
        i = max(0, min(i, len(q) - 1))
        e = q[i]
        done = {
            int(k.split("#")[1]): v
            for (_r, k), v in ctx.store.latest("answers", "unit_id", ctx.rater_id).items()
            if k.startswith(e.run.fixture_id + "#")
        }
        return render(
            request,
            "answers.html",
            e=e,
            i=i,
            n=len(q),
            done=done,
            error_types=ERROR_TYPES,
            subtypes=WRONG_SUBTYPES,
            grades=ANSWER_GRADES,
            t0=time.time(),
        )

    @app.post("/api/answers/claim")
    def rate_claim(r: ClaimRating):
        entry = _entry(r.run_id)
        claims = entry.run.output.answer.claims
        if not 0 <= r.claim_index < len(claims):
            raise HTTPException(422, "claim_index out of range")
        if r.verdict != "ok" and r.error_type not in ERROR_TYPES:
            raise HTTPException(422, f"error_type must be one of {ERROR_TYPES}")
        if r.subtype and r.subtype not in WRONG_SUBTYPES:
            raise HTTPException(422, f"subtype must be one of {WRONG_SUBTYPES}")
        rec = ctx.store.append(
            ctx.rater_id,
            "answers",
            {
                "kind": "claim",
                "unit_id": f"{r.run_id}#{r.claim_index}",
                "claim_type": claims[r.claim_index].type.value,
                **r.model_dump(),
                "error_type": None if r.verdict == "ok" else r.error_type,
            },
        )
        exported = None
        if r.verdict != "ok":
            exported = queues.export_errors(
                ctx.answers(), ctx.store, ctx.fixtures_dir, ctx.reports_dir, ctx.cfg
            )
        return {"saved": rec, "exported": exported}

    @app.post("/api/answers/grade")
    def grade(r: GradeRating):
        _entry(r.run_id)
        if r.grade not in ANSWER_GRADES:
            raise HTTPException(422, f"grade must be one of {ANSWER_GRADES}")
        return {
            "saved": ctx.store.append(ctx.rater_id, "answers", {"kind": "grade", **r.model_dump()})
        }

    def _entry(run_id: str) -> queues.AnswerEntry:
        for e in ctx.answers():
            if e.run.fixture_id == run_id:
                return e
        raise HTTPException(404, f"unknown run {run_id}")

    @app.post("/api/export")
    def export():
        return queues.export_errors(
            ctx.answers(), ctx.store, ctx.fixtures_dir, ctx.reports_dir, ctx.cfg
        )

    # ---- (b) pairs ----------------------------------------------------------------------

    @app.get("/pairs", response_class=HTMLResponse)
    def pairs(request: Request, i: int = 0):
        q = queues.pair_queue(ctx.pairs_path)
        if not q:
            return render(
                request, "empty.html", queue="pairs", why="no pairs file (rav-evals pairs-sample)"
            )
        i = max(0, min(i, len(q) - 1))
        p = q[i]
        left, right = (p["b_ref"], p["a_ref"]) if p["swapped"] else (p["a_ref"], p["b_ref"])
        prev = ctx.store.latest("pairs", "pair_id", ctx.rater_id).get((ctx.rater_id, p["pair_id"]))
        return render(
            request,
            "pairs.html",
            p=p,
            i=i,
            n=len(q),
            left=left,
            right=right,
            left_text=ctx.text_for(left),
            right_text=ctx.text_for(right),
            labels=LABELS,
            directional=sorted(DIRECTIONAL),
            prev=prev,
            t0=time.time(),
        )

    @app.post("/api/pairs/rate")
    def rate_pair(r: PairRating):
        if r.label not in LABELS:
            raise HTTPException(422, f"label must be one of {LABELS}")
        known = {p["pair_id"] for p in queues.pair_queue(ctx.pairs_path)}
        if r.pair_id not in known:
            raise HTTPException(404, f"unknown pair {r.pair_id}")
        if r.label in DIRECTIONAL and r.direction not in ("ab", "ba"):
            raise HTTPException(422, "directional label needs direction ab|ba")
        data = r.model_dump()
        if r.label not in DIRECTIONAL:
            data["direction"] = None
        return {"saved": ctx.store.append(ctx.rater_id, "pairs", data)}

    # ---- (c) knowledge-layer items --------------------------------------------------------

    @app.get("/items", response_class=HTMLResponse)
    def items(request: Request, i: int = 0, diff: int = 0):
        ids = _item_ids()
        if not ids:
            return render(
                request, "empty.html", queue="items", why="no knowledge-layer items to review yet"
            )
        i = max(0, min(i, len(ids) - 1))
        with ctx.session() as s:
            v = queues.item_view(s, ids[i])
            return render(
                request, "items.html", v=v, i=i, n=len(ids), show_diff=bool(diff), t0=time.time()
            )

    @app.post("/api/items/{item_id}/action")
    def item_action(item_id: int, a: ItemAction):
        if ctx.engine is None:
            raise HTTPException(409, "no database configured")
        with ctx.session() as s:
            try:
                res = queues.item_action(
                    s, item_id, a.action, ctx.rater_id, claim_id=a.claim_id, note=a.note
                )
            except KeyError as e:
                raise HTTPException(404, f"unknown item {item_id}") from e
        ctx.store.append(
            ctx.rater_id,
            "items",
            {"item_id": item_id, **a.model_dump(), "result_status": res["status"]},
        )
        return res

    @app.get("/next/{queue}")
    def nxt(queue: str, i: int = 0) -> RedirectResponse:
        return RedirectResponse(f"/{queue}?i={i + 1}", status_code=303)

    @app.exception_handler(ValueError)
    def bad_value(_req: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    return app


def server_config(app: FastAPI, host: str = ALLOWED_HOST, port: int = 8765):
    """uvicorn config bound to 127.0.0.1; any other host is refused."""
    import uvicorn

    if host != ALLOWED_HOST:
        raise ValueError(f"the review server listens on {ALLOWED_HOST} only, not {host!r}")
    return uvicorn.Config(app, host=ALLOWED_HOST, port=port, log_level="info")


def serve(ctx: ReviewContext, port: int | None = None, opener: Callable | None = None) -> None:
    import uvicorn

    cfg = server_config(create_app(ctx), ctx.cfg.review_host, port or ctx.cfg.review_port)
    if opener:
        opener(f"http://{cfg.host}:{cfg.port}/")
    uvicorn.Server(cfg).run()
