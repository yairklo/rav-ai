"""Live run: ``rav-evals run --knowledge off|base|base+lens --split dev``.

Manual only (needs an API key in the environment and a corpus database); never in CI. Imports
the harness lazily: rav-harness depends on rav-evals, not the other way round.

Every result can be saved as a recorded fixture (``--record-fixtures DIR``): question, tool
transcript, answer, the text of every source read and a knowledge-layer snapshot, so the run
becomes a regression case that ``rav-evals regress`` replays without a key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import EvalsConfig, load_config
from .golden import GoldenQuestion, GoldenSet
from .kl_lookup import PostgresKnowledgeLookup
from .kl_metrics import RecordedRun, compute_metrics, evaluate_question
from .runner import RunOutput


def _harness_answer_fn(session, knowledge: str, model: str | None, sink: dict[str, Any]):
    from rav_harness.config import load_config as load_harness_config
    from rav_harness.integration import build_harness
    from rav_harness.llm import LiteLLMClient
    from rav_harness.prompt import PROMPT_VERSION, prompt_hash

    cfg = load_harness_config()
    if model:
        cfg = cfg.with_model(model)
    if hasattr(cfg, "with_knowledge"):
        cfg = cfg.with_knowledge(knowledge)
    elif knowledge != "off":
        raise SystemExit(
            f"--knowledge {knowledge} needs the KL-2 harness (HarnessConfig.with_knowledge); "
            "this checkout only supports --knowledge off"
        )
    llm = LiteLLMClient()

    def answer_fn(q: GoldenQuestion) -> RunOutput:
        res = build_harness(session, llm, cfg, record=True).run(q.question, question_id=q.id)
        sink[q.id] = res
        if res.answer is None:
            raise RuntimeError(res.error or "harness produced no answer")
        return RunOutput(
            answer=res.answer,
            retrieved_refs=res.retrieved_refs,
            model=res.model,
            tool_calls=res.tool_calls,
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash(),
            knowledge=knowledge,
            scope_keys=list(q.scope_keys),
            knowledge_items=list(getattr(res, "knowledge_items", []) or []),
            cost_usd=res.cost_usd,
        )

    return answer_fn


def run_live(
    golden: GoldenSet,
    database_url: str,
    knowledge: str,
    out_dir: Path,
    *,
    model: str | None = None,
    record_fixtures: Path | None = None,
    cfg: EvalsConfig | None = None,
) -> dict[str, Any]:
    from rav_ai.db.session import make_engine, make_session

    from .pg_store import PostgresSourceStore

    cfg = cfg or load_config()
    engine = make_engine(database_url)
    session = make_session(engine)
    store = PostgresSourceStore(engine)
    kn = PostgresKnowledgeLookup(engine)
    sink: dict[str, Any] = {}
    answer_fn = _harness_answer_fn(session, knowledge, model, sink)
    results, syco = [], []
    for q in golden.questions:
        try:
            out = answer_fn(q)
        except Exception as e:  # noqa: BLE001 - one failing question must not abort the run
            print(f"{q.id}: error {e}")
            continue
        res, s = evaluate_question(q, out, store, kn, cfg)
        results.append(res)
        if s is not None:
            syco.append(s)
        if record_fixtures is not None:
            refs = sorted(set(out.retrieved_refs) | out.answer.cited_refs())
            rr = RecordedRun(
                fixture_id=f"live-{knowledge.replace('+', '_')}-{q.id}",
                question=q,
                transcript=list(getattr(sink.get(q.id), "messages", []) or []),
                output=out,
                sources={r: t for r in refs if (t := store.get_text(r)) is not None},
                knowledge=kn.snapshot(out, list(q.scope_keys)),
            )
            record_fixtures.mkdir(parents=True, exist_ok=True)
            (record_fixtures / f"{rr.fixture_id}.json").write_text(
                rr.model_dump_json(indent=2), encoding="utf-8"
            )
    metrics = compute_metrics(results, syco, cfg)
    bundle = {
        "label": f"knowledge={knowledge}",
        "knowledge": knowledge,
        "golden_version": golden.version,
        "metrics": metrics,
        "results": [r.model_dump(mode="json") for r in results],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle
