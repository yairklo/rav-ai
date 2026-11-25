# rav-harness

First skeleton of the agent that answers halakhic questions. LiteLLM as the model layer
(model set in `config/harness.toml`, or `RAV_HARNESS_MODEL`), a thin hand-written loop
(`loop.py`), no LangChain/LangGraph.

- **Tools** (`tools.py`, deterministic, license-gated): `get_segment`, `neighbors`,
  `search_text` (Postgres FTS on form B, GIN index in migration 0002; hybrid/pgvector later),
  `resolve_citation` (`textnorm.find_citations`).
- **Output**: `rav_evals.schema.Answer`, submitted through a `submit_answer` tool. The loop
  rejects (and feeds back) answers whose system_conclusion/chidush claims are not flagged, or
  that cite refs not retrieved in the run; it forces `is_psak=False` and fills a default
  disclaimer.
- **Runs**: every run (also failures) is a row in `runs`: messages, tool calls, tokens, cost.
- **Evals**: `integration.make_answer_fn` / `run_golden`; the rav-evals verifier checks the answer
  against the DB (`rav_evals.pg_store.PostgresSourceStore`, can_index-gated) and its result is written back to the run row.

```
uv run rav-harness ask "..."            # needs a DB and a provider API key
uv run rav-harness eval golden.yaml
```

Tests use a scripted fake LLM (no network); DB tests need `docker compose up -d` (skipped
otherwise; `REQUIRE_DB=1` makes a missing DB a failure).

## Knowledge layer (KL-2)

`config/harness.toml` `knowledge = "off" | "base" | "base+lens"` (CLI: `--knowledge`, `--mode
personal|public`, `--allow-personal-only`). `off` is prompt `v0`, no new tools. Otherwise prompt
`v1-kl`, the deterministic **scope gate** (`[scope]`, `scope.py`: citations in the question +
`find_issues` + `coverage`) runs before the loop and is shown to the model and stored on the run,
and `knowledge_tools.py` adds `coverage`, `get_sugya`, `get_shitot`, `get_psak_chain`,
`find_issues`, `explain_abbrev`, `get_links` (+ `get_chakirot`, `get_method_principles` with the
lens). Reads `rav_ai.db.models` only.

The text is the authority, the layer a map: tool results carry `layer=knowledge` and a notice,
are filtered (current, not stale, verified/generated, `confidence >= min_confidence`,
`can_display_public` in public mode, license-gated), and **never** add to `retrieved_refs`: a
citation must be read with `get_segment`. `validators.py` holds the deterministic checks 4a-4j
(`[KL_...]` codes) run by `validate_answer`. `runs.knowledge_items` stores
`[{tool: "_meta", meta: {knowledge, scope_gate}}, {item_id, version, tool, tier}, ...]`.
