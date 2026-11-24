# rav-evals

Answer schema, deterministic citation verifier (V1-V4), golden-set format and eval runner.

```powershell
cd packages/evals
uv sync
uv run pytest
uv run rav-evals golden/hanukkah.draft.yaml --answer-fn mymod:answer_fn --sources sources.json
```

`answer_fn(GoldenQuestion) -> RunOutput` (answer + retrieved refs + model + tool calls).
`sources.json` is `{"ref": "text"}` for the in-memory store (tests). Against the real corpus use
`--database-url postgresql+psycopg://...` instead: `PostgresSourceStore` answers V1 (ref exists,
indexable version) and V2 (stored `segment_text.text_b`, same normalization policy as
`rav_evals.normalization`), and every result is logged to the `runs` table (migration 0002).

Integration tests need Postgres (`docker compose up -d` in the repo root); they skip without it,
and fail when `REQUIRE_DB=1` (set in CI).
Reports are written to `--out-dir` as `report.json` and `report.md`.

Golden file `golden/hanukkah.draft.yaml` is a DRAFT: every ref is a `TODO` placeholder and
needs human review. Checks that depend on placeholders are reported as `skipped`.
